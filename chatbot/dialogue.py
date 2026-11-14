"""Conversation state machine for the student chatbot.

``LostItemDialogue`` walks a student through reporting a lost item::

    IDLE -> COLLECTING -> ASK_PHOTO -> SEARCHING -> RESULTS -> CLAIM_FORM -> DONE

* COLLECTING - fill the slots (category is required; colour, date and
  location may be answered with "not sure"), one follow-up question at a time.
* ASK_PHOTO - optional reference photo of a similar item.
* SEARCHING / RESULTS - run the matcher and announce the outcome.
* CLAIM_FORM - on a match, collect name, student ID and contact, then create
  a claim and hand out its reference.

Every message is first normalised (short forms, chat words and typos are
fixed, see ``chatbot/normalise.py``). When the bot is still unsure what the
student meant, it asks "Did you mean ...?" (yes / no) instead of giving up,
and ``quick_replies`` lists buttons the page can show for the current step.

All state lives in a plain mapping supplied by the caller. The Streamlit page
passes a dict kept in ``st.session_state``, so the conversation survives
reruns; the tests pass an ordinary dict. The class itself never touches
Streamlit.

PRIVACY: replies must never contain a found item's ID, photo, description or
location. Matches are stored in the state for the claim, but only their
*count* is ever put into a message.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
from typing import Callable, MutableMapping

import config
from chatbot import nlu, normalise

# Dialogue stages
IDLE = "IDLE"
COLLECTING = "COLLECTING"
ASK_PHOTO = "ASK_PHOTO"
SEARCHING = "SEARCHING"
RESULTS = "RESULTS"
CLAIM_FORM = "CLAIM_FORM"
DONE = "DONE"

# Slots in the order they are asked for. Only the category is required.
SLOT_ORDER = ["category", "colour", "date_lost", "location"]
UNKNOWN = "unknown"   # the student answered "not sure"
MAX_RETRIES = 1       # re-ask an optional slot once, then move on

_UNSURE = re.compile(
    r"\b(not sure|unsure|no idea|don'?t know|do not know|dunno|idk|can'?t remember|"
    r"cannot remember|don'?t remember|forgot|skip|unknown|no clue|n/?a)\b")
_NO = re.compile(r"^\s*(no|nope|nah|n|skip|none|don'?t have( one)?|i don'?t( have( one)?)?|"
                 r"no photo|not really|wrong|not that|bukan|tak)\b")
_YES = re.compile(r"^\s*(yes|yeah|yea|yep|yup|ya|yah|y|sure|ok|okay|i do|i have|"
                  r"correct|right|that's it|betul)\b")
_START_OVER = re.compile(r"\b(start over|start again|restart|reset|cancel)\b")
_LOST_WORDS = re.compile(
    r"\b(lost|lose|losing|missing|misplaced|can'?t find|cannot find|couldn'?t find|"
    r"left my|forgot my|dropped my|looking for my)\b")
_FOUND_WORDS = re.compile(
    r"\b(found|find|picked|pick up|came across|come across|hand in|handed|turn in|"
    r"turned in|left behind|left a|left an|someone|somebody|someones|someone's)\b")
_CLAIM_REFERENCE = re.compile(r"\bCL-?(\d{1,6})\b", re.IGNORECASE)

MATCH_MESSAGE = ("Good news, we have {n} possible match{es}. To avoid unnecessary trips, "
                 "please leave your details and bring proof of ownership to the "
                 "Lost & Found counter.")
NO_MATCH_MESSAGE = "No matching item has been handed in yet. Please check again later."

# Intents the bot may double-check with "Did you want to ...?" when unsure.
INTENT_QUESTIONS = {
    "report_lost_item": "report something you lost",
    "check_claim_status": "check the status of a claim",
    "office_hours": "know the opening hours",
    "office_location": "know where the Lost & Found counter is",
    "how_it_works": "know how this assistant works",
    "found_an_item": "hand in something you found",
}
CONFIRM_MIN_CONFIDENCE = 0.2   # below this the bot does not even guess

# Buttons shown when the bot is idle.
MENU_REPLIES = ["I lost something", "Opening hours", "Where is the counter?",
                "Check my claim", "How does this work?"]


def _load_responses() -> dict[str, list[str]]:
    with open(config.INTENTS_FILE, encoding="utf-8") as handle:
        return {intent["tag"]: intent["responses"] for intent in json.load(handle)["intents"]}


class LostItemDialogue:
    """Slot-filling dialogue for one student conversation.

    Parameters
    ----------
    state:
        Mapping that holds the conversation state between messages (pass the
        same one each time, e.g. a dict stored in ``st.session_state``).
    find_matches, add_claim, load_claims, predict_intent, extract_slots, rank_intents:
        Dependencies, injectable for testing. They default to the real
        matcher, storage and NLU functions. (If ``predict_intent`` is
        replaced but ``rank_intents`` is not, the bot never guesses an intent.)
    """

    def __init__(
        self,
        state: MutableMapping | None = None,
        *,
        find_matches: Callable | None = None,
        add_claim: Callable | None = None,
        load_claims: Callable | None = None,
        predict_intent: Callable | None = None,
        extract_slots: Callable | None = None,
        rank_intents: Callable | None = None,
    ) -> None:
        self.state = state if state is not None else {}
        self._find_matches = find_matches
        self._add_claim = add_claim
        self._load_claims = load_claims
        self._predict_intent = predict_intent or nlu.predict_intent
        self._extract_slots = extract_slots or nlu.extract_slots
        self._rank_intents = rank_intents or (nlu.rank_intents if predict_intent is None
                                              else (lambda text: []))
        self._responses = _load_responses()
        if "stage" not in self.state:
            self.reset()

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Forget the current conversation ("Start over")."""
        self.state.clear()
        self.state.update({
            "stage": IDLE,
            "slots": {slot: None for slot in SLOT_ORDER},
            "pending_slot": None,     # slot the bot has just asked about
            "retries": {},            # slot -> number of failed answers
            "description": [],        # the student's own words, for CLIP and the claim
            "photo_embedding": None,
            "matches": [],            # [{"item_id", "score"}] - NEVER shown to the student
            "claim_id": None,
            "confirm": None,          # open "Did you mean ...?" question, see _ask_confirm
        })

    @property
    def stage(self) -> str:
        return self.state["stage"]

    @property
    def awaiting_photo(self) -> bool:
        """True while the bot is waiting for the optional reference photo."""
        return self.stage == ASK_PHOTO

    @property
    def awaiting_claim_form(self) -> bool:
        """True while the claim form (name, student ID, contact) should be shown."""
        return self.stage == CLAIM_FORM

    @property
    def quick_replies(self) -> list[str]:
        """Suggested answers for the current step, for the page to show as buttons.

        Each one is ordinary text the bot understands when sent as a message.
        """
        if self.state.get("confirm"):
            return ["Yes", "No"]
        if self.stage == COLLECTING:
            return {
                "category": list(config.CATEGORIES),
                "colour": list(config.COLOUR_PALETTE) + ["Not sure"],
                "date_lost": ["Today", "Yesterday", "2 days ago", "Last week", "Not sure"],
                "location": list(config.LOCATIONS) + ["Not sure"],
            }.get(self.state["pending_slot"], [])
        if self.stage == ASK_PHOTO:
            return ["No photo"]
        if self.stage in (IDLE, DONE):
            return list(MENU_REPLIES)
        return []

    def _canned(self, intent: str) -> str:
        return random.choice(self._responses.get(intent) or self._responses[nlu.FALLBACK_INTENT])

    # ------------------------------------------------------------------ #
    # Entry points
    # ------------------------------------------------------------------ #
    def handle_message(self, text: str) -> list[str]:
        """Process one chat message from the student; return the bot's replies."""
        text = text.strip()
        if not text:
            return []
        # Fix short forms, chat words and typos once; everything below uses this.
        text = normalise.normalise(text).text or text.lower()
        if _START_OVER.search(text) and self.stage not in (IDLE, DONE):
            self.reset()
            return ["No problem, let's start again. What did you lose?"]

        if self.state.get("confirm"):
            replies = self._answer_confirm(text)
            if replies is not None:
                return replies

        if self.stage == COLLECTING:
            return self._collect(text)
        if self.stage == ASK_PHOTO:
            return self._photo_answer(text)
        if self.stage == CLAIM_FORM:
            return ["Please fill in the form below so staff can contact you, "
                    "or press \"Start over\" to cancel."]
        return self._idle(text)

    def attach_photo(self, photo_embedding) -> list[str]:
        """Use the student's reference photo (as a CLIP vector) and run the search."""
        if not self.awaiting_photo:
            return []
        self.state["photo_embedding"] = photo_embedding
        return ["Thanks, I'll use your photo in the search."] + self._search()

    def submit_claim(self, name: str, student_id: str, contact: str) -> list[str]:
        """Create a claim for the best match and return the confirmation message."""
        if not self.awaiting_claim_form or not self.state["matches"]:
            return []
        name, student_id, contact = name.strip(), student_id.strip(), contact.strip()
        if not (name and student_id and contact):
            return ["Please fill in your name, student ID and a contact (email or phone)."]

        add_claim = self._add_claim
        if add_claim is None:
            from core import storage
            add_claim = storage.add_claim

        best = self.state["matches"][0]
        claim_id = add_claim({
            "item_id": best["item_id"],
            "student_name": name,
            "student_id": student_id,
            "contact": contact,
            "student_description": " ".join(self.state["description"]),
            "match_score": best["score"],
        })
        self.state["claim_id"] = claim_id
        self.state["stage"] = DONE
        return [f"Thank you, {name}. Your claim reference is **{claim_id}**. Please bring "
                "your student ID and proof of ownership to the Lost & Found counter, "
                "where staff will verify the item with you."]

    # ------------------------------------------------------------------ #
    # IDLE / DONE: intents
    # ------------------------------------------------------------------ #
    def _idle(self, text: str) -> list[str]:
        reference = _CLAIM_REFERENCE.search(text)
        if reference:
            return [self._claim_status(f"CL-{int(reference.group(1)):04d}")]

        intent, _ = self._predict_intent(text)
        slots = self._extract_slots(text)
        # "lost and found" is the office's name, not a statement about losing something.
        lowered = re.sub(r"lost\s*(?:and|&|n)\s*found", " ", text.lower())
        # The classifier is trained on little data, so back it up with rules:
        # explicit "lost"-type wording, or an item named in an unclear message.
        is_report = (
            intent == "report_lost_item"
            or (_LOST_WORDS.search(lowered) and intent != "found_an_item")
            or (intent == nlu.FALLBACK_INTENT and slots.get("category"))
        )
        if is_report:
            return self._start_report(text, slots)
        # Handing in an item normally has found-type wording; without it ("wallet",
        # "i fund a phone") the classifier may be right or wrong, so ask.
        if intent == "found_an_item" and not _FOUND_WORDS.search(lowered):
            return self._ask_confirm(
                {"kind": "intent", "intent": intent, "text": text},
                f"Just to check: did you want to {INTENT_QUESTIONS[intent]}?")
        if intent == nlu.FALLBACK_INTENT:
            item = nlu.suggest_slot(text, "category")
            if item:   # e.g. "brela" -> "Did you lose your umbrella?"
                return self._ask_confirm(
                    {"kind": "intent", "intent": "report_lost_item",
                     "text": f"i lost my {item.lower()}"},
                    f"Did you mean you lost your **{item.lower()}**?")
            guess = self._guess_intent(text)
            if guess:
                return self._ask_confirm(
                    {"kind": "intent", "intent": guess, "text": text},
                    f"Sorry, I'm not sure I understood. Did you want to {INTENT_QUESTIONS[guess]}?")
        return [self._canned(intent)]

    def _start_report(self, text: str, slots: dict) -> list[str]:
        self.reset()
        self.state["stage"] = COLLECTING
        self.state["description"].append(text)
        self._merge_slots(slots)
        return [self._canned("report_lost_item")] + self._advance()

    def _guess_intent(self, text: str) -> str | None:
        """The most likely intent if the classifier leans towards one, else ``None``.

        Used when the classifier is not confident enough to act on its own
        but has a reasonable guess, so the bot can ask instead of giving up.
        """
        ranked = self._rank_intents(text)
        if not ranked:
            return None
        intent, confidence = ranked[0]
        if intent in INTENT_QUESTIONS and confidence >= CONFIRM_MIN_CONFIDENCE:
            return intent
        return None

    # ------------------------------------------------------------------ #
    # "Did you mean ...?"
    # ------------------------------------------------------------------ #
    def _ask_confirm(self, question: dict, message: str) -> list[str]:
        """Remember an open yes/no question and return it as the bot's reply.

        ``question`` is ``{"kind": "slot", "slot": ..., "value": ...}`` or
        ``{"kind": "intent", "intent": ..., "text": <the original message>}``.
        """
        self.state["confirm"] = question
        return [message]

    def _answer_confirm(self, text: str) -> list[str] | None:
        """Handle the reply to a "Did you mean ...?" question.

        Returns ``None`` when the student ignored the question and wrote
        something else; that message is then handled normally.
        """
        question = self.state["confirm"]
        self.state["confirm"] = None
        if _YES.search(text):
            if question["kind"] == "slot":
                slot, value = question["slot"], question["value"]
                self.state["slots"][slot] = value
                self.state["retries"].pop(slot, None)
                self.state["description"].append(value.lower())
                return [f"Got it: {value.lower()}."] + self._advance()
            if question["intent"] == "report_lost_item":
                return self._start_report(question["text"], self._extract_slots(question["text"]))
            return [self._canned(question["intent"])]
        if _NO.search(text):
            if question["kind"] == "slot":
                return ["Okay, my mistake. " + self._question(question["slot"])
                        + " You can also pick one of the buttons below."]
            return ["Sorry about that. Could you say it another way? "
                    "You can also pick one of the buttons below."]
        return None

    def _claim_status(self, claim_id: str) -> str:
        load_claims = self._load_claims
        if load_claims is None:
            from core import storage
            load_claims = storage.load_claims
        claims = load_claims()
        row = claims[claims["claim_id"] == claim_id]
        if row.empty:
            return f"I couldn't find a claim with the reference {claim_id}. Please check it and try again."
        status = row.iloc[0]["status"]
        detail = {
            "Pending": "Staff have not verified it yet. Please bring proof of ownership to the counter.",
            "Verified": "The item has been verified and released.",
            "Rejected": "Staff could not confirm ownership. Please visit the counter if you think this is a mistake.",
        }.get(status, "")
        return f"Claim {claim_id} is **{status}**. {detail}".strip()

    # ------------------------------------------------------------------ #
    # COLLECTING: slot filling
    # ------------------------------------------------------------------ #
    def _merge_slots(self, slots: dict) -> None:
        """Copy newly extracted slot values in without overwriting known ones."""
        for slot in SLOT_ORDER:
            if slots.get(slot) and self.state["slots"][slot] in (None, UNKNOWN):
                self.state["slots"][slot] = slots[slot]

    def _collect(self, text: str) -> list[str]:
        pending = self.state["pending_slot"]
        slots = self._extract_slots(text)
        if pending == "date_lost" and not slots.get("date_lost"):
            slots["date_lost"] = self._date_from_answer(text)
        if pending == "location" and not slots.get("location") and text.strip().lower() == "other":
            slots["location"] = "Other"
        self._merge_slots(slots)

        replies: list[str] = []
        answered = pending and self.state["slots"][pending] not in (None, UNKNOWN)
        if answered or any(slots.values()):
            self.state["description"].append(text)

        if pending and not answered:
            unsure = bool(_UNSURE.search(text.lower()))
            guess = None if unsure else nlu.suggest_slot(text, pending)
            if guess:
                return self._ask_confirm({"kind": "slot", "slot": pending, "value": guess},
                                         f"Did you mean **{guess.lower()}**?")
            if pending == "category":
                # Required slot: keep asking, and explain what can be searched.
                options = ", ".join(c.lower() for c in config.CATEGORIES)
                prefix = ("I need to know what the item is to search for it. "
                          if unsure else "Sorry, I didn't recognise that item. ")
                return [prefix + f"I can search for: {options}. Which one is it? "
                        "If your item isn't listed, please ask at the Lost & Found counter."]
            retries = self.state["retries"].get(pending, 0)
            if unsure or retries >= MAX_RETRIES:
                self.state["slots"][pending] = UNKNOWN
                replies.append("No problem, we'll leave that out.")
            else:
                self.state["retries"][pending] = retries + 1
                return ["Sorry, I didn't catch that. " + self._question(pending)
                        + " (You can say \"not sure\".)"]

        return replies + self._advance()

    @staticmethod
    def _date_from_answer(text: str) -> dt.date | None:
        """Parse a direct answer to "when did you lose it?" more leniently."""
        try:
            parsed = nlu._dateparser_parse(text, dt.date.today())
        except Exception:
            return None
        return parsed if parsed and parsed <= dt.date.today() else None

    def _question(self, slot: str) -> str:
        item = self.state["slots"]["category"]
        item = item.lower() if item and item != UNKNOWN else "item"
        return {
            "category": "What kind of item did you lose? (for example: bottle, umbrella, "
                        "calculator, charger, key, phone, bag, laptop or book)",
            "colour": f"What colour is your {item}?",
            "date_lost": f"When did you lose your {item}? (for example: yesterday, last Tuesday, 12 Nov)",
            "location": "Where do you think you lost it? ("
                        + ", ".join(config.LOCATIONS) + ")",
        }[slot]

    def _advance(self) -> list[str]:
        """Ask for the next missing slot, or move on to the photo question."""
        for slot in SLOT_ORDER:
            if self.state["slots"][slot] is None:
                self.state["pending_slot"] = slot
                return [self._question(slot)]
        self.state["pending_slot"] = None
        self.state["stage"] = ASK_PHOTO
        return ["Do you have a photo of a similar item? (optional) "
                "You can upload one in the sidebar, or type \"no\" to continue."]

    # ------------------------------------------------------------------ #
    # ASK_PHOTO / SEARCHING / RESULTS
    # ------------------------------------------------------------------ #
    def _photo_answer(self, text: str) -> list[str]:
        lowered = text.lower()
        if _NO.search(lowered) or _UNSURE.search(lowered):
            return self._search()
        if _YES.search(lowered):
            return ["Great - please upload it using the photo box in the sidebar."]
        return ["Please upload a photo in the sidebar, or type \"no\" to search without one."]

    def _query(self) -> dict:
        """Build the matcher query from the collected slots."""
        slots = {k: (None if v == UNKNOWN else v) for k, v in self.state["slots"].items()}
        return {
            "category": slots["category"],
            "colour": slots["colour"],
            "location": slots["location"],
            "date_lost": slots["date_lost"],
            "free_text": " ".join(self.state["description"]),
            "photo_embedding": self.state["photo_embedding"],
        }

    def _search(self) -> list[str]:
        self.state["stage"] = SEARCHING
        find_matches = self._find_matches
        if find_matches is None:
            from core import matcher
            find_matches = matcher.find_matches

        matches = find_matches(self._query())
        # Keep only what the claim needs; explanations stay out of the session.
        self.state["matches"] = [{"item_id": m.item_id, "score": m.score} for m in matches]
        self.state["stage"] = RESULTS

        if not matches:
            self.state["stage"] = DONE
            return [NO_MATCH_MESSAGE]
        self.state["stage"] = CLAIM_FORM
        count = len(matches)
        return [MATCH_MESSAGE.format(n=count, es="" if count == 1 else "es")]
