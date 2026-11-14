"""Tests for chatbot/dialogue.py.

The matcher and storage are replaced with fakes, so these tests exercise only
the conversation logic - including the privacy rule that a reply never
reveals anything about a found item.
"""

import datetime as dt

import pandas as pd
import pytest

from chatbot import dialogue as dlg
from chatbot.dialogue import LostItemDialogue
from core.matcher import Match

SECRET_ITEM_ID = "LF-0042"
SECRET_EXPLANATION = "colour Black matches; location Library matches"


class Fakes:
    """Records what the dialogue asks of the matcher and storage."""

    def __init__(self, matches):
        self.matches = matches
        self.queries = []
        self.claims = []

    def find_matches(self, query):
        self.queries.append(query)
        return self.matches

    def add_claim(self, record):
        self.claims.append(record)
        return "CL-0007"

    def load_claims(self):
        return pd.DataFrame([{"claim_id": "CL-0007", "status": "Pending"}])


def make(matches=None):
    fakes = Fakes(matches if matches is not None else
                  [Match(SECRET_ITEM_ID, 0.81, SECRET_EXPLANATION)])
    bot = LostItemDialogue({}, find_matches=fakes.find_matches, add_claim=fakes.add_claim,
                           load_claims=fakes.load_claims)
    return bot, fakes


def test_full_flow_with_match():
    bot, fakes = make()
    transcript = []

    transcript += bot.handle_message("I lost my black umbrella at the library yesterday")
    # Every slot was in the first message, so the bot goes straight to the photo question.
    assert bot.awaiting_photo
    assert "photo" in transcript[-1].lower()

    transcript += bot.handle_message("no")
    assert bot.awaiting_claim_form
    assert "1 possible match." in transcript[-1]

    query = fakes.queries[0]
    assert query["category"] == "Umbrella"
    assert query["colour"] == "Black"
    assert query["location"] == "Library"
    assert query["date_lost"] == dt.date.today() - dt.timedelta(days=1)
    assert query["photo_embedding"] is None
    assert "black umbrella" in query["free_text"]

    transcript += bot.submit_claim("Ada Lovelace", "S123", "ada@example.edu")
    assert bot.stage == dlg.DONE
    assert "CL-0007" in transcript[-1]

    claim = fakes.claims[0]
    assert claim["item_id"] == SECRET_ITEM_ID
    assert claim["student_name"] == "Ada Lovelace"
    assert claim["match_score"] == 0.81

    # Privacy: nothing about the found item may reach the student.
    everything = " ".join(transcript)
    assert SECRET_ITEM_ID not in everything
    assert "LF-" not in everything
    assert SECRET_EXPLANATION not in everything


def test_follow_up_questions_one_at_a_time():
    bot, fakes = make()

    replies = bot.handle_message("I lost my bottle")
    assert bot.stage == dlg.COLLECTING
    assert "colour" in replies[-1].lower()

    replies = bot.handle_message("it's blue")
    assert "when" in replies[-1].lower()

    replies = bot.handle_message("not sure")          # optional slot may be skipped
    assert "where" in replies[-1].lower()

    replies = bot.handle_message("in the cafeteria")
    assert bot.awaiting_photo

    bot.handle_message("no")
    query = fakes.queries[0]
    assert (query["category"], query["colour"], query["location"]) == ("Bottle", "Blue", "Cafeteria")
    assert query["date_lost"] is None


def test_category_is_required():
    bot, _ = make()
    bot.handle_message("I lost something")
    assert bot.state["pending_slot"] == "category"

    replies = bot.handle_message("not sure")
    assert bot.state["pending_slot"] == "category"    # still asking
    assert "bottle" in replies[-1].lower()

    bot.handle_message("my wallet")                   # unsupported item
    assert bot.state["pending_slot"] == "category"

    bot.handle_message("actually it was my calculator")
    assert bot.state["slots"]["category"] == "Calculator"


def test_unrecognised_optional_answer_is_retried_once_then_skipped():
    bot, _ = make()
    bot.handle_message("I lost my umbrella")
    assert bot.state["pending_slot"] == "colour"
    bot.handle_message("sort of stripy")
    assert bot.state["pending_slot"] == "colour"      # asked again
    bot.handle_message("stripy")
    assert bot.state["slots"]["colour"] == dlg.UNKNOWN
    assert bot.state["pending_slot"] == "date_lost"


def test_no_match():
    bot, fakes = make(matches=[])
    bot.handle_message("I lost my grey laptop in the lab yesterday")
    replies = bot.handle_message("no")
    assert replies == [dlg.NO_MATCH_MESSAGE]
    assert bot.stage == dlg.DONE
    assert not bot.awaiting_claim_form
    assert bot.submit_claim("Ada", "S1", "a@b.c") == []
    assert fakes.claims == []


def test_plural_match_message():
    bot, _ = make(matches=[Match("LF-0001", 0.9, ""), Match("LF-0002", 0.7, "")])
    bot.handle_message("I lost my black umbrella at the library yesterday")
    assert "2 possible matches." in bot.handle_message("skip")[-1]


def test_photo_is_passed_to_the_matcher():
    bot, fakes = make()
    assert bot.attach_photo("early") == []            # ignored unless a photo was asked for
    bot.handle_message("I lost my black umbrella at the library yesterday")
    bot.attach_photo("EMBEDDING")
    assert fakes.queries[0]["photo_embedding"] == "EMBEDDING"
    assert bot.awaiting_claim_form


def test_claim_form_requires_all_fields():
    bot, fakes = make()
    bot.handle_message("I lost my black umbrella at the library yesterday")
    bot.handle_message("no")
    assert "fill in" in bot.submit_claim("Ada", "", "ada@example.edu")[0]
    assert bot.awaiting_claim_form and fakes.claims == []


def test_start_over_resets():
    bot, _ = make()
    bot.handle_message("I lost my bottle")
    bot.handle_message("start over")
    assert bot.stage == dlg.IDLE
    assert bot.state["slots"]["category"] is None


def test_new_report_after_done_starts_fresh():
    bot, _ = make(matches=[])
    bot.handle_message("I lost my grey laptop in the lab yesterday")
    bot.handle_message("no")
    bot.handle_message("I lost my bottle")
    assert bot.stage == dlg.COLLECTING
    assert bot.state["slots"]["category"] == "Bottle"
    assert bot.state["slots"]["colour"] is None       # nothing carried over


def test_state_survives_between_instances():
    """The Streamlit page rebuilds the object on every rerun around the same dict."""
    state = {}
    LostItemDialogue(state).handle_message("I lost my bottle")
    assert LostItemDialogue(state).stage == dlg.COLLECTING


@pytest.mark.parametrize("text, expected_fragment", [
    ("what are your opening hours", "9:00"),
    ("where is the lost and found office", "B1"),
    ("I found a phone in the lab", "counter"),
])
def test_other_intents_get_canned_responses(text, expected_fragment):
    bot, fakes = make()
    assert expected_fragment in bot.handle_message(text)[0]
    assert bot.stage == dlg.IDLE and fakes.queries == []


def test_claim_status_lookup():
    bot, _ = make()
    assert "Pending" in bot.handle_message("what is the status of CL-0007?")[0]
    assert "couldn't find" in bot.handle_message("check claim cl-99")[0]


# --------------------------------------------------------------------------- #
# Messy input, "Did you mean ...?" and quick replies
# --------------------------------------------------------------------------- #
def test_short_forms_and_typos_fill_every_slot():
    bot, fakes = make()
    bot.handle_message("lost my blck hp at caf ytd")
    assert bot.awaiting_photo
    bot.handle_message("No photo")
    query = fakes.queries[0]
    assert (query["category"], query["colour"], query["location"]) == ("Phone", "Black", "Cafeteria")
    assert query["date_lost"] == dt.date.today() - dt.timedelta(days=1)


def test_unclear_item_asks_did_you_mean():
    bot, _ = make()
    bot.handle_message("I lost something")
    assert "Phone" in bot.quick_replies                      # item buttons
    reply = bot.handle_message("calcu")
    assert "Did you mean **calculator**?" in reply[0]
    assert bot.quick_replies == ["Yes", "No"]
    bot.handle_message("yes")
    assert bot.state["slots"]["category"] == "Calculator"
    assert bot.state["pending_slot"] == "colour"
    assert "Not sure" in bot.quick_replies                   # colour buttons


def test_saying_no_to_did_you_mean_asks_again():
    bot, _ = make()
    bot.handle_message("I lost something")
    bot.handle_message("calcu")
    reply = bot.handle_message("no")
    assert bot.state["slots"]["category"] is None
    assert "What kind of item" in reply[0]


def test_ignoring_did_you_mean_handles_the_new_message():
    bot, _ = make()
    bot.handle_message("I lost something")
    bot.handle_message("calcu")
    bot.handle_message("umbrella")
    assert bot.state["slots"]["category"] == "Umbrella"
    assert bot.state["confirm"] is None


def test_lone_misspelt_item_offers_a_report():
    bot, _ = make()
    assert "lost your **umbrella**" in bot.handle_message("brela")[0]
    bot.handle_message("yes")
    assert bot.stage == dlg.COLLECTING
    assert bot.state["slots"]["category"] == "Umbrella"


def test_unsure_intent_is_confirmed_before_answering():
    bot = LostItemDialogue({}, predict_intent=lambda t: ("fallback", 0.3),
                           rank_intents=lambda t: [("office_hours", 0.3), ("greeting", 0.2)])
    assert "opening hours?" in bot.handle_message("opn tiem?")[0]
    assert "9:00" in bot.handle_message("yes")[0]


def test_no_guess_when_the_message_is_off_topic():
    bot = LostItemDialogue({}, predict_intent=lambda t: ("fallback", 0.9),
                           rank_intents=lambda t: [("fallback", 0.9)])
    bot.handle_message("tell me a joke")
    assert bot.state["confirm"] is None


def test_found_item_without_found_wording_is_double_checked():
    bot, _ = make()
    assert "hand in something you found" in bot.handle_message("wallet")[0]


def test_quick_replies_follow_the_conversation():
    bot, _ = make()
    assert "I lost something" in bot.quick_replies           # idle menu
    bot.handle_message("I lost my umbrella")
    assert "Black" in bot.quick_replies                      # colour step
    for answer in ["Black", "Yesterday", "Library"]:
        bot.handle_message(answer)
    assert bot.quick_replies == ["No photo"]
    bot.handle_message("No photo")
    assert bot.quick_replies == []                           # claim form is showing


@pytest.mark.parametrize("button", dlg.MENU_REPLIES)
def test_every_menu_button_is_understood(button):
    bot, _ = make()
    reply = bot.handle_message(button)
    assert reply and "didn't understand" not in reply[0]
