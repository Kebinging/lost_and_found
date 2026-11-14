"""Natural-language understanding for the student chatbot.

Two jobs:

* **Intent prediction** - which kind of message is this (greeting, lost-item
  report, opening hours, ...)? A TF-IDF + LogisticRegression pipeline trained
  on ``intents.json`` (see ``train_intents.py``).
* **Slot extraction** - pull the item category, colour, location and date out
  of free text using the keyword dictionaries in ``config`` and ``dateparser``.
"""

from __future__ import annotations

import datetime as dt
import logging
import pickle
import re
from functools import lru_cache

import config

logger = logging.getLogger(__name__)

FALLBACK_INTENT = "fallback"

# --------------------------------------------------------------------------- #
# Text preprocessing
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _lemmatizer():
    """Return an NLTK WordNet lemmatiser, or ``None`` if WordNet is unavailable.

    WordNet is downloaded on first use. If that fails (e.g. no internet) the
    chatbot still works, just without lemmatisation.
    """
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        try:
            lemmatizer.lemmatize("keys")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            lemmatizer.lemmatize("keys")
        return lemmatizer
    except Exception:
        logger.warning("NLTK WordNet unavailable; continuing without lemmatisation.")
        return None


def preprocess(text: str) -> str:
    """Lower-case, strip punctuation and lemmatise ``text``.

    Used both when training the intent model and when predicting, so the two
    always see text in the same form.
    """
    tokens = re.sub(r"[^a-z0-9\s]", " ", str(text).lower()).split()
    lemmatizer = _lemmatizer()
    if lemmatizer is not None:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# Intent prediction
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_intent_model():
    """Load the trained intent pipeline, training it first if it is missing.

    The pickle is a generated local file (it is git-ignored and rebuilt from
    ``intents.json``); if it cannot be loaded - for example after a
    scikit-learn upgrade - it is simply retrained.
    """
    from chatbot import train_intents

    if config.INTENT_MODEL_FILE.exists():
        try:
            with open(config.INTENT_MODEL_FILE, "rb") as handle:
                return pickle.load(handle)
        except Exception:
            logger.warning("Could not load %s; retraining.", config.INTENT_MODEL_FILE)
    return train_intents.train_and_save(verbose=False)


def predict_intent(text: str) -> tuple[str, float]:
    """Return ``(intent, confidence)``; low confidence gives ``("fallback", conf)``."""
    cleaned = preprocess(text)
    if not cleaned:
        return FALLBACK_INTENT, 0.0
    model = load_intent_model()
    probabilities = model.predict_proba([cleaned])[0]
    best = int(probabilities.argmax())
    intent, confidence = str(model.classes_[best]), float(probabilities[best])
    if confidence < config.INTENT_CONFIDENCE:
        return FALLBACK_INTENT, confidence
    return intent, confidence


# --------------------------------------------------------------------------- #
# Slot extraction: keywords
# --------------------------------------------------------------------------- #
def _keyword_matches(text: str, synonyms: dict[str, list[str]]) -> list[tuple[int, int, str]]:
    """Find every synonym in ``text`` as ``(start, end, label)``, left to right.

    Matching is on whole words, tolerates plurals, and when two matches overlap
    ("water bottle" / "bottle") the longer one wins.
    """
    found = []
    for label, words in synonyms.items():
        for word in words:
            pattern = r"\b" + r"\s+".join(re.escape(part) for part in word.split()) + r"(?:s|es)?\b"
            found += [(m.start(), m.end(), label) for m in re.finditer(pattern, text)]

    kept: list[tuple[int, int, str]] = []
    for match in sorted(found, key=lambda m: m[1] - m[0], reverse=True):
        if all(match[1] <= other[0] or match[0] >= other[1] for other in kept):
            kept.append(match)
    return sorted(kept)


def extract_category(text: str) -> str | None:
    """Return the item category mentioned in ``text``, if any.

    In a compound such as "phone charger" or "laptop bag" the last word is the
    actual item, so a category immediately followed by another is ignored.
    """
    lowered = text.lower()
    matches = _keyword_matches(lowered, config.CATEGORY_SYNONYMS)
    heads = [
        match for index, match in enumerate(matches)
        if not (index + 1 < len(matches)
                and matches[index + 1][2] != match[2]
                and not lowered[match[1]:matches[index + 1][0]].strip())
    ]
    return heads[0][2] if heads else None


def extract_colour(text: str) -> str | None:
    """Return the first colour mentioned in ``text``, if any."""
    matches = _keyword_matches(text.lower(), config.COLOUR_SYNONYMS)
    return matches[0][2] if matches else None


def extract_location(text: str) -> str | None:
    """Return the first campus location mentioned in ``text``, if any."""
    matches = _keyword_matches(text.lower(), config.LOCATION_SYNONYMS)
    return matches[0][2] if matches else None


# --------------------------------------------------------------------------- #
# Slot extraction: dates
# --------------------------------------------------------------------------- #
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
          r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "couple of": 2, "three": 3,
                 "few": 3, "four": 4, "five": 5, "six": 6, "seven": 7}

# Phrases rewritten into something dateparser understands unambiguously.
_RELATIVE_REWRITES = [
    (r"\bday before yesterday\b", "2 days ago"),
    (r"\b(?:yesterday|last night)\b", "yesterday"),
    (r"\b(?:today|this morning|this afternoon|this evening|tonight|just now)\b", "today"),
    (r"\blast week\b", "1 week ago"),
    (r"\blast month\b", "1 month ago"),
]
_AGO = re.compile(r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")\s+(day|week|month)s?\s+ago\b")
_WEEKDAY = re.compile(r"\b(" + "|".join(_WEEKDAYS) + r")\b")
_EXPLICIT_DATES = [
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),                                   # 2026-11-12
    re.compile(r"\b\d{1,2}[/.-]\d{1,2}(?:[/.-]\d{2,4})?\b"),                    # 12/11, 12/11/2026
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s+({_MONTH})\b(?:,?\s+(\d{{4}}))?"),  # 12 Nov
    re.compile(rf"\b({_MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?"),            # Nov 12
]


def _dateparser_parse(phrase: str, today: dt.date) -> dt.date | None:
    """Parse one date phrase with ``dateparser`` (dates resolve to the past, D/M order)."""
    import dateparser  # imported lazily: it is slow to import

    parsed = dateparser.parse(phrase, languages=["en"], settings={
        "PREFER_DATES_FROM": "past",
        "DATE_ORDER": "DMY",
        "RELATIVE_BASE": dt.datetime.combine(today, dt.time(12, 0)),
    })
    return parsed.date() if parsed else None


def extract_date(text: str, today: dt.date | None = None) -> dt.date | None:
    """Return the date mentioned in ``text``, if any (never a future date).

    Understands "yesterday", "last night", "3 days ago", "last Tuesday",
    "12 Nov", "Nov 12th", "12/11/2026" and "2026-11-12". Only a recognised
    date phrase is handed to ``dateparser``; running it over a whole sentence
    produces too many false positives.
    """
    today = today or dt.date.today()
    lowered = text.lower()
    result: dt.date | None = None

    for pattern, replacement in _RELATIVE_REWRITES:
        if re.search(pattern, lowered):
            result = _dateparser_parse(replacement, today)
            break

    if result is None and (match := _AGO.search(lowered)):
        amount = _NUMBER_WORDS.get(match.group(1)) or int(match.group(1))
        result = _dateparser_parse(f"{amount} {match.group(2)}s ago", today)

    if result is None and (match := _WEEKDAY.search(lowered)):
        # The most recent such weekday strictly before today ("on/last Tuesday").
        days_back = (today.weekday() - _WEEKDAYS.index(match.group(1)) - 1) % 7 + 1
        result = today - dt.timedelta(days=days_back)

    if result is None:
        for index, pattern in enumerate(_EXPLICIT_DATES):
            if not (match := pattern.search(lowered)):
                continue
            if index == 2:    # "12th of nov [2026]" -> "12 nov [2026]"
                phrase = " ".join(g for g in match.groups() if g)
            elif index == 3:  # "nov 12th[, 2026]" -> "12 nov [2026]"
                month, day, year = match.groups()
                phrase = " ".join(g for g in (day, month, year) if g)
            else:
                phrase = match.group(0)
            result = _dateparser_parse(phrase, today)
            if result is not None:
                break

    return result if result is not None and result <= today else None


def extract_slots(text: str, today: dt.date | None = None) -> dict:
    """Extract ``category``, ``colour``, ``location`` and ``date_lost`` from ``text``.

    Slots that are not mentioned are ``None``. ``today`` can be fixed for tests.
    """
    return {
        "category": extract_category(text),
        "colour": extract_colour(text),
        "location": extract_location(text),
        "date_lost": extract_date(text, today),
    }
