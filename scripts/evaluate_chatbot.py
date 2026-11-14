"""Compare the original chatbot with the improved one on messy messages.

Runs every case in ``chatbot/messy_test_set.json`` (typos, short forms, chat
words and Malay words) through two versions of the chatbot's understanding:

* **Before** - the original: whole-word intent model, and item / colour /
  location / date found only by exact keywords.
* **After** - the improved version: short-form dictionary + spelling
  correction (``chatbot/normalise.py``), and the intent model with letter
  n-grams and typo augmentation (``chatbot/train_intents.py``).

It prints the accuracy of each part (useful for the report's Discussion) and
lists the messages the improved version still gets wrong.

Usage (from the project root)::

    python scripts/evaluate_chatbot.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

import config  # noqa: E402
from chatbot import nlu, train_intents  # noqa: E402
from chatbot.normalise import normalise  # noqa: E402

SLOTS = ["category", "colour", "location", "date"]


def load_cases() -> tuple[dt.date, list[dict]]:
    with open(config.MESSY_TEST_FILE, encoding="utf-8") as handle:
        data = json.load(handle)
    return dt.date.fromisoformat(data["today"]), data["cases"]


def classify(model, text: str) -> str:
    """The same decision rule as ``nlu.predict_intent``, for any trained model."""
    probabilities = model.predict_proba([nlu.preprocess(text)])[0]
    best = int(probabilities.argmax())
    if probabilities[best] < config.INTENT_CONFIDENCE:
        return nlu.FALLBACK_INTENT
    return str(model.classes_[best])


def understand(text: str, today: dt.date, model, improved: bool) -> dict:
    """Return the intent and slots one version of the chatbot extracts from ``text``."""
    if improved:
        text = normalise(text).text
    return {
        "intent": classify(model, text),
        "category": nlu.extract_category(text),
        "colour": nlu.extract_colour(text),
        "location": nlu.extract_location(text),
        "date": nlu.extract_date(text, today),
    }


def expected(case: dict, today: dt.date) -> dict:
    offset = case.get("date_offset")
    return {
        "intent": case["intent"],
        "category": case.get("category"),
        "colour": case.get("colour"),
        "location": case.get("location"),
        "date": None if offset is None else today - dt.timedelta(days=offset),
    }


def evaluate() -> dict:
    """Return ``{"before": {...}, "after": {...}, "mistakes": [...]}`` accuracies (0-1)."""
    today, cases = load_cases()
    models = {
        "before": train_intents.train(verbose=False, improved=False),
        "after": train_intents.train(verbose=False, improved=True),
    }
    reports = [case for case in cases if case["intent"] == "report_lost_item"]
    results: dict = {"cases": len(cases), "reports": len(reports), "mistakes": []}

    for version, model in models.items():
        improved = version == "after"
        correct = {"intent": 0, **{slot: 0 for slot in SLOTS}}
        for case in cases:
            got, want = understand(case["text"], today, model, improved), expected(case, today)
            correct["intent"] += got["intent"] == want["intent"]
            if case["intent"] == "report_lost_item":
                for slot in SLOTS:
                    correct[slot] += got[slot] == want[slot]
            if improved:
                wrong = [k for k in got if got[k] != want[k]
                         and (k == "intent" or case["intent"] == "report_lost_item")]
                if wrong:
                    results["mistakes"].append((case["text"], {k: (got[k], want[k]) for k in wrong}))
        results[version] = {"intent": correct["intent"] / len(cases),
                            **{slot: correct[slot] / len(reports) for slot in SLOTS}}
    return results


def main() -> None:
    results = evaluate()
    print(f"Messy test set: {results['cases']} messages "
          f"({results['reports']} lost-item reports used for the slot scores)\n")
    print(f"{'':<22}{'Before':>8}{'After':>8}")
    for key, label in [("intent", "Intent"), ("category", "Item category"),
                       ("colour", "Colour"), ("location", "Location"), ("date", "Date")]:
        print(f"{label + ' accuracy':<22}{results['before'][key]:>8.0%}{results['after'][key]:>8.0%}")
    if results["mistakes"]:
        print("\nStill wrong after the improvements (got, expected):")
        for text, wrong in results["mistakes"]:
            print(f"  {text!r}: {wrong}")


if __name__ == "__main__":
    main()
