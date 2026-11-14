"""Tests for chatbot/normalise.py (short forms, typos) and the messy-input accuracy."""

import pytest

from chatbot import normalise as norm


@pytest.mark.parametrize("text, expected", [
    ("lost my blck hp at caf ytd", "lost my black phone at cafeteria yesterday"),   # short forms + typo
    ("i lsot my umbrela in the libary", "i lost my umbrella in the library"),       # swapped letters, typos
    ("cnt find my calculater", "can't find my calculator"),
    ("hilang botol hitam semalam", "lost bottle black yesterday"),                  # Malay
    ("lost my keyy lah bro", "lost my key"),                                        # repeated letters, fillers
    ("pleaseee help", "please help"),
    ("opning hrs?", "opening hours?"),                                             # punctuation kept
])
def test_normalise(text, expected):
    assert norm.normalise(text).text == expected


@pytest.mark.parametrize("text", [
    "my CL-0007 claim",          # claim reference untouched
    "lost it on 12/11 or the 3rd",  # dates untouched
])
def test_numbers_and_references_are_kept(text):
    assert norm.normalise(text).text == text.lower()


def test_changes_are_reported():
    changes = norm.normalise("lost my hp at libary").changes
    assert ("hp", "phone", "short form") in changes
    assert ("libary", "library", "spelling") in changes


def test_real_words_are_not_corrected():
    # Needs WordNet; without it the protection is weaker, so skip rather than fail.
    if norm.ensure_wordnet() is None:
        pytest.skip("WordNet not available")
    assert norm.normalise("I sat at the bottom table").text == "i sat at the bottom table"


@pytest.mark.parametrize("a, b, distance", [
    ("lost", "lost", 0),
    ("lsot", "lost", 1),         # neighbouring letters swapped = one edit
    ("umbrela", "umbrella", 1),
    ("calculater", "calculator", 1),
    ("bottom", "bottle", 2),
])
def test_edit_distance(a, b, distance):
    assert norm.edit_distance(a, b) == distance


def test_short_words_are_not_guessed():
    assert norm.correct_word("abc") is None


@pytest.mark.parametrize("text, expected", [
    ("calcu", "Calculator"),     # start of a word
    ("brela", "Umbrella"),       # too far to auto-correct, close enough to ask
    ("wallet", None),            # not a supported item
])
def test_suggest(text, expected):
    import config
    assert norm.suggest(text, config.CATEGORY_SYNONYMS) == expected


def test_messy_messages_are_understood():
    """The improved chatbot must stay accurate on chatbot/messy_test_set.json."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import evaluate_chatbot

    results = evaluate_chatbot.evaluate()
    for part, accuracy in results["after"].items():
        assert accuracy >= 0.9, f"{part} accuracy dropped to {accuracy:.0%}"
        assert accuracy >= results["before"][part]
