"""Tests for chatbot/nlu.py."""

import datetime as dt

import pytest

from chatbot import nlu

TODAY = dt.date(2026, 11, 14)  # a Saturday


def test_spec_example():
    slots = nlu.extract_slots("I lost my black umbrella at the library yesterday")
    assert slots == {
        "category": "Umbrella",
        "colour": "Black",
        "location": "Library",
        "date_lost": dt.date.today() - dt.timedelta(days=1),
    }


def test_nothing_mentioned():
    assert nlu.extract_slots("hello there") == {
        "category": None, "colour": None, "location": None, "date_lost": None}


@pytest.mark.parametrize("text, expected", [
    ("my water bottle is gone", "Bottle"),
    ("I lost my keys", "Key"),
    ("lost a power bank", "Charger"),
    ("I can't find my iPhone", "Phone"),
    ("my phone charger is missing", "Charger"),   # compound: the charger, not the phone
    ("I lost my laptop bag", "Bag"),              # compound: the bag, not the laptop
    ("left my bag with my laptop inside", "Bag"),  # first item mentioned
    ("lost my notebook", "Book"),
    ("I lost my wallet", "Wallet"),
    ("my airpods case is missing", "Earbuds"),
    ("lost my student id", "ID Card"),
    ("left my specs in the lab", "Spectacles"),
    ("I lost my usb drive", None),                 # unsupported category
])
def test_category(text, expected):
    assert nlu.extract_category(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("a gray tumbler", "Grey"),
    ("it's navy", "Blue"),
    ("dark RED umbrella", "Red"),
    ("an umbrella", None),
])
def test_colour(text, expected):
    assert nlu.extract_colour(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("left it in the canteen", "Cafeteria"),
    ("somewhere in the computer lab", "Lab"),
    ("at the car park", "Car Park"),
    ("in the Lecture Hall", "Lecture Hall"),
    ("no idea where", None),
])
def test_location(text, expected):
    assert nlu.extract_location(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("yesterday", dt.date(2026, 11, 13)),
    ("I lost it last night", dt.date(2026, 11, 13)),
    ("this morning", dt.date(2026, 11, 14)),
    ("3 days ago", dt.date(2026, 11, 11)),
    ("a week ago", dt.date(2026, 11, 7)),
    ("last Tuesday", dt.date(2026, 11, 10)),
    ("on saturday", dt.date(2026, 11, 7)),       # today is Saturday -> the previous one
    ("12 Nov", dt.date(2026, 11, 12)),
    ("on the 3rd of November", dt.date(2026, 11, 3)),
    ("Nov 12th", dt.date(2026, 11, 12)),
    ("10/11/2026", dt.date(2026, 11, 10)),        # day/month order
    ("2026-11-09", dt.date(2026, 11, 9)),
    ("20 Dec", dt.date(2025, 12, 20)),            # never in the future
    ("I lost my umbrella", None),
])
def test_date(text, expected):
    assert nlu.extract_date(text, today=TODAY) == expected


def test_preprocess():
    assert nlu.preprocess("  I LOST my Keys!!  ").split()[:3] == ["i", "lost", "my"]
    assert nlu.preprocess("?!") == ""


@pytest.mark.parametrize("text, expected", [
    ("I lost my umbrella", "report_lost_item"),
    ("hello", "greeting"),
    ("what are your opening hours", "office_hours"),
    ("where is the lost and found office", "office_location"),
    ("thank you so much", "thanks"),
])
def test_intent(text, expected):
    intent, confidence = nlu.predict_intent(text)
    assert intent == expected
    assert 0.0 <= confidence <= 1.0


def test_gibberish_falls_back():
    assert nlu.predict_intent("zxqv wkrp")[0] == "fallback"
    assert nlu.predict_intent("")[0] == "fallback"
