"""Tests for core/classifier.py (how YOLO and CLIP are combined).

Only the decision logic is tested, with made-up CLIP scores, so the CLIP
model is not needed.
"""

import numpy as np
import pytest

import config
from core import classifier
from core.colour import nearest_palette_colour

YOLO_KNOWS = {"Bottle", "Umbrella", "Calculator", "Charger", "Key", "Phone", "Bag", "Laptop", "Book"}


@pytest.mark.parametrize("yolo, clip, expected", [
    # CLIP is sure it is something YOLO was never trained on -> CLIP wins.
    ("Charger", [("Earbuds", 0.85), ("Charger", 0.10)], ("Earbuds", "clip")),
    # CLIP disagrees about a class YOLO *was* trained on -> trust YOLO.
    ("Bottle", [("Phone", 0.80), ("Bottle", 0.15)], ("Bottle", "yolo")),
    # CLIP not confident enough to overrule YOLO.
    ("Charger", [("Earbuds", 0.50), ("Charger", 0.40)], ("Charger", "yolo")),
    # YOLO found nothing; CLIP fairly sure.
    (None, [("ID Card", 0.55), ("Wallet", 0.30)], ("ID Card", "clip")),
    # Nobody is sure -> Other, staff decide.
    (None, [("Toy", 0.25), ("Bag", 0.20)], ("Other", "other")),
    (None, [], ("Other", "other")),
])
def test_decide(yolo, clip, expected):
    assert classifier.decide(yolo, clip, YOLO_KNOWS) == expected


def test_scores_are_probabilities_best_first():
    labels = ["Bottle", "Bottle", "Phone", "Earbuds"]
    similarities = np.array([0.20, 0.30, 0.25, 0.10])   # Bottle's best phrase is 0.30
    scores = classifier.scores_from_similarities(labels, similarities)
    assert [name for name, _ in scores] == ["Bottle", "Phone", "Earbuds"]
    assert sum(p for _, p in scores) == pytest.approx(1.0)


def test_every_category_except_other_has_prompts():
    assert set(config.CATEGORY_PROMPTS) == set(config.CATEGORIES) - {config.OTHER_CATEGORY}
    assert config.CATEGORIES[-1] == config.OTHER_CATEGORY


@pytest.mark.parametrize("rgb, expected", [
    ((44, 56, 69), "Blue"),       # dim blue bottle photographed indoors (was "Black")
    ((108, 136, 178), "Blue"),    # was "Grey"
    ((62, 62, 62), "Black"),
    ((132, 119, 118), "Grey"),
    ((200, 200, 205), "Silver"),
    ((255, 182, 193), "Pink"),
    ((90, 50, 30), "Brown"),
])
def test_colour_names_from_real_photos(rgb, expected):
    assert nearest_palette_colour(rgb) == expected
