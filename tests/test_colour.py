"""Tests for core/colour.py."""

import numpy as np
import pytest
from PIL import Image

from core.colour import dominant_colour, nearest_palette_colour


@pytest.mark.parametrize("rgb, expected", [
    ((255, 0, 0), "Red"),
    ((0, 0, 0), "Black"),
    ((255, 255, 255), "White"),
    ((0, 0, 255), "Blue"),
])
def test_solid_image(rgb, expected):
    assert dominant_colour(Image.new("RGB", (120, 80), rgb)) == expected


def test_accepts_numpy_array():
    assert dominant_colour(np.zeros((50, 50, 3), dtype=np.uint8)) == "Black"


def test_background_border_is_ignored():
    """A red object on a white background: the border must not win."""
    image = Image.new("RGB", (100, 100), "white")
    image.paste(Image.new("RGB", (84, 84), (220, 20, 20)), (8, 8))
    assert dominant_colour(image) == "Red"


def test_largest_cluster_wins():
    image = Image.new("RGB", (100, 100), (20, 60, 200))        # mostly blue
    image.paste(Image.new("RGB", (30, 30), (240, 220, 40)), (35, 35))  # yellow patch
    assert dominant_colour(image) == "Blue"


def test_nearest_palette_colour():
    assert nearest_palette_colour((250, 250, 250)) == "White"
    assert nearest_palette_colour((10, 10, 10)) == "Black"
