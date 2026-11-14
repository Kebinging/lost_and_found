"""Dominant colour extraction for a cropped object.

The crop is shrunk, its border trimmed (the border is mostly background),
the remaining pixels are clustered with KMeans, and the biggest cluster is
given a colour name from ``config.COLOUR_PALETTE``.

The name comes from the colour's **hue, saturation and brightness** (HSV):
greyish colours are named by brightness (Black / Grey / Silver / White) and
the rest by hue (Red, Orange, Yellow, ...). Indoor phone photos are often
dim, and a dim blue bottle (RGB 44, 56, 69) is still clearly blue by hue,
whereas the plain nearest-colour distance used before called it Black.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

import config

_RESIZE = 100            # crop is resized to _RESIZE x _RESIZE pixels
_CENTRE_FRACTION = 0.8   # keep the central 80% in each direction
_CLUSTERS = 3


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an ``(N, 3)`` array of 0-255 RGB values to true-scale LAB."""
    scaled = (np.asarray(rgb, dtype=np.float32) / 255.0).reshape(-1, 1, 3)
    return cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def nearest_palette_colour_lab(rgb: tuple[float, float, float]) -> str:
    """The original method: palette colour closest to ``rgb`` in LAB space (kept for comparison)."""
    names = list(config.COLOUR_PALETTE)
    palette_lab = _rgb_to_lab(np.array([config.COLOUR_PALETTE[n] for n in names]))
    target_lab = _rgb_to_lab(np.array([rgb]))[0]
    distances = np.linalg.norm(palette_lab - target_lab, axis=1)
    return names[int(np.argmin(distances))]


# Below this saturation a colour counts as grey-ish (named by brightness only).
_GREY_SATURATION = 0.18
# (upper brightness limit, name) for grey-ish colours, darkest first.
_GREY_LEVELS = [(0.25, "Black"), (0.62, "Grey"), (0.85, "Silver"), (1.01, "White")]
# (upper hue limit in degrees, name) for coloured pixels.
_HUES = [(15, "Red"), (40, "Orange"), (70, "Yellow"), (170, "Green"), (260, "Blue"),
         (300, "Purple"), (345, "Pink"), (361, "Red")]


def nearest_palette_colour(rgb: tuple[float, float, float]) -> str:
    """Return the colour name for ``rgb`` from its hue, saturation and brightness."""
    r, g, b = (float(v) / 255.0 for v in rgb)
    brightest, darkest = max(r, g, b), min(r, g, b)
    value = brightest
    saturation = 0.0 if brightest == 0 else (brightest - darkest) / brightest

    if value < 0.12:
        return "Black"
    if saturation < _GREY_SATURATION:
        return next(name for limit, name in _GREY_LEVELS if value < limit)

    hue, _, _ = colorsys_rgb_to_hsv(r, g, b)
    name = next(name for limit, name in _HUES if hue * 360 < limit)
    # Dark orange / red is what people call brown; pale saturated red is pink.
    if name in ("Orange", "Red") and value < 0.6 and saturation > 0.3:
        return "Brown" if name == "Orange" or value < 0.45 else name
    if name == "Red" and value > 0.8 and saturation < 0.5:
        return "Pink"
    return name


def colorsys_rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    import colorsys

    return colorsys.rgb_to_hsv(r, g, b)


def dominant_colour(crop: Image.Image | np.ndarray) -> str:
    """Return the name of the main colour of ``crop`` (a PIL image or RGB array)."""
    image = crop if isinstance(crop, Image.Image) else Image.fromarray(np.asarray(crop))
    image = image.convert("RGB").resize((_RESIZE, _RESIZE))
    pixels = np.asarray(image)

    margin = int(round(_RESIZE * (1 - _CENTRE_FRACTION) / 2))
    centre = pixels[margin:_RESIZE - margin, margin:_RESIZE - margin]
    samples = centre.reshape(-1, 3).astype(np.float32)

    # A flat-coloured crop has fewer distinct colours than clusters requested.
    n_clusters = min(_CLUSTERS, len(np.unique(samples, axis=0)))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(samples)
    largest = int(np.bincount(kmeans.labels_).argmax())
    return nearest_palette_colour(tuple(kmeans.cluster_centers_[largest]))
