"""Dominant colour extraction for a cropped object.

The crop is shrunk, its border trimmed (the border is mostly background),
the remaining pixels are clustered with KMeans, and the biggest cluster is
mapped to the nearest named colour in ``config.COLOUR_PALETTE``. "Nearest" is
measured in LAB space, where distances track how different colours look to a
person far better than they do in RGB.
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


def nearest_palette_colour(rgb: tuple[float, float, float]) -> str:
    """Return the palette colour name closest to ``rgb`` (LAB Euclidean distance)."""
    names = list(config.COLOUR_PALETTE)
    palette_lab = _rgb_to_lab(np.array([config.COLOUR_PALETTE[n] for n in names]))
    target_lab = _rgb_to_lab(np.array([rgb]))[0]
    distances = np.linalg.norm(palette_lab - target_lab, axis=1)
    return names[int(np.argmin(distances))]


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
