"""CLIP embeddings for images and text.

CLIP maps pictures and sentences into the same 512-dimensional space, so a
photo of a found item can be compared both with another photo and with a
written description. The model is loaded once and reused.

``sentence-transformers`` (and PyTorch) are imported lazily so that modules
which only need ``cosine`` - and the unit tests - do not pay the import cost.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image

import config


@lru_cache(maxsize=1)
def get_model():
    """Load the CLIP model once (downloads ~600 MB on first use)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.CLIP_MODEL_NAME)


def embed_image(pil_image: Image.Image) -> np.ndarray:
    """Return the L2-normalised CLIP embedding of an image."""
    vector = get_model().encode(pil_image.convert("RGB"), normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vector, dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
    """Return the L2-normalised CLIP embedding of a piece of text."""
    vector = get_model().encode(text, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vector, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors (0.0 if either is all zeros)."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norm) if norm else 0.0
