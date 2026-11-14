"""Name an item from its picture with CLIP (zero-shot classification).

YOLO can only report the classes it was trained on, so an item outside them
(earbuds, an ID card, a toy) is either missed or forced into the nearest
class. CLIP was trained on hundreds of millions of image-caption pairs and
can compare a picture with *any* written description, without extra
training. This module compares the item's picture with the descriptions in
``config.CATEGORY_PROMPTS`` ("a photo of wireless earbuds", ...) and turns
the similarities into a probability per category.

:func:`decide` then combines YOLO and CLIP into one category:

1. CLIP is confident (``CLIP_OVERRIDE_CONFIDENCE``) that it is a category
   **no YOLO model can detect** -> use CLIP. (YOLO cannot know these, so its
   answer would be a forced guess.)
2. Otherwise, if YOLO found the item -> use YOLO (it was trained for its classes).
3. Otherwise, if CLIP is fairly sure (``CLIP_MIN_CONFIDENCE``) -> use CLIP.
4. Otherwise -> "Other", and staff choose the category.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image

import config


@lru_cache(maxsize=1)
def _prompt_embeddings() -> tuple[list[str], np.ndarray]:
    """Embed every category description once; return ``(category per row, matrix)``."""
    from core import embedder

    labels, texts = [], []
    for category, prompts in config.CATEGORY_PROMPTS.items():
        for prompt in prompts:
            labels.append(category)
            texts.append(config.CLIP_PROMPT_TEMPLATE.format(prompt))
    matrix = np.vstack([embedder.embed_text(text) for text in texts])
    return labels, matrix


def scores_from_similarities(labels: list[str], similarities: np.ndarray) -> list[tuple[str, float]]:
    """Turn description similarities into ``[(category, probability), ...]``, best first.

    Each category keeps its best-matching description; a softmax with CLIP's
    temperature turns the per-category similarities into probabilities that
    add up to 1.
    """
    best: dict[str, float] = {}
    for label, similarity in zip(labels, similarities):
        best[label] = max(best.get(label, -1.0), float(similarity))
    categories = list(best)
    logits = config.CLIP_TEMPERATURE * np.array([best[c] for c in categories])
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return sorted(zip(categories, probabilities.tolist()), key=lambda pair: -pair[1])


def classify(image: Image.Image, image_embedding: np.ndarray | None = None) -> list[tuple[str, float]]:
    """Return ``[(category, probability), ...]`` for a picture of one item, best first.

    Pass ``image_embedding`` if the CLIP vector has already been computed.
    """
    if image_embedding is None:
        from core import embedder

        image_embedding = embedder.embed_image(image)
    labels, matrix = _prompt_embeddings()
    similarities = matrix @ np.asarray(image_embedding, dtype=np.float32).ravel()
    return scores_from_similarities(labels, similarities)


def decide(yolo_category: str | None, clip_scores: list[tuple[str, float]],
           yolo_categories: set[str]) -> tuple[str, str]:
    """Combine YOLO and CLIP into ``(category, source)``; see the module docstring.

    ``yolo_categories`` are the categories the loaded YOLO models can detect.
    ``source`` is "clip", "yolo" or "other".
    """
    clip_category, clip_confidence = clip_scores[0] if clip_scores else (None, 0.0)
    if (clip_category and clip_category not in yolo_categories
            and clip_confidence >= config.CLIP_OVERRIDE_CONFIDENCE):
        return clip_category, "clip"
    if yolo_category:
        return yolo_category, "yolo"
    if clip_category and clip_confidence >= config.CLIP_MIN_CONFIDENCE:
        return clip_category, "clip"
    return config.OTHER_CATEGORY, "other"
