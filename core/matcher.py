"""Score and rank found items against a student's lost-item query.

Pipeline (see ``find_matches``):

1. Take only items whose status is ``Unclaimed``.
2. Hard filters - the category must match, and the item must have been handed
   in between one day before the loss and ``DATE_WINDOW_DAYS`` after it.
3. Score every remaining item between 0 and 1 from image similarity (if the
   student uploaded a photo), text similarity, colour and location.
4. Return the items scoring at least ``MATCH_THRESHOLD``, best first.

The explanations attached to each match are for staff and debugging only and
must never be shown to students.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import config
from core import storage
from core.embedder import cosine

logger = logging.getLogger(__name__)


@dataclass
class Match:
    """One candidate found item with its score breakdown."""

    item_id: str
    score: float
    explanation: str
    components: dict[str, float] = field(default_factory=dict)


def colour_score(query_colour: str | None, item_colour: str | None) -> float:
    """1 for the same colour, 0.5 for easily-confused colours, otherwise 0."""
    if not query_colour or not item_colour:
        return 0.0
    if query_colour == item_colour:
        return 1.0
    if any(query_colour in group and item_colour in group for group in config.COLOUR_GROUPS):
        return 0.5
    return 0.0


def location_score(query_location: str | None, item_location: str | None) -> float:
    """1 if both locations are known and equal, otherwise 0."""
    return 1.0 if query_location and query_location == item_location else 0.0


def scale_text_similarity(raw: float) -> float:
    """Stretch a raw CLIP text-image cosine onto 0..1 (see ``config.TEXT_SIM_RANGE``)."""
    low, high = config.TEXT_SIM_RANGE
    return float(np.clip((raw - low) / (high - low), 0.0, 1.0))


def _default_free_text(query: dict) -> str:
    """Fallback description built from the slots when the student gave no text."""
    words = [query.get("colour"), query.get("category")]
    return " ".join(w.lower() for w in words if w)


def _within_date_window(date_found: dt.date | None, date_lost: dt.date) -> bool:
    if date_found is None:
        return True  # an item without a date cannot be ruled out
    earliest = date_lost - dt.timedelta(days=1)
    latest = date_lost + dt.timedelta(days=config.DATE_WINDOW_DAYS)
    return earliest <= date_found <= latest


def _explain(components: dict[str, float], weights: dict[str, float],
             query: dict, item: dict, has_embedding: bool) -> str:
    """Human-readable score breakdown for staff."""
    parts = [f"category {item['category']} OK"]
    if "image" in weights:
        parts.append(f"photo similarity {components['image']:.2f}")
    parts.append(f"text similarity {components['text']:.2f}"
                 if has_embedding else "no embedding stored (text/photo = 0)")
    if not query.get("colour"):
        parts.append("colour not given")
    elif components["colour"] == 1.0:
        parts.append(f"colour {item['colour']} matches")
    elif components["colour"] == 0.5:
        parts.append(f"colour {query['colour']} is similar to {item['colour']}")
    else:
        parts.append(f"colour differs ({query['colour']} vs {item['colour']})")
    if not query.get("location"):
        parts.append("location not given")
    elif components["location"] == 1.0:
        parts.append(f"location {item['location_found']} matches")
    else:
        parts.append(f"location differs ({query['location']} vs {item['location_found']})")
    return "; ".join(parts)


def find_matches(
    query: dict,
    *,
    threshold: float | None = None,
    max_results: int | None = None,
    embed_text_fn: Callable[[str], np.ndarray] | None = None,
) -> list[Match]:
    """Return the found items that best match ``query``, best first.

    ``query`` keys (all optional): ``category``, ``colour``, ``location``,
    ``date_lost`` (``datetime.date``), ``free_text`` and ``photo_embedding``
    (a CLIP vector of the student's reference photo).

    ``threshold`` and ``max_results`` default to ``config.MATCH_THRESHOLD`` and
    ``config.MAX_RESULTS``; the Inventory page passes ``threshold=0`` to see
    every score. ``embed_text_fn`` defaults to the CLIP text encoder and exists
    so the tests can run without loading the model.
    """
    threshold = config.MATCH_THRESHOLD if threshold is None else threshold
    max_results = config.MAX_RESULTS if max_results is None else max_results

    # 1. Only unclaimed items can be matched.
    items = storage.load_found_items(status="Unclaimed")

    # 2. Hard filters.
    if query.get("category"):
        items = items[items["category"] == query["category"]]
    if query.get("date_lost"):
        items = items[items["date_found"].map(
            lambda found: _within_date_window(found, query["date_lost"]))]
    if items.empty:
        return []

    # 3. Score.
    embeddings = storage.load_embeddings()
    photo_embedding = query.get("photo_embedding")
    weights = (config.MATCH_WEIGHTS_WITH_PHOTO if photo_embedding is not None
               else config.MATCH_WEIGHTS_NO_PHOTO)

    text_embedding = None
    free_text = (query.get("free_text") or "").strip() or _default_free_text(query)
    if free_text and embeddings:
        try:
            if embed_text_fn is None:
                from core.embedder import embed_text as embed_text_fn
            text_embedding = embed_text_fn(free_text)
        except Exception:  # CLIP unavailable: degrade to colour + location only
            logger.exception("Text embedding failed; text similarity set to 0.")

    matches: list[Match] = []
    for item in items.to_dict("records"):
        item_embedding = embeddings.get(item["item_id"])
        has_embedding = item_embedding is not None

        components = {
            "text": 0.0,
            "colour": colour_score(query.get("colour"), item["colour"]),
            "location": location_score(query.get("location"), item["location_found"]),
        }
        if has_embedding and text_embedding is not None:
            components["text"] = scale_text_similarity(cosine(text_embedding, item_embedding))
        if "image" in weights:
            components["image"] = (
                float(np.clip(cosine(photo_embedding, item_embedding), 0.0, 1.0))
                if has_embedding else 0.0
            )

        score = sum(weights[name] * components[name] for name in weights)
        if score >= threshold:
            matches.append(Match(
                item_id=item["item_id"],
                score=round(float(score), 4),
                explanation=_explain(components, weights, query, item, has_embedding),
                components={k: round(v, 4) for k, v in components.items()},
            ))

    # 4. Best first, capped.
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[:max_results]
