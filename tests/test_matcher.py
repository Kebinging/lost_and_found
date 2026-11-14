"""Tests for core/matcher.py using fake items and hand-made embeddings.

No CLIP model is loaded: embeddings are small unit vectors and the text
encoder is replaced by ``fake_embed_text``.
"""

import datetime as dt

import numpy as np
import pytest
from PIL import Image

import config
from core import matcher, storage

LOST_ON = dt.date(2026, 11, 10)


def _unit(index: int) -> np.ndarray:
    vector = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
    vector[index] = 1.0
    return vector


def fake_embed_text(text: str) -> np.ndarray:
    """Pretend text encoder: every description 'looks like' embedding 0."""
    return _unit(0)


def _add(category, colour, location, date_found, embedding_index=None) -> str:
    image = Image.new("RGB", (10, 10))
    item_id = storage.add_found_item(
        {"category": category, "colour": colour, "location_found": location,
         "date_found": date_found}, image, image)
    if embedding_index is not None:
        storage.save_embedding(item_id, _unit(embedding_index))
    return item_id


@pytest.fixture
def inventory(temp_storage):
    return {
        "target": _add("Umbrella", "Black", "Library", LOST_ON, embedding_index=0),
        "other_umbrella": _add("Umbrella", "Yellow", "Cafeteria", LOST_ON, embedding_index=1),
        "bottle": _add("Bottle", "Black", "Library", LOST_ON, embedding_index=0),
    }


def _query(**overrides) -> dict:
    query = {"category": "Umbrella", "colour": "Black", "location": "Library",
             "date_lost": LOST_ON, "free_text": "black umbrella"}
    query.update(overrides)
    return query


def find(query, **kwargs):
    return matcher.find_matches(query, embed_text_fn=fake_embed_text, **kwargs)


def test_correct_item_ranks_first(inventory):
    results = find(_query(), threshold=0.0)
    assert [m.item_id for m in results] == [inventory["target"], inventory["other_umbrella"]]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].score > results[1].score


def test_wrong_category_is_filtered_out(inventory):
    ids = [m.item_id for m in find(_query(), threshold=0.0)]
    assert inventory["bottle"] not in ids


def test_threshold_removes_weak_matches(inventory):
    assert [m.item_id for m in find(_query())] == [inventory["target"]]


def test_no_match_when_nothing_fits(inventory):
    assert find(_query(category="Laptop")) == []


def test_date_window(inventory):
    # Handed in one day before the reported loss date is still allowed...
    assert find(_query(date_lost=LOST_ON + dt.timedelta(days=1)))
    # ...two days before is not, and neither is beyond the window.
    assert not find(_query(date_lost=LOST_ON + dt.timedelta(days=2)))
    assert not find(_query(date_lost=LOST_ON - dt.timedelta(days=config.DATE_WINDOW_DAYS + 1)))
    # Unknown date: no date filter.
    assert find(_query(date_lost=None))


def test_only_unclaimed_items_are_matched(inventory):
    storage.update_item(inventory["target"], status="Pending Claim")
    assert inventory["target"] not in [m.item_id for m in find(_query(), threshold=0.0)]


def test_similar_colour_scores_half(inventory):
    exact = find(_query(), threshold=0.0)[0]
    similar = find(_query(colour="Grey"), threshold=0.0)[0]
    assert similar.components["colour"] == 0.5
    assert similar.score == pytest.approx(
        exact.score - 0.5 * config.MATCH_WEIGHTS_NO_PHOTO["colour"])


def test_photo_changes_the_weights(inventory):
    result = find(_query(photo_embedding=_unit(0)), threshold=0.0)[0]
    assert result.item_id == inventory["target"]
    assert result.components["image"] == pytest.approx(1.0)
    assert result.score == pytest.approx(1.0)

    # A photo of something different drags the score down by the image weight.
    wrong_photo = find(_query(photo_embedding=_unit(5)), threshold=0.0)[0]
    assert wrong_photo.score == pytest.approx(1.0 - config.MATCH_WEIGHTS_WITH_PHOTO["image"])


def test_item_without_embedding_still_scores_on_colour_and_location(temp_storage):
    item_id = _add("Key", "Silver", "Lab", LOST_ON)  # no embedding saved
    result = find(_query(category="Key", colour="Silver", location="Lab"), threshold=0.0)[0]
    assert result.item_id == item_id
    assert result.components["text"] == 0.0
    weights = config.MATCH_WEIGHTS_NO_PHOTO
    assert result.score == pytest.approx(weights["colour"] + weights["location"])


def test_max_results(temp_storage):
    for _ in range(5):
        _add("Bottle", "Blue", "Hostel", LOST_ON, embedding_index=0)
    query = _query(category="Bottle", colour="Blue", location="Hostel")
    assert len(find(query)) == config.MAX_RESULTS
    assert len(find(query, max_results=10)) == 5


def test_text_similarity_scaling():
    low, high = config.TEXT_SIM_RANGE
    assert matcher.scale_text_similarity(low - 0.1) == 0.0
    assert matcher.scale_text_similarity(high + 0.1) == 1.0
    assert matcher.scale_text_similarity((low + high) / 2) == pytest.approx(0.5)
