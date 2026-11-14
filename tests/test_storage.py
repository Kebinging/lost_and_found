"""Tests for core/storage.py (all run against a temporary directory)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import config
from core import storage


def _add_item(**overrides) -> str:
    record = {
        "category": "Bottle",
        "colour": "Black",
        "description": "Black bottle with sticker",
        "location_found": "Library",
        "date_found": dt.date(2026, 11, 10),
        "confidence": 0.91,
        "detector": "custom",
    }
    record.update(overrides)
    return storage.add_found_item(record, Image.new("RGB", (64, 48), "black"),
                                  Image.new("RGB", (20, 20), "black"))


def test_init_creates_folders_and_empty_tables(temp_storage):
    assert config.FOUND_ITEMS_FILE.exists()
    assert config.CLAIMS_FILE.exists()
    assert config.IMAGES_DIR.is_dir() and config.CROPS_DIR.is_dir()

    items, claims = storage.load_found_items(), storage.load_claims()
    assert items.empty and list(items.columns) == storage.ITEM_COLUMNS
    assert claims.empty and list(claims.columns) == storage.CLAIM_COLUMNS


def test_init_does_not_overwrite_existing_data(temp_storage):
    _add_item()
    storage.init_storage()
    assert len(storage.load_found_items()) == 1


def test_add_and_load_item(temp_storage):
    item_id = _add_item()
    assert item_id == "LF-0001"
    assert (config.IMAGES_DIR / "LF-0001.jpg").exists()
    assert (config.CROPS_DIR / "LF-0001.jpg").exists()

    item = storage.load_found_items().iloc[0]
    assert item["item_id"] == "LF-0001"
    assert item["category"] == "Bottle"
    assert item["date_found"] == dt.date(2026, 11, 10)
    assert item["confidence"] == pytest.approx(0.91)
    assert item["status"] == "Unclaimed"
    assert isinstance(item["created_at"], dt.datetime)
    assert storage.resolve_path(item["image_path"]).exists()


def test_ids_auto_increment(temp_storage):
    assert [_add_item(), _add_item(), _add_item()] == ["LF-0001", "LF-0002", "LF-0003"]
    assert storage.next_item_id() == "LF-0004"
    assert storage.next_claim_id() == "CL-0001"


def test_load_items_filtered_by_status(temp_storage):
    first, second = _add_item(), _add_item()
    storage.update_item(second, status="Claimed")
    assert storage.load_found_items(status="Unclaimed")["item_id"].tolist() == [first]
    assert storage.load_found_items(status="Claimed")["item_id"].tolist() == [second]


def test_update_item(temp_storage):
    item_id = _add_item()
    storage.update_item(item_id, colour="Grey", status="Disposed")
    item = storage.load_found_items().iloc[0]
    assert (item["colour"], item["status"]) == ("Grey", "Disposed")

    with pytest.raises(KeyError):
        storage.update_item("LF-9999", status="Claimed")
    with pytest.raises(ValueError):
        storage.update_item(item_id, status="Stolen")
    with pytest.raises(ValueError):
        storage.update_item(item_id, not_a_column=1)


def test_add_claim_marks_item_pending(temp_storage):
    item_id = _add_item()
    claim_id = storage.add_claim({
        "item_id": item_id,
        "student_name": "Ada Lovelace",
        "student_id": "0012345",          # leading zeros must survive Excel
        "contact": "ada@example.edu",
        "student_description": "black bottle, lost in the library",
        "match_score": 0.78,
    })
    assert claim_id == "CL-0001"

    claim = storage.load_claims().iloc[0]
    assert claim["item_id"] == item_id
    assert claim["student_id"] == "0012345"
    assert claim["status"] == "Pending"
    assert claim["match_score"] == pytest.approx(0.78)
    assert storage.load_found_items().iloc[0]["status"] == "Pending Claim"


def test_add_claim_for_unknown_item_fails(temp_storage):
    with pytest.raises(KeyError):
        storage.add_claim({"item_id": "LF-9999"})
    assert storage.load_claims().empty


def test_update_claim(temp_storage):
    claim_id = storage.add_claim({"item_id": _add_item(), "student_name": "Ada"})
    storage.update_claim(claim_id, status="Verified")
    assert storage.load_claims().iloc[0]["status"] == "Verified"
    with pytest.raises(KeyError):
        storage.update_claim("CL-9999", status="Verified")
    with pytest.raises(ValueError):
        storage.update_claim(claim_id, status="Maybe")


def test_embeddings_round_trip(temp_storage):
    assert storage.load_embeddings() == {}
    vector = np.arange(config.EMBEDDING_DIM, dtype=np.float32)
    storage.save_embedding("LF-0001", vector)
    storage.save_embedding("LF-0002", vector * 2)

    loaded = storage.load_embeddings()
    assert set(loaded) == {"LF-0001", "LF-0002"}
    np.testing.assert_array_equal(loaded["LF-0001"], vector)


def test_locked_workbook_gives_friendly_error(temp_storage, monkeypatch):
    """Excel locks an open workbook; that must surface as a clear message."""
    def locked(*args, **kwargs):
        raise PermissionError("file is open in another program")

    monkeypatch.setattr(pd, "ExcelWriter", locked)
    with pytest.raises(storage.StorageError,
                       match="Close found_items.xlsx in Excel and try again."):
        _add_item()
    # The failed save must not leave orphan photos behind.
    assert not list(config.IMAGES_DIR.iterdir())
    assert not list(config.CROPS_DIR.iterdir())
