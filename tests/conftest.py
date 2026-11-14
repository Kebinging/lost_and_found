"""Shared pytest setup.

* Puts the project root on ``sys.path`` so the tests can ``import config``,
  ``core`` and ``chatbot`` however pytest is launched.
* ``temp_storage`` points every data path at a temporary folder, so tests
  never touch the real ``data/`` directory.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from core import storage  # noqa: E402


@pytest.fixture
def temp_storage(tmp_path, monkeypatch):
    """Redirect all storage to ``tmp_path`` and initialise it; return the folder."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "FOUND_ITEMS_FILE", tmp_path / "found_items.xlsx")
    monkeypatch.setattr(config, "CLAIMS_FILE", tmp_path / "claims.xlsx")
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(config, "CROPS_DIR", tmp_path / "crops")
    monkeypatch.setattr(config, "EMBEDDINGS_FILE", tmp_path / "embeddings.npy")
    storage.init_storage()
    return tmp_path
