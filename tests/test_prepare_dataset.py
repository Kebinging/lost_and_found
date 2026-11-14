"""Tests for scripts/prepare_dataset.py on a tiny synthetic dataset.

A wrong class remap would silently poison training, so the cleaning logic is
checked here without needing the real 4,546-image download.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prepare_dataset  # noqa: E402

# Deliberately NOT in the final order, with "None" in the middle.
RAW_NAMES = ["Bottle", "Calculator", "Charger", "Key", "None", "Umbrella"]


@pytest.fixture
def raw_dataset(tmp_path):
    """100 fake images: 77 in train/, 23 in test/, no valid/ (like the original)."""
    raw = tmp_path / "raw"
    for split, count in (("train", 77), ("test", 23)):
        (raw / split / "images").mkdir(parents=True)
        (raw / split / "labels").mkdir(parents=True)
        for index in range(count):
            stem = f"{split}_{index:03d}"
            (raw / split / "images" / f"{stem}.jpg").write_bytes(b"fake image")
            class_id = index % len(RAW_NAMES)
            # Every image also carries one "None" label, which must be removed.
            (raw / split / "labels" / f"{stem}.txt").write_text(
                f"{class_id} 0.5 0.5 0.2 0.2\n4 0.1 0.1 0.05 0.05\n")
    (raw / "data.yaml").write_text(yaml.safe_dump({"nc": 6, "names": RAW_NAMES}))
    return raw


def test_id_map_follows_names_not_positions():
    id_map = prepare_dataset.build_id_map(
        RAW_NAMES, ["Bottle", "Umbrella", "Calculator", "Charger", "Key"], "None")
    assert id_map == {0: 0, 5: 1, 1: 2, 2: 3, 3: 4}   # raw ID 4 ("None") is absent


def test_unexpected_class_is_rejected():
    with pytest.raises(SystemExit):
        prepare_dataset.build_id_map(["Bottle", "Wallet"], ["Bottle"], "None")


def test_remap_label_lines_drops_removed_classes():
    lines = ["5 0.5 0.5 0.2 0.2", "4 0.1 0.1 0.1 0.1", ""]
    assert prepare_dataset.remap_label_lines(lines, {5: 1}) == ["1 0.5 0.5 0.2 0.2"]


def test_prepare_resplits_and_cleans(raw_dataset, tmp_path):
    out = tmp_path / "clean"
    summary = prepare_dataset.prepare(raw_dataset, out)

    assert summary["images"] == {"train": 70, "valid": 20, "test": 10}
    for split, count in summary["images"].items():
        assert len(list((out / split / "images").iterdir())) == count
        assert len(list((out / split / "labels").iterdir())) == count

    # No label may reference a class outside 0..4, and counts must add up.
    label_ids = [int(line.split()[0])
                 for label in out.rglob("labels/*.txt")
                 for line in label.read_text().splitlines()]
    assert set(label_ids) <= {0, 1, 2, 3, 4}
    assert sum(summary["instances"].values()) == len(label_ids)
    assert summary["instances"]["Umbrella"] == label_ids.count(1) > 0

    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == ["Bottle", "Umbrella", "Calculator", "Charger", "Key"]
    assert data["nc"] == 5
    assert (data["train"], data["val"], data["test"]) == (
        "train/images", "valid/images", "test/images")


def test_split_is_reproducible(raw_dataset, tmp_path):
    prepare_dataset.prepare(raw_dataset, tmp_path / "a")
    prepare_dataset.prepare(raw_dataset, tmp_path / "b")
    names = lambda root: sorted(p.name for p in (root / "test" / "images").iterdir())  # noqa: E731
    assert names(tmp_path / "a") == names(tmp_path / "b")
