"""Tests for scripts/merge_datasets.py on tiny synthetic datasets."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import merge_datasets as merge  # noqa: E402


def make_source(folder: Path, names: list[str], images: dict[str, list[str]]) -> Path:
    """Write a YOLO dataset: ``images`` maps a file stem to its label lines."""
    (folder / "train" / "images").mkdir(parents=True)
    (folder / "train" / "labels").mkdir(parents=True)
    (folder / "data.yaml").write_text(yaml.safe_dump({"names": names}), encoding="utf-8")
    for stem, lines in images.items():
        (folder / "train" / "images" / f"{stem}.jpg").write_bytes(f"{folder.name}-{stem}".encode())
        (folder / "train" / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    return folder


@pytest.fixture
def settings(tmp_path):
    make_source(tmp_path / "earbuds", ["Ear_Piece", "junk"], {
        "a": ["0 0.5 0.5 0.2 0.2"],
        "b": ["1 0.5 0.5 0.2 0.2"],                     # only a dropped class -> background
        "c": ["0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3"],     # polygon label
    })
    make_source(tmp_path / "wallets", ["Wallet"], {"w1": ["0 0.4 0.4 0.1 0.1"],
                                                  "w2": ["0 0.6 0.6 0.1 0.1"]})
    make_source(tmp_path / "campus", ["Wallet"], {"real": ["0 0.5 0.5 0.5 0.5"]})
    return {
        "output": str(tmp_path / "combined"),
        "split": {"train": 0.5, "valid": 0.25, "test": 0.25},
        "seed": 1, "max_images_per_class": 100, "background_fraction": 0.5,
        "sources": [
            {"key": "earbuds", "path": str(tmp_path / "earbuds"),
             "classes": {"Ear_Piece": "Earbuds", "junk": None}},
            {"key": "wallets", "path": str(tmp_path / "wallets"), "classes": {"wallet": "Wallet"}},
            {"key": "campus", "path": str(tmp_path / "campus"), "split": "real_test",
             "classes": {"Wallet": "Wallet"}},
        ],
    }


def read_all(settings):
    return [item for s in settings["sources"] for item in merge.read_source(s)]


def test_classes_are_renamed_and_dropped(settings):
    items = {item.image.stem: item for item in read_all(settings)}
    assert items["a"].labels[0][0] == "Earbuds"
    assert items["b"].labels == []                          # "junk" dropped
    assert items["w1"].classes == {"Wallet"}                # case-insensitive match


def test_polygon_becomes_bounding_box(settings):
    item = next(i for i in read_all(settings) if i.image.stem == "c")
    assert item.labels[0][1:] == pytest.approx((0.2, 0.2, 0.2, 0.2))


def test_unmapped_class_stops_the_merge(settings):
    settings["sources"][0]["classes"].pop("junk")
    with pytest.raises(SystemExit, match="junk"):
        merge.read_source(settings["sources"][0])


def test_duplicates_are_removed(settings, tmp_path):
    items = read_all(settings)
    copy = merge.Item(image=items[0].image, source="again")
    kept, dropped = merge.remove_duplicates(items + [copy])
    assert dropped == 1 and len(kept) == len(items)


def test_balance_caps_images_per_class():
    items = [merge.Item(Path(f"{i}.jpg"), "s", [("Wallet", .5, .5, .1, .1)]) for i in range(10)]
    items += [merge.Item(Path(f"bg{i}.jpg"), "s") for i in range(10)]
    kept = merge.balance(items, max_per_class=4, background_fraction=0.2, seed=0)
    assert sum(1 for i in kept if i.labels) == 4
    assert sum(1 for i in kept if not i.labels) == 1        # 20% of 5 images


def test_full_merge_writes_yolo_dataset(settings):
    items, _ = merge.remove_duplicates(read_all(settings))
    kept = merge.balance(items, 100, 0.5, 1)
    splits = merge.assign_splits(kept, settings["split"], 1)
    out = Path(settings["output"])
    summary = merge.write_dataset(splits, out, settings)

    assert summary["classes"] == ["Earbuds", "Wallet"]      # config.CATEGORIES order
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == ["Earbuds", "Wallet"] and data["val"] == "valid/images"
    # Own photos only in the separate real-world test set.
    assert [p.name for p in (out / "real_test" / "images").iterdir()] == ["campus_real.jpg"]
    assert yaml.safe_load((out / "data_real_test.yaml").read_text())["test"] == "real_test/images"
    label = (out / "real_test" / "labels" / "campus_real.txt").read_text().split()
    assert label[0] == "1"                                  # Wallet's new class ID
    assert (out / "SOURCES.md").exists()


def test_sources_file_is_valid():
    settings = merge.load_settings()
    keys = [s["key"] for s in settings["sources"]]
    assert len(keys) == len(set(keys))


def test_unmapped_classes_are_listed_together(settings):
    settings["sources"][0]["classes"].pop("junk")
    assert merge.unmapped_classes(settings["sources"][0]) == ["junk"]
    assert merge.unmapped_classes(settings["sources"][1]) == []
