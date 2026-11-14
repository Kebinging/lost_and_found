"""Combine several YOLO datasets into one training set for the custom model.

The datasets and how their classes map onto our categories are listed in
``dataset_sources.yaml``. This script:

1. (optional, ``--download``) downloads every source that is not on disk yet
   from Roboflow in YOLOv8 format, using the ``ROBOFLOW_API_KEY`` environment
   variable.
2. Reads each source's class names from its own ``data.yaml`` and renames
   them to our categories ("Ear_Piece" -> Earbuds); classes mapped to ``null``
   are dropped. A class missing from the map stops the script.
3. Removes duplicate images (same file content in two datasets).
4. (optional, ``--pseudo-label``) **pseudo-labelling**: each dataset only
   labels its own classes, so a phone in a wallet photo is unlabelled and
   would teach the model that phones are background. The current YOLO models
   label such objects (confidence >= 0.6) for the categories that a source
   does not cover.
5. Balances the classes: at most ``max_images_per_class`` images per class,
   plus ``background_fraction`` images with no objects at all.
6. Splits into train / valid / test (70 / 20 / 10). Sources marked
   ``split: real_test`` (your own campus photos) all go into a separate
   ``real_test`` set that is never trained on.
7. Writes ``data.yaml`` (and ``data_real_test.yaml``), plus ``SOURCES.md``
   listing every dataset and licence for the report's references.

Usage (from the project root)::

    python scripts/merge_datasets.py --download
    python scripts/merge_datasets.py --pseudo-label     # slower; best on a GPU
    python scripts/train.py --data datasets/combined/data.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                    # project root
sys.path.insert(0, str(Path(__file__).parent))   # scripts/

import config  # noqa: E402
import prepare_dataset  # noqa: E402

SOURCES_FILE = ROOT / "dataset_sources.yaml"
REAL_TEST = "real_test"
PSEUDO_LABEL_CONF = 0.6


@dataclass
class Item:
    """One image with its labels, already renamed to our categories."""

    image: Path
    source: str
    labels: list[tuple[str, float, float, float, float]] = field(default_factory=list)
    split: str | None = None   # forced split ("real_test") or None

    @property
    def classes(self) -> set[str]:
        return {label[0] for label in self.labels}


# --------------------------------------------------------------------------- #
# Reading the sources
# --------------------------------------------------------------------------- #
def load_settings(path: Path = SOURCES_FILE) -> dict:
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    targets = set(config.CATEGORIES) - {config.OTHER_CATEGORY}
    for source in settings["sources"]:
        wrong = {t for t in source["classes"].values() if t is not None and t not in targets}
        if wrong:
            raise SystemExit(f"Source {source['key']!r} maps to unknown categories {sorted(wrong)}; "
                             f"use one of {sorted(targets)} or null.")
    return settings


def source_dir(source: dict) -> Path:
    path = Path(source.get("path") or Path("datasets") / "sources" / source["key"])
    return path if path.is_absolute() else ROOT / path


def download_source(source: dict, folder: Path) -> None:
    """Download one source from Roboflow (YOLOv8 format) into ``folder``."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set the ROBOFLOW_API_KEY environment variable first "
                         "(never write the key into a file).")
    from roboflow import Roboflow  # optional dependency: pip install roboflow

    info = source["roboflow"]
    project = Roboflow(api_key=api_key).workspace(info["workspace"]).project(info["project"])
    project.version(info["version"]).download("yolov8", location=str(folder))


def build_name_map(source_names: list[str], mapping: dict, key: str) -> dict[int, str | None]:
    """Map each class ID of a source to our category (``None`` = drop).

    Exact names are tried first, then case-insensitive. Every class must be
    in the map, so a dataset that changed its classes cannot slip through.
    """
    lower = {str(name).strip().lower(): target for name, target in mapping.items()}
    id_map: dict[int, str | None] = {}
    for class_id, name in enumerate(source_names):
        if name in mapping:
            id_map[class_id] = mapping[name]
        elif name.strip().lower() in lower:
            id_map[class_id] = lower[name.strip().lower()]
        else:
            raise SystemExit(f"Source {key!r} has class {name!r}, which is not in its "
                             "'classes' map in dataset_sources.yaml. Add it (or map it to null).")
    return id_map


def to_box(numbers: list[float]) -> tuple[float, float, float, float] | None:
    """Return ``(x_centre, y_centre, width, height)`` from a box or polygon label line."""
    if len(numbers) == 4:
        return tuple(numbers)  # type: ignore[return-value]
    if len(numbers) >= 6 and len(numbers) % 2 == 0:  # segmentation polygon -> its bounding box
        xs, ys = numbers[0::2], numbers[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    return None


def read_source(source: dict) -> list[Item]:
    """Read every image of one source with its labels renamed to our categories."""
    folder = source_dir(source)
    names = prepare_dataset.read_class_names(prepare_dataset.find_data_yaml(folder))
    id_map = build_name_map(names, source["classes"], source["key"])

    items = []
    for image, label in prepare_dataset.collect_pairs(folder):
        item = Item(image=image, source=source["key"], split=source.get("split"))
        lines = label.read_text(encoding="utf-8").splitlines() if label.exists() else []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            target = id_map.get(int(float(parts[0])))
            box = to_box([float(v) for v in parts[1:]])
            if target and box:
                item.labels.append((target, *box))
        items.append(item)
    return items


def remove_duplicates(items: list[Item]) -> tuple[list[Item], int]:
    """Drop images whose file content was already seen; return (kept, number dropped)."""
    seen: set[str] = set()
    kept = []
    for item in items:
        digest = hashlib.sha1(item.image.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            kept.append(item)
    return kept, len(items) - len(kept)


# --------------------------------------------------------------------------- #
# Pseudo-labelling
# --------------------------------------------------------------------------- #
def _iou(a, b) -> float:
    def corners(box):
        x, y, w, h = box
        return x - w / 2, y - h / 2, x + w / 2, y + h / 2
    ax1, ay1, ax2, ay2 = corners(a)
    bx1, by1, bx2, by2 = corners(b)
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def pseudo_label(items: list[Item], covered: dict[str, set[str]]) -> int:
    """Add labels from the current YOLO models for categories a source does not label.

    Returns the number of labels added. Uses YOLO only (not CLIP) and only
    confident boxes that do not overlap an existing label.
    """
    from PIL import Image

    from core import detector

    added = 0
    for item in items:
        missing = (set(config.CATEGORIES) - {config.OTHER_CATEGORY}) - covered[item.source]
        if not missing:
            continue
        with Image.open(item.image) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        for det in detector.detect(image, use_clip=False):
            if det["category"] not in missing or det["confidence"] < PSEUDO_LABEL_CONF:
                continue
            x1, y1, x2, y2 = det["bbox"]
            box = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)
            if all(_iou(box, label[1:]) < 0.5 for label in item.labels):
                item.labels.append((det["category"], *box))
                added += 1
    return added


# --------------------------------------------------------------------------- #
# Balancing, splitting, writing
# --------------------------------------------------------------------------- #
def balance(items: list[Item], max_per_class: int, background_fraction: float,
            seed: int) -> list[Item]:
    """Cap the images per class and the share of images without objects.

    An image is kept while at least one of its classes is still under the cap.
    """
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    counts: dict[str, int] = {}
    labelled, background = [], []
    for item in shuffled:
        if not item.labels:
            background.append(item)
        elif any(counts.get(c, 0) < max_per_class for c in item.classes):
            labelled.append(item)
            for c in item.classes:
                counts[c] = counts.get(c, 0) + 1
    if background_fraction <= 0 or not labelled:
        return labelled
    max_background = int(len(labelled) * background_fraction / (1 - background_fraction))
    return labelled + background[:max_background]


def assign_splits(items: list[Item], ratios: dict[str, float], seed: int) -> dict[str, list[Item]]:
    forced = [item for item in items if item.split == REAL_TEST]
    splits = prepare_dataset.split_pairs([i for i in items if i.split != REAL_TEST], ratios, seed)
    if forced:
        splits[REAL_TEST] = forced
    return splits


def write_dataset(splits: dict[str, list[Item]], out_dir: Path, settings: dict) -> dict:
    """Write images, labels, data.yaml and SOURCES.md; return a summary."""
    classes = [c for c in config.CATEGORIES
               if any(c in item.classes for items in splits.values() for item in items)]
    class_ids = {name: index for index, name in enumerate(classes)}

    if out_dir.exists():
        shutil.rmtree(out_dir)  # always rebuild from scratch for reproducibility
    summary = {"classes": classes, "images": {}, "instances": {}}
    for split, items in splits.items():
        images_dir, labels_dir = out_dir / split / "images", out_dir / split / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        used: set[str] = set()
        instances = {name: 0 for name in classes}
        for item in items:
            stem, counter = f"{item.source}_{item.image.stem}", 1
            while stem in used:
                stem, counter = f"{item.source}_{item.image.stem}_{counter}", counter + 1
            used.add(stem)
            shutil.copy2(item.image, images_dir / f"{stem}{item.image.suffix.lower()}")
            lines = [f"{class_ids[name]} " + " ".join(f"{v:.6f}" for v in box)
                     for name, *box in item.labels]
            for name, *_ in item.labels:
                instances[name] += 1
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                    encoding="utf-8")
        summary["images"][split] = len(items)
        summary["instances"][split] = instances

    # No "path" key: Ultralytics resolves the folders relative to the yaml file.
    base = {"train": "train/images", "val": "valid/images", "nc": len(classes), "names": classes}
    (out_dir / "data.yaml").write_text(
        yaml.safe_dump({**base, "test": "test/images"}, sort_keys=False), encoding="utf-8")
    if REAL_TEST in splits:
        (out_dir / "data_real_test.yaml").write_text(
            yaml.safe_dump({**base, "test": f"{REAL_TEST}/images"}, sort_keys=False),
            encoding="utf-8")

    credits = ["# Datasets in this training set", ""]
    for source in settings["sources"]:
        credits.append(f"- **{source.get('name', source['key'])}** - {source.get('licence', '?')}"
                       + (f" - {source['url']}" if source.get("url") else ""))
    (out_dir / "SOURCES.md").write_text("\n".join(credits) + "\n", encoding="utf-8")
    return summary


def print_summary(summary: dict, out_dir: Path, notes: list[str]) -> None:
    print(f"\nCombined dataset written to {out_dir}")
    for note in notes:
        print(f"  {note}")
    splits = list(summary["images"])
    print("\n" + f"{'images':<14}" + "".join(f"{s:>10}" for s in splits))
    print(f"{'':<14}" + "".join(f"{summary['images'][s]:>10}" for s in splits))
    print("\n" + f"{'instances':<14}" + "".join(f"{s:>10}" for s in splits))
    for name in summary["classes"]:
        print(f"{name:<14}" + "".join(f"{summary['instances'][s][name]:>10}" for s in splits))
    thin = [n for n in summary["classes"] if summary["instances"].get("train", {}).get(n, 0) < 150]
    if thin:
        print(f"\nWarning: few training examples for {thin}. Expect weak results for these; "
              "add more data or keep relying on CLIP for them.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sources", type=Path, default=SOURCES_FILE)
    parser.add_argument("--download", action="store_true",
                        help="download sources that are missing (needs ROBOFLOW_API_KEY)")
    parser.add_argument("--pseudo-label", action="store_true",
                        help="label unlabelled objects with the current YOLO models")
    parser.add_argument("--skip", nargs="*", default=[], help="source keys to leave out")
    args = parser.parse_args()

    settings = load_settings(args.sources)
    sources = [s for s in settings["sources"] if s["key"] not in args.skip]
    notes, items = [], []
    for source in sources:
        folder = source_dir(source)
        has_dataset = folder.exists() and any(folder.rglob("data.yaml"))
        if not has_dataset:
            if args.download and source.get("roboflow"):
                print(f"Downloading {source['key']} ...")
                download_source(source, folder)
            else:
                raise SystemExit(f"{folder} has no dataset. Download it (--download) or "
                                 f"remove / skip source {source['key']!r}.")
        source_items = read_source(source)
        notes.append(f"{source['key']}: {len(source_items)} images")
        items += source_items

    items, duplicates = remove_duplicates(items)
    notes.append(f"duplicates removed: {duplicates}")
    if args.pseudo_label:
        covered = {s["key"]: {t for t in s["classes"].values() if t} for s in sources}
        notes.append(f"pseudo-labels added: {pseudo_label(items, covered)}")

    kept = balance(items, settings.get("max_images_per_class", 1500),
                   settings.get("background_fraction", 0.1), settings.get("seed", 42))
    notes.append(f"kept after balancing: {len(kept)} of {len(items)}")
    splits = assign_splits(kept, settings["split"], settings.get("seed", 42))
    out_dir = ROOT / settings.get("output", "datasets/combined")
    print_summary(write_dataset(splits, out_dir, settings), out_dir, notes)


if __name__ == "__main__":
    main()
