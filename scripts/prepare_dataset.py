"""Clean and re-split the Roboflow "Lost and Found" dataset.

The dataset (https://universe.roboflow.com/rnomo/lost-and-found-nlb4x-e6ayk/dataset/2,
CC BY 4.0) ships with a 77% / 0% / 23% split - no validation set - and a
``None`` class that is not a real object. This script:

1. (optional, ``--download``) downloads it in YOLOv8 format with the
   ``roboflow`` package, using the API key in the ``ROBOFLOW_API_KEY``
   environment variable. Otherwise unzip a manual download into
   ``datasets/raw/`` first.
2. Reads the class names from the dataset's own ``data.yaml``.
3. Removes every ``None`` label and remaps the remaining class IDs to
   ``0 Bottle, 1 Umbrella, 2 Calculator, 3 Charger, 4 Key``.
4. Merges all images and re-splits them randomly (seed 42) into
   70% train / 20% valid / 10% test.
5. Writes a new ``data.yaml``.
6. Prints image counts per split and instance counts per class.

Usage (from the project root)::

    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --download
    python scripts/prepare_dataset.py --raw path/to/unzipped --out path/to/output
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

import config  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ROBOFLOW_WORKSPACE = "rnomo"
ROBOFLOW_PROJECT = "lost-and-found-nlb4x-e6ayk"
ROBOFLOW_VERSION = 2


def download_dataset(raw_dir: Path) -> None:
    """Download the dataset in YOLOv8 format into ``raw_dir``."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set the ROBOFLOW_API_KEY environment variable first "
                         "(never write the key into a file).")
    from roboflow import Roboflow  # optional dependency: pip install roboflow

    project = Roboflow(api_key=api_key).workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    project.version(ROBOFLOW_VERSION).download("yolov8", location=str(raw_dir))


def find_data_yaml(raw_dir: Path) -> Path:
    """Locate the dataset's ``data.yaml`` (the zip may add a wrapper folder)."""
    candidates = sorted(raw_dir.rglob("data.yaml"), key=lambda p: len(p.parts))
    if not candidates:
        raise SystemExit(f"No data.yaml found under {raw_dir}. Unzip the Roboflow "
                         "YOLOv8 download there, or run with --download.")
    return candidates[0]


def read_class_names(data_yaml: Path) -> list[str]:
    """Return the class names from ``data.yaml``, ordered by class ID."""
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=int)]
    return [str(name) for name in names]


def build_id_map(source_names: list[str], keep: list[str], drop: str) -> dict[int, int]:
    """Map each original class ID to its new ID; dropped classes are absent.

    Names are matched case-insensitively. A class that is neither kept nor
    the dropped one is an error, so a changed dataset cannot be silently
    mis-labelled.
    """
    new_ids = {name.lower(): index for index, name in enumerate(keep)}
    id_map: dict[int, int] = {}
    for old_id, name in enumerate(source_names):
        key = name.strip().lower()
        if key in new_ids:
            id_map[old_id] = new_ids[key]
        elif key != drop.lower():
            raise SystemExit(f"Unexpected class {name!r} in the dataset; "
                             f"expected {keep} plus {drop!r}.")
    missing = [name for name in keep if name.lower() not in
               {source_names[i].strip().lower() for i in id_map}]
    if missing:
        raise SystemExit(f"Classes missing from the dataset: {missing}")
    return id_map


def collect_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    """Return ``(image, label)`` path pairs from every split of the raw dataset."""
    pairs = []
    for image in sorted(raw_dir.rglob("*")):
        if image.suffix.lower() not in IMAGE_EXTENSIONS or image.parent.name != "images":
            continue
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        pairs.append((image, label))
    return pairs


def remap_label_lines(lines: list[str], id_map: dict[int, int]) -> list[str]:
    """Drop labels of removed classes and rewrite the class ID of the rest."""
    remapped = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        old_id = int(float(parts[0]))
        if old_id in id_map:
            remapped.append(" ".join([str(id_map[old_id]), *parts[1:]]))
    return remapped


def split_pairs(pairs: list, ratios: dict[str, float], seed: int) -> dict[str, list]:
    """Shuffle deterministically and cut into train / valid / test."""
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * ratios["train"])
    n_valid = int(len(shuffled) * ratios["valid"])
    return {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train:n_train + n_valid],
        "test": shuffled[n_train + n_valid:],
    }


def prepare(raw_dir: Path, out_dir: Path, seed: int = config.SPLIT_SEED) -> dict:
    """Run the whole clean + re-split; return the summary that is printed.

    Images whose labels were all ``None`` are kept with an empty label file:
    YOLO uses them as background (negative) examples.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    classes = config.DATASET_CLASSES
    id_map = build_id_map(read_class_names(find_data_yaml(raw_dir)),
                          classes, config.DATASET_DROP_CLASS)

    pairs = collect_pairs(raw_dir)
    if not pairs:
        raise SystemExit(f"No images found under {raw_dir}.")

    if out_dir.exists():
        shutil.rmtree(out_dir)  # always rebuild from scratch for reproducibility

    summary = {"images": {}, "instances": {name: 0 for name in classes}}
    used_names: set[str] = set()
    for split, split_items in split_pairs(pairs, config.SPLIT_RATIOS, seed).items():
        images_dir = out_dir / split / "images"
        labels_dir = out_dir / split / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        for image, label in split_items:
            # Guard against the same file name appearing in two raw splits.
            stem, counter = image.stem, 1
            while stem in used_names:
                stem, counter = f"{image.stem}_{counter}", counter + 1
            used_names.add(stem)

            lines = label.read_text(encoding="utf-8").splitlines() if label.exists() else []
            remapped = remap_label_lines(lines, id_map)
            for line in remapped:
                summary["instances"][classes[int(line.split()[0])]] += 1

            shutil.copy2(image, images_dir / f"{stem}{image.suffix.lower()}")
            (labels_dir / f"{stem}.txt").write_text(
                "\n".join(remapped) + ("\n" if remapped else ""), encoding="utf-8")
        summary["images"][split] = len(split_items)

    # No "path" key: Ultralytics then resolves the splits relative to this
    # file, so the dataset folder can be moved (e.g. to Colab) unchanged.
    data_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(classes),
        "names": classes,
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False),
                                       encoding="utf-8")
    return summary


def print_summary(summary: dict, out_dir: Path) -> None:
    total = sum(summary["images"].values())
    print(f"\nDataset written to {out_dir}")
    print("Images per split:")
    for split, count in summary["images"].items():
        print(f"  {split:<6}{count:>6}  ({count / total:.0%})")
    print(f"  {'total':<6}{total:>6}")
    print("Instances per class:")
    for name, count in summary["instances"].items():
        print(f"  {name:<12}{count:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", type=Path, default=config.RAW_DATASET_DIR,
                        help="folder holding the unzipped Roboflow download")
    parser.add_argument("--out", type=Path, default=config.DATASET_DIR,
                        help="output folder for the cleaned dataset")
    parser.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    parser.add_argument("--download", action="store_true",
                        help="download with the roboflow package (needs ROBOFLOW_API_KEY)")
    args = parser.parse_args()

    if args.download:
        download_dataset(args.raw)
    print_summary(prepare(args.raw, args.out, args.seed), args.out)


if __name__ == "__main__":
    main()
