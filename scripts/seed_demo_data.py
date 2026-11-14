"""Add ~15 demo found items to the inventory from the dataset's test images.

Each image goes through the same pipeline as the Staff Upload page: detect
the object, crop it, extract the colour, save the record and its CLIP
embedding. Location and date are chosen at random (fixed seed) so the demo
has variety. If the detector finds nothing in an image, the dataset's own
label is used instead and the item is recorded with ``detector = "manual"``.

Usage (from the project root, after preparing the dataset)::

    python scripts/seed_demo_data.py
    python scripts/seed_demo_data.py --count 20 --images path/to/any/photos
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
import sys
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

import config  # noqa: E402
from core import colour, detector, embedder, storage  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def label_fallback(image_path: Path, image: Image.Image) -> tuple[str, tuple] | None:
    """Return ``(category, bbox)`` from the image's YOLO label file, if it has one."""
    label = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if not label.exists():
        return None
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:  # only plain "class cx cy w h" boxes
            continue
        class_id = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:])
        width, height = image.size
        bbox = (int((cx - w / 2) * width), int((cy - h / 2) * height),
                int((cx + w / 2) * width), int((cy + h / 2) * height))
        return config.DATASET_CLASSES[class_id], bbox
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", type=Path, default=config.DATASET_DIR / "test" / "images")
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = sorted(p for p in args.images.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"No images in {args.images}. Run scripts/prepare_dataset.py "
                         "first, or pass --images with a folder of photos.")

    rng = random.Random(args.seed)
    rng.shuffle(paths)
    storage.init_storage()
    locations = [loc for loc in config.LOCATIONS if loc != "Other"]

    added = 0
    for path in paths:
        if added >= args.count:
            break
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")

        detections = detector.detect(image)
        if detections:
            best = detections[0]
            category, bbox = best["category"], best["bbox"]
            confidence, source = best["confidence"], best["source"]
        else:
            fallback = label_fallback(path, image)
            if fallback is None:
                continue  # background image: nothing to add
            category, bbox = fallback
            confidence, source = 0.0, "manual"

        item_crop = detector.crop(image, bbox)
        item_colour = colour.dominant_colour(item_crop)
        item_id = storage.add_found_item({
            "category": category,
            "colour": item_colour,
            "description": f"Demo item seeded from {path.name}",
            "location_found": rng.choice(locations),
            "date_found": dt.date.today() - dt.timedelta(days=rng.randint(0, 10)),
            "confidence": confidence,
            "detector": source,
        }, image, item_crop)
        storage.save_embedding(item_id, embedder.embed_image(item_crop))
        added += 1
        print(f"{item_id}: {item_colour} {category} ({source}, {confidence:.2f})")

    print(f"\nAdded {added} demo item(s) to {config.FOUND_ITEMS_FILE}")


if __name__ == "__main__":
    main()
