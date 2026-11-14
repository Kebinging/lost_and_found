"""Evaluate the custom model on the held-out test split.

Prints mAP50, mAP50-95, precision and recall (overall and per class) and
saves the confusion matrix, PR curve and sample predictions as images in the
run folder - handy for the report.

Usage (from the project root)::

    python scripts/evaluate.py
    python scripts/evaluate.py --weights runs/detect/train/weights/best.pt --split valid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

import config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, default=config.CUSTOM_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=config.DATA_YAML)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    args = parser.parse_args()

    for path, hint in ((args.weights, "train the model first"),
                       (args.data, "run scripts/prepare_dataset.py first")):
        if not path.exists():
            raise SystemExit(f"{path} not found - {hint}.")

    from ultralytics import YOLO

    metrics = YOLO(str(args.weights)).val(
        data=str(args.data), split="val" if args.split == "valid" else args.split,
        plots=True, project=str(config.PROJECT_ROOT / "runs" / "detect"), name="evaluate",
    )

    print(f"\nResults on the {args.split} split")
    print(f"{'class':<12}{'precision':>10}{'recall':>10}{'mAP50':>10}{'mAP50-95':>10}")
    print(f"{'all':<12}{metrics.box.mp:>10.3f}{metrics.box.mr:>10.3f}"
          f"{metrics.box.map50:>10.3f}{metrics.box.map:>10.3f}")
    for index, class_id in enumerate(metrics.box.ap_class_index):
        precision, recall, ap50, ap = metrics.box.class_result(index)
        print(f"{metrics.names[int(class_id)]:<12}{precision:>10.3f}{recall:>10.3f}"
              f"{ap50:>10.3f}{ap:>10.3f}")
    print(f"\nConfusion matrix, PR curve and sample predictions: {metrics.save_dir}")


if __name__ == "__main__":
    main()
