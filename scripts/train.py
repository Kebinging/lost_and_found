"""Train the custom YOLO11 model locally (alternative to the Colab notebook).

Run ``scripts/prepare_dataset.py`` first. Training on a CPU is very slow; use
the notebook on Colab's free GPU unless this machine has one.

Usage (from the project root)::

    python scripts/train.py
    python scripts/train.py --epochs 20 --batch 8
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

import config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=config.DATA_YAML)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found. Run scripts/prepare_dataset.py first.")

    from ultralytics import YOLO

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(config.COCO_MODEL_PATH))  # start from COCO-pretrained weights
    results = model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, seed=args.seed,
        project=str(config.PROJECT_ROOT / "runs" / "detect"), name="train",
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, config.CUSTOM_MODEL_PATH)
    print(f"\nBest weights copied to {config.CUSTOM_MODEL_PATH}")
    print("Next: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
