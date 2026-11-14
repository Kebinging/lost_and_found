"""Object detection with two YOLO models.

* The **custom** model (``models/best.pt``) is trained on the Roboflow
  "Lost and Found" dataset: Bottle, Umbrella, Calculator, Charger, Key.
* The **COCO** model (``models/yolo11n.pt``) adds Phone, Bag, Laptop, Book
  (and can also see bottles and umbrellas).

Both models run on every image. Their results are converted to the unified
category names in ``config.CATEGORIES`` and merged, so the rest of the system
never needs to know which model found an object.

YOLO can only name the classes it was trained on, so each detected object is
then also shown to CLIP (``core/classifier.py``), which can recognise items
YOLO has never seen (earbuds, ID cards, ...). If YOLO finds nothing at all,
CLIP looks at the whole photo. Items neither model is sure about become
"Other".
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import NotRequired, TypedDict

from PIL import Image, ImageDraw, ImageFont

import config

logger = logging.getLogger(__name__)

BBox = tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels


class Detection(TypedDict):
    """One detected object."""

    category: str      # unified category name, e.g. "Bottle"
    confidence: float  # 0..1 (YOLO's box confidence, or CLIP's for a whole-photo guess)
    bbox: BBox
    source: str        # which model found the object: "custom", "coco" or "clip" (whole photo)
    named_by: NotRequired[str]        # who chose the category: "yolo", "clip" or "other"
    yolo_category: NotRequired[str | None]           # YOLO's own label, before CLIP
    clip_scores: NotRequired[list[tuple[str, float]]]  # CLIP's top guesses


@lru_cache(maxsize=1)
def load_models() -> dict:
    """Load both YOLO models once and return ``{"custom": model | None, "coco": model}``.

    The custom model is optional: until ``models/best.pt`` has been trained
    and copied in, detection still works with the COCO categories only.
    """
    from ultralytics import YOLO

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    custom = None
    if config.CUSTOM_MODEL_PATH.exists():
        custom = YOLO(str(config.CUSTOM_MODEL_PATH))
    else:
        logger.warning("Custom model not found at %s - using the COCO model only. "
                       "Train it with notebooks/train_yolo.ipynb.", config.CUSTOM_MODEL_PATH)

    # Ultralytics downloads yolo11n.pt to this path on first use.
    coco = YOLO(str(config.COCO_MODEL_PATH))
    return {"custom": custom, "coco": coco}


def custom_model_available() -> bool:
    """True if the trained custom model file is present."""
    return config.CUSTOM_MODEL_PATH.exists()


def yolo_categories() -> set[str]:
    """The unified categories the loaded YOLO models can detect.

    Read from the models themselves, so it updates when ``best.pt`` is
    retrained with more classes.
    """
    models = load_models()
    found = set(config.COCO_TO_CATEGORY.values())
    if models["custom"] is not None:
        found |= {c for c in map(_custom_category, models["custom"].names.values()) if c}
    return found


def _custom_category(class_name: str) -> str | None:
    """Map a custom-model class name to a unified category (case-insensitive)."""
    lookup = {c.lower(): c for c in config.CATEGORIES}
    return lookup.get(str(class_name).strip().lower())


def _coco_category(class_name: str) -> str | None:
    """Map a COCO class name to a unified category; unknown classes give ``None``."""
    return config.COCO_TO_CATEGORY.get(str(class_name).strip().lower())


def _run_model(model, image: Image.Image, source: str) -> list[Detection]:
    """Run one model and convert its boxes to ``Detection`` dicts."""
    to_category = _custom_category if source == "custom" else _coco_category
    result = model.predict(image, conf=config.DETECTION_CONF, verbose=False)[0]

    detections: list[Detection] = []
    for box in result.boxes:
        category = to_category(result.names[int(box.cls[0])])
        confidence = float(box.conf[0])
        if category is None or confidence < config.DETECTION_CONF:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
        detections.append(Detection(category=category, confidence=confidence,
                                    bbox=(x1, y1, x2, y2), source=source))
    return detections


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
             - intersection)
    return intersection / union if union > 0 else 0.0


def merge_detections(detections: list[Detection]) -> list[Detection]:
    """Sort by confidence and drop duplicates of the same object.

    When two boxes share a category and overlap with IoU above
    ``config.DETECTION_IOU_MERGE`` (typically the custom and COCO models both
    seeing the same bottle), only the more confident one is kept.
    """
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        duplicate = any(
            det["category"] == other["category"]
            and iou(det["bbox"], other["bbox"]) > config.DETECTION_IOU_MERGE
            for other in kept
        )
        if not duplicate:
            kept.append(det)
    return kept


def detect(image: Image.Image, use_clip: bool = True) -> list[Detection]:
    """Detect lost-and-found items in ``image``; best detection first.

    With ``use_clip`` (the default) every object is also named by CLIP and
    the final category is chosen by ``classifier.decide``; if YOLO finds
    nothing, CLIP names the whole photo. If CLIP fails to load, the YOLO
    results are returned unchanged.
    """
    models = load_models()
    image = image.convert("RGB")

    detections: list[Detection] = []
    if models["custom"] is not None:
        detections += _run_model(models["custom"], image, "custom")
    detections += _run_model(models["coco"], image, "coco")
    detections = merge_detections(detections)

    if use_clip:
        try:
            detections = name_with_clip(image, detections)
        except Exception:
            logger.exception("CLIP naming failed; using YOLO labels only.")
    return detections


def padded_bbox(bbox: BBox, margin: float) -> BBox:
    """Grow a box by ``margin`` (a fraction of its width / height) on every side."""
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return (int(x1 - dx), int(y1 - dy), int(x2 + dx), int(y2 + dy))


def name_with_clip(image: Image.Image, detections: list[Detection]) -> list[Detection]:
    """Let CLIP check / correct each detection's category (see ``core/classifier.py``)."""
    from core import classifier

    known = yolo_categories()
    if not detections:
        width, height = image.size
        scores = classifier.classify(image)
        category, named_by = classifier.decide(None, scores, known)
        return [Detection(category=category, confidence=scores[0][1], bbox=(0, 0, width, height),
                          source="clip", named_by=named_by, yolo_category=None,
                          clip_scores=scores[:3])]

    named: list[Detection] = []
    for det in detections:
        scores = classifier.classify(crop(image, padded_bbox(det["bbox"], config.CLIP_CROP_MARGIN)))
        category, named_by = classifier.decide(det["category"], scores, known)
        named.append({**det, "category": category, "named_by": named_by,
                      "yolo_category": det["category"], "clip_scores": scores[:3]})
    return merge_detections(named)


def crop(image: Image.Image, bbox: BBox) -> Image.Image:
    """Cut ``bbox`` out of ``image`` (the box is clamped to the image)."""
    width, height = image.size
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(width, int(bbox[2])), min(height, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return image.copy()
    return image.crop((x1, y1, x2, y2))


def draw_boxes(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """Return a copy of ``image`` with numbered, labelled boxes drawn on it."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, round(max(annotated.size) / 300))
    try:
        font = ImageFont.load_default(size=max(14, round(max(annotated.size) / 45)))
    except TypeError:  # Pillow < 10.1 has no size argument
        font = ImageFont.load_default()

    for number, det in enumerate(detections, start=1):
        colour = {"custom": "#00B050", "coco": "#0070C0"}.get(det["source"], "#C55A11")
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=line_width)
        label = f"{number}. {det['category']} {det['confidence']:.2f}"
        left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
        label_height = bottom - top + 6
        label_y = y1 - label_height if y1 >= label_height else y1
        draw.rectangle((x1, label_y, x1 + (right - left) + 8, label_y + label_height),
                       fill=colour)
        draw.text((x1 + 4, label_y + 2), label, fill="white", font=font)
    return annotated
