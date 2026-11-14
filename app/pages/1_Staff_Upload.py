"""Staff page: register a found item from a photo.

Upload or take a photo -> YOLO detects the item -> category and colour are
pre-filled -> staff confirm / correct -> the record, photos and CLIP embedding
are saved.
"""

import datetime as dt
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

import streamlit as st  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

import config  # noqa: E402
from app.common import get_detector, get_embedder, require_staff_pin, setup_page  # noqa: E402
from core import colour, storage  # noqa: E402
from core import detector as detector_module  # noqa: E402  (helpers only; models load lazily)

setup_page("Register a found item")
require_staff_pin()

if "last_saved_item" in st.session_state:
    st.success(st.session_state.pop("last_saved_item"))

# ---- 1. Photo -------------------------------------------------------------- #
upload_tab, camera_tab = st.tabs(["Upload a photo", "Use the camera"])
with upload_tab:
    uploaded = st.file_uploader("Photo of the found item", type=["jpg", "jpeg", "png"])
with camera_tab:
    captured = st.camera_input("Take a photo of the found item")

source = uploaded or captured
if source is None:
    st.info("Upload or take a photo to begin.")
    st.stop()

image_bytes = source.getvalue()
image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")

# ---- 2. Detection (cached per photo so widget clicks do not re-run YOLO) --- #
photo_key = hashlib.sha256(image_bytes).hexdigest()
if st.session_state.get("detected_photo") != photo_key:
    try:
        detector = get_detector()
        with st.spinner("Detecting the item..."):
            st.session_state["detections"] = detector.detect(image)
    except Exception as error:
        st.session_state["detections"] = []
        st.warning(f"Object detection is unavailable ({error}). You can still enter "
                   "the item manually.")
    st.session_state["detected_photo"] = photo_key
detections = st.session_state["detections"]



def explain_category(det: dict) -> None:
    """Tell staff how the suggested category was chosen, so they can judge it."""
    named_by, yolo_category = det.get("named_by"), det.get("yolo_category")
    if named_by == "clip" and yolo_category and yolo_category != det["category"]:
        st.caption(f"YOLO said **{yolo_category}**, but CLIP is confident it is "
                   f"**{det['category']}**, which YOLO was not trained on.")
    elif named_by == "clip":
        st.caption(f"YOLO found no known object; CLIP thinks the photo shows "
                   f"**{det['category']}**.")
    elif named_by == "other":
        st.caption("Neither YOLO nor CLIP is sure what this is, so it is marked **Other**. "
                   "Choose the right category if there is one.")
    if det.get("clip_scores"):
        guesses = ", ".join(f"{name} {score:.0%}" for name, score in det["clip_scores"])
        st.caption(f"CLIP's top guesses: {guesses}")


photo_column, form_column = st.columns(2)
with photo_column:
    st.image(detector_module.draw_boxes(image, detections),
             caption="Detected objects" if detections else "No object detected")
    st.caption("Green box: custom YOLO model, blue: COCO YOLO model, "
               "orange: no box found, CLIP named the whole photo.")
    if not detector_module.custom_model_available():
        st.caption("Custom model (models/best.pt) not found - only the COCO categories "
                   "(phone, bag, laptop, book, bottle, umbrella) are detected.")

with form_column:
    # ---- 3. Choose the object ---------------------------------------------- #
    if not detections:
        st.warning("Nothing was detected. Choose the category manually.")
        selected = None
        item_crop = image
    else:
        choice = 0
        if len(detections) > 1:
            choice = st.radio(
                "Which object is the found item?",
                options=range(len(detections)),
                format_func=lambda i: (f"{i + 1}. {detections[i]['category']} "
                                       f"({detections[i]['confidence']:.2f})"),
            )
        selected = detections[choice]
        item_crop = detector_module.crop(image, selected["bbox"])
        st.image(item_crop, width=160, caption="Selected object")
        explain_category(selected)

    detected_category = selected["category"] if selected else config.OTHER_CATEGORY
    detected_colour = colour.dominant_colour(item_crop)
    colours = list(config.COLOUR_PALETTE)

    # ---- 4. Auto-filled form ----------------------------------------------- #
    # The key includes the photo and choice so the defaults refresh with them.
    with st.form(f"item_form_{photo_key}_{detections.index(selected) if selected else 'manual'}"):
        category = st.selectbox("Category", config.CATEGORIES,
                                index=config.CATEGORIES.index(detected_category))
        item_colour = st.selectbox("Colour", colours, index=colours.index(detected_colour))
        st.text_input("Detection confidence",
                      value=f"{selected['confidence']:.2f}" if selected else "n/a (manual)",
                      disabled=True)
        location = st.selectbox("Location found", config.LOCATIONS)
        date_found = st.date_input("Date found", value=dt.date.today(),
                                   max_value=dt.date.today())
        description = st.text_area("Description (staff only, never shown to students)",
                                   placeholder="e.g. Black bottle with a sticker on the lid")
        save = st.form_submit_button("Save item", type="primary")

if save:
    # If staff overrode the detected category, the model did not produce it.
    model_category = selected is not None and category == selected["category"]
    record = {
        "category": category,
        "colour": item_colour,
        "description": description,
        "location_found": location,
        "date_found": date_found,
        "confidence": selected["confidence"] if model_category else 0.0,
        # e.g. "custom", "coco+clip" (YOLO box, CLIP name), "clip", "manual"
        "detector": (selected["source"] + ("+clip" if selected.get("named_by") == "clip"
                                           and selected["source"] != "clip" else "")
                     if model_category else "manual"),
    }
    try:
        item_id = storage.add_found_item(record, image, item_crop)
    except storage.StorageError as error:
        st.error(str(error))
        st.stop()

    message = f"Item saved as {item_id}."
    try:
        storage.save_embedding(item_id, get_embedder().embed_image(item_crop))
    except Exception as error:
        message += (f" Warning: its image embedding could not be saved ({error}), "
                    "so matching for this item will rely on colour and location only.")
    st.session_state["last_saved_item"] = message
    st.session_state.pop("detected_photo", None)
    st.rerun()
