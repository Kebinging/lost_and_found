"""Staff page: browse the found-item inventory, change statuses and test the matcher."""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

import streamlit as st  # noqa: E402

import config  # noqa: E402
from app.common import require_staff_pin, setup_page  # noqa: E402
from chatbot import nlu  # noqa: E402
from core import matcher, storage  # noqa: E402

setup_page("Inventory")
require_staff_pin()

try:
    items = storage.load_found_items()
except storage.StorageError as error:
    st.error(str(error))
    st.stop()

if items.empty:
    st.info("No found items yet. Add one on the Staff Upload page.")
    st.stop()

# ---- Filters --------------------------------------------------------------- #
dates = items["date_found"].dropna()
earliest = min(dates) if len(dates) else dt.date.today()
latest = max(max(dates), dt.date.today()) if len(dates) else dt.date.today()

category_column, status_column, date_column = st.columns(3)
categories = category_column.multiselect("Category", config.CATEGORIES)
statuses = status_column.multiselect("Status", config.ITEM_STATUSES)
date_range = date_column.date_input("Date found (from - to)", value=(earliest, latest))

filtered = items
if categories:
    filtered = filtered[filtered["category"].isin(categories)]
if statuses:
    filtered = filtered[filtered["status"].isin(statuses)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[filtered["date_found"].map(lambda d: d is not None and start <= d <= end)]

st.caption(f"{len(filtered)} of {len(items)} item(s)")
st.dataframe(filtered.drop(columns=["image_path", "crop_path"]), hide_index=True)

# ---- Item details ---------------------------------------------------------- #
if not filtered.empty:
    st.subheader("Item details")
    item_id = st.selectbox("Select an item", filtered["item_id"].tolist())
    item = filtered[filtered["item_id"] == item_id].iloc[0]

    photo_column, crop_column, detail_column = st.columns([2, 1, 2])
    for column, key, caption in ((photo_column, "image_path", "Original photo"),
                                 (crop_column, "crop_path", "Detected object")):
        path = storage.resolve_path(item[key])
        if path.exists():
            column.image(str(path), caption=caption)
        else:
            column.warning(f"{caption} file is missing.")

    with detail_column:
        st.markdown(
            f"**{item['item_id']}** - {item['colour']} {item['category']}  \n"
            f"**Status:** {item['status']}  \n"
            f"**Found:** {item['location_found']}, {item['date_found']}  \n"
            f"**Detector:** {item['detector']} (confidence {item['confidence']:.2f})  \n"
            f"**Description:** {item['description'] or '-'}"
        )
        claimed_column, disposed_column = st.columns(2)
        for column, status in ((claimed_column, "Claimed"), (disposed_column, "Disposed")):
            if column.button(f"Mark {status}", disabled=item["status"] == status):
                try:
                    storage.update_item(item_id, status=status)
                except storage.StorageError as error:
                    st.error(str(error))
                else:
                    st.rerun()

# ---- Matcher test bench ---------------------------------------------------- #
with st.expander("Test matcher"):
    st.caption("Type a description as a student would. Every unclaimed item of the "
               "detected category is scored, including those below the match threshold "
               f"({config.MATCH_THRESHOLD}).")
    description = st.text_input("Description",
                                placeholder="I lost my black umbrella at the library yesterday")
    if description:
        slots = nlu.extract_slots(description)
        st.write("Extracted:", {k: str(v) if v else None for k, v in slots.items()})
        try:
            with st.spinner("Scoring..."):
                results = matcher.find_matches(
                    {"category": slots["category"], "colour": slots["colour"],
                     "location": slots["location"], "date_lost": slots["date_lost"],
                     "free_text": description},
                    threshold=0.0, max_results=len(items))
        except storage.StorageError as error:
            st.error(str(error))
        else:
            if not results:
                st.info("No unclaimed item passes the category/date filters.")
            else:
                st.dataframe(
                    [{"item_id": m.item_id, "score": m.score,
                      "match": m.score >= config.MATCH_THRESHOLD,
                      "explanation": m.explanation}
                     for m in results],
                    hide_index=True)
