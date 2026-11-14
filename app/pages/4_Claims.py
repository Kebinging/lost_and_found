"""Staff page: review student claims next to the item they were matched with."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

import streamlit as st  # noqa: E402

from app.common import require_staff_pin, setup_page  # noqa: E402
from core import storage  # noqa: E402

setup_page("Claims")
require_staff_pin()

try:
    claims = storage.load_claims()
    items = storage.load_found_items()
except storage.StorageError as error:
    st.error(str(error))
    st.stop()

if claims.empty:
    st.info("No claims yet. They appear here when a student submits one in the chatbot.")
    st.stop()

only_pending = st.toggle("Show pending claims only", value=True)
shown = claims[claims["status"] == "Pending"] if only_pending else claims
st.dataframe(shown, hide_index=True)

if shown.empty:
    st.info("No pending claims.")
    st.stop()

claim_id = st.selectbox("Select a claim", shown["claim_id"].tolist())
claim = shown[shown["claim_id"] == claim_id].iloc[0]
linked = items[items["item_id"] == claim["item_id"]]

item_column, student_column = st.columns(2)

with item_column:
    st.subheader("Found item")
    if linked.empty:
        st.warning(f"Item {claim['item_id']} is no longer in the inventory.")
    else:
        item = linked.iloc[0]
        photo_path = storage.resolve_path(item["image_path"])
        if photo_path.exists():
            st.image(str(photo_path), caption=item["item_id"])
        st.markdown(
            f"**{item['item_id']}** - {item['colour']} {item['category']}  \n"
            f"**Status:** {item['status']}  \n"
            f"**Found:** {item['location_found']}, {item['date_found']}  \n"
            f"**Staff description:** {item['description'] or '-'}"
        )

with student_column:
    st.subheader("Student's claim")
    st.markdown(
        f"**{claim['claim_id']}** - {claim['status']}  \n"
        f"**Name:** {claim['student_name']}  \n"
        f"**Student ID:** {claim['student_id']}  \n"
        f"**Contact:** {claim['contact']}  \n"
        f"**Submitted:** {claim['created_at']}  \n"
        f"**Match score:** {claim['match_score']:.2f}"
    )
    st.markdown("**Student's description**")
    st.info(claim["student_description"] or "-")

    if claim["status"] == "Pending":
        st.caption("Check the student's ID and proof of ownership before verifying.")
        verify_column, reject_column = st.columns(2)
        verify = verify_column.button("Verify & mark claimed", type="primary")
        reject = reject_column.button("Reject")
        if verify or reject:
            try:
                storage.update_claim(claim_id, status="Verified" if verify else "Rejected")
                if not linked.empty:
                    storage.update_item(claim["item_id"],
                                        status="Claimed" if verify else "Unclaimed")
            except storage.StorageError as error:
                st.error(str(error))
            else:
                st.rerun()
