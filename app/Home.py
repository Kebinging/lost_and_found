"""Home page - entry point of the app.

Run from the project root::

    streamlit run app/Home.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

import streamlit as st  # noqa: E402

from app.common import setup_page  # noqa: E402
from core import storage  # noqa: E402

setup_page()

st.markdown(
    "A university Lost & Found prototype. **Staff** photograph a found item and the "
    "system recognises what it is and its colour. **Students** describe what they lost "
    "to a chatbot, which tells them whether a possible match has been handed in - so "
    "they only visit the counter when it is worth the trip."
)

staff_column, student_column = st.columns(2)
with staff_column:
    st.page_link("pages/1_Staff_Upload.py", label="I found an item (Staff)", icon="📷")
    st.caption("Register a found item from a photo. Requires the staff PIN.")
with student_column:
    st.page_link("pages/2_Student_Chatbot.py", label="I lost something (Student)", icon="💬")
    st.caption("Describe your lost item and check whether it has been handed in.")

st.divider()

try:
    items = storage.load_found_items()
    claims = storage.load_claims()
except storage.StorageError as error:
    st.error(str(error))
else:
    total, unclaimed, pending = st.columns(3)
    total.metric("Total items", len(items))
    unclaimed.metric("Unclaimed items", int((items["status"] == "Unclaimed").sum()))
    pending.metric("Pending claims", int((claims["status"] == "Pending").sum()))

st.caption("For privacy and to prevent false claims, photos and details of found items "
           "are never shown to students. Ownership is verified in person by staff.")
