"""Shared helpers for the Streamlit pages: page header, staff PIN gate and
cached model loaders.

Every page starts by putting the project root on ``sys.path`` (Streamlit only
adds the ``app/`` folder) and then calls ``setup_page``.
"""

from __future__ import annotations

import hmac

import streamlit as st

from core import storage

APP_TITLE = "Smart Lost & Found"
APP_ICON = "🔎"


def setup_page(subtitle: str | None = None) -> None:
    """Configure the page and draw the header shared by every page.

    Must be the first Streamlit call on a page.
    """
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    st.title(f"{APP_ICON} {APP_TITLE}")
    if subtitle:
        st.subheader(subtitle)
    try:
        storage.init_storage()
    except storage.StorageError as error:
        st.error(str(error))
        st.stop()


def _configured_pin() -> str | None:
    """Return ``STAFF_PIN`` from ``.streamlit/secrets.toml``, or ``None`` if unset."""
    try:
        pin = st.secrets.get("STAFF_PIN")
    except Exception:  # no secrets.toml at all
        return None
    return str(pin) if pin else None


def require_staff_pin() -> None:
    """Stop the page unless the staff PIN has been entered this session.

    This is a prototype-level gate only (one shared PIN, no user accounts).
    """
    pin = _configured_pin()
    if pin is None:
        st.error("Staff pages are locked because no staff PIN is configured.")
        st.markdown("Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, "
                    "set `STAFF_PIN`, then restart the app.")
        st.stop()

    if st.session_state.get("staff_authenticated"):
        if st.sidebar.button("Lock staff pages"):
            st.session_state["staff_authenticated"] = False
            st.rerun()
        return

    st.info("This page is for Lost & Found staff. Enter the staff PIN to continue.")
    with st.form("staff_pin_form"):
        entered = st.text_input("Staff PIN", type="password")
        submitted = st.form_submit_button("Unlock")
    if submitted:
        if hmac.compare_digest(entered.encode(), pin.encode()):
            st.session_state["staff_authenticated"] = True
            st.rerun()
        st.error("Incorrect PIN.")
    st.stop()


@st.cache_resource(show_spinner="Loading the object detection models...")
def get_detector():
    """Load the YOLO models once per server process and return the detector module."""
    from core import detector

    detector.load_models()
    return detector


@st.cache_resource(show_spinner="Loading the CLIP model (first run downloads ~600 MB)...")
def get_embedder():
    """Load the CLIP model once per server process and return the embedder module."""
    from core import embedder

    embedder.get_model()
    return embedder
