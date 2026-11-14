"""Student page: chatbot for reporting a lost item.

PRIVACY: this page must never display a found item's photo, description,
location or item ID. It only shows the dialogue's replies (which contain a
match count and a claim reference at most) and the student's own messages.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

import streamlit as st  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from app.common import get_embedder, setup_page  # noqa: E402
from chatbot.dialogue import LostItemDialogue  # noqa: E402
from core import storage  # noqa: E402

setup_page("Lost something? Ask the assistant")

WELCOME = ("Hello! I'm the Lost & Found assistant. Tell me what you lost - for example "
           "\"I lost my black umbrella at the library yesterday\" - and I'll check "
           "whether it has been handed in.")

# Conversation state lives in the session so it survives Streamlit reruns.
if "dialogue_state" not in st.session_state:
    st.session_state["dialogue_state"] = {}
    st.session_state["chat_history"] = [("assistant", WELCOME)]
    st.session_state["photo_uploader_key"] = 0

dialogue = LostItemDialogue(st.session_state["dialogue_state"])
history: list[tuple[str, str]] = st.session_state["chat_history"]


def bot_say(replies: list[str]) -> None:
    history.extend(("assistant", reply) for reply in replies)


def run_safely(action) -> None:
    """Run a dialogue step, turning storage/model problems into a chat message."""
    try:
        bot_say(action())
    except storage.StorageError:
        bot_say(["Sorry, the system is busy right now. Please try again in a moment."])
    except Exception:
        bot_say(["Sorry, something went wrong on our side. Please try again or "
                 "visit the Lost & Found counter."])


# ---- Sidebar: optional photo + start over ---------------------------------- #
with st.sidebar:
    st.header("Reference photo (optional)")
    photo = st.file_uploader(
        "Photo of a similar item",
        type=["jpg", "jpeg", "png"],
        disabled=not dialogue.awaiting_photo,
        key=f"photo_{st.session_state['photo_uploader_key']}",
        help="Enabled when the assistant asks for a photo.",
    )
    if photo is not None and dialogue.awaiting_photo:
        if st.button("Use this photo", type="primary"):
            def use_photo() -> list[str]:
                image = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
                return dialogue.attach_photo(get_embedder().embed_image(image))

            history.append(("user", "*(uploaded a reference photo)*"))
            run_safely(use_photo)
            st.session_state["photo_uploader_key"] += 1  # clears the uploader
            st.rerun()

    st.divider()
    if st.button("Start over"):
        dialogue.reset()
        st.session_state["chat_history"] = [("assistant", WELCOME)]
        st.session_state["photo_uploader_key"] += 1
        st.rerun()

# ---- Chat history ---------------------------------------------------------- #
for role, message in history:
    with st.chat_message(role):
        st.markdown(message)

# ---- Inline claim form (only after a possible match) ------------------------ #
if dialogue.awaiting_claim_form:
    with st.chat_message("assistant"):
        with st.form("claim_form"):
            st.markdown("**Your details**")
            name = st.text_input("Full name")
            student_id = st.text_input("Student ID")
            contact = st.text_input("Email or phone")
            if st.form_submit_button("Submit claim", type="primary"):
                run_safely(lambda: dialogue.submit_claim(name, student_id, contact))
                st.rerun()

# ---- Chat input ------------------------------------------------------------ #
if prompt := st.chat_input("Describe your lost item..."):
    history.append(("user", prompt))
    run_safely(lambda: dialogue.handle_message(prompt))
    st.rerun()
