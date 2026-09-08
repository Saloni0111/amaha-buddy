import streamlit as st

from agent.state import ConversationState
from agent.orchestrator import process_turn

st.set_page_config(page_title="AmahaBuddy", page_icon="🧭")

st.title("AmahaBuddy")
st.caption(
    "Tell me a bit about what's going on, and I'll help you figure out what kind of "
    "support might fit. This isn't therapy or diagnosis -- just a starting point."
)

if "state" not in st.session_state:
    st.session_state.state = ConversationState()
if "recommendation" not in st.session_state:
    st.session_state.recommendation = None

# Render existing history
for msg in st.session_state.state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Show recommendation card if one exists
if st.session_state.recommendation:
    rec = st.session_state.recommendation
    with st.container(border=True):
        st.subheader(f"Suggested next step: {rec['category']}")
        if rec.get("description"):
            st.write(rec["description"])
        st.caption(rec["reasoning"])

user_input = st.chat_input("What's on your mind?")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("..."):
        state, reply, recommendation = process_turn(st.session_state.state, user_input)

    st.session_state.state = state
    if recommendation:
        st.session_state.recommendation = recommendation

    with st.chat_message("assistant"):
        st.write(reply)

    if recommendation:
        st.rerun()

with st.sidebar:
    st.subheader("AmahaBuddy")
    if st.button("Reset conversation"):
        st.session_state.state = ConversationState()
        st.session_state.recommendation = None
        st.rerun()
