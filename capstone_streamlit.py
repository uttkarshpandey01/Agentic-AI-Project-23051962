"""
capstone_streamlit.py — Physics Study Buddy
Run: streamlit run capstone_streamlit.py
"""

import os
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Physics Study Buddy",
    page_icon="⚛️",
    layout="centered",
)

st.title("⚛️ Physics Study Buddy")
st.caption(
    "AI-powered physics tutor — concepts, problem-solving, and live web search. "
    "Powered by LangGraph + ChromaDB + Groq."
)

# ── Guard: GROQ key ────────────────────────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    st.error(
        "**GROQ_API_KEY not set.**\n\n"
        "Create a `.env` file in the project folder with:\n```\nGROQ_API_KEY=your_key_here\n```"
    )
    st.stop()

# ── Load agent (cached so it only builds once per session) ────────────────────
@st.cache_resource(show_spinner="Loading physics knowledge base…")
def load_agent():
    from agent import get_app, DOCUMENTS  # noqa: PLC0415
    app, embedder, collection = get_app()
    return app, embedder, collection, DOCUMENTS


app, embedder, collection, DOCUMENTS = load_agent()

# ── Sidebar: knowledge base topics ────────────────────────────────────────────
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.write(f"**{len(DOCUMENTS)} documents loaded**")
    for doc in DOCUMENTS:
        st.markdown(f"- {doc['topic']}")
    st.divider()
    st.markdown(
        "**Tools available:**\n"
        "- ChromaDB semantic search\n"
        "- DuckDuckGo web search\n"
        "- Conversation memory (last 3 turns)\n"
        "- Faithfulness self-evaluation"
    )
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a physics question…"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            try:
                result = app.invoke({"question": prompt}, config=config)
                answer      = result.get("answer", "Sorry, I couldn't generate an answer.")
                route       = result.get("route", "retrieve")
                sources     = result.get("sources", [])
                faithfulness = result.get("faithfulness", 1.0)
            except Exception as e:
                answer       = f"⚠️ Error: {e}"
                route        = "error"
                sources      = []
                faithfulness = 0.0

        st.markdown(answer)

        # Show metadata in an expander
        with st.expander("🔍 Agent details", expanded=False):
            col1, col2 = st.columns(2)
            col1.metric("Route", route)
            col2.metric("Faithfulness", f"{faithfulness:.2f}")
            if sources:
                st.markdown("**Sources used:**")
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
