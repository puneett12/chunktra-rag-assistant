"""Minimal browser UI: ask a question, read the answer, inspect the sources.

Run:  make ui
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app.generation.answer import answer_question  # noqa: E402
from app.generation.llm import health as llm_health  # noqa: E402
from app.retrieval import store  # noqa: E402

st.set_page_config(page_title="Clause", page_icon="📄", layout="wide")
st.title("Clause")
st.caption("Ask a question about your documents. Every answer cites its source.")

with st.sidebar:
    st.subheader("System")
    st.json(store.stats())
    h = llm_health()
    (st.success if h.get("ok") else st.error)(
        f"LLM: {'connected' if h.get('ok') else h.get('error', 'unavailable')}"
    )
    st.divider()
    top_k = st.slider("Sources in context (k)", 1, 10, 5)
    use_rerank = st.checkbox(
        "Use reranker", value=True,
        help="Turn off to see how much the reranker is contributing.",
    )

question = st.text_input(
    "Question",
    placeholder="What is the notice period for terminating the master services agreement?",
)

if question:
    with st.spinner("Retrieving and generating..."):
        result = answer_question(question, top_k=top_k, use_rerank=use_rerank)

    if result.get("error"):
        st.error(result["error"])
    else:
        st.markdown("### Answer")
        st.write(result["answer"])
        if not result.get("grounded"):
            st.info("The system declined to answer — the corpus does not cover this.")

    st.markdown("### Sources")
    for s in result.get("sources", []):
        label = f"[{s['n']}] {s['title']} — page {s['page']} · {s['section']}"
        with st.expander(label):
            st.caption(
                f"rerank score {s['rerank_score']} · found by {', '.join(s['found_by'])}"
            )
            st.text(s["excerpt"])
            st.caption(s["source_path"])
