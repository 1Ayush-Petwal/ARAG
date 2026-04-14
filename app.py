"""Streamlit UI for Adaptive HybridRAG.

Launch with:
    streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive HybridRAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tweak: tighten passage expanders ──────────────────────────────────────
st.markdown(
    "<style>details summary { font-size: 0.85rem; } </style>",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Adaptive HybridRAG")
    st.caption("Knowledge-Graph + Vector retrieval with self-assessment and web fallback")

    st.divider()
    st.subheader("Settings")
    show_trace   = st.checkbox("Show agent route trace", value=True)
    show_context = st.checkbox("Show retrieved context", value=False)

    st.divider()
    st.subheader("Stack")
    st.markdown("""
| Component | Library |
|---|---|
| LLM | Groq (llama-3.1-70b) |
| Fast LLM | Groq (llama-3.1-8b) |
| Embeddings | all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| Graph DB | Neo4j |
| Web search | DuckDuckGo |
| Orchestration | LangGraph |
""")

    st.divider()
    st.caption("See `scripts/ingest.py` to load documents first.")

# ── Main area ─────────────────────────────────────────────────────────────────
st.header("Ask a question")

question = st.text_input(
    label="Your question",
    placeholder="e.g. What were Apple's total revenues in fiscal year 2023?",
    label_visibility="collapsed",
)

run_btn = st.button("Ask", type="primary", use_container_width=False)

if run_btn and question.strip():
    from src.agent.workflow import run_query

    with st.spinner("Retrieving, reasoning, verifying…"):
        try:
            state = run_query(question)
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

    answer          = state.get("answer") or "_No answer generated._"
    route_log       = state.get("route_log", [])
    grounded        = state.get("grounded")
    web_used        = bool(state.get("web_docs"))
    unsupported     = state.get("unsupported_claims", [])
    graph_facts     = state.get("graph_facts", [])
    vector_docs     = state.get("vector_docs", [])
    grade           = state.get("grade", "")
    iterations      = state.get("iterations", 0)

    # ── Answer ────────────────────────────────────────────────────────────────
    col_ans, col_badges = st.columns([4, 1])
    with col_ans:
        st.subheader("Answer")
    with col_badges:
        if web_used:
            st.info("🌐 Web search")
        if grounded is True:
            st.success("✓ Grounded")
        elif grounded is False:
            st.error("⚠ Unverified")

    st.markdown(answer)

    # ── Unsupported claims ────────────────────────────────────────────────────
    if unsupported:
        with st.expander(f"⚠ {len(unsupported)} unsupported claim(s) detected", expanded=True):
            for claim in unsupported:
                st.markdown(f"- {claim}")

    st.divider()

    # ── Stats row ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grade",         grade.upper() if grade else "—")
    c2.metric("Iterations",    str(iterations))
    c3.metric("Vector docs",   len(vector_docs))
    c4.metric("Graph facts",   len(graph_facts))

    # ── Route trace ───────────────────────────────────────────────────────────
    if show_trace:
        st.markdown("**Agent route:**  `" + "` → `".join(route_log) + "`")

    # ── Retrieved context ─────────────────────────────────────────────────────
    if show_context:
        st.subheader("Retrieved Context")

        if graph_facts:
            with st.expander(f"Knowledge Graph ({len(graph_facts)} fact(s))", expanded=True):
                for fact in graph_facts:
                    st.code(fact, language=None)

        if vector_docs:
            with st.expander(f"Vector Passages ({len(vector_docs)} chunk(s))", expanded=False):
                for i, doc in enumerate(vector_docs, 1):
                    src = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page", "")
                    label = f"[{i}] {Path(src).name}" + (f" p.{page}" if page != "" else "")
                    with st.expander(label):
                        st.text(doc.page_content[:600])

        if web_used:
            web_docs = state.get("web_docs", [])
            with st.expander(f"Web Results ({len(web_docs)} result(s))", expanded=False):
                for doc in web_docs:
                    title = doc.metadata.get("title", "Web result")
                    url   = doc.metadata.get("source", "")
                    st.markdown(f"**{title}**  \n[{url}]({url})")
                    st.text(doc.page_content[:300])
                    st.divider()

elif run_btn and not question.strip():
    st.warning("Please enter a question.")
