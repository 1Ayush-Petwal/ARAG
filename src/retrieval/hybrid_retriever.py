"""Hybrid retriever: fuses vector (ChromaDB) and graph (Neo4j) results.

Fusion strategy:
  - Vector docs are re-ranked using Reciprocal Rank Fusion (RRF), which is
    score-scale-agnostic and outperforms simple score averaging.
  - Graph triples are deduplicated and appended verbatim — they are already
    structured facts, not ranked passages.
  - The final context string is: [Graph Facts block] + [Vector Passages block].
"""
import logging
from typing import Optional

from langchain_core.documents import Document

from src.config import AppConfig, get_config
from src.retrieval.vector_retriever import retrieve_vector
from src.retrieval.graph_retriever import retrieve_graph

logger = logging.getLogger("hybrid_rag.retrieval.hybrid")

RRF_K = 60  # standard constant; higher = smoother rank influence


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


def fuse_results(
    vector_results: list[tuple[Document, float]],
    graph_facts: list[str],
    top_k: int = 6,
) -> tuple[list[Document], list[str]]:
    """
    Fuse vector and graph results.

    Vector documents are re-ranked via RRF (handles multiple queries or
    duplicate chunks gracefully). Graph facts are deduplicated in order.

    Returns:
        (top_k_vector_docs, deduplicated_graph_facts)
    """
    # RRF over vector results
    rrf: dict[str, float] = {}
    by_id: dict[str, Document] = {}

    for rank, (doc, _score) in enumerate(vector_results):
        doc_id = doc.metadata.get("chunk_id", f"vec_{rank}")
        rrf[doc_id] = rrf.get(doc_id, 0.0) + _rrf_score(rank)
        by_id[doc_id] = doc

    sorted_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)
    top_docs = [by_id[did] for did in sorted_ids[:top_k]]

    # Deduplicate graph facts (preserve order)
    unique_facts = list(dict.fromkeys(graph_facts))

    logger.debug(
        f"Fused: {len(top_docs)} vector doc(s), {len(unique_facts)} graph fact(s)"
    )
    return top_docs, unique_facts


def build_context(vector_docs: list[Document], graph_facts: list[str]) -> str:
    """Render vector docs and graph facts into a single context string for the LLM."""
    parts: list[str] = []

    if graph_facts:
        parts.append("### Knowledge Graph Facts")
        parts.extend(f"- {fact}" for fact in graph_facts)

    if vector_docs:
        parts.append("\n### Retrieved Document Passages")
        for i, doc in enumerate(vector_docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            page_str = f" (p.{page})" if page != "" else ""
            parts.append(f"\n[{i}] {source}{page_str}")
            parts.append(doc.page_content.strip())

    return "\n".join(parts) if parts else "No relevant context found."


def hybrid_retrieve(
    question: str,
    config: Optional[AppConfig] = None,
) -> tuple[list[Document], list[str], str]:
    """
    Run vector + graph retrieval in sequence and return fused results.

    Returns:
        (vector_docs, graph_facts, fused_context_string)
    """
    if config is None:
        config = get_config()

    vector_results = retrieve_vector(question, config=config)
    graph_facts = retrieve_graph(question, config=config)

    top_docs, top_facts = fuse_results(
        vector_results, graph_facts, top_k=config.vector_store.top_k
    )
    context = build_context(top_docs, top_facts)

    return top_docs, top_facts, context
