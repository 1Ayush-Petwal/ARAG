"""Hybrid retrieval node.

Runs vector + graph retrieval on the current best version of the question
(rewritten > analyzed > original) and stores the fused results in state.
"""
import logging

from src.agent.state import GraphState
from src.config import get_config
from src.retrieval.hybrid_retriever import hybrid_retrieve

logger = logging.getLogger("hybrid_rag.nodes.retriever")


def retriever_node(state: GraphState) -> dict:
    """
    Execute hybrid (vector + graph) retrieval.

    Prefers rewritten_question (if the rewriter has run) over analyzed_question.
    """
    question = (
        state.get("rewritten_question")
        or state.get("analyzed_question")
        or state["question"]
    )

    config = get_config()
    vector_docs, graph_facts, fused_context = hybrid_retrieve(question, config=config)

    logger.info(
        f"Retrieved {len(vector_docs)} vector doc(s) "
        f"and {len(graph_facts)} graph fact(s)."
    )

    return {
        "vector_docs": vector_docs,
        "graph_facts": graph_facts,
        "fused_context": fused_context,
        "route_log": ["hybrid_retriever"],
    }
