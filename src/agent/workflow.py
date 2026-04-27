"""LangGraph workflow assembly for Adaptive HybridRAG.

Graph topology
──────────────
  query_analyzer
       │
  hybrid_retriever  ◄──────────────────────────┐
       │                                        │
  relevance_grader                              │
       │                                        │
   ┌───┴──────────────┐                         │
   │sufficient │poor  │off_topic                │
   ▼           ▼      ▼                         │
generator  rewriter  web_searcher ──────────────┤
   │           │                                │
   │           └──► hybrid_retriever (loop) ────┘
   │
hallucination_checker  (advisory — tags state, never loops)
   │
   ▼
  END
"""
import logging
from functools import lru_cache
from typing import Optional

from langgraph.graph import END, StateGraph

from src.agent.state import GraphState
from src.config import get_config
from src.graph_nodes.grader import decide_after_grade, grader_node
from src.graph_nodes.generator import generator_node
from src.graph_nodes.hallucination_checker import hallucination_checker_node
from src.graph_nodes.query_analyzer import query_analyzer_node
from src.graph_nodes.retriever_node import retriever_node
from src.graph_nodes.rewriter import rewriter_node
from src.utils.logging import RunLogger

logger = logging.getLogger("hybrid_rag.agent.workflow")
_run_logger = RunLogger()


def _web_search_node(state: GraphState) -> dict:
    """Inline web-search node (keeps workflow.py self-contained)."""
    from src.retrieval.web_searcher import web_search

    config = get_config()
    question = (
        state.get("rewritten_question")
        or state.get("analyzed_question")
        or state["question"]
    )
    docs = web_search(question, config=config)
    return {
        "web_docs": docs,
        "route_log": ["web_searcher"],
        "learned_new_facts": len(docs) > 0,
    }


def build_workflow() -> StateGraph:
    """Construct the LangGraph StateGraph (not yet compiled)."""
    wf = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────────
    wf.add_node("query_analyzer",       query_analyzer_node)
    wf.add_node("hybrid_retriever",     retriever_node)
    wf.add_node("relevance_grader",     grader_node)
    wf.add_node("query_rewriter",       rewriter_node)
    wf.add_node("web_searcher",         _web_search_node)
    wf.add_node("generator",            generator_node)
    wf.add_node("hallucination_checker", hallucination_checker_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    wf.set_entry_point("query_analyzer")

    # ── Fixed edges ───────────────────────────────────────────────────────────
    wf.add_edge("query_analyzer",   "hybrid_retriever")
    wf.add_edge("hybrid_retriever", "relevance_grader")

    # ── Grader conditional routing ────────────────────────────────────────────
    wf.add_conditional_edges(
        "relevance_grader",
        decide_after_grade,
        {
            "generate":   "generator",
            "rewrite":    "query_rewriter",
            "web_search": "web_searcher",
        },
    )

    # ── Retry loop: rewriter → retriever ──────────────────────────────────────
    wf.add_edge("query_rewriter", "hybrid_retriever")

    # ── Web search → generator ────────────────────────────────────────────────
    wf.add_edge("web_searcher", "generator")

    # ── Generator → hallucination check ──────────────────────────────────────
    wf.add_edge("generator", "hallucination_checker")

    # ── Hallucination check is ADVISORY ───────────────────────────────────────
    # Earlier the workflow regenerated on a failed check, but the fast model
    # frequently false-positives on grounded answers, causing infinite loops
    # that returned the same answer while burning rate-limit budget. The
    # check still runs and tags state.grounded / state.unsupported_claims so
    # the UI can surface low-confidence warnings — but it no longer gates
    # the response.
    wf.add_edge("hallucination_checker", END)

    return wf


@lru_cache(maxsize=1)
def get_compiled_workflow():
    """Return a compiled (cached) LangGraph app instance."""
    return build_workflow().compile()


def run_query(question: str) -> dict:
    """
    Run a natural-language question through the full HybridRAG pipeline.

    Args:
        question: the user's question

    Returns:
        Final GraphState dict containing 'answer', 'route_log', 'grounded', etc.
    """
    app = get_compiled_workflow()

    initial: GraphState = {
        "question":           question,
        "analyzed_question":  "",
        "vector_docs":        [],
        "graph_facts":        [],
        "fused_context":      "",
        "grade":              "poor",
        "grade_reason":       "",
        "rewritten_question": None,
        "web_docs":           [],
        "answer":             None,
        "grounded":           None,
        "unsupported_claims": [],
        "iterations":         0,
        "route_log":          [],
        "learned_new_facts":  False,
    }

    logger.info(f"Query: '{question}'")
    final_state = app.invoke(initial)

    try:
        _run_logger.log_run(final_state)
    except Exception:
        pass  # logging failure must never crash the pipeline

    return final_state
