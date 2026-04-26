"""LangGraph state definition for Adaptive HybridRAG.

GraphState is a TypedDict that flows through every node in the workflow.
The `route_log` field uses operator.add so each node can append its name
without overwriting the accumulated log from prior nodes.
"""
import operator
from typing import Annotated, Literal, Optional

from langchain_core.documents import Document
from typing_extensions import TypedDict


class GraphState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    question: str

    # ── After query_analyzer ─────────────────────────────────────────────────
    analyzed_question: str

    # ── After hybrid_retriever ───────────────────────────────────────────────
    vector_docs: list[Document]
    graph_facts: list[str]
    fused_context: str

    # ── After relevance_grader ───────────────────────────────────────────────
    grade: Literal["sufficient", "poor", "off_topic"]
    grade_reason: str

    # ── After query_rewriter ─────────────────────────────────────────────────
    rewritten_question: Optional[str]

    # ── After web_searcher ───────────────────────────────────────────────────
    web_docs: list[Document]

    # ── After generator ──────────────────────────────────────────────────────
    answer: Optional[str]

    # ── After hallucination_checker ──────────────────────────────────────────
    grounded: Optional[bool]
    unsupported_claims: list[str]

    # ── Control flow ─────────────────────────────────────────────────────────
    iterations: int
    # operator.add ensures each node appends rather than replaces the log
    route_log: Annotated[list[str], operator.add]
    learned_new_facts: Optional[bool]

