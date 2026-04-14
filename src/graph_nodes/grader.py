"""Relevance grader node.

Judges whether the retrieved context is sufficient to answer the question.
Uses Pydantic structured output so the grade is always a controlled enum value.
"""
import logging
from typing import Literal

from pydantic import BaseModel

from src.agent.state import GraphState
from src.config import get_config
from src.llm.provider import get_llm
from src.utils.prompts import GRADER_PROMPT

logger = logging.getLogger("hybrid_rag.nodes.grader")


class GradeOutput(BaseModel):
    grade: Literal["sufficient", "poor", "off_topic"]
    reason: str


def grader_node(state: GraphState) -> dict:
    """
    Grade the relevance of retrieved context to the current question.

    Returns updates to: grade, grade_reason, route_log.
    """
    question = (
        state.get("rewritten_question")
        or state.get("analyzed_question")
        or state["question"]
    )
    context = state["fused_context"]
    config = get_config()

    llm = get_llm("fast", config)
    structured_llm = llm.with_structured_output(GradeOutput)

    chain = GRADER_PROMPT | structured_llm
    result: GradeOutput = chain.invoke({"question": question, "context": context})

    logger.info(f"Grade: [{result.grade.upper()}] — {result.reason[:80]}")

    return {
        "grade": result.grade,
        "grade_reason": result.reason,
        "route_log": [f"grader:{result.grade}"],
    }


def decide_after_grade(state: GraphState) -> str:
    """
    Conditional edge function — routes to the next node based on grade.

    Routing logic:
      sufficient → generator
      off_topic  → web_searcher (skip retry loop, go straight to web)
      poor       → query_rewriter if iterations remain, else web_searcher
    """
    grade = state["grade"]
    iterations = state.get("iterations", 0)
    max_iter = get_config().agent.max_iterations

    if grade == "sufficient":
        return "generate"
    if grade == "off_topic":
        return "web_search"
    # grade == "poor"
    if iterations < max_iter - 1:
        return "rewrite"
    return "web_search"
