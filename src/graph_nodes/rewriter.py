"""Query rewriter node.

Reformulates the question when retrieval was graded as 'poor', aiming to
surface better documents on the next retrieval pass. Increments the
iteration counter so the agent can track how many retries have occurred.
"""
import logging

from src.agent.state import GraphState
from src.config import get_config
from src.llm.provider import get_llm
from src.utils.prompts import REWRITER_PROMPT

logger = logging.getLogger("hybrid_rag.nodes.rewriter")


def rewriter_node(state: GraphState) -> dict:
    """
    Rewrite the current question using the grader's failure reason.

    Returns updates to: rewritten_question, iterations, route_log.
    """
    question = (
        state.get("rewritten_question")
        or state.get("analyzed_question")
        or state["question"]
    )
    reason = state.get("grade_reason", "insufficient context retrieved")
    config = get_config()

    llm = get_llm("fast", config)
    chain = REWRITER_PROMPT | llm
    result = chain.invoke({"question": question, "reason": reason})
    rewritten = result.content.strip() or question

    logger.info(f"Rewritten: '{question[:55]}' → '{rewritten[:55]}'")

    return {
        "rewritten_question": rewritten,
        "iterations": state.get("iterations", 0) + 1,
        "route_log": ["query_rewriter"],
    }
