"""Query analyzer node.

Optionally rewrites the raw question to be more retrieval-friendly
before any retrieval is attempted. Uses the fast model to keep latency low.
"""
import logging

from src.agent.state import GraphState
from src.config import get_config
from src.llm.provider import get_llm
from src.utils.prompts import QUERY_ANALYZER_PROMPT

logger = logging.getLogger("hybrid_rag.nodes.query_analyzer")


def query_analyzer_node(state: GraphState) -> dict:
    """
    Analyze and optionally refine the user question.

    Returns updates to: analyzed_question, route_log, iterations (reset to 0).
    """
    question = state["question"]
    config = get_config()
    llm = get_llm("fast", config)

    chain = QUERY_ANALYZER_PROMPT | llm
    result = chain.invoke({"question": question})
    analyzed = result.content.strip() or question

    if analyzed != question:
        logger.info(f"Query refined: '{question[:60]}' → '{analyzed[:60]}'")
    else:
        logger.debug(f"Query unchanged: '{question[:60]}'")

    return {
        "analyzed_question": analyzed,
        "route_log": ["query_analyzer"],
        "iterations": 0,
    }
