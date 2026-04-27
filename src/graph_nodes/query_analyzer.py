"""Query analyzer node.

Optionally rewrites the raw question to be more retrieval-friendly
before any retrieval is attempted. Uses the fast model to keep latency low.
"""
import logging
import re

from src.agent.state import GraphState
from src.config import get_config
from src.llm.provider import get_llm
from src.utils.prompts import QUERY_ANALYZER_PROMPT

logger = logging.getLogger("hybrid_rag.nodes.query_analyzer")

# A question is considered "well-formed" if it's reasonably short, ends with
# a question mark, and starts with a wh-/aux question word. For these we skip
# the LLM rewrite — empirically the fast model often drifts the vocabulary
# (e.g. "last updated" → "current version") which then sinks retrieval.
_QUESTION_WORDS = (
    "who", "what", "when", "where", "why", "how", "which", "whose", "whom",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "should", "would", "will", "may", "might", "list", "name", "give",
    "tell", "show", "explain", "describe", "summarize", "compare",
)
_MAX_PASSTHROUGH_WORDS = 25


def _is_well_formed(question: str) -> bool:
    q = question.strip()
    if not q:
        return False
    if len(q.split()) > _MAX_PASSTHROUGH_WORDS:
        return False
    first = re.split(r"\s+", q.lower(), maxsplit=1)[0].strip("'\"`")
    return first in _QUESTION_WORDS


def query_analyzer_node(state: GraphState) -> dict:
    """
    Analyze and optionally refine the user question.

    Returns updates to: analyzed_question, route_log, iterations (reset to 0).
    """
    question = state["question"]

    # Pass-through fast path: short, well-formed questions are sent to
    # retrieval verbatim. The rewriter still kicks in on grade=poor if
    # retrieval underperforms.
    if _is_well_formed(question):
        logger.debug(f"Query passthrough (well-formed): '{question[:60]}'")
        return {
            "analyzed_question": question,
            "route_log": ["query_analyzer:passthrough"],
            "iterations": 0,
        }

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
