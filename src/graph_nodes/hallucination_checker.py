"""Hallucination checker node.

Verifies that the generated answer is grounded in the retrieved context.
If unsupported claims are found and the iteration budget allows, the
workflow loops back to the generator with an explicit correction prompt.
"""
import logging
from typing import Literal

from pydantic import BaseModel

from src.agent.state import GraphState
from src.config import get_config
from src.llm.provider import get_llm
from src.utils.prompts import HALLUCINATION_CHECKER_PROMPT

logger = logging.getLogger("hybrid_rag.nodes.hallucination_checker")


class HallucinationOutput(BaseModel):
    grounded: bool
    unsupported_claims: list[str]


def hallucination_checker_node(state: GraphState) -> dict:
    """
    Check whether the generated answer is fully grounded in retrieved context.

    Returns updates to: grounded, unsupported_claims, route_log.
    """
    answer = state.get("answer") or ""
    context = state.get("fused_context") or ""
    config = get_config()

    llm = get_llm("fast", config)
    structured_llm = llm.with_structured_output(HallucinationOutput)

    chain = HALLUCINATION_CHECKER_PROMPT | structured_llm
    result: HallucinationOutput = chain.invoke({
        "answer": answer,
        "context": context,
    })

    if result.grounded:
        logger.info("Hallucination check PASSED — answer is grounded.")
    else:
        logger.warning(
            f"Hallucination check FAILED — "
            f"{len(result.unsupported_claims)} unsupported claim(s): "
            + "; ".join(result.unsupported_claims[:2])
        )

    status = "pass" if result.grounded else "fail"
    return {
        "grounded": result.grounded,
        "unsupported_claims": result.unsupported_claims,
        "route_log": [f"hallucination_check:{status}"],
    }


def decide_after_hallucination_check(state: GraphState) -> Literal["end", "regenerate"]:
    """
    Conditional edge — end if grounded; regenerate if ungrounded and budget allows.
    """
    grounded = state.get("grounded", True)
    iterations = state.get("iterations", 0)
    max_iter = get_config().agent.max_iterations

    if grounded or iterations >= max_iter:
        return "end"
    return "regenerate"
