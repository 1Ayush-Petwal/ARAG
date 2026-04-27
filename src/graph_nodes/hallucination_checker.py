"""Hallucination checker node.

Verifies that the generated answer is grounded in the retrieved context.

NOTE: This check is ADVISORY in the current workflow — it tags
state.grounded / state.unsupported_claims so callers can surface a
low-confidence warning, but it does NOT loop the workflow back to the
generator. The previous regenerate loop produced infinite-retry storms
when the fast model false-positived on correctly grounded answers.
"""
import logging

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


