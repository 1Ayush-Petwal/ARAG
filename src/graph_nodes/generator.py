"""Answer generator node.

Generates the final answer from the fused context (graph facts + vector passages,
and optionally web search results). Applies a web disclaimer in the system prompt
when web docs are present so the LLM labels its answer accordingly.
"""
import logging

from src.agent.state import GraphState
from src.config import get_config
from src.llm.provider import get_llm
from src.utils.prompts import GENERATOR_PROMPT, WEB_DISCLAIMER

logger = logging.getLogger("hybrid_rag.nodes.generator")


def generator_node(state: GraphState) -> dict:
    """
    Generate an answer from the available context.

    If web_docs are present, they are appended to the fused context and
    the WEB_DISCLAIMER is injected into the system prompt.

    Returns updates to: answer, route_log.
    """
    question = (
        state.get("rewritten_question")
        or state.get("analyzed_question")
        or state["question"]
    )

    web_docs = state.get("web_docs") or []
    base_context = state.get("fused_context", "")

    if web_docs:
        web_block = "\n\n".join(d.page_content for d in web_docs)
        context = base_context + "\n\n### Live Web Search Results\n" + web_block
        disclaimer = WEB_DISCLAIMER
    else:
        context = base_context
        disclaimer = ""

    # If a previous answer was ungrounded, instruct the model to be more conservative
    prior_answer = state.get("answer")
    ungrounded_claims = state.get("unsupported_claims") or []
    if prior_answer and ungrounded_claims:
        claims_block = "\n".join(f"- {c}" for c in ungrounded_claims)
        question = (
            f"{question}\n\n"
            f"[Note: A previous attempt contained unsupported claims:\n{claims_block}\n"
            f"Strictly avoid these claims. Only use information from the context.]"
        )

    config = get_config()
    llm = get_llm("generation", config)

    chain = GENERATOR_PROMPT | llm
    result = chain.invoke({
        "question": question,
        "context": context,
        "web_disclaimer": disclaimer,
    })

    answer = result.content.strip()
    logger.debug(f"Generated answer ({len(answer)} chars)")

    return {
        "answer": answer,
        "route_log": ["generator"],
    }
