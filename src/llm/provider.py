"""LLM provider abstraction.

Supports Groq, Google Gemini (AI Studio), and Ollama.
Use get_llm("generation") for the heavy answer-generator model and
get_llm("fast") for the cheaper grader/rewriter calls.

The "fast" role can target a different provider than "generation" via
the `llm.fast_provider` config field — useful when a small open-source
model is too weak for structured judgment tasks (graders, hallucination
checker) and we want to route those calls to a stronger small model.
"""
import logging
from functools import lru_cache
from typing import Optional

from langchain_core.language_models import BaseChatModel

from src.config import AppConfig, get_config

logger = logging.getLogger("hybrid_rag.llm.provider")


def _resolve_provider(cfg, model_type: str) -> str:
    """Pick which provider serves a given role."""
    if model_type == "fast" and cfg.fast_provider:
        return cfg.fast_provider
    return cfg.provider


def get_llm(
    model_type: str = "generation",
    config: Optional[AppConfig] = None,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """
    Instantiate and return a chat LLM.

    Args:
        model_type: "generation" for the heavy model, "fast" for light calls.
        config: optional config override (uses singleton if None).
        max_tokens: per-call override of the configured output token cap.
            Use this for callers (e.g. graph extraction) whose tool-call
            output is much larger than the default.
    """
    if config is None:
        config = get_config()

    cfg = config.llm
    provider = _resolve_provider(cfg, model_type)
    model_name = cfg.generation_model if model_type == "generation" else cfg.fast_model
    out_tokens = max_tokens if max_tokens is not None else cfg.max_tokens

    if provider == "groq":
        if not config.groq_api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        from langchain_groq import ChatGroq

        logger.debug(f"Using Groq model: {model_name}")
        return ChatGroq(
            api_key=config.groq_api_key,
            model=model_name,
            temperature=cfg.temperature,
            max_tokens=out_tokens,
        )

    elif provider == "google":
        if not config.google_api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. Add it to your .env file."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.debug(f"Using Google model: {model_name}")
        return ChatGoogleGenerativeAI(
            google_api_key=config.google_api_key,
            model=model_name,
            temperature=cfg.temperature,
            max_output_tokens=out_tokens,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        logger.debug(f"Using Ollama model: {model_name} at {cfg.ollama_base_url}")
        return ChatOllama(
            base_url=cfg.ollama_base_url,
            model=model_name,
            temperature=cfg.temperature,
            num_predict=out_tokens,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            "Set llm.provider (or llm.fast_provider) to one of: "
            "'groq', 'google', 'ollama'."
        )


@lru_cache(maxsize=4)
def get_cached_llm(model_type: str = "generation") -> BaseChatModel:
    """Cached LLM instance — avoids re-initializing on every node call."""
    return get_llm(model_type)
