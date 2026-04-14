"""LLM provider abstraction.

Supports Groq (default, fast free tier) and Ollama (local fallback).
Use get_llm("generation") for the heavy answer-generator model and
get_llm("fast") for the cheaper grader/rewriter calls.
"""
import logging
from functools import lru_cache
from typing import Optional

from langchain_core.language_models import BaseChatModel

from src.config import AppConfig, get_config

logger = logging.getLogger("hybrid_rag.llm.provider")


def get_llm(
    model_type: str = "generation",
    config: Optional[AppConfig] = None,
) -> BaseChatModel:
    """
    Instantiate and return a chat LLM.

    Args:
        model_type: "generation" for the heavy model, "fast" for light calls.
        config: optional config override (uses singleton if None).
    """
    if config is None:
        config = get_config()

    cfg = config.llm
    model_name = cfg.generation_model if model_type == "generation" else cfg.fast_model

    if cfg.provider == "groq":
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
            max_tokens=cfg.max_tokens,
        )

    elif cfg.provider == "ollama":
        from langchain_ollama import ChatOllama

        logger.debug(f"Using Ollama model: {model_name} at {cfg.ollama_base_url}")
        return ChatOllama(
            base_url=cfg.ollama_base_url,
            model=model_name,
            temperature=cfg.temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: {cfg.provider!r}. "
            "Set llm.provider to 'groq' or 'ollama' in config.yaml."
        )


@lru_cache(maxsize=4)
def get_cached_llm(model_type: str = "generation") -> BaseChatModel:
    """Cached LLM instance — avoids re-initializing on every node call."""
    return get_llm(model_type)
