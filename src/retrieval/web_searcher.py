"""DuckDuckGo web-search fallback.

Results are wrapped in <untrusted_source> XML tags so the generator prompt
can signal the LLM to treat them as potentially adversarial content and to
ignore any embedded instructions (prompt-injection mitigation).
"""
import logging
from typing import Optional

from langchain_core.documents import Document

from src.config import AppConfig, get_config

logger = logging.getLogger("hybrid_rag.retrieval.web")


def web_search(
    query: str,
    max_results: Optional[int] = None,
    config: Optional[AppConfig] = None,
) -> list[Document]:
    """
    Search DuckDuckGo and return results as LangChain Documents.

    Args:
        query: the search query (typically the rewritten question)
        max_results: how many results to fetch (default from config)
        config: optional config override

    Returns:
        List of Documents with web content. Empty list on failure.
    """
    if config is None:
        config = get_config()

    n = max_results or config.web_search.max_results

    try:
        # Package was renamed `duckduckgo_search` -> `ddgs`. Prefer the new
        # name; fall back to the old one so existing installs keep working.
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=n))
    except Exception as exc:
        logger.error(f"DuckDuckGo search failed for '{query[:60]}': {exc}")
        return []

    docs: list[Document] = []
    for item in raw:
        title = item.get("title", "")
        body = item.get("body", "")
        href = item.get("href", "")

        # Wrap in untrusted_source to mitigate prompt injection
        content = (
            f"<untrusted_source title='{title}' url='{href}'>\n"
            f"{body}\n"
            f"</untrusted_source>"
        )
        docs.append(Document(
            page_content=content,
            metadata={"source": href, "title": title, "type": "web_search"},
        ))

    logger.info(f"Web search: {len(docs)} result(s) for '{query[:60]}...'")
    return docs
