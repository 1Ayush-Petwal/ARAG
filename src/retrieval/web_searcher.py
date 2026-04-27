"""DuckDuckGo web-search fallback.

Results are wrapped in <untrusted_source> XML tags so the generator prompt
can signal the LLM to treat them as potentially adversarial content and to
ignore any embedded instructions (prompt-injection mitigation).
"""
import hashlib
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

        chunk_id = f"web_{hashlib.md5(href.encode('utf-8')).hexdigest()[:8]}"

        # Wrap in untrusted_source to mitigate prompt injection
        content = (
            f"<untrusted_source title='{title}' url='{href}'>\n"
            f"{body}\n"
            f"</untrusted_source>"
        )
        docs.append(Document(
            page_content=content,
            metadata={"source": href, "title": title, "type": "web_search", "chunk_id": chunk_id},
        ))

    logger.info(f"Web search: {len(docs)} result(s) for '{query[:60]}...'")

    # Self-healing: optionally pipe new web docs into the Neo4j graph so
    # follow-up queries can be answered from the graph instead of going
    # back to the web. Gated behind config because the LLM extraction is
    # slow and consumes generation-LLM budget. Import lazily to avoid a
    # circular import (graph_indexer → llm → config → web_searcher).
    if docs and config.web_search.auto_ingest:
        from src.ingestion.graph_indexer import ingest_to_graph

        logger.info("Self-healing: ingesting web docs into Neo4j...")
        try:
            ingest_to_graph(docs, config=config, batch_size=len(docs))
        except Exception as exc:
            logger.error(f"Neo4j auto-ingest for web docs failed: {exc}")

    return docs
