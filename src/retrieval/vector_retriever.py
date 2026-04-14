"""Vector-based document retrieval using ChromaDB similarity search."""
import logging
from typing import Optional

from langchain_core.documents import Document

from src.config import AppConfig, get_config
from src.ingestion.vector_indexer import get_vector_store

logger = logging.getLogger("hybrid_rag.retrieval.vector")


def retrieve_vector(
    query: str,
    top_k: Optional[int] = None,
    config: Optional[AppConfig] = None,
) -> list[tuple[Document, float]]:
    """
    Retrieve the top-k most relevant chunks from ChromaDB.

    Returns:
        List of (Document, relevance_score) tuples.
        Relevance score is in [0, 1] — higher means more similar.
    """
    if config is None:
        config = get_config()

    k = top_k or config.vector_store.top_k
    store = get_vector_store(config)

    results = store.similarity_search_with_relevance_scores(query, k=k)

    logger.debug(
        f"Vector retrieval: {len(results)} result(s) for '{query[:60]}...'"
    )
    return results
