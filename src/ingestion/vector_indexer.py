"""ChromaDB vector store ingestion.

Uses chunk_id as the Chroma document ID so re-ingestion is idempotent
(duplicate chunks are safely overwritten rather than duplicated).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from langchain_core.documents import Document

from src.config import AppConfig, get_config

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("hybrid_rag.ingestion.vector_indexer")

_embeddings_cache: Optional["HuggingFaceEmbeddings"] = None


def _get_embeddings(config: AppConfig) -> "HuggingFaceEmbeddings":
    from langchain_huggingface import HuggingFaceEmbeddings  # lazy import

    global _embeddings_cache
    if _embeddings_cache is None:
        logger.info(f"Loading embedding model: {config.embedding.model}")
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=config.embedding.model,
            model_kwargs={"device": config.embedding.device},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_cache


def get_vector_store(config: Optional[AppConfig] = None) -> "Chroma":
    """Return the persistent ChromaDB collection (creates it if absent)."""
    if config is None:
        config = get_config()

    from langchain_chroma import Chroma  # lazy import

    persist_dir = Path(config.vector_store.persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=config.vector_store.collection_name,
        embedding_function=_get_embeddings(config),
        persist_directory=str(persist_dir),
    )


def ingest_chunks(
    chunks: list[Document],
    config: Optional[AppConfig] = None,
) -> Chroma:
    """
    Embed and store document chunks in ChromaDB.

    Re-ingestion is idempotent: existing chunks with the same chunk_id
    are overwritten.
    """
    if config is None:
        config = get_config()

    if not chunks:
        logger.warning("No chunks provided — skipping vector ingestion.")
        return get_vector_store(config)

    store = get_vector_store(config)
    ids = [c.metadata.get("chunk_id", str(i)) for i, c in enumerate(chunks)]

    logger.info(
        f"Ingesting {len(chunks)} chunks into ChromaDB "
        f"collection '{config.vector_store.collection_name}'..."
    )
    store.add_documents(documents=chunks, ids=ids)

    count = store._collection.count()
    logger.info(f"Vector ingestion complete. Collection now contains {count} documents.")
    return store
