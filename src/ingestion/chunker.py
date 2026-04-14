"""Text chunking for ingestion pipeline.

Chunks get a stable `chunk_id` that is shared between the vector store
and the knowledge graph, enabling cross-store provenance linking.
"""
import hashlib
import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_config

logger = logging.getLogger("hybrid_rag.ingestion.chunker")


def _make_chunk_id(doc: Document, index: int) -> str:
    """Produce a deterministic chunk ID from source path + content hash."""
    source = doc.metadata.get("source", "unknown")
    content_hash = hashlib.md5(doc.page_content[:200].encode()).hexdigest()[:8]
    return f"{source}::chunk_{index}::{content_hash}"


def chunk_documents(
    documents: list[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Document]:
    """
    Split documents into overlapping text chunks.

    Each chunk receives a `chunk_id` metadata field that is consistent
    across both the ChromaDB vector store and Neo4j graph store, so
    graph nodes can reference the originating text passage.
    """
    config = get_config()
    size = chunk_size or config.chunker.chunk_size
    overlap = chunk_overlap or config.chunker.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = _make_chunk_id(chunk, i)

    logger.info(
        f"Split {len(documents)} document(s) into {len(chunks)} chunks "
        f"(size={size}, overlap={overlap})"
    )
    return chunks
