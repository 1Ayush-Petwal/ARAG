#!/usr/bin/env python3
"""CLI ingestion script.

Usage:
    # Ingest a whole folder (vector + graph)
    python -m scripts.ingest data/raw

    # Vector-only (no Neo4j required)
    python -m scripts.ingest data/raw --vector-only

    # Graph-only (vector store already populated)
    python -m scripts.ingest data/raw --graph-only

    # Smaller LLM batch if Groq rate-limits you
    python -m scripts.ingest data/raw --batch-size 5
"""
import argparse
import sys
from pathlib import Path

# Allow execution from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.loader import load_directory, load_file
from src.ingestion.chunker import chunk_documents
from src.ingestion.vector_indexer import ingest_chunks
from src.ingestion.graph_indexer import ingest_to_graph
from src.utils.logging import logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Adaptive HybridRAG stores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", help="File or directory to ingest.")
    parser.add_argument(
        "--vector-only",
        action="store_true",
        help="Skip graph ingestion (no Neo4j connection required).",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Skip vector ingestion (ChromaDB already populated).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        metavar="N",
        help="Chunks per LLM batch for graph extraction (default: 10).",
    )
    args = parser.parse_args()

    if args.vector_only and args.graph_only:
        parser.error("--vector-only and --graph-only are mutually exclusive.")

    input_path = Path(args.path).resolve()

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading documents from: {input_path}")
    if input_path.is_dir():
        docs = load_directory(input_path)
    elif input_path.is_file():
        docs = load_file(input_path)
    else:
        logger.error(f"Path does not exist: {input_path}")
        sys.exit(1)

    if not docs:
        logger.error("No documents loaded. Check the path and file formats.")
        sys.exit(1)

    # ── Chunk ─────────────────────────────────────────────────────────────────
    logger.info("Chunking documents...")
    chunks = chunk_documents(docs)
    logger.info(f"{len(chunks)} chunk(s) ready for ingestion.")

    # ── Vector store ──────────────────────────────────────────────────────────
    if not args.graph_only:
        logger.info("Ingesting into ChromaDB (vector store)...")
        ingest_chunks(chunks)
        logger.info("Vector ingestion complete.")

    # ── Graph store ───────────────────────────────────────────────────────────
    if not args.vector_only:
        logger.info("Ingesting into Neo4j (knowledge graph)...")
        ingest_to_graph(chunks, batch_size=args.batch_size)
        logger.info("Graph ingestion complete.")

    logger.info("All ingestion done.")


if __name__ == "__main__":
    main()
