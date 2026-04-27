"""Neo4j knowledge-graph ingestion via LLMGraphTransformer.

Extracts typed entities and relationships from document chunks using an LLM
and stores them in Neo4j. Each node carries a `source_chunk_id` linking back
to the originating ChromaDB chunk for provenance.

Run this ONCE per corpus (it's expensive — batch-processes with the LLM).
Results persist in Neo4j, so subsequent runs are cheap if chunks haven't changed.
"""
import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph

from src.config import AppConfig, get_config
from src.llm.provider import get_llm

logger = logging.getLogger("hybrid_rag.ingestion.graph_indexer")


def get_graph(config: Optional[AppConfig] = None) -> Neo4jGraph:
    """Open and return a Neo4j graph connection."""
    if config is None:
        config = get_config()
    return Neo4jGraph(
        url=config.neo4j_uri,
        username=config.neo4j_username,
        password=config.neo4j_password,
        database=config.neo4j_database,
        enhanced_schema=True,
    )


def _create_indexes(graph: Neo4jGraph) -> None:
    """Create fulltext index and ensure the entity id constraint exists.

    add_graph_documents(baseEntityLabel=True) internally creates a uniqueness
    constraint on (:__Entity__ {id}).  If a prior run left a plain index on
    that property, Neo4j rejects the constraint with IndexAlreadyExists.
    We drop the plain index first so the constraint can be (re-)created cleanly.
    """
    # Drop the plain index that conflicts with the uniqueness constraint
    try:
        graph.query("DROP INDEX entity_id_index IF EXISTS")
    except Exception as exc:
        logger.debug(f"entity_id_index drop note: {exc}")

    # Uniqueness constraint on __Entity__.id — same property add_graph_documents
    # needs; creating it here makes subsequent calls idempotent.
    try:
        graph.query(
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
            "FOR (n:__Entity__) REQUIRE n.id IS UNIQUE"
        )
    except Exception as exc:
        logger.debug(f"entity_id_unique constraint note: {exc}")

    # Fulltext index on entity names (used by graph_retriever fallback)
    try:
        graph.query(
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
            "FOR (n:__Entity__) ON EACH [n.id]"
        )
    except Exception as exc:
        logger.debug(f"entity_fulltext index note: {exc}")

    logger.info("Neo4j indexes ensured.")


def ingest_to_graph(
    chunks: list[Document],
    config: Optional[AppConfig] = None,
    batch_size: int = 4,
) -> Neo4jGraph:
    """
    Extract entities + relationships from chunks and store them in Neo4j.

    Args:
        chunks: document chunks produced by chunker.py
        config: optional config override
        batch_size: number of chunks per LLM extraction batch

    Returns:
        The Neo4jGraph connection (ready for querying).
    """
    if config is None:
        config = get_config()

    # Graph extraction emits a large structured tool-call (all nodes +
    # relationships per batch). The default max_tokens (2048) truncates that
    # JSON mid-stream and Groq returns 400 tool_use_failed. Give this LLM a
    # much bigger output budget so the function call can complete.
    llm = get_llm("generation", config, max_tokens=8192)
    graph = get_graph(config)

    # Drop conflicting plain index and ensure uniqueness constraint before
    # add_graph_documents runs — otherwise Neo4j raises IndexAlreadyExists.
    _create_indexes(graph)

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=config.graph_store.allowed_nodes,
        allowed_relationships=config.graph_store.allowed_relationships,
        node_properties=["description"],
        relationship_properties=["description"],
        strict_mode=False,
    )

    total_nodes = 0
    total_rels = 0
    num_batches = (len(chunks) + batch_size - 1) // batch_size

    logger.info(
        f"Starting graph extraction: {len(chunks)} chunks, "
        f"{num_batches} batch(es) of {batch_size}..."
    )

    for batch_idx in range(num_batches):
        batch = chunks[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        logger.info(f"  Batch {batch_idx + 1}/{num_batches} ({len(batch)} chunks)...")

        try:
            graph_docs = transformer.convert_to_graph_documents(batch)

            for gd in graph_docs:
                # Attach chunk_id provenance to all nodes
                chunk_id = gd.source.metadata.get("chunk_id", "")
                for node in gd.nodes:
                    if node.properties is None:
                        node.properties = {}
                    node.properties["source_chunk_id"] = chunk_id
                total_nodes += len(gd.nodes)
                total_rels += len(gd.relationships)

            graph.add_graph_documents(
                graph_docs,
                baseEntityLabel=True,
                include_source=True,
            )
        except Exception as exc:
            logger.error(f"  Batch {batch_idx + 1} failed: {exc} — skipping.")
            continue

    logger.info(
        f"Graph ingestion complete: "
        f"{total_nodes} node(s), {total_rels} relationship(s) extracted."
    )

    return graph
