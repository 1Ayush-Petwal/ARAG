"""Graph-based retrieval from the Neo4j knowledge graph.

Strategy:
  1. Extract named entities from the question via a fast LLM call.
  2. For each entity, query its 1-hop neighbourhood in the graph and
     serialise the result as human-readable triples.
  3. Fall back to a fulltext search across all entity names when
     entity-linking returns nothing.
"""
from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from src.config import AppConfig, get_config
from src.llm.provider import get_llm
from src.utils.prompts import ENTITY_EXTRACTOR_PROMPT

if TYPE_CHECKING:
    from langchain_neo4j import Neo4jGraph

logger = logging.getLogger("hybrid_rag.retrieval.graph")


def _extract_entities(question: str, config: AppConfig) -> list[str]:
    """Use a fast LLM call to pull named entities from the question."""
    llm = get_llm("fast", config)
    chain = ENTITY_EXTRACTOR_PROMPT | llm
    result = chain.invoke({"question": question})
    raw = result.content.strip()
    if not raw:
        return []
    return [e.strip() for e in raw.split(",") if e.strip()]


def _query_entity_neighbourhood(graph: "Neo4jGraph", entity_name: str) -> list[str]:
    """Return 1-hop triples for a given entity name (case-insensitive match)."""
    try:
        rows = graph.query(
            """
            MATCH (n:__Entity__)
            WHERE toLower(n.id) CONTAINS toLower($name)
            WITH n LIMIT 3
            OPTIONAL MATCH (n)-[r]-(m:__Entity__)
            RETURN
                n.id           AS source,
                type(r)        AS rel,
                m.id           AS target,
                n.description  AS src_desc,
                m.description  AS tgt_desc
            LIMIT 30
            """,
            {"name": entity_name},
        )
    except Exception as exc:
        logger.error(f"Graph neighbourhood query failed for '{entity_name}': {exc}")
        return []

    triples: list[str] = []
    for row in rows:
        src = row.get("source") or "?"
        rel = row.get("rel") or "RELATES_TO"
        tgt = row.get("target")
        if tgt:
            triples.append(f"{src} --[{rel}]--> {tgt}")
    return triples


def _fulltext_fallback(graph: "Neo4jGraph", question: str) -> list[str]:
    """Full-text search over entity IDs when entity-linking finds nothing."""
    search_term = " ".join(question.split()[:6])  # first 6 words
    try:
        rows = graph.query(
            """
            CALL db.index.fulltext.queryNodes("entity_fulltext", $term)
            YIELD node, score
            WHERE score > 0.3
            WITH node LIMIT 5
            OPTIONAL MATCH (node)-[r]-(m:__Entity__)
            RETURN node.id AS source, type(r) AS rel, m.id AS target
            LIMIT 20
            """,
            {"term": search_term},
        )
    except Exception as exc:
        logger.debug(f"Fulltext fallback failed: {exc}")
        return []

    triples: list[str] = []
    for row in rows:
        src = row.get("source") or "?"
        rel = row.get("rel") or "RELATES_TO"
        tgt = row.get("target")
        if tgt:
            triples.append(f"{src} --[{rel}]--> {tgt}")
    return triples


def retrieve_graph(
    question: str,
    config: Optional[AppConfig] = None,
) -> list[str]:
    """
    Retrieve relevant knowledge-graph facts for a question.

    Returns:
        Deduplicated list of triple strings,
        e.g. ``["Apple --[REPORTS]--> Revenue_383B", ...]``
    """
    if config is None:
        config = get_config()

    from src.ingestion.graph_indexer import get_graph  # lazy — avoids Neo4j import at module load
    graph = get_graph(config)
    entities = _extract_entities(question, config)

    seen: set[str] = set()
    triples: list[str] = []

    for entity in entities:
        for t in _query_entity_neighbourhood(graph, entity):
            if t not in seen:
                seen.add(t)
                triples.append(t)

    # Fallback if entity-linking found nothing
    if not triples:
        logger.debug("Entity-linking empty — trying fulltext fallback.")
        for t in _fulltext_fallback(graph, question):
            if t not in seen:
                seen.add(t)
                triples.append(t)

    logger.debug(f"Graph retrieval: {len(triples)} triple(s) for '{question[:60]}...'")
    return triples
