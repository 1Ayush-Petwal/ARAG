"""Configuration loader for Adaptive HybridRAG.

Loads config.yaml for structural settings and .env for secrets.
Use get_config() to get the singleton config instance.
"""
import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).parent.parent


class LLMConfig(BaseModel):
    provider: str = "groq"
    generation_model: str = "llama-3.1-70b-versatile"
    fast_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.0
    max_tokens: int = 2048
    ollama_base_url: str = "http://localhost:11434"


class EmbeddingConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"


class VectorStoreConfig(BaseModel):
    collection_name: str = "hybrid_rag_docs"
    persist_directory: str = str(BASE_DIR / "data" / "vectorstore")
    top_k: int = 6


class GraphStoreConfig(BaseModel):
    allowed_nodes: list[str] = Field(default_factory=lambda: [
        "Company", "Person", "Metric", "Product",
        "Organization", "Location", "Date", "Event",
    ])
    allowed_relationships: list[str] = Field(default_factory=lambda: [
        "REPORTS", "EMPLOYS", "MENTIONS", "LOCATED_IN",
        "OWNED_BY", "PRODUCED_BY", "RELATES_TO", "FILED_BY", "COMPETES_WITH",
    ])


class ChunkerConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 150


class AgentConfig(BaseModel):
    max_iterations: int = 3


class WebSearchConfig(BaseModel):
    max_results: int = 3


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    graph_store: GraphStoreConfig = Field(default_factory=GraphStoreConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)

    # Secrets — always from env, never from YAML
    groq_api_key: str = ""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"  # Aura: set to your instance ID


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load config from YAML + env vars. Env vars take precedence for secrets."""
    if config_path is None:
        config_path = str(BASE_DIR / "config.yaml")

    data: dict = {}
    if Path(config_path).exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    return AppConfig(
        **data,
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )


_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Return the module-level singleton config (lazy-loaded)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
