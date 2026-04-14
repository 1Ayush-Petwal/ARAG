"""Structured logging utilities for Adaptive HybridRAG."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger to use Rich handler."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    return logging.getLogger("hybrid_rag")


logger = setup_logging()


class RunLogger:
    """Appends structured JSON records of each agent run to a daily log file."""

    def __init__(self, log_dir: str = "data/logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_run(self, state: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": state.get("question"),
            "answer": state.get("answer"),
            "grade": state.get("grade"),
            "grounded": state.get("grounded"),
            "iterations": state.get("iterations"),
            "route_log": state.get("route_log", []),
            "web_used": bool(state.get("web_docs")),
            "unsupported_claims": state.get("unsupported_claims", []),
        }
        log_file = self.log_dir / f"run_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
