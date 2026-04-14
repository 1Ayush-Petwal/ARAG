#!/usr/bin/env python3
"""RAGAS-based evaluation of the Adaptive HybridRAG pipeline.

Runs every question in eval/qa_set.jsonl through the full pipeline and
scores with RAGAS metrics: faithfulness, answer_relevancy,
context_precision, context_recall.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --qa-path eval/qa_set.jsonl --output eval/results.json
    python -m eval.run_eval --types single_hop multi_hop   # filter by type
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from rich.console import Console
from rich.table import Table

from src.agent.workflow import run_query
from src.utils.logging import logger

console = Console()


def load_qa_set(path: str, types: list[str] | None = None) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if types and item.get("type") not in types:
                    continue
                entries.append(item)
    return entries


def run_evaluation(
    qa_path: str = "eval/qa_set.jsonl",
    output_path: str = "eval/results.json",
    types: list[str] | None = None,
) -> dict:
    qa_pairs = load_qa_set(qa_path, types=types)
    logger.info(f"Loaded {len(qa_pairs)} QA pair(s) from '{qa_path}'")

    questions: list[str] = []
    answers: list[str] = []
    ground_truths: list[str] = []
    contexts: list[list[str]] = []
    meta: list[dict] = []

    for i, item in enumerate(qa_pairs):
        q = item["question"]
        gt = item["ground_truth"]
        logger.info(f"[{i+1}/{len(qa_pairs)}] {q[:70]}...")

        try:
            state = run_query(q)
            answer = state.get("answer") or ""
            context = state.get("fused_context") or ""
            route = state.get("route_log", [])
            web_used = bool(state.get("web_docs"))
        except Exception as exc:
            logger.error(f"  Query failed: {exc}")
            answer = ""
            context = ""
            route = ["ERROR"]
            web_used = False

        questions.append(q)
        answers.append(answer)
        ground_truths.append(gt)
        contexts.append([context])
        meta.append({
            "id": item.get("id", str(i)),
            "type": item.get("type", "unknown"),
            "route": route,
            "web_used": web_used,
        })

    # ── RAGAS evaluation ──────────────────────────────────────────────────────
    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "ground_truth": ground_truths,
        "contexts":     contexts,
    })

    logger.info("Running RAGAS scoring…")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    scores = dict(result)

    # ── Console table ─────────────────────────────────────────────────────────
    table = Table(title="RAGAS Evaluation Results", box=None, show_header=True)
    table.add_column("Metric", style="cyan", min_width=24)
    table.add_column("Score", style="bold", justify="right")
    table.add_column("Target", style="dim", justify="right")

    targets = {
        "faithfulness":      0.85,
        "answer_relevancy":  0.80,
        "context_precision": 0.75,
        "context_recall":    0.75,
    }
    for metric, score in scores.items():
        target = targets.get(metric, "-")
        hit = "✓" if isinstance(score, float) and score >= target else "✗"
        table.add_row(metric, f"{score:.4f}  {hit}", str(target))

    console.print("\n")
    console.print(table)

    # ── Type-level breakdown ──────────────────────────────────────────────────
    type_counts: dict[str, int] = {}
    web_counts: dict[str, int] = {}
    for m in meta:
        t = m["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
        if m["web_used"]:
            web_counts[t] = web_counts.get(t, 0) + 1

    console.print("\n[bold]Questions by type:[/bold]")
    for t, count in sorted(type_counts.items()):
        web = web_counts.get(t, 0)
        console.print(f"  {t:20s} {count:3d} questions   web_fallback={web}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "scores": scores,
        "targets": targets,
        "n_questions": len(qa_pairs),
        "per_question": [
            {
                "id": meta[i]["id"],
                "type": meta[i]["type"],
                "question": questions[i],
                "answer": answers[i],
                "route": meta[i]["route"],
                "web_used": meta[i]["web_used"],
            }
            for i in range(len(qa_pairs))
        ],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to '{output_path}'")

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Adaptive HybridRAG with RAGAS.")
    parser.add_argument("--qa-path",  default="eval/qa_set.jsonl", help="Path to QA JSONL file.")
    parser.add_argument("--output",   default="eval/results.json",  help="Where to write JSON results.")
    parser.add_argument("--types",    nargs="*",
                        help="Filter question types, e.g. --types single_hop multi_hop")
    args = parser.parse_args()

    run_evaluation(args.qa_path, args.output, args.types)


if __name__ == "__main__":
    main()
