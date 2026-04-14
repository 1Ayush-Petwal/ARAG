#!/usr/bin/env python3
"""CLI query script.

Usage:
    # One-shot
    python -m scripts.ask "What were Apple's revenues in FY2023?"

    # Interactive REPL
    python -m scripts.ask

    # Show retrieved context + trace
    python -m scripts.ask "..." --verbose
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.agent.workflow import run_query

console = Console()


def _render_result(state: dict, verbose: bool) -> None:
    """Pretty-print one pipeline result to the terminal."""
    answer = state.get("answer") or "[No answer generated]"
    route_log = state.get("route_log", [])
    grounded = state.get("grounded")
    web_used = bool(state.get("web_docs"))
    unsupported = state.get("unsupported_claims", [])

    # ── Answer panel ──────────────────────────────────────────────────────────
    badges: list[str] = []
    if web_used:
        badges.append("[yellow]web search used[/yellow]")
    if grounded is True:
        badges.append("[green]grounded ✓[/green]")
    elif grounded is False:
        badges.append("[red]unverified ⚠[/red]")

    title = "Answer" + (f"  ({', '.join(badges)})" if badges else "")
    console.print(Panel(answer, title=title, border_style="cyan", padding=(1, 2)))

    # ── Route trace ───────────────────────────────────────────────────────────
    console.print(f"\n[dim]Route: {' → '.join(route_log)}[/dim]\n")

    # ── Unsupported claims warning ────────────────────────────────────────────
    if unsupported:
        console.print(
            Panel(
                "\n".join(f"• {c}" for c in unsupported),
                title="[red]Unsupported claims detected[/red]",
                border_style="red",
            )
        )

    # ── Verbose: context ──────────────────────────────────────────────────────
    if verbose:
        graph_facts = state.get("graph_facts", [])
        vector_docs = state.get("vector_docs", [])

        if graph_facts:
            console.print("\n[bold]Graph Facts:[/bold]")
            for fact in graph_facts:
                console.print(f"  [cyan]•[/cyan] {fact}")

        if vector_docs:
            console.print("\n[bold]Vector Passages:[/bold]")
            for i, doc in enumerate(vector_docs, 1):
                src = doc.metadata.get("source", "unknown")
                snippet = doc.page_content[:300].replace("\n", " ")
                console.print(f"  [[dim]{i}[/dim]] [italic]{Path(src).name}[/italic]")
                console.print(f"       {snippet}{'...' if len(doc.page_content) > 300 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the Adaptive HybridRAG pipeline.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask. Omit for interactive REPL.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show retrieved graph facts and vector passages.",
    )
    args = parser.parse_args()

    if args.question:
        # One-shot mode
        with console.status("[bold green]Retrieving and reasoning…[/bold green]"):
            state = run_query(args.question)
        _render_result(state, args.verbose)
    else:
        # Interactive REPL
        console.print("[bold cyan]Adaptive HybridRAG — Interactive Mode[/bold cyan]")
        console.print("[dim]Type 'exit' or press Ctrl-C to quit.[/dim]\n")
        while True:
            try:
                question = console.input("[bold]>[/bold] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Bye![/dim]")
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                console.print("[dim]Bye![/dim]")
                break
            with console.status("[bold green]Thinking…[/bold green]"):
                try:
                    state = run_query(question)
                except Exception as exc:
                    console.print(f"[red]Error:[/red] {exc}")
                    continue
            _render_result(state, args.verbose)


if __name__ == "__main__":
    main()
