"""
app.py -- Interactive CLI for the AI Customer Support Agent.

Usage:
    python app.py demo                 # Run on 5 built-in example tweets
    python app.py chat                 # Interactive tweet-by-tweet mode
    python app.py index [--kaggle-csv path/to/twcs.csv]  # Build RAG index
    python app.py evaluate             # Run full benchmark (mock mode)

The CLI is designed for interviewers to reproduce key results in < 2 minutes.
"""

from __future__ import annotations

import os
# Fix Windows console Unicode encoding before any Rich output
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import json
import logging
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
# Force Rich to use modern colour mode (avoids legacy Windows cp1252 codec crash)
_console_kwargs = {"highlight": False}
from rich.table import Table
from rich import box

# ── Setup path so src/ is importable ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# Copy .env.example → .env if not present (offline mock mode by default)
_env_path = Path(__file__).parent / ".env"
_env_example = Path(__file__).parent / ".env.example"
if not _env_path.exists() and _env_example.exists():
    import shutil
    shutil.copy(_env_example, _env_path)

from src.config import settings
from src.models import Action

app = typer.Typer(help="Spotify AI Customer Support Agent -- Hiver SDE Assignment")
console = Console(force_terminal=True, **_console_kwargs)

logging.basicConfig(level=getattr(logging, settings.log_level, "INFO"))

# ─────────────────────────────────────────────────────────────────────────────
# Demo tweets
# ─────────────────────────────────────────────────────────────────────────────
_DEMO_TWEETS = [
    "My Spotify keeps pausing randomly! I've reinstalled twice already. iPhone 14 user.",
    "Someone hacked my account and changed my email address! I can't log in!",
    "You charged me twice this month for premium. I want a refund now!",
    "How do I enable crossfade between songs? Can't find it anywhere in settings.",
    "Great update Spotify, love having all my downloaded songs deleted (sarcasm)",
]


def _print_reply(tweet: str, reply) -> None:
    """Pretty-print a single AgentReply to the console."""
    console.print(Panel(
        f"[bold yellow]Tweet:[/bold yellow] {tweet}",
        border_style="dim white",
    ))

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Key", style="bold cyan", width=22)
    table.add_column("Value", style="white")
    table.add_row("Intent", reply.intent.value)
    table.add_row("Sentiment", reply.classification.sentiment.value)
    table.add_row("Urgency", reply.classification.urgency.value)
    table.add_row("Confidence", f"{reply.classification.confidence:.0%}")
    action_colour = "red" if reply.action == Action.ESCALATE_TO_HUMAN else "green"
    table.add_row("Action", f"[{action_colour}]{reply.action.value}[/{action_colour}]")
    table.add_row("Policy Triggered", reply.escalation.policy_triggered or "(default auto-handle)")
    console.print(table)

    if reply.retrieved_contexts:
        console.print(f"[dim]{len(reply.retrieved_contexts)} historical context(s) retrieved[/dim]")

    if reply.drafted_reply:
        console.print(Panel(
            f"[bold green]Drafted Reply:[/bold green]\n{reply.drafted_reply}",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"[bold red]ESCALATED:[/bold red]\n{reply.escalation_reason}",
            border_style="red",
        ))

    console.print(f"[dim]{reply.latency_ms:.0f}ms | mode: {settings.agent_mode.value}[/dim]\n")


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def demo() -> None:
    """Run the agent on 5 built-in example tweets to showcase all intents."""
    from src.pipeline import run as pipeline_run

    console.rule("[bold green]🎵 Spotify Support Agent — DEMO[/bold green]")
    console.print(f"[dim]Mode: {settings.agent_mode.value} | Model: {settings.agent_model}[/dim]\n")

    for i, tweet in enumerate(_DEMO_TWEETS, 1):
        console.print(f"[bold]── Example {i}/{len(_DEMO_TWEETS)} ──[/bold]")
        reply = pipeline_run(tweet, tweet_id=f"demo-{i:02d}")
        _print_reply(tweet, reply)


@app.command()
def chat() -> None:
    """Interactive mode — type any tweet and get an instant AI response."""
    from src.pipeline import run as pipeline_run

    console.rule("[bold green]🎵 Spotify Support Agent — Interactive Chat[/bold green]")
    console.print("[dim]Type a customer tweet and press Enter. Ctrl+C to quit.[/dim]\n")
    idx = 1
    while True:
        try:
            tweet = typer.prompt(f"[Tweet #{idx}]")
            if not tweet.strip():
                continue
            reply = pipeline_run(tweet, tweet_id=f"chat-{idx:03d}")
            _print_reply(tweet, reply)
            idx += 1
        except (KeyboardInterrupt, typer.Abort):
            console.print("\n[dim]Goodbye! 🎵[/dim]")
            break


@app.command()
def index(
    kaggle_csv: str = typer.Option(None, "--kaggle-csv", help="Path to Kaggle twcs.csv"),
    max_pairs: int = typer.Option(5000, help="Max resolution pairs to index"),
) -> None:
    """Build the ChromaDB vector index from Kaggle data or bundled sample."""
    from src.data_loader import load_resolution_pairs
    from src.retriever import index_resolution_pairs, collection_size

    console.rule("[bold cyan]📚 Building RAG Index[/bold cyan]")
    console.print(f"[dim]Loading up to {max_pairs} resolution pairs...[/dim]")

    pairs = load_resolution_pairs(kaggle_csv=kaggle_csv, max_pairs=max_pairs)
    console.print(f"[white]Loaded {len(pairs)} pairs. Indexing into ChromaDB...[/white]")

    n = index_resolution_pairs(pairs)
    console.print(f"[bold green]✅ Indexed {n} pairs | Collection size: {collection_size()}[/bold green]")


@app.command()
def evaluate(
    max_examples: int = typer.Option(0, help="Limit examples (0 = all)"),
    skip_judge: bool = typer.Option(True, help="Skip LLM-as-judge scoring"),
    output: str = typer.Option("results/benchmark_results.json", help="Output JSON path"),
) -> None:
    """Run full benchmark: Full Agent vs Baseline 1 vs Baseline 2."""
    from evaluation.benchmark_runner import main as benchmark_main
    benchmark_main(
        output=output,
        skip_judge=skip_judge,
        max_examples=max_examples,
    )


@app.command()
def info() -> None:
    """Print current configuration and project summary."""
    console.rule("[bold green]ℹ️  Agent Configuration[/bold green]")
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Setting", style="bold cyan", width=28)
    table.add_column("Value", style="white")
    table.add_row("AGENT_MODE", settings.agent_mode.value)
    table.add_row("AGENT_MODEL", settings.agent_model)
    table.add_row("EMBEDDING_PROVIDER", settings.embedding_provider.value)
    table.add_row("RAG_TOP_K", str(settings.rag_top_k))
    table.add_row("AUTO_HANDLE_THRESHOLD", f"{settings.auto_handle_confidence_threshold:.0%}")
    table.add_row("MAX_REPLY_CHARS", str(settings.max_reply_chars))
    table.add_row("Golden Eval Set", str(settings.golden_eval_path))
    console.print(table)


if __name__ == "__main__":
    app()
