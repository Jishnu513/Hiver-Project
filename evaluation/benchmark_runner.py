"""
evaluation/benchmark_runner.py — Full benchmark: Agent vs Baseline 1 vs Baseline 2.

Runs all three systems on the golden eval set and produces a comparative
results table in both JSON and human-readable rich console output.

Usage:
    python -m evaluation.benchmark_runner
    python -m evaluation.benchmark_runner --output results/benchmark_results.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from evaluation.harness import run_evaluation
from evaluation.judge import judge_batch
from src.baselines import simple_baseline, trivial_baseline
from src.config import settings
from src.data_loader import load_golden_eval_set
from src.models import IncomingTweet
from src.pipeline import run as agent_run

app = typer.Typer()
console = Console()
logger = logging.getLogger(__name__)


def _run_system(system_name: str, run_fn, examples: list[dict]) -> tuple[list[dict], float]:
    """Run a system on all examples and return (replies_as_dicts, wall_time_s)."""
    console.print(f"\n[bold cyan]Running {system_name}...[/bold cyan]")
    replies = []
    t0 = time.time()
    for ex in examples:
        try:
            reply = run_fn(ex["tweet_text"], tweet_id=ex["example_id"])
            replies.append(reply.model_dump(mode="json"))
        except Exception as e:
            logger.error(f"Error on {ex['example_id']}: {e}")
            replies.append({
                "tweet_id": ex["example_id"],
                "intent": "OUT_OF_SCOPE_OR_CHITCHAT",
                "action": "AUTO_HANDLE",
                "escalation_reason": "Pipeline error",
                "drafted_reply": None,
                "latency_ms": 0,
                "tokens_used": 0,
                "cost_usd": 0,
            })
    elapsed = time.time() - t0
    console.print(f"[green]✓ {system_name}: {len(replies)} replies in {elapsed:.1f}s[/green]")
    return replies, elapsed


def _print_comparison_table(results: list[dict]) -> None:
    """Print a rich comparison table of all systems."""
    table = Table(
        title="📊 Benchmark Results: Agent vs Baselines",
        show_header=True,
        header_style="bold magenta",
    )
    cols = [
        ("System", "cyan"),
        ("Intent Macro F1", "green"),
        ("Escalation F1", "green"),
        ("Escalation F2 (cost-wtd)", "yellow"),
        ("Escalation FNR", "red"),
        ("ROUGE-L", "blue"),
        ("BLEU-4", "blue"),
        ("Judge Score (avg)", "magenta"),
        ("Avg Latency (ms)", "white"),
        ("Cost / 1k queries ($)", "white"),
    ]
    for col, style in cols:
        table.add_column(col, style=style, justify="right")

    for r in results:
        judge_score = r.get("judge_overall_mean", "N/A")
        judge_str = f"{judge_score:.3f}" if isinstance(judge_score, float) else judge_score
        table.add_row(
            r["system"],
            f"{r['intent']['macro_f1']:.3f}",
            f"{r['escalation']['escalation_f1']:.3f}",
            f"{r['escalation']['escalation_f2_cost_weighted']:.3f}",
            f"{r['escalation']['false_negative_rate']:.3f}",
            f"{r['response']['rouge_l_mean']:.3f}",
            f"{r['response']['bleu4_mean']:.3f}",
            judge_str,
            f"{r['operational']['avg_latency_ms']:.0f}",
            f"${r['operational']['cost_per_1k_queries_usd']:.4f}",
        )

    console.print(table)


@app.command()
def main(
    output: str = typer.Option(
        "results/benchmark_results.json",
        help="Path to save JSON results",
    ),
    skip_judge: bool = typer.Option(False, help="Skip LLM-as-judge (faster)"),
    max_examples: int = typer.Option(0, help="Limit examples (0 = all)"),
) -> None:
    """Run full benchmark comparing Agent vs Trivial Baseline vs Simple Baseline."""
    console.rule("[bold green]🎵 Spotify Support Agent — Benchmark Runner[/bold green]")

    # Load golden set
    golden_objs = load_golden_eval_set()
    examples = [ex.model_dump(mode="json") for ex in golden_objs]
    if max_examples > 0:
        examples = examples[:max_examples]
    console.print(f"[white]Golden examples loaded: {len(examples)}[/white]")

    # Run all three systems
    agent_replies, agent_time = _run_system(
        "Full Agent (RAG + Policy)", agent_run, examples
    )
    trivial_replies, trivial_time = _run_system(
        "Trivial Baseline (Keywords + Macros)",
        trivial_baseline.run,
        examples,
    )
    simple_replies, simple_time = _run_system(
        "Simple Baseline (Zero-Shot LLM)",
        simple_baseline.run,
        examples,
    )

    # Evaluate all systems
    all_results = []
    for system_name, replies in [
        ("Full Agent (RAG + Policy)", agent_replies),
        ("Trivial Baseline (Keywords + Macros)", trivial_replies),
        ("Simple Baseline (Zero-Shot LLM)", simple_replies),
    ]:
        result = run_evaluation(examples, replies, system_name=system_name)

        if not skip_judge:
            judge_scores = judge_batch(examples, replies)
            judge_dicts = [s.model_dump() for s in judge_scores]
            overall_scores = [s["overall_score"] for s in judge_dicts]
            result["judge_scores"] = judge_dicts
            result["judge_overall_mean"] = round(
                sum(overall_scores) / len(overall_scores), 3
            )
        all_results.append(result)

    # Print rich table
    _print_comparison_table(all_results)

    # Save JSON results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    console.print(f"\n[bold green]✅ Results saved to {output_path}[/bold green]")


if __name__ == "__main__":
    app()
