"""
evaluation/harness.py — Automated metrics runner.

Computes:
  - Intent classification: Precision, Recall, Macro F1, Confusion Matrix
  - Escalation decision: Precision, Recall, F1 (weighted by asymmetric cost)
  - Response quality: ROUGE-L, BLEU-4 (vs reference replies in golden set)
  - Operational: avg latency, total tokens, estimated API cost
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_bleu(reference: str, hypothesis: str) -> float:
    """Compute BLEU-4 using nltk. Returns 0.0 if references are too short."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        if len(ref_tokens) < 4:
            return 0.0
        return sentence_bleu(
            [ref_tokens], hyp_tokens,
            smoothing_function=SmoothingFunction().method1
        )
    except Exception:
        return 0.0


def _safe_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return scores["rougeL"].fmeasure
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_intent_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str] | None = None
) -> dict[str, Any]:
    """
    Compute intent classification metrics.

    Returns dict with macro F1, per-class report, and confusion matrix.
    """
    y_true_str = [x.value if hasattr(x, "value") else str(x) for x in y_true]
    y_pred_str = [x.value if hasattr(x, "value") else str(x) for x in y_pred]
    report = classification_report(y_true_str, y_pred_str, labels=labels, output_dict=True, zero_division=0)
    macro_f1 = f1_score(y_true_str, y_pred_str, average="macro", zero_division=0)
    cm = confusion_matrix(y_true_str, y_pred_str, labels=labels or sorted(set(y_true_str)))

    return {
        "macro_f1": round(macro_f1, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels or sorted(set(y_true_str)),
    }


def compute_escalation_metrics(
    y_true: list[str], y_pred: list[str]
) -> dict[str, float]:
    """
    Compute escalation decision metrics.

    Uses 'ESCALATE_TO_HUMAN' as the positive class (safety-critical).
    Includes cost-weighted F-beta score (beta=2 penalises False Negatives 4x more).
    """
    from sklearn.metrics import fbeta_score

    pos_label = "ESCALATE_TO_HUMAN"
    y_true_str = [x.value if hasattr(x, "value") else str(x) for x in y_true]
    y_pred_str = [x.value if hasattr(x, "value") else str(x) for x in y_pred]

    precision = precision_score(y_true_str, y_pred_str, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true_str, y_pred_str, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true_str, y_pred_str, pos_label=pos_label, zero_division=0)
    # F2: weights recall more heavily than precision (safety-first)
    f2 = fbeta_score(y_true_str, y_pred_str, beta=2, pos_label=pos_label, zero_division=0)

    # False Negative Rate: dangerous — auto-handled a critical issue
    true_positives = sum(1 for t, p in zip(y_true_str, y_pred_str) if t == pos_label and p == pos_label)
    false_negatives = sum(1 for t, p in zip(y_true_str, y_pred_str) if t == pos_label and p != pos_label)
    fnr = false_negatives / max(true_positives + false_negatives, 1)

    return {
        "escalation_precision": round(precision, 4),
        "escalation_recall": round(recall, 4),
        "escalation_f1": round(f1, 4),
        "escalation_f2_cost_weighted": round(f2, 4),
        "false_negative_rate": round(fnr, 4),
        "false_negatives": false_negatives,
    }


def compute_response_metrics(
    predictions: list[str | None], references: list[str | None]
) -> dict[str, float]:
    """
    Compute BLEU-4 and ROUGE-L scores for generated replies.

    Skips examples where either prediction or reference is None (escalated).
    """
    bleu_scores, rouge_scores = [], []
    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            continue
        bleu_scores.append(_safe_bleu(ref, pred))
        rouge_scores.append(_safe_rouge_l(ref, pred))

    return {
        "bleu4_mean": round(float(np.mean(bleu_scores)) if bleu_scores else 0.0, 4),
        "rouge_l_mean": round(float(np.mean(rouge_scores)) if rouge_scores else 0.0, 4),
        "n_responses_evaluated": len(bleu_scores),
    }


def compute_operational_metrics(replies: list[dict]) -> dict[str, Any]:
    """Compute latency, token usage, and cost stats across a run."""
    latencies = [r.get("latency_ms", 0.0) for r in replies if r.get("latency_ms")]
    tokens = [r.get("tokens_used", 0) for r in replies]
    costs = [r.get("cost_usd", 0.0) for r in replies]

    return {
        "avg_latency_ms": round(float(np.mean(latencies)) if latencies else 0.0, 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)) if latencies else 0.0, 2),
        "total_tokens": int(sum(tokens)),
        "total_cost_usd": round(sum(costs), 6),
        "cost_per_1k_queries_usd": round(sum(costs) / max(len(replies), 1) * 1000, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation run
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    golden_examples: list[dict],
    agent_replies: list[dict],
    system_name: str = "agent",
) -> dict[str, Any]:
    """
    Run full evaluation of agent replies against golden examples.

    Args:
        golden_examples: List of GoldenExample dicts (ground truth).
        agent_replies: List of AgentReply dicts (predictions), matched by index.
        system_name: Label for this system in the results dict.

    Returns:
        Nested results dict with all metric categories.
    """
    assert len(golden_examples) == len(agent_replies), (
        f"Length mismatch: {len(golden_examples)} examples vs {len(agent_replies)} replies"
    )

    y_true_intent = [
        ex["ground_truth_intent"].value if hasattr(ex["ground_truth_intent"], "value") else str(ex["ground_truth_intent"])
        for ex in golden_examples
    ]
    y_pred_intent = [
        r["intent"].value if hasattr(r["intent"], "value") else str(r["intent"])
        for r in agent_replies
    ]

    y_true_action = [
        ex["ground_truth_action"].value if hasattr(ex["ground_truth_action"], "value") else str(ex["ground_truth_action"])
        for ex in golden_examples
    ]
    y_pred_action = [
        r["action"].value if hasattr(r["action"], "value") else str(r["action"])
        for r in agent_replies
    ]

    ref_replies = [ex.get("reference_reply") for ex in golden_examples]
    pred_replies = [r.get("drafted_reply") for r in agent_replies]

    intent_metrics = compute_intent_metrics(y_true_intent, y_pred_intent)
    escalation_metrics = compute_escalation_metrics(y_true_action, y_pred_action)
    response_metrics = compute_response_metrics(pred_replies, ref_replies)
    operational_metrics = compute_operational_metrics(agent_replies)

    results = {
        "system": system_name,
        "n_examples": len(golden_examples),
        "intent": intent_metrics,
        "escalation": escalation_metrics,
        "response": response_metrics,
        "operational": operational_metrics,
    }

    logger.info(
        f"[{system_name}] Intent Macro F1: {intent_metrics['macro_f1']:.3f} | "
        f"Escalation F1: {escalation_metrics['escalation_f1']:.3f} | "
        f"ROUGE-L: {response_metrics['rouge_l_mean']:.3f}"
    )
    return results
