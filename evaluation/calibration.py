"""
evaluation/calibration.py — Human-judge agreement analysis.

Computes inter-rater reliability between human scores and LLM-judge scores:
  - Cohen's Kappa (κ) for categorical escalation decisions
  - Pearson r and Spearman ρ for 1–5 rubric dimension scores
  - Mean Absolute Error (MAE) per rubric dimension

This provides the "human agreement evidence" required by the assignment rubric.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)


def load_human_scores(path: str | Path) -> list[dict]:
    """
    Load human-annotated scores from a JSON file.

    Expected format: list of dicts with keys matching JudgeScore fields
    plus 'example_id' and 'human_escalation_decision'.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_escalation_kappa(
    human_decisions: list[str],
    llm_decisions: list[str],
) -> dict[str, float]:
    """
    Cohen's Kappa for escalation decision agreement.

    Both lists should contain 'AUTO_HANDLE' or 'ESCALATE_TO_HUMAN'.
    """
    kappa = cohen_kappa_score(human_decisions, llm_decisions)
    # Simple % agreement
    agreement_pct = sum(h == l for h, l in zip(human_decisions, llm_decisions)) / len(human_decisions)

    interpretation = (
        "Poor" if kappa < 0.20 else
        "Fair" if kappa < 0.40 else
        "Moderate" if kappa < 0.60 else
        "Substantial" if kappa < 0.80 else
        "Almost Perfect"
    )

    return {
        "cohens_kappa": round(kappa, 4),
        "interpretation": interpretation,
        "percent_agreement": round(agreement_pct, 4),
        "n_examples": len(human_decisions),
    }


def compute_rubric_agreement(
    human_scores: list[dict],
    llm_scores: list[dict],
    dimensions: list[str] | None = None,
) -> dict[str, Any]:
    """
    Pearson r, Spearman ρ, and MAE for each rubric dimension.

    Args:
        human_scores: List of dicts with dimension score keys (1–5 ints).
        llm_scores:   Matching list of JudgeScore dicts.
        dimensions:   Which dimensions to evaluate. Defaults to all 4.

    Returns:
        Per-dimension stats and overall summary.
    """
    dims = dimensions or [
        "factual_grounding",
        "escalation_appropriateness",
        "brand_tone_empathy",
        "actionability",
    ]

    results: dict[str, Any] = {"dimensions": {}, "overall": {}}

    all_pearson, all_spearman, all_mae = [], [], []

    for dim in dims:
        h_vals = np.array([s[dim] for s in human_scores], dtype=float)
        l_vals = np.array([s[dim] for s in llm_scores], dtype=float)

        pearson_r, pearson_p = stats.pearsonr(h_vals, l_vals)
        spearman_r, spearman_p = stats.spearmanr(h_vals, l_vals)
        mae = float(np.mean(np.abs(h_vals - l_vals)))

        results["dimensions"][dim] = {
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p": round(float(pearson_p), 4),
            "spearman_rho": round(float(spearman_r), 4),
            "spearman_p": round(float(spearman_p), 4),
            "mae": round(mae, 4),
            "human_mean": round(float(np.mean(h_vals)), 3),
            "llm_mean": round(float(np.mean(l_vals)), 3),
        }

        all_pearson.append(float(pearson_r))
        all_spearman.append(float(spearman_r))
        all_mae.append(mae)

    results["overall"] = {
        "mean_pearson_r": round(float(np.mean(all_pearson)), 4),
        "mean_spearman_rho": round(float(np.mean(all_spearman)), 4),
        "mean_mae": round(float(np.mean(all_mae)), 4),
        "n_examples": len(human_scores),
    }

    return results


def run_calibration_report(
    human_scores_path: str | Path,
    llm_judge_scores: list[dict],
) -> dict[str, Any]:
    """
    Full calibration report comparing human vs LLM judge scores.

    Args:
        human_scores_path: Path to human_scores.json (40 examples).
        llm_judge_scores: List of JudgeScore dicts for the same 40 examples.

    Returns:
        Combined report with escalation kappa and rubric agreement stats.
    """
    human_data = load_human_scores(human_scores_path)
    assert len(human_data) == len(llm_judge_scores), (
        f"Length mismatch: {len(human_data)} human vs {len(llm_judge_scores)} LLM"
    )

    human_decisions = [h.get("human_escalation_decision", "AUTO_HANDLE") for h in human_data]
    llm_decisions = [s["action"] for s in llm_judge_scores]

    kappa_results = compute_escalation_kappa(human_decisions, llm_decisions)

    # Map JudgeScore dicts for rubric agreement
    llm_rubric = [
        {
            "factual_grounding": s["factual_grounding"],
            "escalation_appropriateness": s["escalation_appropriateness"],
            "brand_tone_empathy": s["brand_tone_empathy"],
            "actionability": s["actionability"],
        }
        for s in llm_judge_scores
    ]
    human_rubric = [
        {
            "factual_grounding": h.get("factual_grounding", 3),
            "escalation_appropriateness": h.get("escalation_appropriateness", 3),
            "brand_tone_empathy": h.get("brand_tone_empathy", 3),
            "actionability": h.get("actionability", 3),
        }
        for h in human_data
    ]
    rubric_results = compute_rubric_agreement(human_rubric, llm_rubric)

    report = {
        "escalation_agreement": kappa_results,
        "rubric_agreement": rubric_results,
        "summary": (
            f"Escalation Kappa: κ={kappa_results['cohens_kappa']:.3f} "
            f"({kappa_results['interpretation']}) | "
            f"Mean Pearson r={rubric_results['overall']['mean_pearson_r']:.3f} | "
            f"Mean MAE={rubric_results['overall']['mean_mae']:.3f}"
        ),
    }

    logger.info(report["summary"])
    return report
