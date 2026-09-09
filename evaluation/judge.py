"""
evaluation/judge.py — LLM-as-judge evaluator.

Scores each AgentReply on 4 dimensions (1–5 scale):
  1. Factual Grounding (hallucination resistance)
  2. Escalation Appropriateness
  3. Brand Tone & Empathy
  4. Actionability

In MOCK mode: returns deterministic scored stubs to allow full pipeline runs
without any API cost.

Design note: Using a different model as judge than the generator (when possible)
reduces self-preference bias — documented in decision_log.md.
"""

from __future__ import annotations

import json
import logging
import re

from src.config import AgentMode, settings
from src.models import JudgeScore

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """\
You are an expert quality auditor for a Spotify customer support team.
Evaluate the AI-drafted support reply against the given rubric.
Be critical and specific. Respond ONLY with a valid JSON object.
"""

_JUDGE_USER_TEMPLATE = """\
Evaluate this AI-drafted customer support reply for Spotify:

=== ORIGINAL CUSTOMER TWEET ===
{tweet_text}

=== GROUND TRUTH (What an ideal reply should cover) ===
Intent: {ground_truth_intent}
Required Key Points: {key_points}
Correct Action: {ground_truth_action}

=== AI SYSTEM'S RESPONSE ===
Action taken: {predicted_action}
Escalation reason: {escalation_reason}
Drafted reply: {drafted_reply}

=== SCORING RUBRIC (score each 1-5) ===

1. factual_grounding (1=hallucinated content/wrong steps, 5=all facts verified & grounded in Spotify's real procedures)
2. escalation_appropriateness (1=critical failure routing, 3=suboptimal, 5=flawless routing with clear justification)
3. brand_tone_empathy (1=robotic/corporate, 3=passable, 5=warm/empathetic/concise/Spotify voice)
4. actionability (1=vague/dead-end, 3=partially helpful, 5=specific next steps clearly given)

=== OUTPUT FORMAT ===
Respond with ONLY this JSON (no markdown wrapper):
{{
  "factual_grounding": <int 1-5>,
  "escalation_appropriateness": <int 1-5>,
  "brand_tone_empathy": <int 1-5>,
  "actionability": <int 1-5>,
  "judge_rationale": "<2-3 sentences explaining the scores>"
}}
"""

# Dimension weights for overall score (escalation_appropriateness is highest — safety-critical)
_WEIGHTS = {
    "factual_grounding": 0.30,
    "escalation_appropriateness": 0.35,
    "brand_tone_empathy": 0.15,
    "actionability": 0.20,
}

# Mock scores for systematic testing (slightly varied to avoid flat distributions)
_MOCK_SCORE_SETS = [
    {"factual_grounding": 4, "escalation_appropriateness": 5, "brand_tone_empathy": 4, "actionability": 4},
    {"factual_grounding": 3, "escalation_appropriateness": 4, "brand_tone_empathy": 3, "actionability": 3},
    {"factual_grounding": 5, "escalation_appropriateness": 5, "brand_tone_empathy": 5, "actionability": 4},
    {"factual_grounding": 2, "escalation_appropriateness": 3, "brand_tone_empathy": 3, "actionability": 2},
    {"factual_grounding": 4, "escalation_appropriateness": 4, "brand_tone_empathy": 4, "actionability": 5},
]
_mock_score_idx = 0


def _compute_overall(scores: dict) -> float:
    return round(
        sum(_WEIGHTS[k] * scores[k] for k in _WEIGHTS), 3
    )


def judge_reply(
    example_id: str,
    tweet_text: str,
    ground_truth_intent: str,
    ground_truth_action: str,
    key_points: list[str],
    predicted_action: str,
    escalation_reason: str,
    drafted_reply: str | None,
) -> JudgeScore:
    """
    Score a single AgentReply using the LLM-as-judge rubric.

    Returns a JudgeScore with dimension scores and overall weighted average.
    """
    global _mock_score_idx

    if settings.agent_mode == AgentMode.MOCK:
        scores = _MOCK_SCORE_SETS[_mock_score_idx % len(_MOCK_SCORE_SETS)]
        _mock_score_idx += 1
        return JudgeScore(
            example_id=example_id,
            **scores,
            overall_score=_compute_overall(scores),
            judge_rationale="[MOCK] Deterministic stub scores for offline testing.",
            judge_model="mock/mock-judge",
        )

    # Live LLM judge call
    import litellm

    prompt = _JUDGE_USER_TEMPLATE.format(
        tweet_text=tweet_text[:300],
        ground_truth_intent=ground_truth_intent,
        key_points=", ".join(key_points) if key_points else "N/A",
        ground_truth_action=ground_truth_action,
        predicted_action=predicted_action,
        escalation_reason=escalation_reason[:200],
        drafted_reply=(drafted_reply or "[No reply — escalated]")[:280],
    )

    response = litellm.completion(
        model=settings.litellm_model(),
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        api_key=settings.api_key_for_provider(),
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=300,
    )
    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}

    scores = {
        "factual_grounding": int(data.get("factual_grounding", 3)),
        "escalation_appropriateness": int(data.get("escalation_appropriateness", 3)),
        "brand_tone_empathy": int(data.get("brand_tone_empathy", 3)),
        "actionability": int(data.get("actionability", 3)),
    }

    return JudgeScore(
        example_id=example_id,
        **scores,
        overall_score=_compute_overall(scores),
        judge_rationale=str(data.get("judge_rationale", "")),
        judge_model=settings.litellm_model(),
    )


def judge_batch(
    examples: list[dict],
    replies: list[dict],
) -> list[JudgeScore]:
    """
    Run LLM-as-judge over a full list of examples and replies.

    Args:
        examples: List of GoldenExample dicts.
        replies: List of AgentReply dicts (aligned by index).

    Returns:
        List of JudgeScore objects.
    """
    assert len(examples) == len(replies)
    judge_scores = []
    for ex, rep in zip(examples, replies):
        score = judge_reply(
            example_id=ex["example_id"],
            tweet_text=ex["tweet_text"],
            ground_truth_intent=ex["ground_truth_intent"],
            ground_truth_action=ex["ground_truth_action"],
            key_points=ex.get("key_points_required", []),
            predicted_action=rep["action"],
            escalation_reason=rep.get("escalation_reason", ""),
            drafted_reply=rep.get("drafted_reply"),
        )
        judge_scores.append(score)
    return judge_scores
