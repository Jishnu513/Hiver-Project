"""
src/baselines/simple_baseline.py — Baseline 2: Zero-shot ungrounded LLM prompt.

A single LLM call with no:
  - Intent taxonomy
  - RAG context / historical resolutions
  - Deterministic escalation policy
  - Brand voice guidelines

Purpose: Demonstrates that raw LLMs without structure hallucinate
non-existent features, produce generic advice, and fail to enforce
safe escalation boundaries. Our full agent should outperform this on
factual grounding and escalation precision.

In MOCK mode: returns a plausible-looking but generic, ungrounded reply.
"""

from __future__ import annotations

import json
import logging
import re
import time

from src.config import AgentMode, settings
from src.models import (
    Action,
    AgentReply,
    ClassificationResult,
    EscalationDecision,
    Intent,
    SentimentLabel,
    UrgencyLevel,
)

logger = logging.getLogger(__name__)

# The "minimal" prompt — no taxonomy, no RAG, no explicit policy
_ZERO_SHOT_PROMPT = """\
You are a customer support assistant. A customer has tweeted the following message.

Customer tweet:
"{tweet_text}"

Please:
1. Decide whether to handle this automatically or escalate to a human agent.
2. If auto-handling, draft a helpful reply.

Respond with valid JSON only:
{{
  "action": "AUTO_HANDLE" or "ESCALATE_TO_HUMAN",
  "escalation_reason": "<reason if escalating, else empty string>",
  "reply": "<drafted reply, max 280 chars, or empty string if escalating>"
}}
"""

_MOCK_GENERIC_REPLIES = [
    "Hi! Thanks for reaching out. We're sorry to hear you're having trouble. "
    "Please visit our support page at support.spotify.com for assistance. ^SP",
    "Hello! We apologize for the inconvenience. Our team is here to help — "
    "please check the Spotify Help Center for troubleshooting steps. ^SP",
    "Thanks for contacting us! We understand this is frustrating. "
    "Try logging out and back in, then let us know if that helps! ^SP",
]
_mock_idx = 0


def run(tweet_text: str, tweet_id: str = "simple-00") -> AgentReply:
    """
    Simple baseline: single zero-shot LLM call, no taxonomy or RAG.
    """
    global _mock_idx

    # Stub classification (no real classifier called)
    stub_classification = ClassificationResult(
        intent=Intent.GENERAL_INQUIRY_HOW_TO,
        confidence=0.5,
        sentiment=SentimentLabel.NEUTRAL,
        urgency=UrgencyLevel.LOW,
        secondary_intent=None,
        reasoning="Simple baseline: no intent classification performed.",
    )

    if settings.agent_mode == AgentMode.MOCK:
        # Deterministic rotation through generic replies
        reply = _MOCK_GENERIC_REPLIES[_mock_idx % len(_MOCK_GENERIC_REPLIES)]
        _mock_idx += 1
        escalation = EscalationDecision(
            action=Action.AUTO_HANDLE,
            reason="Simple baseline MOCK: generic auto-reply, no policy check.",
            policy_triggered=None,
            confidence_override=False,
        )
        return AgentReply(
            tweet_id=tweet_id,
            intent=Intent.GENERAL_INQUIRY_HOW_TO,
            action=Action.AUTO_HANDLE,
            escalation_reason=escalation.reason,
            drafted_reply=reply,
            retrieved_contexts=[],
            classification=stub_classification,
            escalation=escalation,
            latency_ms=0.0,
            tokens_used=0,
            cost_usd=0.0,
        )

    # Live LLM zero-shot call
    import litellm

    prompt = _ZERO_SHOT_PROMPT.format(tweet_text=tweet_text[:400])
    t0 = time.time()
    response = litellm.completion(
        model=settings.litellm_model(),
        messages=[{"role": "user", "content": prompt}],
        api_key=settings.api_key_for_provider(),
        temperature=0.3,
        response_format={"type": "json_object"},
        max_tokens=200,
    )
    latency = (time.time() - t0) * 1000
    tokens = response.usage.total_tokens if response.usage else 0
    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}

    action_str = data.get("action", "AUTO_HANDLE")
    action = Action.ESCALATE_TO_HUMAN if "ESCALATE" in action_str else Action.AUTO_HANDLE
    escalation_reason = data.get("escalation_reason", "No reason provided.")
    reply_text = data.get("reply", "")

    escalation = EscalationDecision(
        action=action,
        reason=escalation_reason,
        policy_triggered=None,
        confidence_override=False,
    )

    return AgentReply(
        tweet_id=tweet_id,
        intent=Intent.GENERAL_INQUIRY_HOW_TO,  # Simple baseline doesn't classify
        action=action,
        escalation_reason=escalation_reason,
        drafted_reply=reply_text if reply_text else None,
        retrieved_contexts=[],
        classification=stub_classification,
        escalation=escalation,
        latency_ms=round(latency, 2),
        tokens_used=tokens,
        cost_usd=round(tokens * 0.075 / 1_000_000, 8),
    )
