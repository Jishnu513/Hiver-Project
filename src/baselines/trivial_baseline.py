"""
src/baselines/trivial_baseline.py — Baseline 1: Rule/Keyword-based + Canned Macros.

This is the simplest possible approach:
- Intent: regex/keyword matching
- Escalation: keyword trigger list
- Reply: fixed macro template per intent

Purpose: Establishes a minimum floor. Our full system should significantly
outperform this on every metric, especially LLM-as-judge rubric scores.
"""

from __future__ import annotations

from src.models import (
    Action,
    AgentReply,
    ClassificationResult,
    EscalationDecision,
    Intent,
    RetrievedContext,
    SentimentLabel,
    UrgencyLevel,
)
from src.classifier import _keyword_classify  # reuse keyword logic
from src.policy import _SECURITY_KEYWORDS, _BILLING_ESCALATION_KEYWORDS


# ─────────────────────────────────────────────────────────────────────────────
# Canned macro templates (fixed, not grounded in any historical data)
# ─────────────────────────────────────────────────────────────────────────────

_CANNED_REPLIES: dict[Intent, str] = {
    Intent.BILLING_AND_SUBSCRIPTION: (
        "Hi! For billing questions, please visit spotify.com/account or contact "
        "our support at support.spotify.com. ^SP"
    ),
    Intent.PLAYBACK_AND_APP_BUGS: (
        "Hi there! Please try: 1) Restarting the app 2) Clearing cache "
        "3) Reinstalling Spotify. Let us know if the issue persists! ^SP"
    ),
    Intent.ACCOUNT_ACCESS_AND_SECURITY: (
        "For account issues, please visit support.spotify.com or send us a DM. ^SP"
    ),
    Intent.CONTENT_AND_CATALOG: (
        "Some content may not be available in your region. Visit "
        "support.spotify.com for more information. ^SP"
    ),
    Intent.FEATURE_REQUEST_AND_FEEDBACK: (
        "Thanks for the feedback! You can share feature ideas at "
        "community.spotify.com. ^SP"
    ),
    Intent.GENERAL_INQUIRY_HOW_TO: (
        "For help, please visit support.spotify.com or check our help centre. ^SP"
    ),
    Intent.OUT_OF_SCOPE_OR_CHITCHAT: (
        "Hi! Thanks for reaching out. For support, visit support.spotify.com. ^SP"
    ),
}

_ESCALATION_KEYWORDS = _SECURITY_KEYWORDS | _BILLING_ESCALATION_KEYWORDS | frozenset([
    "human", "agent", "real person", "speak to someone",
])


def run(tweet_text: str, tweet_id: str = "trivial-00") -> AgentReply:
    """
    Trivial baseline pipeline:
      - Keyword intent classification
      - Keyword escalation check
      - Fixed canned reply template
    """
    lower = tweet_text.lower()

    # Classify intent using keyword heuristics
    classification = _keyword_classify(tweet_text)

    # Escalation: simple keyword check
    should_escalate = any(kw in lower for kw in _ESCALATION_KEYWORDS)

    if should_escalate:
        action = Action.ESCALATE_TO_HUMAN
        reason = "Escalation keyword detected in tweet text."
        reply = "Please send us a DM and our team will look into this for you. ^SP"
    else:
        action = Action.AUTO_HANDLE
        reason = "No escalation keyword found — using canned macro response."
        reply = _CANNED_REPLIES.get(
            classification.intent,
            "Hi! For help, please visit support.spotify.com. ^SP",
        )

    escalation = EscalationDecision(
        action=action,
        reason=reason,
        policy_triggered="TRIVIAL_KEYWORD_RULE",
        confidence_override=False,
    )

    return AgentReply(
        tweet_id=tweet_id,
        intent=classification.intent,
        action=action,
        escalation_reason=reason,
        drafted_reply=reply,
        retrieved_contexts=[],
        classification=classification,
        escalation=escalation,
        latency_ms=0.0,
        tokens_used=0,
        cost_usd=0.0,
    )
