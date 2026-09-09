"""
src/policy.py — Escalation & triage rules engine.

Implements deterministic business rules that OVERRIDE the LLM confidence score.
This is critical for safety and audit compliance.

Design decisions:
  - Decision #5: Hard rules always win over LLM suggestions
  - Decision #6: Asymmetric cost function on escalation (False Negative = brand disaster)
"""

from __future__ import annotations

import re

from src.models import (
    Action,
    ClassificationResult,
    EscalationDecision,
    Intent,
    SentimentLabel,
    UrgencyLevel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Keyword indicators for hard escalation rules
# ─────────────────────────────────────────────────────────────────────────────

_SECURITY_KEYWORDS = frozenset([
    "hack", "hacked", "compromised", "stolen account", "someone else logged",
    "unauthorized access", "identity theft", "phishing", "suspicious login",
    "2fa bypass", "two factor", "account taken",
])

_BILLING_ESCALATION_KEYWORDS = frozenset([
    "unauthorized charge", "fraudulent charge", "charge i didn't make",
    "refund", "chargeback", "dispute", "double charged", "overcharged",
    "billing fraud", "legal action", "sue", "lawsuit",
])

_EXPLICIT_HUMAN_REQUEST = frozenset([
    "speak to a human", "speak to agent", "speak to someone",
    "human agent", "real person", "talk to a person",
    "live agent", "live support",
])


# ─────────────────────────────────────────────────────────────────────────────
# Policy Rules (ordered by priority — first match wins)
# ─────────────────────────────────────────────────────────────────────────────

def _rule_security_threat(
    tweet_text: str, classification: ClassificationResult
) -> EscalationDecision | None:
    """CRITICAL: Any security-related issue must be escalated immediately."""
    lower = tweet_text.lower()
    if (
        classification.intent == Intent.ACCOUNT_ACCESS_AND_SECURITY
        or any(kw in lower for kw in _SECURITY_KEYWORDS)
    ):
        return EscalationDecision(
            action=Action.ESCALATE_TO_HUMAN,
            reason=(
                "Security risk detected: account compromise or unauthorized access "
                "requires identity verification and secure channel communication. "
                "AI must never attempt to handle credentials or account security changes."
            ),
            policy_triggered="RULE_SECURITY_THREAT",
            confidence_override=True,
        )
    return None


def _rule_billing_fraud(
    tweet_text: str, classification: ClassificationResult
) -> EscalationDecision | None:
    """HIGH: Refunds, unauthorized charges, and billing disputes require human agent."""
    lower = tweet_text.lower()
    if any(kw in lower for kw in _BILLING_ESCALATION_KEYWORDS):
        return EscalationDecision(
            action=Action.ESCALATE_TO_HUMAN,
            reason=(
                "Billing dispute / refund request detected. "
                "Refund approvals and unauthorized charge investigations require "
                "a human agent with payment system access."
            ),
            policy_triggered="RULE_BILLING_FRAUD",
            confidence_override=True,
        )
    return None


def _rule_explicit_human_request(
    tweet_text: str, classification: ClassificationResult
) -> EscalationDecision | None:
    """Respect explicit user request for a human agent."""
    lower = tweet_text.lower()
    if any(kw in lower for kw in _EXPLICIT_HUMAN_REQUEST):
        return EscalationDecision(
            action=Action.ESCALATE_TO_HUMAN,
            reason=(
                "Customer explicitly requested a human agent. "
                "Escalating to respect user preference."
            ),
            policy_triggered="RULE_EXPLICIT_HUMAN_REQUEST",
            confidence_override=True,
        )
    return None


def _rule_critical_urgency(
    tweet_text: str, classification: ClassificationResult
) -> EscalationDecision | None:
    """Escalate any CRITICAL urgency that the classifier detected."""
    if classification.urgency == UrgencyLevel.CRITICAL:
        return EscalationDecision(
            action=Action.ESCALATE_TO_HUMAN,
            reason=(
                f"Urgency level is CRITICAL (sentiment: {classification.sentiment.value}). "
                "High-stakes situation requires immediate human review."
            ),
            policy_triggered="RULE_CRITICAL_URGENCY",
            confidence_override=True,
        )
    return None


def _rule_low_confidence(
    tweet_text: str, classification: ClassificationResult
) -> EscalationDecision | None:
    """Escalate if classifier confidence is too low to act reliably."""
    if classification.confidence < 0.55:
        return EscalationDecision(
            action=Action.ESCALATE_TO_HUMAN,
            reason=(
                f"Intent classification confidence is low ({classification.confidence:.0%}). "
                "Ambiguous query escalated to prevent incorrect automated resolution."
            ),
            policy_triggered="RULE_LOW_CONFIDENCE",
            confidence_override=False,
        )
    return None


def _rule_out_of_scope(
    tweet_text: str, classification: ClassificationResult
) -> EscalationDecision | None:
    """Don't draft replies for out-of-scope / chitchat tweets."""
    if classification.intent == Intent.OUT_OF_SCOPE_OR_CHITCHAT:
        return EscalationDecision(
            action=Action.ESCALATE_TO_HUMAN,
            reason=(
                "Tweet classified as out-of-scope or chitchat. "
                "No actionable support issue detected — routing for human review or archival."
            ),
            policy_triggered="RULE_OUT_OF_SCOPE",
            confidence_override=False,
        )
    return None


def _default_auto_handle(
    tweet_text: str, classification: ClassificationResult
) -> EscalationDecision:
    """Default: auto-handle if no escalation rule fired."""
    return EscalationDecision(
        action=Action.AUTO_HANDLE,
        reason=(
            f"No escalation policy triggered. "
            f"Intent '{classification.intent.value}' is safe to auto-handle "
            f"(confidence: {classification.confidence:.0%})."
        ),
        policy_triggered=None,
        confidence_override=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

# Ordered list of policy rules (evaluated top-to-bottom, first match wins)
_POLICY_RULES = [
    _rule_security_threat,
    _rule_billing_fraud,
    _rule_explicit_human_request,
    _rule_critical_urgency,
    _rule_low_confidence,
    _rule_out_of_scope,
]


def evaluate_escalation(
    tweet_text: str,
    classification: ClassificationResult,
) -> EscalationDecision:
    """
    Evaluate all policy rules and return the escalation decision.

    Deterministic rules are evaluated in priority order.
    If no rule fires, the default is AUTO_HANDLE.

    Args:
        tweet_text: The cleaned incoming customer tweet.
        classification: Output from the classifier module.

    Returns:
        EscalationDecision with action, reason, and audit trail.
    """
    for rule_fn in _POLICY_RULES:
        decision = rule_fn(tweet_text, classification)
        if decision is not None:
            return decision
    return _default_auto_handle(tweet_text, classification)
