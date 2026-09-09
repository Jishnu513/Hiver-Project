"""tests/test_policy.py"""

import pytest
from src.models import (
    Action, ClassificationResult, Intent, SentimentLabel, UrgencyLevel
)
from src.policy import evaluate_escalation


def make_classification(
    intent=Intent.GENERAL_INQUIRY_HOW_TO,
    confidence=0.85,
    sentiment=SentimentLabel.NEUTRAL,
    urgency=UrgencyLevel.LOW,
) -> ClassificationResult:
    return ClassificationResult(
        intent=intent,
        confidence=confidence,
        sentiment=sentiment,
        urgency=urgency,
        reasoning="test",
    )


class TestEscalationPolicy:

    def test_security_always_escalates(self):
        cl = make_classification(intent=Intent.ACCOUNT_ACCESS_AND_SECURITY, confidence=0.99)
        decision = evaluate_escalation("my account was hacked", cl)
        assert decision.action == Action.ESCALATE_TO_HUMAN
        assert decision.confidence_override is True
        assert decision.policy_triggered == "RULE_SECURITY_THREAT"

    def test_hack_keyword_escalates_regardless_of_intent(self):
        cl = make_classification(intent=Intent.PLAYBACK_AND_APP_BUGS, confidence=0.90)
        decision = evaluate_escalation("Someone hacked my Spotify and changed my playlist", cl)
        assert decision.action == Action.ESCALATE_TO_HUMAN
        assert "RULE_SECURITY_THREAT" in decision.policy_triggered

    def test_refund_keyword_escalates(self):
        cl = make_classification(intent=Intent.BILLING_AND_SUBSCRIPTION, confidence=0.88)
        decision = evaluate_escalation("I want a refund for the unauthorized charge!", cl)
        assert decision.action == Action.ESCALATE_TO_HUMAN
        assert decision.policy_triggered == "RULE_BILLING_FRAUD"

    def test_explicit_human_request_escalates(self):
        cl = make_classification(confidence=0.80)
        decision = evaluate_escalation("I need to speak to a human agent please", cl)
        assert decision.action == Action.ESCALATE_TO_HUMAN
        assert decision.policy_triggered == "RULE_EXPLICIT_HUMAN_REQUEST"

    def test_critical_urgency_escalates(self):
        cl = make_classification(urgency=UrgencyLevel.CRITICAL, confidence=0.80)
        decision = evaluate_escalation("normal tweet text", cl)
        assert decision.action == Action.ESCALATE_TO_HUMAN
        assert decision.policy_triggered == "RULE_CRITICAL_URGENCY"

    def test_low_confidence_escalates(self):
        cl = make_classification(confidence=0.45)
        decision = evaluate_escalation("hmm idk whats going on", cl)
        assert decision.action == Action.ESCALATE_TO_HUMAN
        assert decision.policy_triggered == "RULE_LOW_CONFIDENCE"

    def test_out_of_scope_escalates(self):
        cl = make_classification(intent=Intent.OUT_OF_SCOPE_OR_CHITCHAT, confidence=0.70)
        decision = evaluate_escalation("lol random tweet", cl)
        assert decision.action == Action.ESCALATE_TO_HUMAN
        assert decision.policy_triggered == "RULE_OUT_OF_SCOPE"

    def test_normal_playback_issue_auto_handles(self):
        cl = make_classification(
            intent=Intent.PLAYBACK_AND_APP_BUGS,
            confidence=0.88,
            urgency=UrgencyLevel.MEDIUM,
        )
        decision = evaluate_escalation("The app keeps skipping songs", cl)
        assert decision.action == Action.AUTO_HANDLE
        assert decision.policy_triggered is None

    def test_how_to_question_auto_handles(self):
        cl = make_classification(
            intent=Intent.GENERAL_INQUIRY_HOW_TO,
            confidence=0.90,
            urgency=UrgencyLevel.LOW,
        )
        decision = evaluate_escalation("How do I enable crossfade?", cl)
        assert decision.action == Action.AUTO_HANDLE

    def test_security_beats_high_confidence(self):
        """Security rule should override even a 99% confident auto-handle prediction."""
        cl = make_classification(intent=Intent.ACCOUNT_ACCESS_AND_SECURITY, confidence=0.99)
        decision = evaluate_escalation("I can't log in to my account", cl)
        assert decision.action == Action.ESCALATE_TO_HUMAN
        assert decision.confidence_override is True
