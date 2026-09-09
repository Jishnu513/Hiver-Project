"""tests/test_pipeline.py — End-to-end integration tests in MOCK mode."""

import pytest
from src.models import Action, AgentReply, Intent
from src.pipeline import preprocess, run


class TestPreprocess:

    def test_strips_mentions(self):
        result = preprocess("@SpotifyCares please fix this!")
        assert "@SpotifyCares" not in result
        assert "fix this" in result

    def test_strips_urls(self):
        result = preprocess("See https://spotify.com/account for details")
        assert "https://" not in result

    def test_normalises_whitespace(self):
        result = preprocess("  too   many    spaces  ")
        assert result == "too many spaces"

    def test_preserves_emoji(self):
        result = preprocess("App is crashing 😤")
        assert "😤" in result


class TestPipelineMock:
    """Integration tests using MOCK mode — no API calls, fully deterministic."""

    def test_returns_agent_reply_type(self):
        reply = run("The app keeps skipping songs every 30 seconds!")
        assert isinstance(reply, AgentReply)

    def test_playback_bug_auto_handled(self):
        reply = run("Spotify crashes every time I open it on my phone.")
        assert reply.action == Action.AUTO_HANDLE
        assert reply.drafted_reply is not None
        assert len(reply.drafted_reply) > 10

    def test_hacked_account_escalates(self):
        reply = run("Someone hacked my account and changed my password!")
        assert reply.action == Action.ESCALATE_TO_HUMAN
        assert reply.escalation_reason  # Should have a reason

    def test_refund_request_escalates(self):
        reply = run("I want a refund for an unauthorized charge on my card!")
        assert reply.action == Action.ESCALATE_TO_HUMAN

    def test_explicit_human_request_escalates(self):
        reply = run("I need to speak to a human agent about my account.")
        assert reply.action == Action.ESCALATE_TO_HUMAN

    def test_reply_within_char_limit(self):
        reply = run("How do I set up Spotify Connect on my speaker?")
        if reply.drafted_reply:
            assert len(reply.drafted_reply) <= 560  # 2x Twitter limit for safety

    def test_tweet_id_preserved(self):
        reply = run("Test tweet", tweet_id="test-123")
        assert reply.tweet_id == "test-123"

    def test_latency_recorded(self):
        reply = run("Something is wrong with my app.")
        assert reply.latency_ms is not None
        assert reply.latency_ms >= 0

    def test_classification_in_reply(self):
        reply = run("My premium plan is not working, I was charged twice.")
        assert reply.classification is not None
        assert reply.classification.intent is not None

    def test_out_of_scope_escalated(self):
        reply = run("lol spotify is life")
        assert reply.action == Action.ESCALATE_TO_HUMAN
