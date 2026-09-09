"""tests/test_classifier.py"""

import pytest
from src.models import Intent, SentimentLabel, UrgencyLevel
from src.classifier import _keyword_classify


class TestKeywordClassifier:

    def test_billing_intent(self):
        result = _keyword_classify("You charged me twice this month, I want a refund!")
        assert result.intent == Intent.BILLING_AND_SUBSCRIPTION
        assert result.confidence > 0.5

    def test_security_intent_hacked(self):
        result = _keyword_classify("My account got hacked, someone changed my email!")
        assert result.intent == Intent.ACCOUNT_ACCESS_AND_SECURITY

    def test_playback_bug_intent(self):
        result = _keyword_classify("The app keeps crashing every time I open it on Android.")
        assert result.intent == Intent.PLAYBACK_AND_APP_BUGS

    def test_content_catalog_intent(self):
        result = _keyword_classify("Why is the album greyed out? I can't play it.")
        assert result.intent == Intent.CONTENT_AND_CATALOG

    def test_feature_request_intent(self):
        result = _keyword_classify("You should add hi-fi audio quality please!")
        assert result.intent == Intent.FEATURE_REQUEST_AND_FEEDBACK

    def test_how_to_intent(self):
        result = _keyword_classify("How do I transfer my playlists from Apple Music?")
        assert result.intent == Intent.GENERAL_INQUIRY_HOW_TO

    def test_out_of_scope_intent(self):
        result = _keyword_classify("lol Spotify better than other apps")
        assert result.intent == Intent.OUT_OF_SCOPE_OR_CHITCHAT

    def test_angry_sentiment_detected(self):
        result = _keyword_classify("This is absolutely terrible, I hate the new update!")
        assert result.sentiment == SentimentLabel.ANGRY

    def test_positive_sentiment_detected(self):
        result = _keyword_classify("I love this feature, it's amazing!")
        assert result.sentiment == SentimentLabel.POSITIVE

    def test_critical_urgency_for_hack(self):
        result = _keyword_classify("Someone hacked my account and stole everything!")
        assert result.urgency == UrgencyLevel.CRITICAL

    def test_secondary_intent_multi_topic(self):
        result = _keyword_classify("App keeps crashing AND you charged me twice!")
        # Should have a secondary intent since both billing and playback keywords present
        assert result.secondary_intent is not None

    def test_confidence_out_of_scope_lower(self):
        result_oos = _keyword_classify("just vibing to some tunes")
        result_real = _keyword_classify("I want a refund please")
        assert result_oos.confidence < result_real.confidence
