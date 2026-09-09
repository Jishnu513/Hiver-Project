"""
src/classifier.py — Intent & urgency classification engine.

In MOCK mode: returns deterministic intent from keyword heuristics.
In LLM mode: uses structured JSON output from Gemini/OpenAI/Groq.

Design decision #4: Classification is a separate step from generation
so that the escalation engine can intercept before any reply is drafted.
"""

from __future__ import annotations

import json
import logging
import re
import time

from src.config import AgentMode, settings
from src.models import (
    ClassificationResult,
    Intent,
    SentimentLabel,
    UrgencyLevel,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Keyword rules for MOCK / Trivial mode
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_PATTERNS: list[tuple[Intent, list[str]]] = [
    (Intent.ACCOUNT_ACCESS_AND_SECURITY, [
        "hack", "hacked", "compromised", "stolen", "unauthorized", "can't log in",
        "cannot log in", "locked out", "password reset", "suspicious", "2fa",
        "two factor", "account stolen", "someone else",
    ]),
    (Intent.BILLING_AND_SUBSCRIPTION, [
        "charge", "charged", "billing", "refund", "payment", "invoice", "receipt",
        "premium", "subscription", "cancel", "family plan", "student", "duo",
        "price", "fee", "credit card", "debit", "paid",
    ]),
    (Intent.PLAYBACK_AND_APP_BUGS, [
        "crash", "crashing", "freeze", "frozen", "skip", "skipping", "pause",
        "pausing", "offline", "download", "cache", "bluetooth", "carplay",
        "airplay", "stuttering", "buffering", "won't play", "not playing",
        "bug", "glitch", "error", "reinstall",
    ]),
    (Intent.CONTENT_AND_CATALOG, [
        "missing song", "missing album", "greyed out", "grayed out", "not available",
        "removed", "explicit", "lyrics", "podcast", "audiobook", "regional",
        "licensing", "grey", "gray", "unavailable",
    ]),
    (Intent.FEATURE_REQUEST_AND_FEEDBACK, [
        "feature", "request", "wish", "should add", "please add", "bring back",
        "update", "new design", "new layout", "hate the new", "love the old",
        "ui", "ux", "interface", "idea",
    ]),
    (Intent.GENERAL_INQUIRY_HOW_TO, [
        "how to", "how do i", "how can i", "how do you", "steps", "guide",
        "help me", "transfer", "share", "collaborate", "equalizer", "eq",
        "connect", "spotify connect", "crossfade", "settings",
    ]),
]

_SENTIMENT_PATTERNS = {
    SentimentLabel.ANGRY: ["hate", "worst", "terrible", "disgusting", "outrageous",
                            "ridiculous", "unacceptable", "wtf", "awful", "horrible"],
    SentimentLabel.NEGATIVE: ["bad", "poor", "disappointed", "sad", "annoyed",
                               "frustrated", "issue", "problem", "broken", "fail"],
    SentimentLabel.POSITIVE: ["love", "great", "amazing", "awesome", "fantastic",
                               "brilliant", "perfect", "excellent", "thanks"],
}

_URGENCY_ESCALATION_KEYWORDS = {
    "hack", "hacked", "compromised", "stolen", "unauthorized charge",
    "legal action", "fraud", "identity theft",
}


def _keyword_classify(text: str) -> ClassificationResult:
    """Fast, deterministic keyword-based classification (no API call)."""
    lower = text.lower()

    # Intent detection (first match wins based on priority order)
    detected_intent = Intent.OUT_OF_SCOPE_OR_CHITCHAT
    for intent, keywords in _INTENT_PATTERNS:
        if any(kw in lower for kw in keywords):
            detected_intent = intent
            break

    # Secondary intent (check remaining patterns)
    secondary = None
    for intent, keywords in _INTENT_PATTERNS:
        if intent != detected_intent and any(kw in lower for kw in keywords):
            secondary = intent
            break

    # Sentiment
    sentiment = SentimentLabel.NEUTRAL
    for label, kws in _SENTIMENT_PATTERNS.items():
        if any(kw in lower for kw in kws):
            sentiment = label
            break

    # Urgency
    if any(kw in lower for kw in _URGENCY_ESCALATION_KEYWORDS):
        urgency = UrgencyLevel.CRITICAL
    elif detected_intent == Intent.BILLING_AND_SUBSCRIPTION:
        urgency = UrgencyLevel.HIGH
    elif detected_intent in (Intent.PLAYBACK_AND_APP_BUGS, Intent.CONTENT_AND_CATALOG):
        urgency = UrgencyLevel.MEDIUM
    elif detected_intent == Intent.GENERAL_INQUIRY_HOW_TO:
        urgency = UrgencyLevel.LOW
    else:
        urgency = UrgencyLevel.LOW

    return ClassificationResult(
        intent=detected_intent,
        confidence=0.85 if detected_intent != Intent.OUT_OF_SCOPE_OR_CHITCHAT else 0.60,
        sentiment=sentiment,
        urgency=urgency,
        secondary_intent=secondary,
        reasoning=(
            f"Keyword match → {detected_intent.value}; "
            f"sentiment={sentiment.value}; urgency={urgency.value}"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM-based classifier
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFY_SYSTEM_PROMPT = """\
You are an expert customer support analyst for Spotify. Your task is to classify
incoming customer tweets with precision and consistency.

Available intents (choose EXACTLY one primary):
- BILLING_AND_SUBSCRIPTION: Premium tiers, payment failures, refunds, cancellation, family/student plans
- PLAYBACK_AND_APP_BUGS: Crashes, skipping, offline sync failures, Bluetooth/CarPlay issues, cache bugs
- ACCOUNT_ACCESS_AND_SECURITY: Login failures, hacked accounts, suspicious activity, password resets
- CONTENT_AND_CATALOG: Missing/greyed-out songs, explicit filter, regional licensing, podcast issues
- FEATURE_REQUEST_AND_FEEDBACK: UI complaints, feature requests, general product feedback
- GENERAL_INQUIRY_HOW_TO: How-to questions, setup help, step-by-step guidance
- OUT_OF_SCOPE_OR_CHITCHAT: Memes, praise, rants without actionable issue, spam

Respond ONLY with a valid JSON object. No markdown, no extra text.
"""

_CLASSIFY_USER_TEMPLATE = """\
Classify this customer tweet:
<tweet>
{tweet_text}
</tweet>

Respond with this JSON schema:
{{
  "intent": "<ONE of the 7 intents>",
  "confidence": <float 0.0-1.0>,
  "sentiment": "<positive|neutral|negative|angry>",
  "urgency": "<low|medium|high|critical>",
  "secondary_intent": "<intent or null>",
  "reasoning": "<one-line explanation>"
}}
"""


def _llm_classify(text: str) -> ClassificationResult:
    """Call LLM with structured JSON prompt for intent classification."""
    import litellm

    prompt = _CLASSIFY_USER_TEMPLATE.format(tweet_text=text[:400])
    t0 = time.time()

    response = litellm.completion(
        model=settings.litellm_model(),
        messages=[
            {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        api_key=settings.api_key_for_provider(),
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=256,
    )
    latency = (time.time() - t0) * 1000
    raw = response.choices[0].message.content.strip()
    logger.debug(f"Classifier LLM response ({latency:.0f}ms): {raw[:200]}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Graceful fallback: extract JSON substring
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            logger.error("LLM returned non-JSON — falling back to keyword classifier")
            return _keyword_classify(text)

    # Coerce and validate with Pydantic
    return ClassificationResult(
        intent=Intent(data.get("intent", Intent.OUT_OF_SCOPE_OR_CHITCHAT.value)),
        confidence=float(data.get("confidence", 0.5)),
        sentiment=SentimentLabel(data.get("sentiment", "neutral")),
        urgency=UrgencyLevel(data.get("urgency", "low")),
        secondary_intent=(
            Intent(data["secondary_intent"])
            if data.get("secondary_intent")
            else None
        ),
        reasoning=str(data.get("reasoning", "")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def classify(text: str) -> ClassificationResult:
    """
    Classify the intent and urgency of an incoming tweet.

    Automatically routes to:
      - Keyword classifier in MOCK mode (zero API cost, deterministic)
      - LLM classifier in all other modes
    """
    if settings.agent_mode == AgentMode.MOCK:
        logger.debug("MOCK mode: using keyword classifier")
        return _keyword_classify(text)
    return _llm_classify(text)
