"""
src/generator.py — Response drafting engine with RAG grounding and brand guardrails.

Builds a context-rich prompt from:
  1. The classified intent + urgency
  2. Retrieved historical @SpotifyCares resolution pairs
  3. Spotify brand voice guidelines

Returns a drafted reply ≤ 280 characters in MOCK mode, or a full LLM-generated
empathetic reply in live mode.

Design decisions:
  - Decision #7: Ground on resolution PAIRS (not raw queries)
  - Decision #12: Enforce 280-char Twitter output length
"""

from __future__ import annotations

import logging
import re
import time

from src.config import AgentMode, settings
from src.models import (
    Action,
    ClassificationResult,
    EscalationDecision,
    Intent,
    RetrievedContext,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Spotify brand voice guidelines injected into every prompt
# ─────────────────────────────────────────────────────────────────────────────

_BRAND_GUIDELINES = """\
Spotify brand voice for support:
- Warm, friendly, and empathetic — not robotic or corporate
- Conversational and concise — this is Twitter, not an email
- Use contractions (we're, you'll, let's)
- Acknowledge the customer's frustration before jumping to solutions
- End with an offer to help further: "Let us know if this helps! 🎵"
- Sign off with ^[Agent initials], e.g. "^SP"
- NEVER promise refunds, credits, or account changes — escalate those
- NEVER share or ask for passwords or payment details
- Keep the reply under 280 characters (Twitter limit)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Canned escalation messages per intent
# ─────────────────────────────────────────────────────────────────────────────

_ESCALATION_TEMPLATES: dict[Intent, str] = {
    Intent.ACCOUNT_ACCESS_AND_SECURITY: (
        "Hey! We take account security very seriously 🔒 A member of our "
        "specialized security team will reach out to you via DM shortly. "
        "Please don't share your password anywhere. ^SP"
    ),
    Intent.BILLING_AND_SUBSCRIPTION: (
        "Hi there! We'd love to look into this billing concern for you. "
        "Our team is reviewing your case and will follow up via DM. "
        "Thanks for your patience! ^SP"
    ),
    Intent.OUT_OF_SCOPE_OR_CHITCHAT: (
        "Hey! Thanks for reaching out to Spotify 🎵 "
        "If you have a support question, we're here to help — "
        "just let us know! ^SP"
    ),
}

_DEFAULT_ESCALATION_MSG = (
    "Hi! We've received your message and our team will follow up with you "
    "directly. Thanks for reaching out to Spotify Support! ^SP"
)

# ─────────────────────────────────────────────────────────────────────────────
# MOCK mode canned replies per intent
# ─────────────────────────────────────────────────────────────────────────────

_MOCK_REPLIES: dict[Intent, str] = {
    Intent.PLAYBACK_AND_APP_BUGS: (
        "Hey, sorry to hear that! 😟 Let's fix this: 1) Log out & back in "
        "2) Clear app cache (Settings → Storage) 3) Reinstall Spotify. "
        "Does that help? Let us know! 🎵 ^SP"
    ),
    Intent.CONTENT_AND_CATALOG: (
        "Hi! Some tracks may be unavailable due to regional licensing 🌍 "
        "Check if your explicit filter is on (Settings → Explicit Content). "
        "Let us know if a specific song is missing! ^SP"
    ),
    Intent.FEATURE_REQUEST_AND_FEEDBACK: (
        "Thanks for the feedback! 🙏 We're always working to improve Spotify. "
        "Share your idea at community.spotify.com — our product team reads every post! ^SP"
    ),
    Intent.GENERAL_INQUIRY_HOW_TO: (
        "Happy to help! 😊 Check our step-by-step guide at support.spotify.com — "
        "search for what you need and we'll walk you through it. "
        "Any other questions? ^SP"
    ),
    Intent.BILLING_AND_SUBSCRIPTION: (
        "Hi! For subscription questions, visit spotify.com/account or the "
        "Spotify app under Settings → Plan. Our support team can also help "
        "via DM! ^SP"
    ),
}

_DEFAULT_MOCK_REPLY = (
    "Hey! Thanks for reaching out 🎵 We'd love to help sort this out. "
    "Could you share a bit more detail so we can look into it? ^SP"
)


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompt construction
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a Spotify customer support agent operating on Twitter. "
    "You draft concise, empathetic, and factually grounded replies to customer tweets.\n\n"
    + _BRAND_GUIDELINES
)


def _build_user_prompt(
    tweet_text: str,
    classification: ClassificationResult,
    contexts: list[RetrievedContext],
) -> str:
    ctx_block = ""
    if contexts:
        ctx_block = "\n\n--- HISTORICAL @SpotifyCares RESOLUTIONS (use as reference) ---\n"
        for i, ctx in enumerate(contexts[:3], 1):
            ctx_block += (
                f"\n[Example {i}] Customer: \"{ctx.source_tweet[:200]}\"\n"
                f"           Spotify replied: \"{ctx.resolution_reply[:200]}\"\n"
            )
        ctx_block += "---\n"

    return (
        f"Customer tweet:\n\"{tweet_text}\"\n"
        f"\nClassified intent: {classification.intent.value}"
        f"\nSentiment: {classification.sentiment.value}"
        f"\nUrgency: {classification.urgency.value}"
        f"{ctx_block}"
        f"\nDraft a Spotify support reply. STRICT RULES:\n"
        f"- ≤ {settings.max_reply_chars} characters total\n"
        f"- Stay grounded in the historical examples above\n"
        f"- Do NOT promise refunds, credit, or password changes\n"
        f"- Return ONLY the reply text, no explanation or wrapper\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_reply(
    tweet_text: str,
    classification: ClassificationResult,
    escalation: EscalationDecision,
    contexts: list[RetrievedContext],
) -> tuple[str | None, float, int]:
    """
    Generate a customer-facing reply draft (or None if escalating).

    Returns:
        (reply_text | None, latency_ms, tokens_used)
    """
    # If escalating — return a canned escalation acknowledgment, no full generation
    if escalation.action == Action.ESCALATE_TO_HUMAN:
        msg = _ESCALATION_TEMPLATES.get(classification.intent, _DEFAULT_ESCALATION_MSG)
        return msg, 0.0, 0

    # MOCK mode — return canned intent-specific reply
    if settings.agent_mode == AgentMode.MOCK:
        reply = _MOCK_REPLIES.get(classification.intent, _DEFAULT_MOCK_REPLY)
        return _truncate(reply), 0.0, 0

    # Live LLM mode
    import litellm

    prompt = _build_user_prompt(tweet_text, classification, contexts)
    t0 = time.time()
    response = litellm.completion(
        model=settings.litellm_model(),
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        api_key=settings.api_key_for_provider(),
        temperature=0.4,
        max_tokens=120,
    )
    latency = (time.time() - t0) * 1000
    tokens = response.usage.total_tokens if response.usage else 0
    reply = response.choices[0].message.content.strip()
    logger.debug(f"Generator LLM response ({latency:.0f}ms, {tokens} tokens): {reply[:100]}")
    return _truncate(reply), latency, tokens


def _truncate(text: str, max_chars: int | None = None) -> str:
    """Ensure reply stays within the configured character limit."""
    limit = max_chars or settings.max_reply_chars
    if len(text) <= limit:
        return text
    # Truncate at last word boundary
    truncated = text[:limit - 3].rsplit(" ", 1)[0]
    return truncated + "..."
