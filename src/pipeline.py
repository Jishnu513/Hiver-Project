"""
src/pipeline.py — End-to-end orchestrator for the AI customer support agent.

Wires together: Preprocessing → Classification → Policy/Triage → RAG Retrieval
                → Response Generation → Structured AgentReply output.

Usage:
    from src.pipeline import run

    reply = run("My Spotify keeps crashing on iPhone 14 after the latest update!")
    print(reply.action, reply.drafted_reply)
"""

from __future__ import annotations

import logging
import re
import time
import uuid

from src.classifier import classify
from src.config import settings
from src.generator import generate_reply
from src.models import AgentReply, IncomingTweet
from src.policy import evaluate_escalation
from src.retriever import retrieve

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing
# ─────────────────────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")


def preprocess(text: str) -> str:
    """Strip @mentions, URLs, and normalise whitespace. Preserve emoji."""
    text = _MENTION_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run(
    tweet_text: str,
    tweet_id: str | None = None,
    author_id: str | None = None,
) -> AgentReply:
    """
    Run the full AI customer support pipeline on a single incoming tweet.

    Steps:
      1. Preprocess (strip noise, normalise)
      2. Classify intent, sentiment, urgency
      3. Evaluate escalation policy (deterministic rules first)
      4. Retrieve RAG context (only if auto-handling)
      5. Generate reply (or return canned escalation acknowledgment)
      6. Package into structured AgentReply

    Args:
        tweet_text: Raw customer tweet text (max 560 chars).
        tweet_id:   Optional tweet ID (auto-generated UUID if not provided).
        author_id:  Optional author ID.

    Returns:
        AgentReply — the complete structured pipeline output.
    """
    t_start = time.time()
    tid = tweet_id or str(uuid.uuid4())[:8]

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    cleaned = preprocess(tweet_text)
    logger.info(f"[{tid}] Cleaned: '{cleaned[:80]}...'")

    # ── Step 2: Classify ─────────────────────────────────────────────────────
    classification = classify(cleaned)
    logger.info(
        f"[{tid}] Intent: {classification.intent.value} "
        f"(conf={classification.confidence:.0%}, "
        f"urgency={classification.urgency.value})"
    )

    # ── Step 3: Escalation Policy ────────────────────────────────────────────
    escalation = evaluate_escalation(cleaned, classification)
    logger.info(f"[{tid}] Action: {escalation.action.value} | {escalation.reason[:60]}...")

    # ── Step 4: RAG Retrieval (skip if escalating — no point grounding) ──────
    contexts = []
    if escalation.action.value == "AUTO_HANDLE":
        try:
            contexts = retrieve(cleaned, top_k=settings.rag_top_k)
            logger.info(f"[{tid}] Retrieved {len(contexts)} context(s)")
        except Exception as exc:
            logger.warning(f"[{tid}] RAG retrieval failed (non-fatal): {exc}")

    # ── Step 5: Generate reply ────────────────────────────────────────────────
    drafted_reply, gen_latency_ms, tokens_used = generate_reply(
        cleaned, classification, escalation, contexts
    )
    logger.info(f"[{tid}] Reply drafted ({gen_latency_ms:.0f}ms): '{str(drafted_reply)[:60]}...'")

    total_latency_ms = (time.time() - t_start) * 1000

    # ── Step 6: Package output ────────────────────────────────────────────────
    return AgentReply(
        tweet_id=tid,
        intent=classification.intent,
        action=escalation.action,
        escalation_reason=escalation.reason,
        drafted_reply=drafted_reply,
        retrieved_contexts=contexts,
        classification=classification,
        escalation=escalation,
        latency_ms=round(total_latency_ms, 2),
        tokens_used=tokens_used,
        cost_usd=_estimate_cost(tokens_used),
    )


def _estimate_cost(tokens: int) -> float:
    """Rough cost estimate. Gemini 1.5 Flash: ~$0.075 per 1M tokens."""
    return round(tokens * 0.075 / 1_000_000, 8)


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing (for evaluation runs)
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(tweets: list[IncomingTweet]) -> list[AgentReply]:
    """Process a list of IncomingTweet objects and return all AgentReplies."""
    results = []
    for tweet in tweets:
        try:
            reply = run(tweet.text, tweet_id=tweet.tweet_id, author_id=tweet.author_id)
        except Exception as exc:
            logger.error(f"Pipeline failed for tweet {tweet.tweet_id}: {exc}")
            continue
        results.append(reply)
    return results
