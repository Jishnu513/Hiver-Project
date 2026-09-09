"""
src/models.py — Pydantic v2 schemas for every data contract in the pipeline.

These are the single source of truth for:
  - Incoming customer tweets
  - Intent + urgency classification outputs
  - Escalation decisions
  - Agent reply responses
  - Evaluation example records
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class Intent(str, Enum):
    """
    Seven MECE intents derived from @SpotifyCares Twitter data.
    Documented in: decision_log.md → Decision #2
    """
    BILLING_AND_SUBSCRIPTION = "BILLING_AND_SUBSCRIPTION"
    PLAYBACK_AND_APP_BUGS = "PLAYBACK_AND_APP_BUGS"
    ACCOUNT_ACCESS_AND_SECURITY = "ACCOUNT_ACCESS_AND_SECURITY"
    CONTENT_AND_CATALOG = "CONTENT_AND_CATALOG"
    FEATURE_REQUEST_AND_FEEDBACK = "FEATURE_REQUEST_AND_FEEDBACK"
    GENERAL_INQUIRY_HOW_TO = "GENERAL_INQUIRY_HOW_TO"
    OUT_OF_SCOPE_OR_CHITCHAT = "OUT_OF_SCOPE_OR_CHITCHAT"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Action(str, Enum):
    """The triage decision: handle automatically or route to a human agent."""
    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline I/O Schemas
# ─────────────────────────────────────────────────────────────────────────────

class IncomingTweet(BaseModel):
    """Raw customer tweet arriving at the pipeline."""
    tweet_id: str = Field(..., description="Unique tweet identifier")
    text: str = Field(..., min_length=1, max_length=560, description="Raw tweet text")
    author_id: Optional[str] = None
    created_at: Optional[str] = None

    model_config = {"str_strip_whitespace": True}


class ClassificationResult(BaseModel):
    """Output from the intent classifier module."""
    intent: Intent
    confidence: float = Field(..., ge=0.0, le=1.0)
    sentiment: SentimentLabel
    urgency: UrgencyLevel
    secondary_intent: Optional[Intent] = Field(
        None,
        description="Set when the tweet spans two intents (e.g., billing + app bug)"
    )
    reasoning: str = Field(..., description="One-line explanation of classification rationale")


class EscalationDecision(BaseModel):
    """Output from the policy / escalation rules engine."""
    action: Action
    reason: str = Field(..., description="Human-readable audit trail for the routing decision")
    policy_triggered: Optional[str] = Field(
        None,
        description="Name of the hard-coded policy rule that fired, if any"
    )
    confidence_override: bool = Field(
        False,
        description="True if a deterministic rule overrode the LLM confidence score"
    )


class RetrievedContext(BaseModel):
    """A single historical resolution pair retrieved from the vector store."""
    source_tweet: str
    resolution_reply: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class AgentReply(BaseModel):
    """
    The complete, structured output of the AI customer support agent pipeline.
    This is what gets delivered to the front-end / ticketing system.
    """
    tweet_id: str
    intent: Intent
    action: Action
    escalation_reason: str
    drafted_reply: Optional[str] = Field(
        None,
        description="The drafted customer-facing reply. None when escalating to human."
    )
    retrieved_contexts: list[RetrievedContext] = Field(default_factory=list)
    classification: ClassificationResult
    escalation: EscalationDecision
    # Observability
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Schemas
# ─────────────────────────────────────────────────────────────────────────────

class GoldenExample(BaseModel):
    """
    A single hand-labelled record in the golden evaluation set.
    Schema mirrors AgentReply but adds ground-truth fields for comparison.
    """
    example_id: str
    tweet_text: str

    # Ground truth labels (set by human annotator)
    ground_truth_intent: Intent
    ground_truth_action: Action
    ground_truth_escalation_reason: str
    key_points_required: list[str] = Field(
        default_factory=list,
        description="Content checklist: bullet points the ideal reply must address"
    )
    human_annotator_notes: Optional[str] = None

    # Difficulty tags for slice-based analysis
    is_multi_intent: bool = False
    is_sarcastic: bool = False
    is_ambiguous: bool = False
    is_security_risk: bool = False
    is_out_of_scope: bool = False

    # Optionally: the historical brand reply to use as reference for BLEU/ROUGE
    reference_reply: Optional[str] = None


class JudgeScore(BaseModel):
    """LLM-as-judge evaluation scores for a single AgentReply."""
    example_id: str
    factual_grounding: int = Field(..., ge=1, le=5)
    escalation_appropriateness: int = Field(..., ge=1, le=5)
    brand_tone_empathy: int = Field(..., ge=1, le=5)
    actionability: int = Field(..., ge=1, le=5)
    overall_score: float  # Weighted average
    judge_rationale: str
    judge_model: str  # e.g. "gemini/gemini-1.5-flash"
