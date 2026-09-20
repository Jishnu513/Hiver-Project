"""
src/ui_server.py — FastAPI Web Dashboard & API Server for Spotify Support Agent.

Provides:
  - Interactive Live Triage Studio
  - Real-time RAG Context Inspector
  - Evaluation Benchmark Visualizer
  - Golden Dataset Explorer
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.data_loader import load_golden_eval_set
from src.pipeline import run as run_pipeline
from src.retriever import collection_size

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm pipeline, classifier, and vector retriever on server startup."""
    logger.info("⚡ Pre-warming Spotify Support AI pipeline...")
    try:
        run_pipeline("Spotify app crashing on Android", tweet_id="warmup-init")
        logger.info("✅ Pipeline successfully pre-warmed and ready!")
    except Exception as e:
        logger.warning(f"Pipeline warmup notice: {e}")
    yield


app = FastAPI(
    title="Spotify AI Support Agent — Dashboard",
    description="Interactive Web UI and API for the Spotify Customer Support Agent pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


class ProcessRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=560, description="Customer tweet text")
    tweet_id: Optional[str] = Field("web-demo", description="Optional tweet ID identifier")


PRESET_SCENARIOS = [
    {
        "id": "preset-1",
        "title": "🔒 Account Hacked",
        "category": "Security Threat",
        "badge": "ESCALATE",
        "badge_color": "crimson",
        "text": "Someone hacked my Spotify account and changed my email address! I can't log in anymore!",
        "expected_intent": "ACCOUNT_ACCESS_AND_SECURITY",
        "expected_action": "ESCALATE_TO_HUMAN",
        "description": "Critical security invariant: triggers RULE_SECURITY_THREAT with hard confidence override.",
    },
    {
        "id": "preset-2",
        "title": "💳 Double Billing",
        "category": "Billing Dispute",
        "badge": "ESCALATE",
        "badge_color": "crimson",
        "text": "You charged me $9.99 twice this month for my Premium family subscription. I need a refund immediately.",
        "expected_intent": "BILLING_AND_SUBSCRIPTION",
        "expected_action": "ESCALATE_TO_HUMAN",
        "description": "Financial compliance: Refund & dispute keywords trigger RULE_BILLING_FRAUD.",
    },
    {
        "id": "preset-3",
        "title": "🎧 App Crashing",
        "category": "Playback & App Bug",
        "badge": "AUTO-HANDLE",
        "badge_color": "green",
        "text": "My Spotify keeps crashing every time I open it on Android 14. Tried restarting but still broken.",
        "expected_intent": "PLAYBACK_AND_APP_BUGS",
        "expected_action": "AUTO_HANDLE",
        "description": "Standard bug: retrieves cache-clearing steps from historical ChromaDB resolutions.",
    },
    {
        "id": "preset-4",
        "title": "🎵 Crossfade Guide",
        "category": "How-To Inquiry",
        "badge": "AUTO-HANDLE",
        "badge_color": "green",
        "text": "How do I enable the crossfade feature between songs in the latest mobile app update?",
        "expected_intent": "GENERAL_INQUIRY_HOW_TO",
        "expected_action": "AUTO_HANDLE",
        "description": "Knowledge retrieval: routes to step-by-step help navigation.",
    },
    {
        "id": "preset-5",
        "title": "🙄 Sarcastic Bug",
        "category": "Failure Analysis Edge Case",
        "badge": "EDGE CASE",
        "badge_color": "amber",
        "text": "Great update Spotify, love having ALL my downloaded songs deleted automatically (sarcasm) 🙄",
        "expected_intent": "FEATURE_REQUEST_AND_FEEDBACK",
        "expected_action": "AUTO_HANDLE",
        "description": "Failure mode analysis: tests nuanced sentiment vs complaint resolution.",
    },
    {
        "id": "preset-6",
        "title": "💬 Chitchat / Out of Scope",
        "category": "Out of Scope",
        "badge": "ARCHIVE/ESCALATE",
        "badge_color": "slate",
        "text": "just vibing to some good music today lol Spotify is life 🎶",
        "expected_intent": "OUT_OF_SCOPE_OR_CHITCHAT",
        "expected_action": "ESCALATE_TO_HUMAN",
        "description": "Non-actionable noise: detects low support signal and prevents unnecessary troubleshooting replies.",
    },
]


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the single-page application dashboard."""
    if not INDEX_HTML.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dashboard static file not found at {INDEX_HTML}",
        )
    return FileResponse(INDEX_HTML)


@app.get("/api/status")
async def get_status():
    """Return pipeline metadata, model configuration, and vector store stats."""
    return {
        "status": "online",
        "agent_mode": settings.agent_mode.value,
        "agent_model": settings.agent_model,
        "embedding_provider": settings.embedding_provider,
        "rag_top_k": settings.rag_top_k,
        "auto_handle_threshold": f"{int(settings.auto_handle_confidence_threshold * 100)}%",
        "max_reply_chars": settings.max_reply_chars,
        "vector_store_items": collection_size(),
        "brand": "@SpotifyCares",
    }


@app.get("/api/presets")
async def get_presets():
    """Return 1-click demonstration scenarios."""
    return PRESET_SCENARIOS


@app.post("/api/process")
async def process_tweet(req: ProcessRequest):
    """Run full AI support agent pipeline on an incoming tweet."""
    try:
        reply = run_pipeline(req.text, tweet_id=req.tweet_id)
        reply_dict = reply.model_dump(mode="json")
        reply_text = reply_dict.get("drafted_reply") or ""
        reply_dict["char_count"] = len(reply_text)
        reply_dict["is_within_limit"] = len(reply_text) <= settings.max_reply_chars
        return reply_dict
    except Exception as e:
        logger.exception(f"Error processing tweet: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/benchmark")
async def get_benchmark():
    """Return comparative benchmark results across Full Agent, Trivial, and Simple baselines."""
    results_path = Path("results/benchmark_results.json")
    if results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {results_path}: {e}")

    # Fallback pre-computed benchmark numbers
    return [
        {
            "system": "Full Agent (RAG + Policy)",
            "intent": {"macro_f1": 0.588},
            "escalation": {
                "escalation_f1": 0.706,
                "escalation_f2_cost_weighted": 0.789,
                "false_negative_rate": 0.143,
            },
            "response": {"rouge_l_mean": 0.219, "bleu4_mean": 0.034},
            "operational": {"avg_latency_ms": 1694, "cost_per_1k_queries_usd": 0.0},
        },
        {
            "system": "Trivial Baseline (Keywords + Macros)",
            "intent": {"macro_f1": 0.588},
            "escalation": {
                "escalation_f1": 0.727,
                "escalation_f2_cost_weighted": 0.625,
                "false_negative_rate": 0.429,
            },
            "response": {"rouge_l_mean": 0.177, "bleu4_mean": 0.013},
            "operational": {"avg_latency_ms": 0, "cost_per_1k_queries_usd": 0.0},
        },
        {
            "system": "Simple Baseline (Zero-Shot LLM)",
            "intent": {"macro_f1": 0.021},
            "escalation": {
                "escalation_f1": 0.000,
                "escalation_f2_cost_weighted": 0.000,
                "false_negative_rate": 1.000,
            },
            "response": {"rouge_l_mean": 0.166, "bleu4_mean": 0.013},
            "operational": {"avg_latency_ms": 0, "cost_per_1k_queries_usd": 0.0},
        },
    ]


@app.get("/api/golden")
async def get_golden_set():
    """Return the golden evaluation set examples."""
    golden_objs = load_golden_eval_set()
    return [g.model_dump(mode="json") for g in golden_objs]
