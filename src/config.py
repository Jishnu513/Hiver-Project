"""
src/config.py — Centralised application settings loaded from environment.

All modules import `settings` from here; nothing reads os.environ directly.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Resolve project root (.env lives next to src/)
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


class AgentMode(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    GROQ = "groq"
    MOCK = "mock"  # Fully offline — no API calls, deterministic stubs


class EmbeddingProvider(str, Enum):
    LOCAL = "local"   # sentence-transformers/all-MiniLM-L6-v2
    OPENAI = "openai"


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    agent_mode: AgentMode = AgentMode.MOCK
    agent_model: str = "gemini/gemini-1.5-flash"

    gemini_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    embedding_model_local: str = "all-MiniLM-L6-v2"
    embedding_model_openai: str = "text-embedding-3-small"

    # ── RAG & Retrieval ───────────────────────────────────────────────────────
    rag_top_k: int = Field(default=5, ge=1, le=20)
    chroma_persist_dir: str = str(_PROJECT_ROOT / "data" / "chroma_store")

    # ── Escalation Policy ─────────────────────────────────────────────────────
    auto_handle_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    # ── Output ────────────────────────────────────────────────────────────────
    max_reply_chars: int = Field(default=280, ge=50, le=560)
    log_level: str = "INFO"

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: Path = _PROJECT_ROOT / "data"
    sample_tweets_path: Path = _PROJECT_ROOT / "data" / "sample_spotify_tweets.csv"
    golden_eval_path: Path = _PROJECT_ROOT / "data" / "golden_eval_set.json"

    @field_validator("agent_mode", mode="before")
    @classmethod
    def _lowercase_mode(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v

    def litellm_model(self) -> str:
        """Return the fully-qualified model string for litellm."""
        overrides = {
            AgentMode.GEMINI: self.agent_model or "gemini/gemini-1.5-flash",
            AgentMode.OPENAI: self.agent_model or "gpt-4o-mini",
            AgentMode.GROQ: self.agent_model or "groq/llama-3.1-70b-versatile",
            AgentMode.MOCK: "mock/mock-model",
        }
        return overrides[self.agent_mode]

    def api_key_for_provider(self) -> str:
        mapping = {
            AgentMode.GEMINI: self.gemini_api_key,
            AgentMode.OPENAI: self.openai_api_key,
            AgentMode.GROQ: self.groq_api_key,
            AgentMode.MOCK: "mock-key",
        }
        return mapping[self.agent_mode]

    model_config = {
        "env_file": str(_PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# ── Singleton ─────────────────────────────────────────────────────────────────
settings = Settings()
