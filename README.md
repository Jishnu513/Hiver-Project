---
title: Spotify AI Support Agent
emoji: 🎵
colorFrom: green
colorTo: black
sdk: docker
pinned: false
license: mit
short_description: 5-stage AI triage pipeline for Spotify customer support tweets
---

<div align="center">

# 🎵 Spotify AI Support Agent

### Production-grade AI triage pipeline for `@SpotifyCares` Twitter support

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Render-1DB954?style=for-the-badge)](https://spotify-ai-support-agent-gu6r.onrender.com)
[![GitHub](https://img.shields.io/badge/GitHub-Jishnu513%2FHiver--Project-181717?style=for-the-badge&logo=github)](https://github.com/Jishnu513/Hiver-Project)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> A 5-stage autonomous AI pipeline that classifies, triages, and drafts replies for Spotify customer support tweets — with a safety policy engine, ChromaDB RAG retrieval, sarcasm detection, and an interactive real-time dashboard.

**[🚀 Try the Live Demo →](https://spotify-ai-support-agent-gu6r.onrender.com)**

</div>

---

## 📸 Dashboard Preview

> **Triage Studio** — Paste any customer tweet and watch the full pipeline run in real time.

The interactive dashboard shows:
- 🎯 Intent classification with confidence score
- 🔒 Safety policy engine decision (6 hard-coded rules)
- 📚 ChromaDB RAG retrieved context chunks
- 💬 AI-drafted reply (≤280 chars, Spotify brand voice)
- 📊 Benchmark results across 3 systems
- 🗂️ Full golden evaluation dataset (200 examples)

---

## 🏗️ Pipeline Architecture

```
Incoming Customer Tweet (@SpotifyCares mention)
          │
          ▼
┌─────────────────────────┐
│  1. Preprocessor        │ → Strip @mentions, URLs, normalize whitespace, preserve emojis
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  2. Intent Classifier   │ → 7 MECE intents + sentiment + urgency
│     MOCK: keyword rules │   (LLM mode: Gemini/Claude/Groq via LiteLLM)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  3. Safety Policy Engine│ → 6 deterministic hard rules (override LLM)
│     (always runs first) │   RULE_SECURITY_THREAT, RULE_BILLING_FRAUD,
└────────┬────────┬───────┘   RULE_EXPLICIT_HUMAN, RULE_CRITICAL_URGENCY,
         │        │            RULE_LOW_CONFIDENCE, RULE_OUT_OF_SCOPE
         ▼        ▼
  ESCALATE_TO  AUTO_HANDLE
    _HUMAN          │
         │          ▼
         │  ┌───────────────────┐
         │  │  4. ChromaDB RAG  │ → Cosine similarity retrieval over
         │  │     Retriever     │   30 @SpotifyCares resolution pairs
         │  └────────┬──────────┘
         │           │
         │           ▼
         │  ┌───────────────────┐
         │  │  5. Reply         │ → Context-grounded, ≤280 chars,
         │  │     Generator     │   Spotify brand voice, ^SP signature
         │  └────────┬──────────┘
         │           │
         └─────┬─────┘
               ▼
         AgentReply (JSON)
```

---

## 🎯 7 Intent Classes

| Intent | Example Tweet |
|--------|---------------|
| `BILLING_AND_SUBSCRIPTION` | "You charged me twice this month!" |
| `PLAYBACK_AND_APP_BUGS` | "Spotify keeps crashing on Android 14" |
| `ACCOUNT_ACCESS_AND_SECURITY` | "Someone hacked my account!" |
| `CONTENT_AND_CATALOG` | "This song is greyed out in my region" |
| `FEATURE_REQUEST_AND_FEEDBACK` | "Please bring back the old UI" |
| `GENERAL_INQUIRY_HOW_TO` | "How do I enable crossfade?" |
| `OUT_OF_SCOPE_OR_CHITCHAT` | "just vibing to music lol 🎶" |

---

## 🔒 Safety Policy Engine (6 Hard Rules)

Rules are evaluated **in priority order** — first match wins and **always overrides** LLM confidence:

| Rule | Trigger | Action |
|------|---------|--------|
| `RULE_SECURITY_THREAT` | Hacked account, phishing, compromised | ESCALATE |
| `RULE_BILLING_FRAUD` | Refund, unauthorized charge, dispute | ESCALATE |
| `RULE_EXPLICIT_HUMAN_REQUEST` | "I want to speak to a human" | ESCALATE |
| `RULE_CRITICAL_URGENCY` | Urgency = CRITICAL | ESCALATE |
| `RULE_LOW_CONFIDENCE` | Classifier confidence < 55% | ESCALATE |
| `RULE_OUT_OF_SCOPE` | Chitchat / memes / non-actionable | ESCALATE |

---

## ✨ Special Features

### 🙄 Sarcasm Detection
The keyword classifier detects explicit sarcasm signals (`🙄`, `"(sarcasm)"`, `"love having"`, `"great job"`) and:
- Forces intent → `FEATURE_REQUEST_AND_FEEDBACK`
- Forces sentiment → `NEGATIVE` (overrides "love" → POSITIVE)
- Uses an empathetic sarcasm-specific reply template

### 🔁 Keep-Alive (No Cold Starts)
A background thread pings `/api/status` every **10 minutes** on Render to prevent the free-tier spin-down. Activated automatically via `RENDER_EXTERNAL_URL` env var.

### 🛡️ Resilient RAG Fallback
If ChromaDB vector store fails to load (memory-constrained free tier), the retriever falls back to **token-overlap scoring** over bundled resolution pairs — the pipeline never crashes.

---

## 📊 Benchmark Results

Evaluated on **200 hand-labelled golden examples** across 7 intent classes:

| System | Intent Macro F1 | Escalation F1 | Escalation F2 (cost-wtd) | False Neg Rate | ROUGE-L |
|--------|----------------|---------------|--------------------------|----------------|---------|
| **Full Agent (RAG + Policy)** | **0.588** | **0.706** | **0.789** | **0.143** | **0.219** |
| Trivial Baseline (Keywords) | 0.588 | 0.727 | 0.625 | 0.429 | 0.177 |
| Simple Baseline (Zero-Shot LLM) | 0.021 | 0.000 | 0.000 | 1.000 | 0.166 |

> **Key insight**: The Full Agent achieves the **lowest false-negative rate (14.3%)** on escalation — critical for brand safety. The cost-weighted F2 metric (2× penalty on missed escalations) shows the biggest advantage over baselines.

---

## 🖥️ Live Dashboard Features

| Tab | What You Can Do |
|-----|----------------|
| ⚡ **Triage Studio** | Paste any tweet or pick from 6 presets. Watch real-time pipeline triage. |
| 📊 **Benchmark & Metrics** | View comparative results across all 3 systems |
| 🎯 **Golden Set (200)** | Browse and search all 200 hand-labelled evaluation examples |
| 🏗️ **Architecture** | Interactive pipeline diagram with stage-by-stage explanation |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- API key: Gemini (free at [aistudio.google.com](https://aistudio.google.com)) or OpenAI or Groq

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/Jishnu513/Hiver-Project.git
cd Hiver-Project

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies (CPU-only torch to save space)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — set GEMINI_API_KEY and AGENT_MODE=full (or keep AGENT_MODE=mock for demo)

# 5. Run the dashboard
python app.py ui
# Open http://localhost:8000
```

### CLI Commands

```bash
# Run the interactive dashboard
python app.py ui

# Process a single tweet
python app.py process "My Spotify keeps crashing on iPhone"

# Run full evaluation benchmark
python app.py evaluate

# Run benchmark with limit
python app.py evaluate --limit 20

# Index resolution pairs into ChromaDB
python app.py index
```

### Run Tests

```bash
python -m pytest tests/ -v
# ✅ 36/36 tests passing
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MODE` | `mock` | `mock` (offline demo) or `gemini` / `openai` / `groq` |
| `AGENT_MODEL` | `gemini/gemini-1.5-flash` | LiteLLM model string |
| `GEMINI_API_KEY` | — | Free at [aistudio.google.com](https://aistudio.google.com) |
| `OPENAI_API_KEY` | — | Optional, for OpenAI embeddings |
| `GROQ_API_KEY` | — | Optional, Groq LLM provider |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers) or `openai` |
| `RAG_TOP_K` | `5` | Number of RAG context pairs to retrieve |
| `AUTO_HANDLE_CONFIDENCE_THRESHOLD` | `0.75` | Min confidence to auto-handle |
| `MAX_REPLY_CHARS` | `280` | Twitter character limit |

---

## 🐳 Docker / Deployment

### Run with Docker

```bash
docker build -t spotify-support-agent .
docker run -p 8000:8000 -e AGENT_MODE=mock spotify-support-agent
# Open http://localhost:8000
```

### Deploy to Render (Free)

1. Fork this repo to your GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Select **Docker** as runtime
5. Add environment variable: `AGENT_MODE=mock`
6. Deploy! Your URL: `https://your-app.onrender.com`

> **Note**: Free tier has cold starts (~60-80s first request). The app has a built-in keep-alive ping every 10 minutes to minimize this.

---

## 📁 Project Structure

```
Hiver-Project/
├── app.py                      # CLI entrypoint (Typer)
├── Dockerfile                  # Container definition
├── Procfile                    # Render/Heroku process file
├── requirements.txt            # Python dependencies
├── koyeb.yaml                  # Koyeb deployment config
│
├── src/
│   ├── pipeline.py             # 5-stage orchestrator
│   ├── classifier.py           # Intent + sarcasm classifier
│   ├── policy.py               # Safety policy engine (6 rules)
│   ├── generator.py            # Reply drafting engine
│   ├── retriever.py            # ChromaDB RAG + fallback
│   ├── models.py               # Pydantic data schemas
│   ├── config.py               # Settings from environment
│   ├── data_loader.py          # CSV/JSON data loading
│   ├── ui_server.py            # FastAPI server + keep-alive
│   ├── static/index.html       # Single-page dashboard
│   └── baselines/
│       ├── trivial_baseline.py # Keyword + macro baseline
│       └── simple_baseline.py  # Zero-shot LLM baseline
│
├── evaluation/
│   ├── harness.py              # Metrics computation
│   ├── benchmark_runner.py     # 3-system comparison runner
│   ├── judge.py                # LLM-as-judge scorer
│   └── calibration.py         # Confidence calibration
│
├── tests/                      # 36 unit tests (pytest)
├── data/                       # Golden eval set + ChromaDB store
├── results/                    # Benchmark output JSONs
└── decision_log.md             # Architecture decision records
```

---

## 📝 Architecture Decisions

Key design decisions are documented in [`decision_log.md`](decision_log.md):

- **Decision #2**: 7 MECE intent taxonomy derived from @SpotifyCares data
- **Decision #4**: Classification is a separate step — policy engine intercepts before generation
- **Decision #5**: Hard rules always override LLM confidence (safety invariant)
- **Decision #7**: Embed resolution PAIRS (query + reply) not raw queries
- **Decision #12**: Enforce 280-char output limit at generation time

---

## 👤 Author

**S. Jishnu** — Hiver SDE Intern Assignment

[![GitHub](https://img.shields.io/badge/GitHub-Jishnu513-181717?logo=github)](https://github.com/Jishnu513)

---

<div align="center">

**[🚀 Live Demo](https://spotify-ai-support-agent-gu6r.onrender.com)** · **[GitHub](https://github.com/Jishnu513/Hiver-Project)** · MIT License

*Built with FastAPI · ChromaDB · LiteLLM · sentence-transformers · Pydantic v2*

</div>
