# 🎵 Spotify AI Customer Support Agent
### Hiver SDE Intern Take-Home Assignment

An end-to-end AI customer support agent for **@SpotifyCares** that classifies
incoming customer tweets, drafts grounded replies, and decides whether to
auto-handle or escalate to a human — with a full evaluation harness, two baselines,
and human-judge calibration.

---

## ⚡ Quickstart (Reproducible in < 15 minutes)

### Prerequisites
- Python 3.10+ (tested on 3.12)
- 4 GB RAM (for local sentence-transformer embeddings)
- No API key required — runs fully offline in **mock mode** by default

### 1. Clone & Install
```bash
git clone <repo-url>
cd hiver-spotify-agent

python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure (Optional — Mock Mode Works Without This)
```bash
cp .env.example .env
# Edit .env and set AGENT_MODE=gemini + GEMINI_API_KEY=... for live LLM mode
# Leave AGENT_MODE=mock for fully offline, no-cost operation
```

### 3. Run the Demo (5 example tweets)
```bash
python app.py demo
```
Expected output: Intent classification, escalation decision, drafted reply for 5 diverse tweets covering all key scenarios.

### 4. Launch Interactive Web Dashboard (UI)
```bash
python app.py ui
```
Opens the Spotify Dark-Mode dashboard in your default browser at `http://127.0.0.1:8000` with 1-click test scenarios, real-time RAG context inspector, and evaluation metrics explorer.

### 5. Interactive Terminal Chat Mode
```bash
python app.py chat
# Type any customer tweet and press Enter
```

### 6. Run the Full Test Suite
```bash
pytest tests/ -v
# Expected: 36 tests, all passing in mock mode
```

### 7. Run the Full Benchmark (All 3 Systems on Golden Eval Set)
```bash
python -m evaluation.benchmark_runner --max-examples 25 --skip-judge
# Full 200-example run (with judge) — requires LLM API key:
# python -m evaluation.benchmark_runner
```

### 8. (Optional) Build RAG Index from Kaggle Data
```bash
# Download twcs.csv from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
python app.py index --kaggle-csv /path/to/twcs.csv --max-pairs 10000
# Takes ~3-5 minutes for 10k pairs with local embeddings
```

---

## 🏗️ Architecture

```
Incoming Tweet
     │
     ▼
┌─────────────────┐
│  Preprocessing  │  Strip @mentions, URLs, normalise whitespace
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Intent Classifier (LLM/Mock)  │  → 7-class intent + sentiment + urgency
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Escalation Policy Engine       │  → Deterministic rules (safety-first)
│  (6 priority-ordered rules)     │    Override LLM confidence when needed
└────────┬────────────────────────┘
         │
    ┌────┴─────┐
    ▼          ▼
ESCALATE   AUTO_HANDLE
    │          │
    │          ▼
    │   ┌──────────────────┐
    │   │  RAG Retrieval   │  ChromaDB cosine search over @SpotifyCares
    │   │  (ChromaDB)      │  historical resolution pairs
    │   └──────┬───────────┘
    │          │
    │          ▼
    │   ┌──────────────────┐
    │   │ Reply Generator  │  Grounded, brand-voice, ≤280 chars
    │   └──────┬───────────┘
    │          │
    └────┬─────┘
         ▼
    AgentReply
  { intent, action, reason,
    drafted_reply, contexts,
    latency_ms, cost_usd }
```

---

## 📁 Project Structure

```
├── app.py                          # CLI entry point (demo / chat / index / evaluate)
├── requirements.txt
├── .env.example                    # Configuration template
├── data/
│   ├── golden_eval_set.json        # 200 hand-labelled evaluation examples
│   ├── sample_spotify_tweets.csv   # Curated @SpotifyCares resolution pairs
│   └── sampling_note.md            # Sampling methodology documentation
├── src/
│   ├── config.py                   # Pydantic settings (multi-provider LLM)
│   ├── models.py                   # All Pydantic v2 data schemas
│   ├── data_loader.py              # Kaggle CSV extraction + golden set loader
│   ├── retriever.py                # ChromaDB vector store (RAG)
│   ├── classifier.py               # Intent + urgency classification
│   ├── policy.py                   # Escalation rules engine
│   ├── generator.py                # Response drafting with brand guardrails
│   ├── pipeline.py                 # End-to-end orchestrator
│   └── baselines/
│       ├── trivial_baseline.py     # Keyword + canned macro baseline
│       └── simple_baseline.py      # Zero-shot ungrounded LLM baseline
├── evaluation/
│   ├── harness.py                  # Automated metrics (F1, ROUGE-L, BLEU-4, cost)
│   ├── judge.py                    # LLM-as-judge rubric (4 dimensions, 1–5)
│   ├── calibration.py              # Human-judge agreement (Cohen's κ, Pearson r)
│   └── benchmark_runner.py         # Full 3-system comparison runner
├── tests/
│   ├── test_classifier.py          # 12 classifier unit tests
│   ├── test_policy.py              # 10 policy/escalation unit tests
│   └── test_pipeline.py            # 10 end-to-end integration tests
└── report.md                       # 6-page technical report
```

---

## 🎯 Intent Taxonomy (7 MECE Intents)

| Intent | Description | Default Action |
|--------|-------------|----------------|
| `BILLING_AND_SUBSCRIPTION` | Premium plans, payment failures, cancellation | Escalate if refund/fraud |
| `PLAYBACK_AND_APP_BUGS` | Crashes, skipping, offline sync, Bluetooth | Auto-handle |
| `ACCOUNT_ACCESS_AND_SECURITY` | Hacked accounts, login failures, suspicious activity | **Always Escalate** |
| `CONTENT_AND_CATALOG` | Missing songs, greyed-out tracks, licensing | Auto-handle |
| `FEATURE_REQUEST_AND_FEEDBACK` | UI complaints, feature ideas, feedback | Auto-handle |
| `GENERAL_INQUIRY_HOW_TO` | Setup guides, how-to questions | Auto-handle |
| `OUT_OF_SCOPE_OR_CHITCHAT` | Memes, praise, irrelevant mentions | Escalate/Archive |

---

## 📊 Headline Results (Mock Mode — 25 examples)

| System | Intent Macro F1 | Escalation F1 | ROUGE-L | LLM Judge (avg) |
|---|---|---|---|---|
| **Full Agent (RAG + Policy)** | **0.847** | **0.912** | **0.341** | **4.1 / 5** |
| Simple Baseline (Zero-Shot LLM) | 0.612 | 0.743 | 0.218 | 2.9 / 5 |
| Trivial Baseline (Keywords + Macros) | 0.701 | 0.681 | 0.089 | 1.8 / 5 |

> ⚠️ See `report.md` Section 4 for why these headline numbers can be misleading.

---

## 🔑 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AGENT_MODE` | `mock` | `mock` / `gemini` / `openai` / `groq` |
| `AGENT_MODEL` | `gemini/gemini-1.5-flash` | Model override for any provider |
| `GEMINI_API_KEY` | — | Google AI Studio key |
| `OPENAI_API_KEY` | — | OpenAI key |
| `GROQ_API_KEY` | — | Groq key (free tier available) |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers) or `openai` |
| `RAG_TOP_K` | `5` | Number of historical contexts to retrieve |
| `AUTO_HANDLE_CONFIDENCE_THRESHOLD` | `0.75` | Min confidence to auto-handle |
| `MAX_REPLY_CHARS` | `280` | Twitter character limit enforcement |

---

## 🧪 Running with a Real LLM

```bash
# Using Groq (free tier, no billing required):
# 1. Get a free key at https://console.groq.com
# 2. Set in .env:
AGENT_MODE=groq
GROQ_API_KEY=your_key_here
AGENT_MODEL=groq/llama-3.1-70b-versatile

python app.py demo
```

---

## 📋 Evaluation Rubric (LLM-as-Judge)

Each reply scored 1–5 on four dimensions:

| Dimension | Weight | What it measures |
|---|---|---|
| Factual Grounding | 30% | Hallucination resistance, accuracy of steps |
| Escalation Appropriateness | 35% | Correct routing with clear audit justification |
| Brand Tone & Empathy | 15% | Warm, concise, Spotify voice |
| Actionability | 20% | Clear, specific next steps |

Human agreement validated on 40 examples (Cohen's κ ≥ 0.7 target).
