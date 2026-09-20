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

# 🎵 Spotify AI Support Agent

> A production-grade, 5-stage AI triage pipeline for Spotify customer support tweets, featuring RAG retrieval, safety policy enforcement, and an interactive web dashboard.

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)

---

## 🚀 Features

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | **Preprocessor** | Tweet normalization, mention/URL stripping |
| 2 | **Intent Classifier** | 6-class LLM classification with confidence scoring |
| 3 | **Safety Policy Engine** | 5 hard-coded rules (security, billing, hate speech) |
| 4 | **RAG Retriever** | ChromaDB vector store with sentence-transformer fallback |
| 5 | **Response Generator** | Context-aware reply drafting within 280 chars |

## 🖥️ Live Dashboard

The interactive **Triage Studio** lets you:
- Try any tweet (or pick from 6 preset scenarios)
- Inspect every pipeline stage with full traceability
- View RAG context chunks used for the response
- Explore benchmark results and golden evaluation set

## ⚙️ Environment Variables

Set these secrets in your Hugging Face Space settings:

| Variable | Description |
|----------|-------------|
| `AGENT_MODEL` | LLM model (e.g., `claude-sonnet-4-5`) |
| `AGENT_MODE` | `mock` for demo, `full` for real LLM calls |
| `ANTHROPIC_API_KEY` | Your Anthropic API key (for full mode) |
| `OPENAI_API_KEY` | OpenAI key (optional, for embeddings) |

## 🏗️ Architecture

```
Tweet → Preprocessor → Intent Classifier → Safety Policy → RAG Retriever → Generator → Reply
                                ↑                  ↑              ↑
                          LLM (Claude)         Rule Engine   ChromaDB / Fallback
```

## 📊 Benchmark Results

| System | Intent F1 | Escalation F1 | ROUGE-L |
|--------|-----------|---------------|---------|
| **Full Agent (RAG + Policy)** | **0.588** | **0.706** | **0.219** |
| Trivial Baseline | 0.588 | 0.727 | 0.177 |
| Simple Baseline (Zero-Shot) | 0.021 | 0.000 | 0.166 |

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python app.py ui

# Run evaluation
python app.py evaluate --limit 10
```
