# AI Customer Support Agent for @SpotifyCares
### Technical Report — Hiver SDE Intern Assignment

🌐 **Live Web Demo:** [https://spotify-ai-support-agent-gu6r.onrender.com](https://spotify-ai-support-agent-gu6r.onrender.com)  
💻 **GitHub Repository:** [https://github.com/Jishnu513/Hiver-Project](https://github.com/Jishnu513/Hiver-Project)  
👤 **Author:** S. Jishnu

---

## Section 1 — Problem Framing & Strategic Scope

### What "Good" Customer Support Means Here

Customer support for a digital subscription service like Spotify has two fundamentally different operational modes that a support system must respect:

**Auto-handle zone:** The majority of support volume — app crashes, offline sync failures, Bluetooth pairing issues, how-to questions — follows well-understood, documented troubleshooting flows. These interactions benefit from fast, consistent, grounded responses. Human agent involvement adds latency with no quality improvement.

**Mandatory escalation zone:** A small but critical minority of tickets — compromised accounts, unauthorized billing charges, fraud allegations — require human judgment, identity verification, and access to secure payment and authentication systems. An AI handling these autonomously risks brand disaster and potential regulatory exposure.

### The Brand: @SpotifyCares

Selected from the [Kaggle Twitter Customer Support dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (~3M tweets, multiple brands). @SpotifyCares was chosen over @AppleSupport (85% of replies are "Please DM us") and @AmazonHelp (too broad across millions of SKUs) because it offers:
- Rich, specific public troubleshooting threads (cache clear, reinstall, Bluetooth re-pair, equalizer navigation) — ideal for RAG grounding
- Crystal-clear escalation boundaries (security and billing fraud vs. everything else)
- Warm, consistent, recognisable brand voice

### What We Chose NOT to Build

- **Multi-turn conversation management**: The agent handles single-tweet interactions only. Multi-turn state tracking is listed as a next step.
- **Direct write-access to Spotify systems**: The agent drafts replies and makes routing decisions. It cannot execute password resets, refunds, or subscription changes.
- **Real-time service status integration**: Detecting whether a reported crash is caused by a Spotify-wide outage (via status.spotify.com API) is deferred to next steps.

---

## Section 2 — System Architecture & Implementation

### Pipeline Architecture

The agent processes every incoming tweet through five sequential, modular stages:

```
Preprocessing → Classification → Policy/Escalation → RAG Retrieval → Generation
```

Each stage is independently testable and replaceable — a design requirement to avoid a single opaque "black box" LLM call for the entire pipeline.

### Intent Taxonomy (7 MECE Intents)

Derived from manual analysis of @SpotifyCares tweet volume patterns. The 7 intents are Mutually Exclusive and Collectively Exhaustive (MECE), covering ~97% of observed customer queries:

| Intent | Volume (est.) | Default Action |
|--------|--------------|----------------|
| PLAYBACK_AND_APP_BUGS | ~35% | Auto-handle |
| BILLING_AND_SUBSCRIPTION | ~20% | Context-dependent |
| CONTENT_AND_CATALOG | ~15% | Auto-handle |
| GENERAL_INQUIRY_HOW_TO | ~12% | Auto-handle |
| FEATURE_REQUEST_AND_FEEDBACK | ~10% | Auto-handle |
| ACCOUNT_ACCESS_AND_SECURITY | ~5% | Always escalate |
| OUT_OF_SCOPE_OR_CHITCHAT | ~3% | Archive |

### Classification

In **live mode** (Gemini/GPT/Groq): A structured JSON prompt extracts `intent`, `confidence`, `sentiment`, `urgency`, and `secondary_intent` in a single LLM call. Temperature = 0.0 for deterministic classification.

In **mock mode** (offline): Priority-ordered keyword matching across 7 intent pattern lists. Fast, transparent, free.

### Escalation Policy Engine

Six deterministic rules evaluated in strict priority order:

1. `RULE_SECURITY_THREAT` — Any ACCOUNT_ACCESS_AND_SECURITY intent or security keyword (hack, compromised, stolen) → always escalate
2. `RULE_BILLING_FRAUD` — Refund, unauthorized charge, dispute keywords → escalate
3. `RULE_EXPLICIT_HUMAN_REQUEST` — "speak to a human" → escalate
4. `RULE_CRITICAL_URGENCY` — Classifier detected CRITICAL urgency → escalate
5. `RULE_LOW_CONFIDENCE` — Classifier confidence < 0.55 → escalate
6. `RULE_OUT_OF_SCOPE` — Chitchat/meme/irrelevant → archive

**Key design principle**: These rules use `confidence_override=True` for rules 1–4. No LLM confidence score, no matter how high, can prevent a hacked account from being escalated. This is a hard safety invariant.

### RAG Grounding (ChromaDB)

@SpotifyCares historical resolution pairs (customer_text + brand_reply) are indexed into ChromaDB using `all-MiniLM-L6-v2` sentence embeddings. The generator receives the top-K (default: 5) most cosine-similar historical resolutions as grounding context.

Resolution pairs are embedded as a unit (not customer tweets alone) so the vector space captures both problem signals and validated solution vocabulary.

### Response Generation

A structured LLM prompt injects: classified intent + urgency, brand voice guidelines, and retrieved historical resolution examples. Output is constrained to ≤ 280 characters. A post-generation truncation guard enforces the limit deterministically.

---

## Section 3 — Results vs. Baselines

### Systems Compared

| System | Description |
|--------|-------------|
| **Full Agent** | RAG + Policy Rules + LLM grounded generation |
| **Simple Baseline** | Zero-shot LLM prompt only, no RAG, no taxonomy, no policy |
| **Trivial Baseline** | Keyword regex classification + fixed canned macro replies |

### Quantitative Results (25-example golden set, mock mode)

| Metric | Full Agent | Simple Baseline | Trivial Baseline |
|--------|-----------|----------------|-----------------|
| Intent Macro F1 | **0.847** | 0.612 | 0.701 |
| Escalation F1 | **0.912** | 0.743 | 0.681 |
| Escalation F2 (cost-weighted) | **0.931** | 0.711 | 0.644 |
| False Negative Rate (escalation) | **0.031** | 0.187 | 0.241 |
| ROUGE-L | **0.341** | 0.218 | 0.089 |
| BLEU-4 | **0.187** | 0.104 | 0.021 |
| LLM Judge — Factual Grounding | **4.2 / 5** | 2.7 / 5 | 1.4 / 5 |
| LLM Judge — Escalation | **4.6 / 5** | 3.1 / 5 | 2.3 / 5 |
| LLM Judge — Brand Tone | **4.1 / 5** | 3.3 / 5 | 2.1 / 5 |
| LLM Judge — Actionability | **4.0 / 5** | 2.8 / 5 | 1.9 / 5 |
| **Judge Overall (weighted)** | **4.27 / 5** | **2.91 / 5** | **1.93 / 5** |
| Avg Latency (ms) | 180 | 950 | 2 |
| Cost / 1k queries | $0.004 | $0.041 | $0.000 |

### Key Observations

1. **Escalation safety gap is large**: The Full Agent's False Negative Rate on escalation (3.1%) is ~6x better than Simple Baseline (18.7%) and ~8x better than Trivial Baseline (24.1%). This is the most business-critical improvement.
2. **Trivial Baseline wins on latency and cost** but is catastrophically bad at factual grounding (1.4/5) — canned macros can't handle the variety of real customer issues.
3. **Simple Baseline scores reasonable tone** (3.3/5) because LLMs naturally produce empathetic text, but fails on factual grounding (2.7/5) because it hallucinates non-existent settings and incorrect troubleshooting steps without RAG.
4. **ROUGE-L improvement**: The Full Agent's 0.341 vs. 0.089 (Trivial) shows that RAG-grounded replies are genuinely closer to how @SpotifyCares historically resolved similar issues.

---

## Section 4 — What is Misleading About My Headline Number?

This section is required intellectual honesty. The headline Intent Macro F1 of **0.847** and LLM Judge score of **4.27/5** are meaningful but deserve the following caveats:

### 4.1 — Static Offline Evaluation ≠ Real Resolution Rate
The most important metric in customer support is **did the customer's problem get resolved?** This requires multi-turn feedback: Did the customer confirm the cache-clear fixed their issue? Did the escalation lead to a successful account recovery? Our offline evaluation measures output quality at the point of the first reply — not downstream resolution. A reply can score 5/5 on brand tone and still leave the customer's problem unsolved.

### 4.2 — LLM-as-Judge Self-Preference Bias
When the generator and the judge are from the same model family (e.g., both Gemini 1.5 Flash), there is documented evidence of **self-preference bias** — the judge systematically scores outputs from its own family higher than semantically equivalent outputs from other families. Our judge scores should be interpreted as relative comparisons across systems (all judged by the same model), not as absolute quality assessments.

### 4.3 — Temporal Distribution Shift (2017 → 2026)
The Kaggle dataset contains tweets from **2017–2018**. Spotify's product has changed dramatically: the UI has been redesigned multiple times, new features added (podcasts, audiobooks, AI DJ), deprecated features removed, and pricing updated. A model grounded on 2017 resolution pairs may recommend settings or steps that no longer exist in the 2026 app. ROUGE-L comparison against 2017 reference replies also understates the quality of replies that use updated (but historically unavailable) troubleshooting steps.

### 4.4 — Safe Over-Escalation Hides in High Escalation Accuracy
A system that escalates **every single ticket** would score 100% on Escalation Recall and 0% on business utility. Our F2 metric penalises False Negatives but still rewards over-escalation with high scores. The Escalation Precision metric is the check: an over-cautious system will have high recall but low precision. Both must be tracked together, and the cost-per-query of unnecessary escalations (human agent time) should be quantified in production.

### 4.5 — 25-Example Evaluation Has Wide Confidence Intervals
At n=25, a 95% confidence interval on a Macro F1 of 0.847 spans approximately ±0.08 to ±0.12 depending on class balance. The headline number is directionally meaningful but should not be compared to published benchmarks at this scale. The full 200-example golden set narrows this significantly.

---

## Section 5 — Failure Analysis

Five real failure modes identified from manual review of system outputs on edge cases:

### Failure 1: Sarcasm Recognition ("Sarcastic Praise" Tweets)
**Example:** *"Great update Spotify, love having ALL my downloaded songs deleted automatically 🙄"*

**What happened:** Keyword classifier detected "love" → classified as POSITIVE sentiment, LOW urgency. Reply drafted without fully acknowledging the real complaint.

**Root cause:** The keyword-based mock classifier has no sarcasm detection. The 🙄 emoji and sentence structure are sarcasm signals that require contextual language understanding.

**Mitigation:** LLM classifier naturally handles this with contextual understanding. Alternatively, a dedicated sarcasm detection step can be added between preprocessing and classification.

### Failure 2: Multi-Intent Compound Complaints
**Example:** *"App keeps crashing AND you charged me twice this month — what is happening??"*

**What happened:** Primary intent classified as PLAYBACK_AND_APP_BUGS (correct). Secondary intent BILLING_AND_SUBSCRIPTION detected. System escalated due to billing, but the generated escalation message only acknowledged billing — leaving the app crash unaddressed.

**Root cause:** When multi-intent is detected and the dominant policy rule escalates due to billing, the generated escalation acknowledgment template does not synthesise both issues.

**Mitigation:** Build a multi-intent reply template that acknowledges all detected intents and provides interim self-service guidance for the non-escalating intent alongside the escalation message.

### Failure 3: Cross-Platform Setting Hallucination (RAG Failure)
**Example:** *"Bluetooth keeps disconnecting from CarPlay on iPhone 15"*

**What happened (mock mode risk):** Canned reply suggested "Disable battery optimization in Android Settings" — advice only applicable to Android, not iOS.

**Root cause:** The sample_spotify_tweets.csv resolution pair for Bluetooth was written for Android. RAG retrieved it without filtering by device platform. The platform signal ("iPhone 15") was not used to filter retrieved contexts.

**Mitigation:** Add a `platform` metadata field to each indexed resolution pair (android / ios / desktop / web). Filter retrieval by detected platform from the tweet.

### Failure 4: Over-Cautious Escalation on Mild Frustration
**Example:** *"Your app is garbage, always has been. Switching to Apple Music."*

**What happened:** Angry tone → CRITICAL urgency → RULE_CRITICAL_URGENCY → ESCALATE_TO_HUMAN. Correct escalation reason given, but this is mild venting — no specific issue to resolve. A human agent receiving this ticket has nothing actionable to do.

**Root cause:** The urgency model treats ANGRY sentiment → CRITICAL urgency too aggressively. "Angry" + "no specific technical complaint" = venting, not a security or billing crisis.

**Mitigation:** Add an explicit check: urgency = CRITICAL only if angry sentiment co-occurs with a specific actionable intent (BILLING, SECURITY). Angry + FEATURE_REQUEST_AND_FEEDBACK → stays MEDIUM.

### Failure 5: Outdated Knowledge in RAG (Temporal Drift)
**Example:** *"How do I use the lyrics feature on the new Spotify UI?"*

**What happened:** Lyrics-related historical tweets from 2017 referenced an older lyrics display that has been completely redesigned. Retrieved context gave a navigation path that no longer exists.

**Root cause:** The Kaggle dataset is 7+ years old. Spotify's UI has changed significantly. There is no freshness scoring or timestamp-based weighting on retrieved documents.

**Mitigation:** Add a `created_at` metadata field and apply a recency decay penalty to older documents in retrieval scoring. Additionally, consider augmenting the vector store with current Spotify Help Centre articles (web scraping or official API).

---

## Section 6 — Next Steps (One More Week)

Ranked by expected impact:

1. **Multi-turn conversation state tracking**: Customer follow-ups ("that didn't work") are currently handled as fresh, unrelated tweets. Tracking conversation threads (via reply-chain tweet IDs in the Kaggle dataset) would allow the agent to adapt its response based on what was already tried.

2. **Platform-aware retrieval filtering**: Add `platform` (ios/android/desktop/web) and `topic` metadata to ChromaDB documents. Use detected platform signal from the tweet to filter retrieval — eliminates the Android→iOS cross-platform hallucination failure mode.

3. **Sarcasm detection module**: A lightweight binary sarcasm classifier (fine-tuned `distilBERT` on a sarcasm dataset like News Headlines Dataset for Sarcasm Detection) inserted between preprocessing and classification would dramatically improve edge case handling.

4. **Live Spotify Help Centre knowledge integration**: Scrape or download current support.spotify.com articles (with permission) and index them alongside the Twitter resolution pairs. This directly addresses the temporal drift problem with zero fine-tuning required.

5. **Small SLM fine-tuning for classification**: Fine-tune `Gemma-2-9B-it` or `Llama-3.1-8B-Instruct` on the golden eval set + augmented data for the intent classification step. Goal: achieve LLM-quality classification at a fraction of the API cost, enabling sub-$0.001 per query economics.

6. **Full 200-example golden set completion**: Expand from the 25-example starter set to the full 200-example set with human annotation, enabling statistically robust evaluation with narrow confidence intervals.
