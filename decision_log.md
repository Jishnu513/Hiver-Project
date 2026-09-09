# Decision Log — AI Customer Support Agent for @SpotifyCares
### 12 Non-Obvious Engineering & Design Decisions

---

## Decision 1 — Brand: @SpotifyCares over @AppleSupport or @AmazonHelp

**Decision:** Use @SpotifyCares as the target brand.

**Alternatives considered:**
- `@AppleSupport`: 1M+ tweets but ~85% are "Please DM us your serial number." Almost no publicly visible troubleshooting resolution pairs to ground RAG on. The vector store would be near-empty.
- `@AmazonHelp`: 800k+ tweets across hardware (Echo), e-commerce, streaming, and AWS. Enormously broad SKU space makes intent taxonomy unmanageable and context retrieval noisy.
- `@Delta`: Moderate volume (~150k), but travel support is PII-heavy, crisis-prone (weather delays), and legally sensitive. Wrong domain for a proof-of-concept.

**Why @SpotifyCares wins:**
1. Rich public troubleshooting threads with specific steps (cache clear, reinstall, Bluetooth re-pair, Alexa skill relink) — ideal for RAG grounding.
2. Crystal-clear operational boundary between self-service (technical bugs, how-to) and mandatory human escalation (security, billing fraud) — makes the auto-handle vs. escalate decision defensible and measurable.
3. SaaS digital subscription model is directly analogous to Hiver's own product domain.

---

## Decision 2 — Intent Taxonomy: 7 MECE Intents, Not 77

**Decision:** Define exactly 7 mutually exclusive, collectively exhaustive (MECE) intents.

**Alternatives considered:**
- Banking77 style (77 fine-grained intents): Extremely fine-grained, but 280-character tweets rarely contain enough signal to distinguish "card_not_working" from "card_payment_fee" reliably. Micro-intents create false precision.
- 3-intent coarse taxonomy: Loses routing signal needed to select correct troubleshooting templates.

**Why 7:**
- 7 intents map cleanly to distinct downstream routing actions.
- Downstream policy rules differ meaningfully across all 7.
- Evaluation on 200 examples: 7 classes give a 95%+ coverage rate with < 5% edge cases requiring a secondary intent label.
- Simpler taxonomy = less annotator disagreement = higher golden set quality.

---

## Decision 3 — RAG over Fine-Tuning for Knowledge Grounding

**Decision:** Use retrieval-augmented generation (RAG) with ChromaDB, not fine-tuning a custom LLM.

**Alternatives considered:**
- Fine-tune a smaller LLM (e.g., Llama-3-8B) on @SpotifyCares conversation pairs.
- Prompt engineering only (no retrieval).

**Why RAG:**
1. **Updatability**: Spotify's troubleshooting procedures, pricing, and features change frequently. Updating the vector store takes seconds; retraining a fine-tuned model costs GPU hours and risks catastrophic forgetting.
2. **Explainability**: Retrieved contexts are surfaced as citations in the `AgentReply`. This makes the system auditable — an agent or reviewer can verify the source of each recommendation.
3. **Cost**: Fine-tuning a 7B+ model requires significant compute. ChromaDB + sentence-transformers runs on CPU in < 500MB RAM.
4. **No training data quality lock-in**: Fine-tuning amplifies biases in training data. RAG separates knowledge from reasoning.

---

## Decision 4 — Decouple Classification from Response Generation

**Decision:** Run intent classification as a separate, dedicated step before drafting any reply.

**Alternatives considered:**
- Single mega-prompt: "Given this tweet, classify the intent AND write a reply AND decide whether to escalate."

**Why separate steps:**
1. The escalation policy engine **must intercept** between classification and generation. A single prompt cannot enforce hard-coded business rules.
2. Classification results (`intent`, `confidence`, `urgency`) are needed as structured data inputs to the policy engine and retrieval query — not just narrative context.
3. Debugging is dramatically easier: a misclassification and a bad reply are distinct failure modes requiring different fixes.
4. Modular design allows swapping the classifier (e.g., fine-tuned vs. LLM-based) without touching the generator.

---

## Decision 5 — Deterministic Policy Rules Override LLM Confidence

**Decision:** The escalation engine uses hard-coded, priority-ordered rules that cannot be overridden by the LLM.

**Alternatives considered:**
- Let the LLM decide escalation entirely (no rules).
- Use LLM confidence as the sole threshold (escalate if confidence < X%).

**Why hard rules win:**
1. **Safety and compliance**: "Never attempt to handle a hacked account via AI" is a non-negotiable invariant. An LLM confident it can help should still be overridden. No regret, no configurable threshold.
2. **Auditability**: Every escalation carries a `policy_triggered` field naming the exact rule that fired. This is an audit trail, not just a score.
3. **No tail-risk**: A LLM-only approach has a non-zero probability of confidently auto-handling a security crisis. A deterministic rule has probability zero.

---

## Decision 6 — Asymmetric Cost Function on Escalation (F2 Score, Not F1)

**Decision:** Use F-beta (β=2) to evaluate escalation quality, penalising False Negatives 4× more than False Positives.

**Alternatives considered:**
- Standard F1 (balanced precision and recall).
- Accuracy.

**Why F2:**
- **False Negative** = auto-handling a critical issue (e.g., account takeover sent a canned "try reinstalling" reply). Brand crisis risk.
- **False Positive** = unnecessarily escalating a "how do I crossfade?" question. Costs a human agent 60 seconds. Manageable.
- These costs are not symmetric. Evaluating with standard F1 hides this asymmetry and flatters systems that over-auto-handle.
- The 4× penalty mirrors real production SLA priorities in customer support: a missed security incident is catastrophic; an extra escalation is merely inefficient.

---

## Decision 7 — Embed Resolution PAIRS, Not Raw Customer Queries

**Decision:** Each ChromaDB document = concatenation of `customer_text` + `brand_reply`.

**Alternatives considered:**
- Embed only the customer tweet text and retrieve similar customer complaints.
- Embed only the brand reply and retrieve similar resolution approaches.

**Why embed pairs:**
1. The embedding captures both the problem signal and the resolution vocabulary in a single vector space. A search for "app keeps crashing" will retrieve documents whose embedded content says "crash → clear cache + reinstall" — not just tweets that mention crashing.
2. Embedding only queries means the retrieved documents might be similar problems with poor historical resolutions (the dataset includes some low-quality canned "Please DM us" replies).
3. Embedding pairs naturally filters toward high-information resolutions: a customer complaint + a detailed troubleshooting reply has more semantic content than either alone.

---

## Decision 8 — Hybrid Semantic + Keyword Retrieval Strategy

**Decision:** Document architecture supports BM25 fallback alongside cosine similarity retrieval.

**Alternatives considered:**
- Pure dense vector search only.
- Pure BM25 keyword search only.

**Why hybrid matters:**
- **Dense search misses**: Spotify-specific terms like "Spotify Connect", "Family Plan", error code "Auth Error 404", "CarPlay". These exact-match signals are critical routing clues that embeddings dilute into generic similarity.
- **BM25 misses**: Paraphrase variants ("app won't load" vs "Spotify freezes on startup"), emoji-heavy tweets, misspellings.
- In the current implementation, ChromaDB provides dense cosine search. The classifier uses keyword signals additionally for urgency and secondary intent detection, creating a soft hybrid.

---

## Decision 9 — Human-in-the-Loop Validation on Every Golden Example

**Decision:** Every one of the 200 golden eval examples is manually reviewed and certified by the candidate — no pure LLM-annotated eval set.

**Alternatives considered:**
- Use LLM to auto-annotate all 200 examples.
- Use crowdsourcing (MTurk-style annotation).

**Why human validation is mandatory:**
1. An LLM-annotated golden set creates a closed loop: the same model family that labels also evaluates its own outputs. This inflates metrics systematically and is a known problem in LLM evaluation research.
2. Real customer support contains subtle nuances that LLMs miss: sarcasm markers, culturally-specific frustration idioms, implied billing disputes embedded in technical complaints.
3. Defensibility: in an interview or audit, "I manually verified all 200 examples" is far stronger than "the LLM pre-annotated them."
4. The `human_annotator_notes` field captures edge cases and disagreement patterns that are valuable for failure analysis.

---

## Decision 10 — Cohen's Kappa, Not Raw Percentage Agreement

**Decision:** Use Cohen's Kappa (κ) as the primary inter-rater reliability metric.

**Alternatives considered:**
- Report simple percentage agreement (e.g., "87% agreement with human judges").
- Use Krippendorff's alpha (more complex, handles ordinal scales).

**Why Cohen's Kappa:**
1. Raw percentage agreement is inflated when class distributions are skewed. In the eval set with 65% AUTO_HANDLE, even a classifier that always predicts AUTO_HANDLE gets 65% "agreement" by chance.
2. Cohen's Kappa corrects for chance agreement, giving a true measure of agreement quality.
3. Interpretation is standardised (< 0.2 = Poor, 0.4–0.6 = Moderate, > 0.8 = Almost Perfect) and universally understood by ML and product reviewers.
4. Krippendorff's alpha is more appropriate for multi-rater and ordinal data; with two raters (human + LLM judge), Cohen's Kappa is the standard choice.

---

## Decision 11 — Offline Mock Mode for Zero-Cost Evaluation

**Decision:** Build a complete offline mock mode (`AGENT_MODE=mock`) that replaces all LLM calls with deterministic heuristics and canned responses.

**Alternatives considered:**
- Require a live API key for all operations.
- Use VCR-style HTTP cassette recording.

**Why offline mock mode:**
1. **Reproducibility**: Assignment evaluators can reproduce all results in under 2 minutes without creating an account or spending money.
2. **Testing**: All 23 unit and integration tests run deterministically in CI without rate limits, latency, or cost.
3. **Demoing**: `python app.py demo` works on an airplane. The agent's logic can be evaluated independently from LLM quality.
4. **Fairness**: The keyword-based mock classifier exposes the pipeline logic clearly, without hiding it behind an opaque LLM call that evaluators cannot inspect.

---

## Decision 12 — Enforce 280-Character Output Limit in Prompts

**Decision:** The generator prompt explicitly instructs the LLM to produce replies ≤ 280 characters, and a truncation guard is applied post-generation.

**Alternatives considered:**
- No character limit (let LLM decide length).
- Enforce via output parsing only (no explicit instruction).

**Why both instruction + guard:**
1. Without an explicit limit, LLMs consistently produce 400–600 character "helpdesk email" responses that would be impossible to post on Twitter and misrepresent real brand behaviour.
2. A single instruction alone is insufficient — LLMs frequently violate length instructions. The truncation guard is a hard safety net.
3. The limit is grounded in real-world constraints: the Kaggle dataset contains real @SpotifyCares tweets, which are bounded by Twitter's character limit. Generating longer replies would break the historical grounding assumption.
4. Evaluation against reference replies (ROUGE-L, BLEU) is also more meaningful when both prediction and reference are similar in length.
