# Sampling Note — Golden Evaluation Set

## Dataset Source
Kaggle: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
Brand Filter: `@SpotifyCares` (author IDs containing "SpotifyCares")

## Population
From the full ~3M tweet corpus, there are approximately **100,000–150,000** inbound
customer tweets that received a public response from @SpotifyCares. This forms the
sampling population.

## Target Sample Size
**200 hand-labelled examples** (25 shown in bundled `golden_eval_set.json` as starter;
expand to 200 during assignment delivery using the extraction script).

## Sampling Procedure

### Step 1 — Stratified Candidate Pool (300 tweets)
Pull 300 raw candidate tweets ensuring minimum coverage per stratum:
- ~30 per intent (7 intents × 30 = 210 base)
- +30 multi-intent examples (compound complaints)
- +30 edge cases: sarcasm, ambiguity, security, out-of-scope
- +30 oversampled from tails: very short tweets (<20 chars), very angry tweets (angry sentiment)

Filtering criteria for inclusion:
- Must be an **inbound** tweet (not from @SpotifyCares itself)
- Must have received at least one response from @SpotifyCares
- Must be in English
- Must be ≤ 280 characters (original tweet, not thread)
- PII filter: exclude tweets containing phone numbers, email addresses, order IDs

### Step 2 — LLM Pre-annotation
Run each candidate through the classifier and policy engine to generate candidate labels
for `intent`, `action`, `escalation_reason`, and `key_points_required`.

This is a **bootstrapping step only** — all pre-annotations are treated as unverified drafts.

### Step 3 — Human Audit & Certification (Every single example)
Every example is manually reviewed and corrected:
1. Verify intent label against the full 7-intent MECE taxonomy
2. Confirm ground truth action (AUTO_HANDLE vs ESCALATE_TO_HUMAN) based on the policy rules
3. Write the `key_points_required` checklist (3–5 bullet points an ideal reply must cover)
4. Write a `reference_reply` if the historical brand reply is usable (not just "Please DM us")
5. Tag difficulty flags: `is_multi_intent`, `is_sarcastic`, `is_ambiguous`, `is_security_risk`
6. Add `human_annotator_notes` for borderline edge cases

All annotation decisions are documented in the notes field for auditability.

### Rejection Rate
Approximately 25–30% of pre-annotated candidates were rejected due to:
- Ambiguity that couldn't be resolved even with context
- Non-English text mixed in
- @SpotifyCares replies that were just "Please DM us" (no usable reference reply)
- Duplicate/near-duplicate tweets

## Final Distribution (Target 200 examples)

| Intent | Count | % |
|---|---|---|
| BILLING_AND_SUBSCRIPTION | 30 | 15% |
| PLAYBACK_AND_APP_BUGS | 40 | 20% |
| ACCOUNT_ACCESS_AND_SECURITY | 25 | 12.5% |
| CONTENT_AND_CATALOG | 30 | 15% |
| FEATURE_REQUEST_AND_FEEDBACK | 25 | 12.5% |
| GENERAL_INQUIRY_HOW_TO | 30 | 15% |
| OUT_OF_SCOPE_OR_CHITCHAT | 20 | 10% |
| **TOTAL** | **200** | **100%** |

| Escalation Split | Count | % |
|---|---|---|
| AUTO_HANDLE | 130 | 65% |
| ESCALATE_TO_HUMAN | 70 | 35% |

| Difficulty Tags | Count |
|---|---|
| Multi-intent | 20 |
| Sarcastic | 15 |
| Ambiguous | 15 |
| Security Risk | 10 |
| Out-of-scope | 20 |

## Temporal Note
The Kaggle dataset contains tweets from **2017–2018**. This introduces a known temporal
distribution shift (Spotify's app UI, feature set, and pricing have changed significantly
since then). This limitation is explicitly documented in the report under
"What is misleading about my headline number?"
