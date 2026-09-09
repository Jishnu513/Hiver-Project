"""
src/data_loader.py — Load, filter, and prepare @SpotifyCares tweets.

Supports two modes:
1. Kaggle mode  — filters twcs.csv (downloaded from Kaggle) for @SpotifyCares.
2. Sample mode  — loads the curated sample_spotify_tweets.csv bundled in /data.

Usage:
    from src.data_loader import load_resolution_pairs, load_golden_eval_set

    pairs = load_resolution_pairs()          # for indexing into ChromaDB
    examples = load_golden_eval_set()        # for evaluation runs
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import settings
from src.models import GoldenExample

logger = logging.getLogger(__name__)

_BRAND_HANDLE = "SpotifyCares"


# ─────────────────────────────────────────────────────────────────────────────
# Raw Kaggle CSV Processing
# ─────────────────────────────────────────────────────────────────────────────

def load_kaggle_twcs(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the raw Kaggle Twitter Customer Support dataset (twcs.csv).
    Filters to only @SpotifyCares conversations.

    Args:
        csv_path: Path to twcs.csv downloaded from Kaggle.

    Returns:
        DataFrame with columns: [tweet_id, author_id, inbound, text, created_at,
                                  response_tweet_id, in_response_to_tweet_id]
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Kaggle dataset not found at: {path}\n"
            "Download from: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter\n"
            "Then pass the path to this function."
        )
    logger.info(f"Loading Kaggle dataset from {path} ...")
    df = pd.read_csv(path, dtype=str)
    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    logger.info(f"Total rows: {len(df):,}")
    return df


def extract_spotify_pairs(
    df: pd.DataFrame,
    max_pairs: int = 15_000,
) -> list[dict]:
    """
    Extract (customer_tweet, @SpotifyCares_reply) resolution pairs from raw data.

    Strategy:
      1. Identify all outbound tweets authored by @SpotifyCares.
      2. Join back to the inbound customer tweet they were responding to.
      3. Return as list of dicts with keys: customer_text, brand_reply, tweet_id.
    """
    # All author IDs that belong to SpotifyCares brand
    brand_authors = set(
        df[df["author_id"].str.contains(_BRAND_HANDLE, case=False, na=False)]["author_id"]
    )
    logger.info(f"Found {len(brand_authors)} @SpotifyCares author IDs")

    outbound = df[df["author_id"].isin(brand_authors)].copy()
    outbound = outbound[outbound["in_response_to_tweet_id"].notna()]
    logger.info(f"Outbound @SpotifyCares replies: {len(outbound):,}")

    # Map tweet_id → text for fast lookup
    id_to_text = dict(zip(df["tweet_id"], df["text"]))

    pairs = []
    for _, row in outbound.iterrows():
        customer_id = row["in_response_to_tweet_id"]
        customer_text = id_to_text.get(customer_id, "")
        if not customer_text.strip():
            continue
        brand_reply = _clean_tweet(str(row["text"]))
        customer_text = _clean_tweet(customer_text)
        if len(customer_text) < 10 or len(brand_reply) < 10:
            continue
        pairs.append(
            {
                "tweet_id": str(row["tweet_id"]),
                "customer_text": customer_text,
                "brand_reply": brand_reply,
                "author_id": str(row["author_id"]),
                "created_at": str(row.get("created_at", "")),
            }
        )
        if len(pairs) >= max_pairs:
            break

    logger.info(f"Extracted {len(pairs):,} resolution pairs")
    return pairs


def save_sample_csv(pairs: list[dict], output_path: str | Path) -> None:
    """Save a curated subsample of resolution pairs to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs).to_csv(path, index=False)
    logger.info(f"Saved {len(pairs)} pairs to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Bundled Sample Data (no Kaggle download required)
# ─────────────────────────────────────────────────────────────────────────────

def load_resolution_pairs(
    kaggle_csv: Optional[str | Path] = None,
    max_pairs: int = 10_000,
) -> list[dict]:
    """
    Load resolution pairs for indexing into ChromaDB.

    Priority order:
      1. If kaggle_csv is provided → extract fresh pairs from twcs.csv.
      2. Otherwise → fall back to the bundled sample_spotify_tweets.csv.
    """
    if kaggle_csv is not None:
        df = load_kaggle_twcs(kaggle_csv)
        return extract_spotify_pairs(df, max_pairs=max_pairs)

    sample_path = settings.sample_tweets_path
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Bundled sample not found at {sample_path}.\n"
            "Run: python -m src.data_loader --kaggle-csv /path/to/twcs.csv"
        )
    logger.info(f"Loading bundled sample from {sample_path}")
    df = pd.read_csv(sample_path, dtype=str)
    return df.to_dict(orient="records")


def load_golden_eval_set() -> list[GoldenExample]:
    """Load and validate the 200 hand-labelled golden examples."""
    path = settings.golden_eval_path
    if not path.exists():
        raise FileNotFoundError(
            f"Golden eval set not found at {path}.\n"
            "Expected: data/golden_eval_set.json"
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    examples = [GoldenExample(**item) for item in raw]
    logger.info(f"Loaded {len(examples)} golden evaluation examples")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_tweet(text: str) -> str:
    """Strip @mentions, URLs, excessive whitespace from a tweet."""
    text = re.sub(r"@\w+", "", text)           # remove @mentions
    text = re.sub(r"http\S+|www\.\S+", "", text)  # remove URLs
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point (python -m src.data_loader --kaggle-csv twcs.csv)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract @SpotifyCares data from Kaggle CSV")
    parser.add_argument("--kaggle-csv", required=True, help="Path to twcs.csv from Kaggle")
    parser.add_argument("--max-pairs", type=int, default=10_000)
    parser.add_argument("--output", default=str(settings.sample_tweets_path))
    args = parser.parse_args()

    pairs = load_resolution_pairs(kaggle_csv=args.kaggle_csv, max_pairs=args.max_pairs)
    save_sample_csv(pairs, args.output)
    print(f"✅ Saved {len(pairs)} pairs to {args.output}")
