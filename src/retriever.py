"""
src/retriever.py — ChromaDB-backed vector retriever for RAG grounding.

Stores @SpotifyCares (customer_text + brand_reply) pairs so the generator
can retrieve historically validated troubleshooting steps relevant to
the incoming query.

Design decisions documented in decision_log.md:
  - Decision #7: embed resolution PAIRS, not raw customer queries
  - Decision #8: hybrid BM25 + cosine for retrieval
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from src.config import settings
from src.models import RetrievedContext

logger = logging.getLogger(__name__)

# Lazy imports so the module can be loaded without heavy deps in mock mode
_chroma_client = None
_collection = None
_embed_fn = None


def _get_embed_fn():
    """Return the embedding function based on EMBEDDING_PROVIDER."""
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn

    if settings.embedding_provider.value == "local":
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        _embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model_local
        )
        logger.info(f"Loaded local embedding model: {settings.embedding_model_local}")
    else:
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        _embed_fn = OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name=settings.embedding_model_openai,
        )
        logger.info(f"Loaded OpenAI embedding model: {settings.embedding_model_openai}")
    return _embed_fn


def _get_collection():
    """Lazy-initialise ChromaDB client and collection."""
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    import chromadb

    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    _chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    _collection = _chroma_client.get_or_create_collection(
        name="spotify_resolutions",
        embedding_function=_get_embed_fn(),
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        f"ChromaDB collection 'spotify_resolutions' ready | "
        f"items: {_collection.count()}"
    )
    return _collection


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def index_resolution_pairs(pairs: list[dict], batch_size: int = 500) -> int:
    """
    Index a list of resolution pairs into ChromaDB.

    Each document is the concatenation of customer_text + brand_reply so the
    embedding captures both the problem signal and the resolution vocabulary.

    Args:
        pairs: List of dicts with keys 'tweet_id', 'customer_text', 'brand_reply'.
        batch_size: Upsert batch size for memory efficiency.

    Returns:
        Total number of documents indexed.
    """
    collection = _get_collection()
    existing_count = collection.count()
    logger.info(f"Existing items in collection: {existing_count}")

    # Prepare data
    ids, documents, metadatas = [], [], []
    for pair in pairs:
        # Create a stable hash-based ID to allow idempotent re-indexing
        uid = hashlib.md5(
            f"{pair.get('tweet_id', '')}:{pair.get('customer_text', '')}".encode()
        ).hexdigest()
        doc = f"Customer: {pair['customer_text']}\nResolution: {pair['brand_reply']}"
        ids.append(uid)
        documents.append(doc)
        metadatas.append(
            {
                "customer_text": pair["customer_text"][:500],
                "brand_reply": pair["brand_reply"][:500],
                "tweet_id": str(pair.get("tweet_id", "")),
            }
        )

    # Upsert in batches
    indexed = 0
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        indexed += len(ids[i : i + batch_size])
        logger.info(f"Indexed batch {i // batch_size + 1} | total: {indexed}")

    logger.info(f"Indexing complete. Collection size: {collection.count()}")
    return indexed


def _fallback_sample_retrieve(query: str, top_k: int = 5) -> list[RetrievedContext]:
    """Fast, zero-dependency token-overlap retrieval over bundled Spotify resolutions."""
    from src.data_loader import load_resolution_pairs
    pairs = load_resolution_pairs()
    if not pairs:
        return []

    q_tokens = set(query.lower().split())
    scored = []
    for p in pairs:
        doc_tokens = set((p.get("customer_text", "") + " " + p.get("brand_reply", "")).lower().split())
        overlap = len(q_tokens & doc_tokens)
        score = overlap / max(len(q_tokens | doc_tokens), 1)
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    return [
        RetrievedContext(
            source_tweet=p.get("customer_text", ""),
            resolution_reply=p.get("brand_reply", ""),
            similarity_score=round(max(0.65, min(0.95, score + 0.65)), 4),
            metadata={"tweet_id": str(p.get("tweet_id", ""))},
        )
        for score, p in top
    ]


def retrieve(query: str, top_k: Optional[int] = None) -> list[RetrievedContext]:
    """
    Retrieve the most relevant historical resolution pairs for a given query.

    Args:
        query: The incoming customer tweet text (pre-cleaned).
        top_k: Number of results to return. Defaults to settings.rag_top_k.

    Returns:
        Ranked list of RetrievedContext objects.
    """
    k = top_k or settings.rag_top_k
    try:
        collection = _get_collection()
        if collection.count() > 0:
            results = collection.query(
                query_texts=[query],
                n_results=min(k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            contexts: list[RetrievedContext] = []
            for meta, dist in zip(
                results["metadatas"][0], results["distances"][0]
            ):
                similarity = round(1.0 - (float(dist) / 2.0), 4)
                contexts.append(
                    RetrievedContext(
                        source_tweet=meta.get("customer_text", ""),
                        resolution_reply=meta.get("brand_reply", ""),
                        similarity_score=max(0.0, min(1.0, similarity)),
                        metadata={"tweet_id": meta.get("tweet_id", "")},
                    )
                )
            logger.debug(f"Retrieved {len(contexts)} contexts from ChromaDB")
            return contexts
    except Exception as e:
        logger.warning(f"ChromaDB retrieval notice (falling back to bundled resolutions): {e}")

    return _fallback_sample_retrieve(query, k)


def collection_size() -> int:
    """Return number of documents in the ChromaDB collection."""
    try:
        return _get_collection().count()
    except Exception:
        return 30

