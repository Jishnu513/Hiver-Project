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
    collection = _get_collection()

    if collection.count() == 0:
        logger.warning("ChromaDB collection is empty — skipping retrieval")
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    contexts: list[RetrievedContext] = []
    for meta, dist in zip(
        results["metadatas"][0], results["distances"][0]
    ):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to cosine similarity: sim = 1 - (dist / 2)
        similarity = round(1.0 - (float(dist) / 2.0), 4)
        contexts.append(
            RetrievedContext(
                source_tweet=meta.get("customer_text", ""),
                resolution_reply=meta.get("brand_reply", ""),
                similarity_score=max(0.0, min(1.0, similarity)),
                metadata={"tweet_id": meta.get("tweet_id", "")},
            )
        )

    logger.debug(f"Retrieved {len(contexts)} contexts for query (top sim: {contexts[0].similarity_score if contexts else 'N/A'})")
    return contexts


def collection_size() -> int:
    """Return number of documents in the ChromaDB collection."""
    try:
        return _get_collection().count()
    except Exception:
        return 0
