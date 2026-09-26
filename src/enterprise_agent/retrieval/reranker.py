"""Jev-based reranker using TypeSafe Choice primitive.

Reranks candidate document chunks by presenting all candidates as options
in a single TypeSafe Choice call.  The probability distribution across
candidates is used to sort and select the top_n most relevant documents.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from typesafe_sdk import Choice, TypeSafeClient

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "jev-1.12"
DEFAULT_TOP_N = 5


def get_rerank_model() -> str:
    """Get the configured rerank model name from RERANK_MODEL env var."""
    return os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL).strip()


def get_typesafe_api_key() -> str | None:
    """Retrieve TypeSafe API key from environment variables."""
    key = os.getenv("TYPESAFE_API_KEY")
    if key:
        key = key.strip()
        if key.lower().startswith("your-") or key.lower() in ("placeholder", "none", ""):
            return None
    return key or None


def is_rerank_enabled() -> bool:
    """Check if reranking is enabled in configuration."""
    raw = os.getenv("RERANK_ENABLED", "").strip().lower()
    if raw in ("false", "0", "no"):
        return False
    if raw in ("true", "1", "yes"):
        return True
    # Auto-enable if TypeSafe API key is present
    return bool(get_typesafe_api_key())


def get_top_n() -> int:
    """Get default top_n count for reranking."""
    raw = os.getenv("RERANK_TOP_N", str(DEFAULT_TOP_N)).strip()
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_TOP_N


def rerank_documents(
    query: str,
    documents: list[dict[str, Any]],
    top_n: int | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Rerank candidate document dictionaries using Jev Choice.

    Each input document is expected to have a ``text`` key containing its passage.
    All candidates are presented as Choice options in a single TypeSafe API call.
    Returns the top_n documents sorted by probability, with ``rerank_score`` attached.

    If reranking is disabled, the API key is missing, or an error occurs, gracefully
    falls back to the original document ordering.
    """
    if not documents:
        return []

    target_top_n = top_n if top_n is not None else get_top_n()
    target_top_n = max(1, target_top_n)

    # If only 1 document, return immediately
    if len(documents) == 1:
        return documents[:target_top_n]

    key = api_key or get_typesafe_api_key()
    if not is_rerank_enabled() or not key:
        logger.debug(
            "Jev rerank skipped (enabled=%s, key_present=%s). Returning raw candidates.",
            is_rerank_enabled(),
            bool(key),
        )
        return documents[:target_top_n]

    model_name = model or get_rerank_model()

    # Extract text strings and build candidate map
    texts: list[str] = [str(doc.get("text", "")) for doc in documents]
    candidate_keys = [f"doc_{i}" for i in range(len(texts))]
    candidate_state = {k: t for k, t in zip(candidate_keys, texts)}
    criteria = {k: None for k in candidate_keys}

    try:
        client = TypeSafeClient(api_key=key)
        response = client.system_one(
            state={"query": query, "candidates": candidate_state},
            questions={
                "best_match": Choice(
                    instructions="Which candidate passage best answers the user's query?",
                    criteria=criteria,
                ),
            },
            model=model_name,
        )

        probabilities = response.answers["best_match"].probabilities
        ranked = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        reranked_docs: list[dict[str, Any]] = []
        for key_name, score in ranked[:target_top_n]:
            idx = int(key_name.split("_")[1])
            if 0 <= idx < len(documents):
                doc_copy = dict(documents[idx])
                doc_copy["rerank_score"] = float(score)
                reranked_docs.append(doc_copy)

        if reranked_docs:
            return reranked_docs

    except Exception as exc:
        logger.warning(
            "Jev rerank call failed (%s: %s). Falling back to un-reranked order.",
            type(exc).__name__,
            exc,
        )

    # Fallback to un-reranked vector similarity order
    logger.warning("Reranking unsuccessful. Falling back to un-reranked vector similarity order.")
    return documents[:target_top_n]
