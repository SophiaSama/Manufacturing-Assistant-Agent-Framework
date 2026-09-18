"""Cross-encoder reranker integration using LiteLLM and Hugging Face.

Reranks candidate document chunks based on cross-attention semantic relevance
using Hugging Face models (e.g. BAAI/bge-reranker-base).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "huggingface/BAAI/bge-reranker-base"
DEFAULT_TOP_N = 5


def get_hf_token() -> str | None:
    """Retrieve Hugging Face token from environment variables."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if token:
        token = token.strip()
        if token.lower().startswith("your-") or token.lower() in ("placeholder", "none", ""):
            return None
    return token or None


def is_rerank_enabled() -> bool:
    """Check if reranking is enabled in configuration."""
    raw = os.getenv("RERANK_ENABLED", "").strip().lower()
    if raw in ("false", "0", "no"):
        return False
    if raw in ("true", "1", "yes"):
        return True
    # Auto-enable if HF token is present
    return bool(get_hf_token())


def get_rerank_model() -> str:
    """Get the configured rerank model name."""
    model = os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL).strip()
    if not model.startswith("huggingface/") and "/" in model and not model.startswith("cohere/"):
        model = f"huggingface/{model}"
    return model


def get_top_n() -> int:
    """Get default top_n count for reranking."""
    raw = os.getenv("RERANK_TOP_N", str(DEFAULT_TOP_N)).strip()
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_TOP_N


def _hf_serverless_rerank(
    query: str,
    texts: list[str],
    model: str,
    token: str,
    top_n: int,
) -> list[tuple[int, float]]:
    """Direct fallback to Hugging Face Serverless text-pair classification pipeline."""
    import httpx

    clean_model = model.removeprefix("huggingface/")
    models_to_try = [clean_model]
    if clean_model != "BAAI/bge-reranker-base":
        models_to_try.append("BAAI/bge-reranker-base")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": [{"text": query, "text_pair": t} for t in texts]}

    for m in models_to_try:
        url = f"https://router.huggingface.co/hf-inference/models/{m}"
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=15.0)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    scores = [item["score"] for item in data[0]]
                elif isinstance(data, list):
                    scores = [item["score"] for item in data]
                else:
                    scores = []
                indexed_scores = list(enumerate(scores))
                indexed_scores.sort(key=lambda x: x[1], reverse=True)
                logger.info("Successfully reranked with Hugging Face Serverless model: %s", m)
                return indexed_scores[:top_n]
            else:
                logger.debug("HF serverless model %s returned status %d: %s", m, resp.status_code, resp.text[:100])
        except Exception as e:
            logger.debug("HF serverless request for %s failed: %s", m, e)

    return []


def rerank_documents(
    query: str,
    documents: list[dict[str, Any]],
    top_n: int | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Rerank candidate document dictionaries using LiteLLM / Hugging Face.

    Each input document is expected to have a 'text' key containing its passage.
    Returns the top_n documents sorted by relevance score, with 'rerank_score' attached.
    If reranking is disabled, token is missing, or an error occurs, gracefully falls back
    to the original document ordering.
    """
    if not documents:
        return []

    target_top_n = top_n if top_n is not None else get_top_n()
    target_top_n = max(1, target_top_n)

    # If only 1 document, return immediately
    if len(documents) == 1:
        return documents[:target_top_n]

    token = api_key or get_hf_token()
    if not is_rerank_enabled() or not token:
        logger.debug(
            "LiteLLM rerank skipped (enabled=%s, token_present=%s). Returning raw candidates.",
            is_rerank_enabled(),
            bool(token),
        )
        return documents[:target_top_n]

    model_name = model or get_rerank_model()

    # Extract text strings for reranking
    texts: list[str] = [str(doc.get("text", "")) for doc in documents]

    custom_base = os.getenv("HF_API_BASE") or os.getenv("HUGGINGFACE_API_BASE")
    is_hf_model = model_name.startswith("huggingface/") or "/" in model_name

    # Route 1: For Hugging Face serverless models, call HF Inference Router directly
    if is_hf_model and not custom_base:
        try:
            hf_scores = _hf_serverless_rerank(
                query=query,
                texts=texts,
                model=model_name,
                token=token,
                top_n=target_top_n,
            )
            if hf_scores:
                reranked_docs = []
                for idx, score in hf_scores:
                    if 0 <= idx < len(documents):
                        doc_copy = dict(documents[idx])
                        doc_copy["rerank_score"] = float(score)
                        reranked_docs.append(doc_copy)
                return reranked_docs[:target_top_n]
        except Exception as exc:
            logger.debug("Native Hugging Face inference call failed (%s); trying LiteLLM...", exc)

    # Route 2: Try litellm.rerank (for custom TEI, Cohere, Jina, Together, etc.)
    try:
        import litellm

        rerank_kwargs: dict[str, Any] = {
            "model": model_name,
            "query": query,
            "documents": texts,
            "top_n": min(target_top_n, len(documents)),
            "api_key": token,
        }
        if custom_base:
            rerank_kwargs["api_base"] = custom_base

        response = litellm.rerank(**rerank_kwargs)

        results_list = getattr(response, "results", None)
        if results_list is None and isinstance(response, dict):
            results_list = response.get("results", [])

        if results_list:
            reranked_docs: list[dict[str, Any]] = []
            for item in results_list:
                if isinstance(item, dict):
                    idx = item.get("index", 0)
                    score = item.get("relevance_score", 0.0)
                else:
                    idx = getattr(item, "index", 0)
                    score = getattr(item, "relevance_score", 0.0)

                if 0 <= idx < len(documents):
                    doc_copy = dict(documents[idx])
                    doc_copy["rerank_score"] = float(score)
                    reranked_docs.append(doc_copy)

            if reranked_docs:
                return reranked_docs[:target_top_n]

    except Exception as exc:
        logger.info(
            "LiteLLM rerank call was not supported or timed out (%s: %s). Trying Hugging Face inference pipeline...",
            type(exc).__name__,
            exc,
        )

    # Attempt 2: Direct Hugging Face Serverless pipeline
    try:
        hf_scores = _hf_serverless_rerank(
            query=query,
            texts=texts,
            model=model_name,
            token=token,
            top_n=target_top_n,
        )
        if hf_scores:
            reranked_docs = []
            for idx, score in hf_scores:
                if 0 <= idx < len(documents):
                    doc_copy = dict(documents[idx])
                    doc_copy["rerank_score"] = float(score)
                    reranked_docs.append(doc_copy)
            return reranked_docs[:target_top_n]
    except Exception as exc:
        logger.warning("Hugging Face serverless rerank failed (%s: %s).", type(exc).__name__, exc)

    # Fallback to un-reranked vector similarity order
    logger.warning("Reranking unsuccessful. Falling back to un-reranked vector similarity order.")
    return documents[:target_top_n]
