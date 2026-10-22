"""Custom LangChain Embeddings adapter for OpenRouter.

OpenRouter's embedding endpoint for some models (e.g. liquid/lfm-2.5-embedding-*)
requires ``{"input": "single string"}`` rather than the OpenAI-standard
``{"input": ["array", "of", "strings"]}``.

This adapter sends one request per text (sequential) using the correct single-
string body format, making it compatible with those models while still
implementing the full LangChain ``Embeddings`` interface.

It also adds proactive inter-request throttling (``request_interval``) to stay
under free-tier rate limits, plus exponential-backoff retries with a cap.
"""

from __future__ import annotations

import logging
import time
from typing import List

import httpx
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_BATCH_BASE_URL = "https://openrouter.ai/api/beta"
_RETRY_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 8
_BACKOFF_BASE = 2.0   # seconds
_BACKOFF_MAX = 60.0   # cap per retry

_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}


class OpenRouterEmbeddings(Embeddings):
    """LangChain-compatible embeddings that call OpenRouter with a single-string
    ``input`` field per request (required by Liquid/Qwen and some other models).

    Parameters
    ----------
    model:
        The OpenRouter model slug, e.g. ``"liquid/lfm-2.5-embedding-350m:free"``.
    api_key:
        Your ``sk-or-v1-…`` key.
    base_url:
        OpenRouter base URL (defaults to ``https://openrouter.ai/api/v1``).
    timeout:
        HTTP request timeout in seconds.
    request_interval:
        Seconds to sleep between consecutive embedding calls to stay under
        free-tier rate limits (default: 1.5 s ≈ 40 req/min).
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 60.0,
        request_interval: float = 1.5,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.request_interval = request_interval
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_one(self, text: str) -> list[float]:
        """Embed a single piece of text, with exponential-backoff retry."""
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": text}

        for attempt in range(_MAX_RETRIES):
            response = self._client.post(url, json=payload)

            if response.status_code == 200:
                data = response.json()
                return data["data"][0]["embedding"]

            if response.status_code in _RETRY_CODES:
                wait = min(_BACKOFF_BASE ** attempt, _BACKOFF_MAX)
                logger.warning(
                    "OpenRouter embeddings HTTP %s (attempt %d/%d) — retrying in %.1fs",
                    response.status_code,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                continue

            # Non-retryable error — raise immediately
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(
                f"OpenRouter embeddings error {response.status_code}: {detail}"
            )

        raise RuntimeError(
            f"OpenRouter embeddings failed after {_MAX_RETRIES} retries "
            f"(last status: {response.status_code})"
        )

    # ------------------------------------------------------------------
    # LangChain Embeddings interface
    # ------------------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents, one request per text.

        A ``request_interval`` sleep is inserted between requests to avoid
        hitting free-tier rate limits.
        """
        result: list[list[float]] = []
        for i, text in enumerate(texts):
            if i > 0 and self.request_interval > 0:
                time.sleep(self.request_interval)
            logger.debug("Embedding document %d/%d", i + 1, len(texts))
            result.append(self._embed_one(text))
        return result

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self._embed_one(text)


# ---------------------------------------------------------------------------
# Async-batch adapter — uses /api/beta/batches
# ---------------------------------------------------------------------------


class BatchOpenRouterEmbeddings(Embeddings):
    """LangChain-compatible embeddings that use OpenRouter's Batch API.

    All texts passed to ``embed_documents`` are submitted as a single batch job
    to ``POST /api/beta/batches`` (``endpoint="/v1/embeddings"``).  The method
    then polls ``GET /api/beta/batches/:id`` until the job completes and returns
    the embedding vectors in the same order as the input texts.

    Benefits over the sequential ``OpenRouterEmbeddings``:

    * ~50 % lower cost (OpenRouter batch discount).
    * No per-request rate-limit pressure at ingestion time.
    * Only 1 HTTP round-trip to submit + polling instead of N sequential calls.

    ``embed_query`` bypasses the batch path and uses a fast synchronous single
    call because real-time query latency cannot tolerate polling delays.

    Parameters
    ----------
    model:
        OpenRouter model slug, e.g. ``"qwen/qwen3-embedding-4b"``.
    api_key:
        Your ``sk-or-v1-…`` key.
    base_url:
        Sync embeddings base URL (defaults to ``https://openrouter.ai/api/v1``).
        Used only by ``embed_query``.
    batch_base_url:
        Batch API base URL (defaults to ``https://openrouter.ai/api/beta``).
    timeout:
        Per-HTTP-request timeout in seconds (applies to submit + each poll).
    poll_interval:
        Seconds to sleep between status polls (default 10).
    max_wait:
        Maximum total seconds to wait for the batch to complete before raising
        ``TimeoutError`` (default 1 800 = 30 min).
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        batch_base_url: str = _BATCH_BASE_URL,
        timeout: float = 60.0,
        poll_interval: float = 10.0,
        max_wait: float = 1800.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.batch_base_url = batch_base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_wait = max_wait

        _auth_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(headers=_auth_headers, timeout=timeout)

    # ------------------------------------------------------------------
    # Batch lifecycle helpers
    # ------------------------------------------------------------------

    def _submit_batch(self, texts: list[str]) -> str:
        """Submit texts as a batch job and return the batch ``id``."""
        requests_payload = [
            {"custom_id": str(i), "body": {"input": text}}
            for i, text in enumerate(texts)
        ]
        payload = {
            "endpoint": "/v1/embeddings",
            "model": self.model,
            "requests": requests_payload,
        }

        url = f"{self.batch_base_url}/batches"
        for attempt in range(_MAX_RETRIES):
            response = self._client.post(url, json=payload)
            if response.status_code in (200, 201):
                batch_id: str = response.json()["id"]
                logger.info(
                    "Submitted embedding batch (%d texts) → batch_id=%s",
                    len(texts),
                    batch_id,
                )
                return batch_id

            if response.status_code in _RETRY_CODES:
                wait = min(_BACKOFF_BASE ** attempt, _BACKOFF_MAX)
                logger.warning(
                    "Batch submit HTTP %s (attempt %d/%d) — retrying in %.1fs",
                    response.status_code,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                continue

            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(
                f"OpenRouter batch submit error {response.status_code}: {detail}"
            )

        raise RuntimeError(
            f"OpenRouter batch submit failed after {_MAX_RETRIES} retries"
        )

    def _poll_batch(self, batch_id: str) -> dict:
        """Poll until the batch reaches a terminal status; return the batch object."""
        url = f"{self.batch_base_url}/batches/{batch_id}"
        deadline = time.monotonic() + self.max_wait
        poll_count = 0

        while True:
            response = self._client.get(url)
            if response.status_code != 200:
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                raise RuntimeError(
                    f"Batch status poll error {response.status_code}: {detail}"
                )

            batch = response.json()
            status: str = batch.get("status", "unknown")
            poll_count += 1
            logger.info(
                "Batch %s status=%s (poll #%d)", batch_id, status, poll_count
            )

            if status in _TERMINAL_STATUSES:
                return batch

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Batch {batch_id!r} did not complete within "
                    f"{self.max_wait:.0f}s (last status: {status!r})"
                )

            time.sleep(self.poll_interval)

    def _parse_results(self, batch: dict, n: int) -> list[list[float]]:
        """Extract and reorder embedding vectors from a completed batch object.

        Raises ``RuntimeError`` if the batch failed or any individual result
        returned a non-200 status code.
        """
        status = batch.get("status")
        if status != "completed":
            raise RuntimeError(
                f"Batch {batch.get('id')!r} ended with status {status!r}. "
                "Check the OpenRouter dashboard for details."
            )

        results: list[dict] = batch.get("results", [])
        # Map custom_id (== original position index as string) → embedding
        index_to_embedding: dict[int, list[float]] = {}

        for entry in results:
            custom_id: str = entry["custom_id"]
            idx = int(custom_id)
            resp = entry.get("response", {})
            status_code = resp.get("status_code", 0)

            if status_code != 200:
                body = resp.get("body", {})
                raise RuntimeError(
                    f"Batch result for custom_id={custom_id!r} returned "
                    f"status {status_code}: {body}"
                )

            body = resp["body"]
            embedding: list[float] = body["data"][0]["embedding"]
            index_to_embedding[idx] = embedding

        if len(index_to_embedding) != n:
            missing = [i for i in range(n) if i not in index_to_embedding]
            raise RuntimeError(
                f"Batch returned {len(index_to_embedding)} results but expected "
                f"{n}. Missing indices: {missing}"
            )

        return [index_to_embedding[i] for i in range(n)]

    # ------------------------------------------------------------------
    # Synchronous single-call path (used by embed_query)
    # ------------------------------------------------------------------

    def _embed_one(self, text: str) -> list[float]:
        """Embed a single text synchronously (no batch), with retries."""
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": text}

        for attempt in range(_MAX_RETRIES):
            response = self._client.post(url, json=payload)
            if response.status_code == 200:
                return response.json()["data"][0]["embedding"]

            if response.status_code in _RETRY_CODES:
                wait = min(_BACKOFF_BASE ** attempt, _BACKOFF_MAX)
                logger.warning(
                    "Sync embedding HTTP %s (attempt %d/%d) — retrying in %.1fs",
                    response.status_code,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                continue

            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(
                f"OpenRouter sync embedding error {response.status_code}: {detail}"
            )

        raise RuntimeError(
            f"OpenRouter sync embedding failed after {_MAX_RETRIES} retries"
        )

    # ------------------------------------------------------------------
    # LangChain Embeddings interface
    # ------------------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents via the OpenRouter Batch API (async, polled).

        All texts are submitted in one batch job.  Blocks until the job
        completes or ``max_wait`` is exceeded.
        """
        if not texts:
            return []

        batch_id = self._submit_batch(texts)
        batch = self._poll_batch(batch_id)
        return self._parse_results(batch, len(texts))

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string synchronously (bypasses batch API)."""
        return self._embed_one(text)
