"""RBAC-filtered hybrid vector + knowledge graph retrieval for NGA corpus."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

# NGA document categories for retrieval routing
VALID_CATEGORIES = (
    "OPR",   # Operator SOPs
    "TEC",   # Technician SOPs + Machine Specs
    "FA",    # Failure Analysis + Escalation
    "QCR",   # Recall & Quality
    "WO",    # Maintenance Work Orders
    "SQ",    # Supplier Quality
    "TR",    # Training Records
)


def normalize_categories(category: str | list[str] | None) -> list[str]:
    """Normalize category input into a deduplicated list."""
    if category is None:
        return list(VALID_CATEGORIES)
    raw = [category] if isinstance(category, str) else list(category)
    seen: set[str] = set()
    result: list[str] = []
    for v in raw:
        upper = v.strip().upper()
        if upper in VALID_CATEGORIES and upper not in seen:
            seen.add(upper)
            result.append(upper)
    return result or list(VALID_CATEGORIES)


def _search_one_category(
    store: Any,
    query: str,
    category: str,
    k: int,
    user_level: int = 1,
) -> list[dict[str, Any]]:
    """Execute RBAC-filtered similarity search for one document category."""
    chroma_filter: dict[str, Any] = {
        "$and": [
            {"category": {"$eq": category}},
            {"level_rank": {"$lte": user_level}},
        ]
    }
    docs = store.similarity_search(query, k=k, filter=chroma_filter)
    return [
        {
            "category": category,
            "doc_id": doc.metadata.get("doc_id"),
            "doc_name": doc.metadata.get("doc_name"),
            "section": doc.metadata.get("section"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "text": doc.page_content,
            "access_level": doc.metadata.get("access_level", "operator"),
            "level_rank": doc.metadata.get("level_rank", 1),
            "corpus_source": doc.metadata.get("corpus_source", "canonical"),
        }
        for doc in docs
    ]


def retrieve_documents(
    store: Any,
    query: str,
    category: str | list[str] | None,
    k: int = 5,
    user_level: int = 1,
    cache: Any | None = None,
) -> list[dict[str, Any]]:
    """Run parallel RBAC-filtered similarity search across categories.

    When `cache` is provided (or an active cache_scope is set), results are
    cached per (query, categories, k, user_level) with corpus-version and
    RBAC scoping (docs/cache-design.md §3.2/§4).
    """
    categories = normalize_categories(category)

    def _run() -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        errors: list[Exception] = []
        with ThreadPoolExecutor(max_workers=min(len(categories), 4)) as executor:
            futures = {
                executor.submit(
                    _search_one_category, store, query, cat, k, user_level
                ): cat
                for cat in categories
            }
            for future, cat in futures.items():
                try:
                    results.extend(future.result())
                except Exception as exc:
                    errors.append(exc)
                    logger.debug(
                        "Retrieval failed for query=%r category=%r: %s",
                        query, cat, exc,
                    )
        if errors and len(errors) >= len(categories):
            # All categories failed — surface the likely cause loudly instead
            # of silently returning empty context (agent would answer blind).
            from nga.ingestion.build_vector_store import _is_dimension_mismatch

            dim_errors = [e for e in errors if _is_dimension_mismatch(e)]
            if dim_errors:
                logger.error(
                    "ALL retrieval categories failed: embedding dimension "
                    "mismatch for query=%r. The vector store was built with a "
                    "different embedding model than the runtime EMBEDDING_MODEL "
                    "— rebuild the store with the matching model.",
                    query,
                )
            else:
                logger.error(
                    "ALL retrieval categories failed for query=%r: %s",
                    query, errors[0],
                )
        if results:
            from enterprise_agent.retrieval.reranker import rerank_documents
            results = rerank_documents(query=query, documents=results, top_n=k)
        return results

    from nga.cache.layers import cached_retrieve

    return cached_retrieve(
        _run,
        query=query,
        categories=categories,
        k=k,
        user_level=user_level,
        cache=cache,
    )


def retrieve_graph_evidence(
    graph: Any,
    embeddings: Any,
    query: str,
    *,
    k_entities: int = 5,
    max_hops: int = 2,
    user_level: int = 1,
    cache: Any | None = None,
) -> dict[str, Any]:
    """Run RBAC-filtered GraphRAG local search (cached per query params)."""
    def _run() -> dict[str, Any]:
        try:
            from nga.graphrag.search import build_graph_evidence
            return build_graph_evidence(
                graph, query, embeddings,
                k_entities=k_entities, max_hops=max_hops, user_level=user_level,
            )
        except Exception:
            logger.exception("Graph retrieval failed for query=%r", query)
            return {"entities": [], "relations": [], "community_summaries": []}

    from nga.cache.layers import cached_graph_evidence

    return cached_graph_evidence(
        _run,
        query=query,
        k_entities=k_entities,
        max_hops=max_hops,
        user_level=user_level,
        cache=cache,
    )


def build_retrieval_payload(
    query: str,
    categories: list[str],
    results: list[dict[str, Any]],
    graph_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    references: list[dict[str, Any]] = []
    for r in results:
        ref = {
            "doc_id": r.get("doc_id"),
            "doc_name": r.get("doc_name"),
            "section": r.get("section"),
            "chunk_id": r.get("chunk_id"),
            "category": r.get("category"),
            "corpus_source": r.get("corpus_source", "canonical"),
        }
        if "rerank_score" in r:
            ref["rerank_score"] = r["rerank_score"]
        references.append(ref)
    payload: dict[str, Any] = {
        "query": query,
        "categories": categories,
        "results": results,
        "references": references,
    }
    if graph_evidence:
        payload["graph_evidence"] = graph_evidence
    return payload


def format_retrieval_results(
    results: list[dict[str, Any]],
    graph_evidence: dict[str, Any] | None = None,
) -> str:
    """Render retrieval results as an LLM-readable context block."""
    if not results:
        block = "(no relevant document passages found)"
    else:
        lines = []
        for r in results:
            doc_id = r.get("doc_id", "")
            section = r.get("section", "")
            cat = r.get("category", "")
            source = r.get("corpus_source", "canonical")
            prefix = f"[{cat}]" if cat else ""
            ref = f"[{doc_id}]" if doc_id else ""
            sec = f" §{section}" if section else ""
            src_tag = "[VARIANT]" if source == "variant" else ""
            src = f" {src_tag}" if src_tag else ""
            lines.append(f"{prefix}{src}{ref}{sec} {r['text']}")
        block = "\n\n".join(lines)

    if graph_evidence:
        try:
            from nga.graphrag.search import format_graph_evidence
            graph_block = format_graph_evidence(graph_evidence)
            if graph_block:
                block = f"{block}\n\n{graph_block}"
        except Exception:
            pass

    return block
