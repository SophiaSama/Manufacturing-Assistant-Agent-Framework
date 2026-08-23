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
        }
        for doc in docs
    ]


def retrieve_documents(
    store: Any,
    query: str,
    category: str | list[str] | None,
    k: int = 5,
    user_level: int = 1,
) -> list[dict[str, Any]]:
    """Run parallel RBAC-filtered similarity search across categories."""
    categories = normalize_categories(category)
    results: list[dict[str, Any]] = []
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
            except Exception:
                logger.exception(
                    "Retrieval failed for query=%r category=%r", query, cat
                )
    return results


def retrieve_graph_evidence(
    graph: Any,
    embeddings: Any,
    query: str,
    *,
    k_entities: int = 5,
    max_hops: int = 2,
    user_level: int = 1,
) -> dict[str, Any]:
    """Run RBAC-filtered GraphRAG local search."""
    try:
        from nga.graphrag.search import build_graph_evidence
        return build_graph_evidence(
            graph, query, embeddings,
            k_entities=k_entities, max_hops=max_hops, user_level=user_level,
        )
    except Exception:
        logger.exception("Graph retrieval failed for query=%r", query)
        return {"entities": [], "relations": [], "community_summaries": []}


def build_retrieval_payload(
    query: str,
    categories: list[str],
    results: list[dict[str, Any]],
    graph_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable retrieval result payload."""
    references = [
        {
            "doc_id": r.get("doc_id"),
            "doc_name": r.get("doc_name"),
            "section": r.get("section"),
            "chunk_id": r.get("chunk_id"),
            "category": r.get("category"),
        }
        for r in results
    ]
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
            prefix = f"[{cat}]" if cat else ""
            ref = f"[{doc_id}]" if doc_id else ""
            sec = f" §{section}" if section else ""
            lines.append(f"{prefix}{ref}{sec} {r['text']}")
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
