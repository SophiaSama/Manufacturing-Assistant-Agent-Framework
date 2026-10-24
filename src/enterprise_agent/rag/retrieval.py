"""Generic hybrid retrieval tool and vector store loader for Enterprise Agent."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from enterprise_agent.config.domain_config import DocumentConfig, DomainConfig

logger = logging.getLogger("enterprise_agent.rag.retrieval")


def build_or_open_vector_store(
    domain_config: DomainConfig,
    documents_dir: Path | str | None = None,
    persist_directory: str | None = None,
    embeddings: Any = None,
) -> Any:
    """Create or load a Chroma vector store for a domain pack's documents."""
    from langchain_chroma import Chroma

    from enterprise_agent.documents.loader import DocumentIngestionPipeline

    data_dir_val = documents_dir or getattr(domain_config.domain, "data_dir", "data")
    dir_path = Path(data_dir_val).resolve() if data_dir_val else Path("data").resolve()
    collection_name = f"{domain_config.domain.id}_reference_docs"

    if persist_directory and Path(persist_directory).exists():
        return Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )

    pipeline = DocumentIngestionPipeline(domain_config.documents, domain_config.rbac)
    docs = pipeline.load_directory(dir_path) if dir_path.exists() else []

    if not docs:
        logger.debug("No documents found at %s; returning empty vector store.", dir_path)
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


def make_generic_retrieval_tool(
    store: Any,
    graph: Any = None,
    embeddings: Any = None,
    user_level: int = 1,
    doc_config: DocumentConfig | None = None,
    retrieve_docs_fn: Any = None,
    retrieve_graph_fn: Any = None,
    cache: Any | None = None,
    headroom_mgr: Any | None = None,
    session_id: str = "default",
) -> BaseTool:
    """Create a hybrid document retrieval tool with RBAC enforcement and optional compression."""
    cfg = doc_config or DocumentConfig()
    tool_name = cfg.tool_name or "search_documents"

    cat_descriptions = []
    for c in cfg.categories:
        cat_descriptions.append(f"{c.code} ({c.name})")
    cat_str = ", ".join(cat_descriptions) if cat_descriptions else "All categories"

    description = cfg.tool_description or (
        f"Search reference documents and policies.\n"
        f"Categories: {cat_str}.\n"
        f"Omit categories to search all available documents."
    )

    def _default_retrieve_docs(query: str, categories: list[str] | None = None) -> list[Any]:
        where_clause: dict[str, Any] = {"level_rank": {"$lte": user_level}}
        if categories:
            where_clause = {
                "$and": [
                    {"level_rank": {"$lte": user_level}},
                    {"category": {"$in": categories}},
                ]
            }
        if hasattr(store, "similarity_search_with_relevance_scores"):
            try:
                results = store.similarity_search_with_relevance_scores(
                    query, k=5, filter=where_clause
                )
                return [doc for doc, score in results]
            except Exception:
                pass
        if hasattr(store, "similarity_search"):
            return store.similarity_search(query, k=5, filter=where_clause)
        return []

    retrieve_fn = retrieve_docs_fn or _default_retrieve_docs

    @tool(tool_name, description=description)
    def retrieval_tool_func(query: str, categories: list[str] | None = None) -> str:
        """Search reference documents and policies."""
        docs = retrieve_fn(query, categories)
        doc_candidates = []
        for i, d in enumerate(docs):
            text = d.page_content if hasattr(d, "page_content") else str(d.get("text", d))
            meta = d.metadata if hasattr(d, "metadata") else (d if isinstance(d, dict) else {})
            doc_id = meta.get("doc_id", "DOC")
            chunk_id = meta.get("chunk_id", f"{doc_id}:c{i}:can")
            cat = meta.get("category", "")
            source = meta.get("corpus_source", "canonical")
            section = meta.get("section", "")
            doc_name = meta.get("doc_name") or meta.get("file_name", "")
            level_rank = meta.get("level_rank", 1)

            doc_candidates.append({
                "text": text,
                "metadata": meta,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "category": cat,
                "corpus_source": source,
                "section": section,
                "doc_name": doc_name,
                "level_rank": level_rank,
            })

        from enterprise_agent.retrieval.reranker import rerank_documents
        reranked = rerank_documents(query=query, documents=doc_candidates, top_n=5)

        doc_payloads = []
        references = []
        for i, r in enumerate(reranked):
            meta = r.get("metadata", {})
            doc_id = r.get("doc_id") or meta.get("doc_id", "DOC")
            chunk_id = r.get("chunk_id") or meta.get("chunk_id", f"{doc_id}:c{i}:can")
            cat = r.get("category") or meta.get("category", "")
            source = r.get("corpus_source") or meta.get("corpus_source", "canonical")
            section = r.get("section") or meta.get("section", "")
            text = r.get("text") or meta.get("text", "")
            doc_name = r.get("doc_name") or meta.get("doc_name") or meta.get("file_name", "")
            rerank_score = r.get("rerank_score")

            ref: dict[str, Any] = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "section": section,
                "chunk_id": chunk_id,
                "category": cat,
                "corpus_source": source,
            }
            if rerank_score is not None:
                ref["rerank_score"] = rerank_score
            references.append(ref)

            doc_payload: dict[str, Any] = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "section": section,
                "chunk_id": chunk_id,
                "category": cat,
                "corpus_source": source,
                "text": text,
                "content": text,  # backwards-compatibility alias
                "metadata": meta,
            }
            if rerank_score is not None:
                doc_payload["rerank_score"] = rerank_score
            doc_payloads.append(doc_payload)

        payload: dict[str, Any] = {
            "query": query,
            "categories": categories or [],
            "results": doc_payloads,
            "references": references,
        }

        if graph is not None and embeddings is not None and retrieve_graph_fn is not None:
            graph_ev = retrieve_graph_fn(graph, embeddings, query, user_level=user_level, cache=cache)
            if graph_ev:
                payload["graph_evidence"] = graph_ev

        payload_str = json.dumps(payload, default=str)
        if headroom_mgr is not None:
            return headroom_mgr.compress_tool_output(tool_name, payload_str, session_id=session_id)
        return payload_str

    retrieval_tool_func.description = description
    return retrieval_tool_func
