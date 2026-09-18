"""Generic LangChain tool factory for SQL and hybrid retrieval."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from enterprise_agent.config.domain_config import DocumentConfig
from enterprise_agent.database.engine import DatabaseAdapter, SqlValidationError


def make_generic_sql_tool(
    db_adapter: DatabaseAdapter,
    tool_name: str = "query_database",
    tool_description: str | None = None,
    cache: Any | None = None,
):
    """Create a LangChain read-only SQL tool using a generic DatabaseAdapter."""
    schema_desc = db_adapter.describe_schema()
    description = tool_description or (
        f"Run a read-only SELECT query against the relational database.\n"
        f"Available tables and columns:\n{schema_desc}\n\n"
        f"Argument `sql` must be a single SELECT statement."
    )

    @tool(tool_name)
    def sql_tool_func(sql: str) -> str:
        def _run() -> str:
            try:
                result = db_adapter.run_query(sql)
            except SqlValidationError as e:
                return f"Query not allowed: {e}"
            except Exception as e:
                return f"Database error: {e}"
            return json.dumps(result, default=str)

        if cache is not None:
            from nga.cache.layers import cached_run_query
            return cached_run_query(
                _run,
                sql=sql,
                db_path=getattr(db_adapter, "db_path", "db"),
                cache=cache,
            )

        return _run()

    # Override description dynamically
    sql_tool_func.description = description
    return sql_tool_func


def make_generic_retrieval_tool(
    store: Any,
    graph: Any = None,
    embeddings: Any = None,
    user_level: int = 1,
    doc_config: DocumentConfig | None = None,
    retrieve_docs_fn: Any = None,
    retrieve_graph_fn: Any = None,
    cache: Any | None = None,
):
    """Create a hybrid document retrieval tool with RBAC enforcement."""
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
        results = store.similarity_search_with_relevance_scores(
            query, k=5, filter=where_clause
        )
        return [doc for doc, score in results]

    retrieve_fn = retrieve_docs_fn or _default_retrieve_docs

    @tool(tool_name)
    def retrieval_tool_func(query: str, categories: list[str] | None = None) -> str:
        docs = retrieve_fn(query, categories)
        doc_payloads = []
        references = []
        for d in docs:
            doc_id = d.metadata.get("doc_id", "DOC")
            category = d.metadata.get("category", "")
            references.append(f"{doc_id} [{category}]")
            doc_payloads.append({
                "content": d.page_content,
                "metadata": d.metadata,
            })

        payload: dict[str, Any] = {
            "query": query,
            "categories": categories or [],
            "results": doc_payloads,
            "references": sorted(set(references)),
        }

        if graph is not None and embeddings is not None and retrieve_graph_fn is not None:
            graph_ev = retrieve_graph_fn(graph, embeddings, query, user_level=user_level, cache=cache)
            if graph_ev:
                payload["graph_evidence"] = graph_ev

        return json.dumps(payload, default=str)

    retrieval_tool_func.description = description
    return retrieval_tool_func
