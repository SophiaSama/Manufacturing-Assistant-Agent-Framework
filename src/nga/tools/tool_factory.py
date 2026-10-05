"""LangChain @tool wrappers for the NGA SQL and retrieval tools."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from nga.tools.retrieval_tool import (
    build_retrieval_payload,
    normalize_categories,
    retrieve_documents,
    retrieve_graph_evidence,
)
from nga.tools.sql_tool import SqlValidationError, run_query


def make_sql_tool(
    db_path: str,
    cache: Any | None = None,
    headroom_mgr: Any | None = None,
    session_id: str = "default",
):
    """Create a read-only SQL tool over nga.db (optionally cached, L2, and compressed)."""

    @tool
    def query_nga_database(sql: str) -> str:
        """Run a read-only SELECT query against the NGA production database.

        The nga.db contains 17 tables: lines, stations, machines, shifts,
        vehicles, build_records, torque_readings, quality_checks, defects,
        nc_records, work_orders, maintenance_logs, escalations, field_actions,
        suppliers, parts, training_records.

        The `sql` argument must be a single SELECT statement.
        Returns JSON: {query, rows, row_count, tables}.
        """
        from nga.cache.layers import cached_run_query

        def _run() -> str:
            try:
                result = run_query(db_path, sql)
            except SqlValidationError as e:
                return f"Query not allowed: {e}"
            return json.dumps(result, default=str)

        # Invalid SQL is rejected BEFORE cache lookup — the cache only ever
        # stores validated read-only query results (design §3.3).
        try:
            from nga.tools.sql_tool import validate_select_only
            validate_select_only(sql)
        except SqlValidationError as e:
            return f"Query not allowed: {e}"

        cached = cached_run_query(_run, sql=sql, db_path=db_path, cache=cache)
        if headroom_mgr is not None:
            return headroom_mgr.compress_tool_output(
                "query_nga_database", cached, session_id=session_id
            )
        return cached

    return query_nga_database


def make_retrieval_tool(
    store,
    graph=None,
    embeddings=None,
    user_level: int = 1,
    cache: Any | None = None,
    headroom_mgr: Any | None = None,
    session_id: str = "default",
):
    """Create a hybrid document retrieval tool with RBAC enforcement and optional compression.

    When a knowledge graph is provided, results are enriched with graph
    entities, relations, and community summaries.

    user_level: caller's max level_rank (1=operator, 2=technician,
                3=engineer, 4=manager).
    cache: optional NgaCache; when None the active cache_scope is used
           (eval hot mode) or caching is disabled (cold).
    headroom_mgr: optional HeadroomManager for context compression.
    """

    def _run(query: str, categories: list[str]) -> dict:
        if graph is not None and embeddings is not None:
            docs = retrieve_documents(
                store, query, category=categories, k=5, user_level=user_level,
                cache=cache,
            )
            graph_ev = retrieve_graph_evidence(
                graph, embeddings, query, user_level=user_level, cache=cache,
            )
            return build_retrieval_payload(query, categories, docs, graph_ev)
        docs = retrieve_documents(store, query, category=categories, k=5,
                                  user_level=user_level, cache=cache)
        return build_retrieval_payload(query, categories, docs)

    @tool
    def search_sop_documents(
        query: str, categories: list[str] | str | None = None
    ) -> str:
        """Search NGA reference documents: SOPs, machine specs, failure analysis,
        recall criteria, work orders, supplier quality, and training records.

        Categories: OPR (operator SOPs), TEC (technician SOPs + machine specs),
        FA (failure analysis + escalation), QCR (recall criteria), WO (work orders),
        SQ (supplier quality), TR (training records).
        Omit categories to search all.

        Returns JSON: {query, categories, results, references, graph_evidence?}
        """
        cats = normalize_categories(categories)
        payload = _run(query, cats)
        raw_json = json.dumps(payload, default=str)
        if headroom_mgr is not None:
            return headroom_mgr.compress_tool_output(
                "search_sop_documents", raw_json, session_id=session_id
            )
        return raw_json

    return search_sop_documents


def make_headroom_retrieve_tool(
    headroom_mgr: Any,
    session_id: str = "default",
):
    """Create tool allowing the agent to fetch verbatim uncompressed evidence by hash."""

    @tool
    def retrieve_uncompressed_evidence(content_hash: str) -> str:
        """Retrieve original verbatim uncompressed text using a Headroom content hash.

        Use this tool when you need exact numerical tolerances, strict safety compliance clauses,
        or full procedure steps for Class A decisions that may be referenced with [hash=...].
        """
        raw = headroom_mgr.store.get(session_id, content_hash)
        if raw is not None:
            return raw
        return f"Error: No uncompressed content found for hash '{content_hash}' in session '{session_id}'."

    return retrieve_uncompressed_evidence
