"""LangChain @tool wrappers for the NGA SQL and retrieval tools."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from nga.tools.retrieval_tool import (
    build_retrieval_payload,
    normalize_categories,
    retrieve_documents,
    retrieve_graph_evidence,
)
from nga.tools.sql_tool import SqlValidationError, run_query


def make_sql_tool(db_path: str):
    """Create a read-only SQL tool over nga.db."""

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
        try:
            result = run_query(db_path, sql)
        except SqlValidationError as e:
            return f"Query not allowed: {e}"
        return json.dumps(result, default=str)

    return query_nga_database


def make_retrieval_tool(store, graph=None, embeddings=None, user_level: int = 1):
    """Create a hybrid document retrieval tool with RBAC enforcement.

    When a knowledge graph is provided, results are enriched with graph
    entities, relations, and community summaries.

    user_level: caller's max level_rank (1=operator, 2=technician,
                3=engineer, 4=manager).
    """

    def _run(query: str, categories: list[str]) -> dict:
        if graph is not None and embeddings is not None:
            docs = retrieve_documents(
                store, query, category=categories, k=5, user_level=user_level
            )
            graph_ev = retrieve_graph_evidence(
                graph, embeddings, query, user_level=user_level
            )
            return build_retrieval_payload(query, categories, docs, graph_ev)
        docs = retrieve_documents(store, query, category=categories, k=5,
                                  user_level=user_level)
        return build_retrieval_payload(query, categories, docs)

    @tool
    def search_sop_documents(
        query: str, categories: list[str] | None = None
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
        return json.dumps(payload, default=str)

    return search_sop_documents
