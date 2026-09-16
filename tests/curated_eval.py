#!/usr/bin/env python3
"""Run a curated real-model eval and save the official report.

Usage:
    EMBEDDING_MODEL=qwen/qwen3-embedding-4b python -m nga.curated_eval

Produces reports/eval/<label>.md + .json via the standard pipeline.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

logging.basicConfig(level=logging.INFO)

# Curated subset spanning all categories (ids from eval-questions/questions.json)
CURATED_IDS = ["R1", "R12", "M1", "M3", "S1", "S5", "E1", "E6", "SQL2", "SQL10"]


def main() -> None:
    from nga.config import Settings
    from nga.evaluation.eval_runner import (
        load_eval_questions,
        run_evaluation_suite,
    )
    from nga.graph.orchestrator import build_orchestrator
    from nga.ingestion.build_graph import load_graph
    from nga.ingestion.build_vector_store import open_vector_store
    from nga.memory.checkpointer import build_checkpointer
    from nga.memory.decision_log import init_decision_log
    from nga.providers.factory import make_embeddings
    from nga.tools.tool_factory import make_retrieval_tool, make_sql_tool

    settings = Settings.from_env()
    init_decision_log(settings.app_state_db_path)

    embeddings = make_embeddings(settings)
    store = open_vector_store(settings, embeddings=embeddings, profile="main")
    graph_data = load_graph(settings.graph_store_dir)
    sql_tool = make_sql_tool(settings.nga_db_path)
    retrieval_tool = make_retrieval_tool(
        store, graph=graph_data, embeddings=embeddings, user_level=4
    )
    checkpointer = build_checkpointer(settings.app_state_db_path)

    def _auto_approver(recommendation_summary: str, **kwargs) -> tuple[str, str | None]:
        """Automated eval: approve recommendations (no interactive console)."""
        return "approved", "auto-approved (eval run)"

    graph = build_orchestrator(
        settings, checkpointer, sql_tool, retrieval_tool, approver=_auto_approver
    )

    questions = load_eval_questions(limit=10**9)
    curated = [q for q in questions if q.get("id") in CURATED_IDS]
    print(f"Running {len(curated)} curated questions with {settings.openrouter_model}")

    summary, results, md_path, json_path = run_evaluation_suite(
        agent_graph=graph,
        questions=curated,
        run_label=f"curated_real_{len(curated)}q",
        role="manager",
        cache_mode="cold",
    )
    print(f"\nReport: {md_path}\nJSON:   {json_path}")
    print(f"Pass rate: {summary['pass_rate']*100:.0f}%  "
          f"avg score: {summary['avg_score']:.3f}  "
          f"avg latency: {summary['avg_latency_s']:.1f}s")
    for cat in summary.get("by_category", []):
        print(f"  {cat['category']:<10} {cat['passed']}/{cat['total']} "
              f"({cat['pass_rate']*100:.0f}%) score={cat['avg_score']:.2f}")


if __name__ == "__main__":
    main()
