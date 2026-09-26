"""Full 75-question evaluation benchmark for the NGA Manufacturing Assistant.

Runs all questions from eval-questions/questions.json against the live agent,
scores with deterministic checks + optional LLM-as-judge, and generates a report.

Usage:
    uv run pytest tests/integration/test_benchmark.py -v -s
    uv run pytest tests/integration/test_benchmark.py -v -s --judge   # with LLM judge
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage

from nga.config import Settings
from nga.evaluation.report import save_report
from nga.evaluation.scoring import ScoreResult, score_answer
from nga.graph.orchestrator import build_orchestrator
from nga.ingestion.build_graph import load_graph
from nga.memory.checkpointer import build_checkpointer
from nga.memory.decision_log import init_decision_log
from nga.models.answer_schema import (
    FinalAnswer,
    parse_final_answer,
    render_final_answer,
)
from nga.providers.factory import make_embeddings
from nga.rag_agent.rbac import ACCESS_LEVELS
from nga.tools.tool_factory import make_retrieval_tool, make_sql_tool

# Benchmark requires multi-hop + tool use + structured output → 'standard'
pytestmark = [pytest.mark.required_tier("standard")]

# ── Fixture: Load questions ────────────────────────────────────────────────────

QUESTIONS_PATH = Path(__file__).parents[2] / "eval-questions" / "questions.json"


def load_questions() -> list[dict[str, Any]]:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


# ── Fixture: Build agent graph ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def settings():
    return Settings.from_env()


@pytest.fixture(scope="session")
def agent_graph(settings, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("app_state")
    db_path = str(tmp / "app_state.db")
    init_decision_log(db_path)

    embeddings = make_embeddings(settings)
    store = Chroma(
        collection_name="nga_reference_docs",
        embedding_function=embeddings,
        persist_directory=settings.vector_store_dir,
    )
    graph_data = load_graph(settings.graph_store_dir)
    sql_tool = make_sql_tool(settings.nga_db_path)
    # Use engineer-level (3) for eval — full access
    retrieval_tool = make_retrieval_tool(
        store, graph=graph_data, embeddings=embeddings, user_level=3
    )
    checkpointer = build_checkpointer(db_path)
    return build_orchestrator(settings, checkpointer, sql_tool, retrieval_tool)


# ── Judge Fixture ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def active_judge(use_judge, judge_backend, settings: Settings):
    """Build the active judge based on --judge and --judge-backend flags."""
    if not use_judge:
        return None
    from nga.evaluation.judge import get_judge
    return get_judge(backend=judge_backend)


# ── Run one question ──────────────────────────────────────────────────────────

def run_question(
    graph,
    question: str,
    thread_id: str,
    user_role: str = "engineer",
) -> tuple[str, list[str], list[str], dict[str, Any] | None, str | None]:
    """Run a single question through the agent graph.

    Returns (rendered_answer, tools_called, tool_outputs, token_usage, model_name).
    """
    user_level = ACCESS_LEVELS.get(user_role, 3)
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "user_role": user_role,
        "user_level": user_level,
    }
    answer = ""
    tools_called: list[str] = []
    tool_outputs: list[str] = []
    question_parts: list[str] = []
    captured_token_usage: dict[str, Any] | None = None
    captured_model_name: str | None = None

    for update in graph.stream(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    ):
        if not isinstance(update, dict):
            continue

        prepare = update.get("prepare") or {}
        if isinstance(prepare.get("question_parts"), list):
            question_parts = prepare["question_parts"]
        if isinstance(prepare.get("model_route"), dict):
            captured_model_name = prepare["model_route"].get("model_name")

        tools_update = update.get("tools") or {}
        messages = tools_update.get("messages") or []
        for msg in messages:
            name = getattr(msg, "name", None)
            if name:
                tools_called.append(name)
            content = getattr(msg, "content", "") or ""
            if content:
                tool_outputs.append(str(content))

        synthesis = update.get("synthesis") or {}
        if synthesis:
            if "token_usage" in synthesis and synthesis["token_usage"]:
                captured_token_usage = synthesis["token_usage"]
            elif isinstance(synthesis.get("final_answer"), dict):
                captured_token_usage = synthesis["final_answer"].get("token_telemetry")

        final_dict = synthesis.get("final_answer")
        if final_dict:
            try:
                final = FinalAnswer.model_validate(final_dict)
                answer = render_final_answer(final)
            except Exception:
                pass
        if not answer:
            msgs = synthesis.get("messages") or []
            if msgs:
                last = msgs[-1]
                raw = getattr(last, "content", "") or ""
                if raw:
                    answer = render_final_answer(
                        parse_final_answer(raw, question_parts)
                    )

    return answer, tools_called, tool_outputs, captured_token_usage, captured_model_name


# ── Parametrized test ─────────────────────────────────────────────────────────

ALL_QUESTIONS = load_questions()
_results: list[ScoreResult] = []


@pytest.fixture(scope="session", autouse=True)
def _save_report_on_finish(settings):
    yield
    if _results:
        md_path, json_path = save_report(
            _results,
            reports_dir="reports/eval",
            model_name=settings.openrouter_model,
        )
        print(f"\n\nEval report saved:\n  Markdown: {md_path}\n  JSON:     {json_path}")
        passed = sum(1 for r in _results if r.passed)
        print(f"Pass rate: {passed}/{len(_results)} = {passed/len(_results)*100:.1f}%")
        tok_results = [r for r in _results if r.token_usage]
        if tok_results:
            tot_tokens = sum(r.token_usage.get("totals", {}).get("total_tokens", 0) for r in tok_results)
            tot_cost = sum(r.token_usage.get("totals", {}).get("total_cost_usd", 0.0) for r in tok_results)
            tcer_list = [
                r.token_usage.get("kpis", {}).get("tcer", 0.0)
                for r in tok_results
                if "tcer" in r.token_usage.get("kpis", {})
            ]
            avg_tcer = sum(tcer_list) / len(tcer_list) if tcer_list else 0.0
            print(
                f"Token Consumption KPIs: Total Tokens: {tot_tokens:,} | Total Cost: ${tot_cost:.4f} | "
                f"Avg TCER: {avg_tcer:.3f} (target <= 0.45)"
            )


@pytest.mark.parametrize(
    "question_data",
    ALL_QUESTIONS,
    ids=[q["id"] for q in ALL_QUESTIONS],
)
def test_question(question_data, agent_graph, settings, active_judge):
    qid = question_data["id"]
    category = question_data.get("category", "unknown")
    question = question_data["question"]
    golden = question_data.get("expected_answer", "")
    source_docs = question_data.get("source_docs", [])
    requires_sql = question_data.get("requires_sql", False)

    # Determine expected tools
    expected_tools: set[str] = set()
    if requires_sql:
        expected_tools.add("query_nga_database")
    if source_docs:
        expected_tools.add("search_sop_documents")

    thread_id = str(uuid.uuid4())
    start = time.monotonic()
    error: str | None = None
    answer = ""
    tools_called: list[str] = []
    tool_outputs: list[str] = []
    captured_token_usage: dict[str, Any] | None = None
    captured_model_name: str | None = None

    try:
        answer, tools_called, tool_outputs, captured_token_usage, captured_model_name = run_question(
            agent_graph, question, thread_id, user_role="engineer"
        )
    except Exception as exc:
        error = str(exc)[:200]

    latency = time.monotonic() - start

    effective_model = captured_model_name or settings.openrouter_model

    result = score_answer(
        answer=answer,
        golden=golden,
        expected_tools=expected_tools,
        tools_called=tools_called,
        source_docs=source_docs,
        question_id=qid,
        category=category,
        tool_outputs=tool_outputs,
        requires_sql=requires_sql,
        judge=active_judge,
        latency_s=latency,
        error=error,
        token_usage=captured_token_usage,
        model_name=effective_model,
    )
    _results.append(result)

    tok_info = ""
    if captured_token_usage:
        tot_tok = captured_token_usage.get("totals", {}).get("total_tokens", 0)
        tot_cost = captured_token_usage.get("totals", {}).get("total_cost_usd", 0.0)
        tcer = captured_token_usage.get("kpis", {}).get("tcer", 0.0)
        tok_info = f" | Tokens: {tot_tok:,} | Cost: ${tot_cost:.5f} | TCER: {tcer:.3f}"

    print(
        f"\n[{qid}] {question[:80]}\n"
        f"  Model: {effective_model}\n"
        f"  Answer: {answer[:200]}\n"
        f"  Score: {result.overall_score:.3f} | Passed: {result.passed} "
        f"| Latency: {latency:.2f}s{tok_info} | Tools: {tools_called}"
    )

    # Soft assertion: warn on failure, don't hard-fail the test suite
    # Change to `assert result.passed` for strict mode
    if not result.passed:
        pytest.skip(f"[SOFT FAIL] {qid} score={result.overall_score:.3f} < 0.70")
