"""Evaluation runner, benchmark executor, and evaluation comparison engine."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import HumanMessage, ToolMessage

from nga.evaluation.report import build_summary, save_report
from nga.evaluation.scoring import ScoreResult, score_answer
from nga.models.answer_schema import FinalAnswer, parse_final_answer, render_final_answer

logger = logging.getLogger("nga.evaluation.eval_runner")


def load_eval_questions(
    questions_path: str = "eval-questions/questions.json",
    category: str | None = None,
    limit: int | None = None,
    smoke_test: bool = False,
) -> list[dict[str, Any]]:
    """Load and optionally filter evaluation questions."""
    path = Path(questions_path)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    questions: list[dict[str, Any]] = data.get("questions", [])

    if category and category.lower() != "all":
        questions = [q for q in questions if q.get("category", "").lower() == category.lower()]

    if smoke_test:
        # Pick 1 representative question from each available category
        seen_cats = set()
        smoke_questions = []
        for q in questions:
            cat = q.get("category", "general")
            if cat not in seen_cats:
                smoke_questions.append(q)
                seen_cats.add(cat)
            if len(smoke_questions) >= 5:
                break
        return smoke_questions

    if limit and limit > 0:
        questions = questions[:limit]

    return questions


def _extract_tools_and_outputs(messages: list[Any]) -> tuple[list[str], list[str]]:
    tools_called: list[str] = []
    tool_outputs: list[str] = []
    for msg in messages:
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                t_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                if t_name and t_name not in tools_called:
                    tools_called.append(t_name)
        if isinstance(msg, ToolMessage):
            tool_outputs.append(str(msg.content))
    return tools_called, tool_outputs


def _render_agent_response(synthesis_update: dict, question_parts: list[str]) -> str:
    payload = synthesis_update.get("final_answer")
    if isinstance(payload, dict):
        try:
            return render_final_answer(FinalAnswer.model_validate(payload))
        except Exception:
            pass
    messages = synthesis_update.get("messages", [])
    if messages:
        last = messages[-1]
        raw = last.content if hasattr(last, "content") else str(last)
        if isinstance(raw, str):
            return render_final_answer(parse_final_answer(raw, question_parts))
        return str(raw)
    return ""


def run_single_eval_question(
    agent_graph: Any,
    question_data: dict[str, Any],
    role: str = "manager",
    judge: Callable[[str, str], int] | None = None,
) -> ScoreResult:
    """Execute a single evaluation question on the agent graph and score it."""
    q_id = question_data.get("id", "UNKNOWN")
    category = question_data.get("category", "general")
    question_text = question_data.get("question", "")
    expected_answer = question_data.get("expected_answer", "")
    source_docs = question_data.get("source_docs", [])
    requires_sql = bool(question_data.get("requires_sql", False))

    expected_tools: set[str] = set()
    if requires_sql:
        expected_tools.add("query_nga_database")
    if source_docs:
        expected_tools.add("search_sop_documents")

    thread_id = str(uuid.uuid4())
    initial_state = {
        "messages": [HumanMessage(content=question_text)],
        "user_role": role,
        "user_level": 4,  # full clearance for eval benchmarking
    }

    t0 = time.perf_counter()
    answer_text = ""
    error_msg: str | None = None
    question_parts: list[str] = []
    collected_messages: list[Any] = []

    try:
        for update in agent_graph.stream(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="updates",
        ):
            if not isinstance(update, dict):
                continue
            prepare_update = update.get("prepare")
            if prepare_update and isinstance(prepare_update.get("question_parts"), list):
                question_parts = [
                    p for p in prepare_update["question_parts"] if isinstance(p, str) and p.strip()
                ]

            agent_update = update.get("agent")
            if agent_update and "messages" in agent_update:
                collected_messages.extend(agent_update["messages"])

            tools_update = update.get("tools")
            if tools_update and "messages" in tools_update:
                collected_messages.extend(tools_update["messages"])

            synthesis_update = update.get("synthesis")
            if synthesis_update:
                answer_text = _render_agent_response(synthesis_update, question_parts)
                if "messages" in synthesis_update:
                    collected_messages.extend(synthesis_update["messages"])

        if not answer_text and collected_messages:
            last = collected_messages[-1]
            answer_text = last.content if hasattr(last, "content") else str(last)

    except Exception as exc:
        logger.exception("Eval question %s failed with exception", q_id)
        error_msg = str(exc)
        answer_text = ""

    latency_s = round(time.perf_counter() - t0, 3)
    tools_called, tool_outputs = _extract_tools_and_outputs(collected_messages)

    return score_answer(
        answer=answer_text,
        golden=expected_answer,
        expected_tools=expected_tools,
        tools_called=tools_called,
        source_docs=source_docs,
        question_id=q_id,
        category=category,
        tool_outputs=tool_outputs,
        requires_sql=requires_sql,
        judge=judge,
        latency_s=latency_s,
        error=error_msg,
    )


def run_evaluation_suite(
    agent_graph: Any,
    questions: list[dict[str, Any]],
    run_label: str = "",
    reports_dir: str = "reports/eval",
    role: str = "manager",
    judge: Callable[[str, str], int] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[ScoreResult], Path, Path]:
    """Run full evaluation suite, stream progress, compute summary, and save reports."""
    label = run_label or f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    total = len(questions)
    results: list[ScoreResult] = []

    logger.info("Starting evaluation suite '%s' with %d questions", label, total)

    for idx, q in enumerate(questions, start=1):
        result = run_single_eval_question(agent_graph, q, role=role, judge=judge)
        results.append(result)

        if progress_callback:
            progress_callback({
                "type": "progress",
                "current": idx,
                "total": total,
                "percent": round((idx / total) * 100, 1),
                "question_id": result.question_id,
                "category": result.category,
                "passed": result.passed,
                "score": result.overall_score,
                "latency_s": result.latency_s,
                "error": result.error,
            })

    md_path, json_path = save_report(results, reports_dir=reports_dir, run_label=label)
    summary = build_summary(results)
    summary["run_label"] = label
    summary["md_report_path"] = str(md_path)
    summary["json_report_path"] = str(json_path)

    if progress_callback:
        progress_callback({
            "type": "completed",
            "run_label": label,
            "summary": summary,
        })

    return summary, results, md_path, json_path


def list_evaluation_reports(reports_dir: str = "reports/eval") -> list[dict[str, Any]]:
    """List all saved evaluation reports with summary metadata."""
    out_dir = Path(reports_dir)
    if not out_dir.exists():
        return []

    reports = []
    for json_file in sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            summary = data.get("summary", {})
            reports.append({
                "run_label": data.get("run_label") or json_file.stem,
                "timestamp": datetime.fromtimestamp(
                    json_file.stat().st_mtime, timezone.utc
                ).isoformat(),
                "file_path": str(json_file),
                "total": summary.get("total", len(data.get("results", []))),
                "passed": summary.get("passed", 0),
                "pass_rate": summary.get("pass_rate", 0.0),
                "avg_score": summary.get("avg_score", 0.0),
                "avg_latency_s": summary.get("avg_latency_s", 0.0),
                "categories": summary.get("by_category", []),
            })
        except Exception as exc:
            logger.warning("Failed to parse eval report %s: %s", json_file, exc)
    return reports


def get_evaluation_report(run_label: str, reports_dir: str = "reports/eval") -> dict[str, Any] | None:
    """Retrieve full evaluation report JSON by run_label."""
    json_path = Path(reports_dir) / f"{run_label}.json"
    if not json_path.exists():
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Error reading report %s: %s", json_path, exc)
        return None


def compare_evaluation_runs(
    report_a: dict[str, Any],
    report_b: dict[str, Any],
) -> dict[str, Any]:
    """Compare two evaluation runs (Baseline A vs Candidate B) and compute comprehensive deltas."""
    label_a = report_a.get("run_label", "Run A (Baseline)")
    label_b = report_b.get("run_label", "Run B (Candidate)")

    sum_a = report_a.get("summary", {})
    sum_b = report_b.get("summary", {})

    pass_rate_a = float(sum_a.get("pass_rate", 0.0))
    pass_rate_b = float(sum_b.get("pass_rate", 0.0))
    score_a = float(sum_a.get("avg_score", 0.0))
    score_b = float(sum_b.get("avg_score", 0.0))
    lat_a = float(sum_a.get("avg_latency_s", 0.0))
    lat_b = float(sum_b.get("avg_latency_s", 0.0))

    summary_delta = {
        "pass_rate": {
            "a": pass_rate_a,
            "b": pass_rate_b,
            "delta": round(pass_rate_b - pass_rate_a, 3),
            "pct_delta": round((pass_rate_b - pass_rate_a) * 100, 1),
        },
        "avg_score": {
            "a": score_a,
            "b": score_b,
            "delta": round(score_b - score_a, 3),
        },
        "avg_latency_s": {
            "a": lat_a,
            "b": lat_b,
            "delta": round(lat_b - lat_a, 2),
        },
        "total_questions": {
            "a": sum_a.get("total", 0),
            "b": sum_b.get("total", 0),
        },
        "passed_questions": {
            "a": sum_a.get("passed", 0),
            "b": sum_b.get("passed", 0),
            "delta": int(sum_b.get("passed", 0)) - int(sum_a.get("passed", 0)),
        },
    }

    # Category comparisons
    cats_a = {c["category"]: c for c in sum_a.get("by_category", [])}
    cats_b = {c["category"]: c for c in sum_b.get("by_category", [])}
    all_cat_names = sorted(set(cats_a.keys()) | set(cats_b.keys()))

    category_deltas = []
    for cat in all_cat_names:
        ca = cats_a.get(cat, {"total": 0, "passed": 0, "pass_rate": 0.0, "avg_score": 0.0, "avg_latency_s": 0.0})
        cb = cats_b.get(cat, {"total": 0, "passed": 0, "pass_rate": 0.0, "avg_score": 0.0, "avg_latency_s": 0.0})
        category_deltas.append({
            "category": cat,
            "total_a": ca.get("total", 0),
            "total_b": cb.get("total", 0),
            "pass_rate_a": ca.get("pass_rate", 0.0),
            "pass_rate_b": cb.get("pass_rate", 0.0),
            "pass_rate_delta": round(cb.get("pass_rate", 0.0) - ca.get("pass_rate", 0.0), 3),
            "score_a": ca.get("avg_score", 0.0),
            "score_b": cb.get("avg_score", 0.0),
            "score_delta": round(cb.get("avg_score", 0.0) - ca.get("avg_score", 0.0), 3),
            "latency_a": ca.get("avg_latency_s", 0.0),
            "latency_b": cb.get("avg_latency_s", 0.0),
            "latency_delta": round(cb.get("avg_latency_s", 0.0) - ca.get("avg_latency_s", 0.0), 2),
        })

    # Question-by-question transition matrix
    res_a = {r["id"]: r for r in report_a.get("results", [])}
    res_b = {r["id"]: r for r in report_b.get("results", [])}
    common_ids = sorted(set(res_a.keys()) | set(res_b.keys()))

    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    maintained_pass: list[dict[str, Any]] = []
    maintained_fail: list[dict[str, Any]] = []
    all_question_comparisons: list[dict[str, Any]] = []

    for qid in common_ids:
        ra = res_a.get(qid)
        rb = res_b.get(qid)

        pass_a = bool(ra and ra.get("passed"))
        pass_b = bool(rb and rb.get("passed"))
        score_a_val = float(ra.get("overall_score", 0.0)) if ra else 0.0
        score_b_val = float(rb.get("overall_score", 0.0)) if rb else 0.0
        lat_a_val = float(ra.get("latency_s", 0.0)) if ra else 0.0
        lat_b_val = float(rb.get("latency_s", 0.0)) if rb else 0.0
        cat = (rb or ra or {}).get("category", "general")

        status = "unknown"
        if pass_a and not pass_b:
            status = "regression"
        elif not pass_a and pass_b:
            status = "improvement"
        elif pass_a and pass_b:
            status = "maintained_pass"
        elif not pass_a and not pass_b:
            status = "maintained_fail"

        item = {
            "id": qid,
            "category": cat,
            "status": status,
            "passed_a": pass_a,
            "passed_b": pass_b,
            "score_a": score_a_val,
            "score_b": score_b_val,
            "score_delta": round(score_b_val - score_a_val, 3),
            "latency_a": lat_a_val,
            "latency_b": lat_b_val,
            "latency_delta": round(lat_b_val - lat_a_val, 2),
            "tools_a": (ra or {}).get("tools_called", []),
            "tools_b": (rb or {}).get("tools_called", []),
            "checks_a": (ra or {}).get("checks", {}),
            "checks_b": (rb or {}).get("checks", {}),
            "error_a": (ra or {}).get("error"),
            "error_b": (rb or {}).get("error"),
        }

        all_question_comparisons.append(item)
        if status == "regression":
            regressions.append(item)
        elif status == "improvement":
            improvements.append(item)
        elif status == "maintained_pass":
            maintained_pass.append(item)
        elif status == "maintained_fail":
            maintained_fail.append(item)

    return {
        "run_a": {
            "label": label_a,
            "summary": sum_a,
        },
        "run_b": {
            "label": label_b,
            "summary": sum_b,
        },
        "summary_delta": summary_delta,
        "category_deltas": category_deltas,
        "counts": {
            "regressions": len(regressions),
            "improvements": len(improvements),
            "maintained_pass": len(maintained_pass),
            "maintained_fail": len(maintained_fail),
            "total_compared": len(common_ids),
        },
        "regressions": regressions,
        "improvements": improvements,
        "maintained_pass": maintained_pass,
        "maintained_fail": maintained_fail,
        "all_questions": all_question_comparisons,
    }
