"""Hybrid deterministic + LLM-as-judge scoring for NGA evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

PASS_THRESHOLD = 0.70


@dataclass(frozen=True)
class ScoreResult:
    question_id: str
    category: str
    checks: dict[str, bool]
    deterministic_score: float
    judge_score: int | None
    overall_score: float
    passed: bool
    tools_called: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    error: str | None = None
    cache_stats: dict | None = None  # per-layer hit/miss/ms when cache hot
    tier: str = "L2"


def _parsed_tool_payloads(tool_outputs: list[str] | None) -> list[dict]:
    payloads: list[dict] = []
    for output in tool_outputs or []:
        try:
            p = json.loads(output)
            if isinstance(p, dict):
                payloads.append(p)
        except (TypeError, ValueError):
            pass
    return payloads


def _has_sql_tool_output(tool_outputs: list[str] | None) -> bool:
    for p in _parsed_tool_payloads(tool_outputs):
        if "query" in p and "rows" in p:
            return True
    return False


def _has_document_tool_output(tool_outputs: list[str] | None) -> bool:
    for p in _parsed_tool_payloads(tool_outputs):
        refs = p.get("references")
        if isinstance(refs, list) and refs:
            return True
        results = p.get("results")
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict) and r.get("doc_id"):
                    return True
    return False


def _check_source_doc_cited(
    answer: str, source_docs: list[str]
) -> bool:
    """Check if at least one expected source doc ID appears in the answer."""
    if not source_docs:
        return True   # no expectation set
    answer_lower = answer.lower()
    return any(doc.lower() in answer_lower for doc in source_docs)


def deterministic_checks(
    answer: str,
    expected_tools: set[str],
    tools_called: list[str],
    source_docs: list[str],
    tool_outputs: list[str] | None = None,
    requires_sql: bool = False,
) -> dict[str, bool]:
    lowered = answer.lower()
    checks: dict[str, bool] = {}

    # 1. Expected tools were called
    checks["expected_tools_covered"] = expected_tools.issubset(set(tools_called))

    # 2. SQL tool produced evidence (only if question requires SQL)
    if requires_sql:
        checks["has_sql_source"] = (
            "sql:" in lowered
            or "nga" in lowered
            or _has_sql_tool_output(tool_outputs)
        )

    # 3. Document tool produced evidence (only if source_docs specified)
    if source_docs:
        checks["has_doc_source"] = (
            _has_document_tool_output(tool_outputs)
            or _check_source_doc_cited(answer, source_docs)
        )

    # 4. Answer is not empty
    checks["non_empty_answer"] = len(answer.strip()) > 20

    return checks


def score_answer(
    answer: str,
    golden: str,
    expected_tools: set[str],
    tools_called: list[str],
    source_docs: list[str],
    question_id: str,
    category: str,
    tool_outputs: list[str] | None = None,
    requires_sql: bool = False,
    judge: Callable[[str, str], int] | None = None,
    latency_s: float = 0.0,
    error: str | None = None,
    cache_stats: dict | None = None,
    tier: str = "L2",
) -> ScoreResult:
    checks = deterministic_checks(
        answer, expected_tools, tools_called, source_docs,
        tool_outputs, requires_sql,
    )
    deterministic_score = sum(1 for v in checks.values() if v) / max(len(checks), 1)

    judge_score: int | None = None
    if judge is not None and not error:
        try:
            judge_score = int(judge(answer, golden))
            overall = (deterministic_score + judge_score / 5) / 2
        except Exception:
            overall = deterministic_score
    else:
        overall = deterministic_score

    overall_score = round(overall, 4)
    return ScoreResult(
        question_id=question_id,
        category=category,
        checks=checks,
        deterministic_score=round(deterministic_score, 4),
        judge_score=judge_score,
        overall_score=overall_score,
        passed=overall_score >= PASS_THRESHOLD,
        tools_called=tools_called,
        latency_s=latency_s,
        error=error,
        cache_stats=cache_stats,
        tier=tier,
    )
