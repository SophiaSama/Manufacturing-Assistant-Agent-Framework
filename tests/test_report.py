"""Tests for report generation — covers the generate_markdown_report path
that previously crashed with an undefined `err` variable."""

from nga.evaluation.report import generate_markdown_report, build_summary
from nga.evaluation.scoring import ScoreResult


def _make_result(**overrides) -> ScoreResult:
    defaults = dict(
        question_id="TST-001",
        category="STR1",
        checks={"has_answer": True},
        deterministic_score=0.85,
        judge_score=5,
        overall_score=0.85,
        passed=True,
        tools_called=["search_documents"],
        latency_s=1.2,
        error=None,
        tier="L1",
    )
    defaults.update(overrides)
    return ScoreResult(**defaults)


def test_generate_markdown_report_no_crash():
    """Regression test: generating markdown must not raise NameError on `err`."""
    results = [
        _make_result(question_id="Q1", passed=True, error=None, tier="L1"),
        _make_result(question_id="Q2", passed=False, error="LLM timeout", tier="L3",
                     overall_score=0.3),
    ]
    md = generate_markdown_report(results, run_label="regression-test")
    assert "Q1" in md
    assert "Q2" in md
    # The error column should render the actual error string, not blow up
    assert "LLM timeout" in md
    # A passing result with no error should render a dash
    assert "—" in md


def test_build_summary_includes_tiers():
    results = [
        _make_result(question_id=f"Q{i}", tier=tier)
        for i, tier in enumerate(["L1", "L1", "L2", "L3", "L4"])
    ]
    summary = build_summary(results)
    assert summary["total"] == 5
    tier_labels = {t["tier"] for t in summary.get("by_tier", [])}
    assert "L1" in tier_labels
