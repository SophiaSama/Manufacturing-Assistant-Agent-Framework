"""Tests for Automated Hard Release Criteria and Regression Safeguards."""

import pytest

from nga.evaluation.release_gate import evaluate_release_gate


@pytest.fixture
def passing_baseline_and_candidate():
    baseline = {
        "run_label": "baseline_vector_pure",
        "summary": {
            "pass_rate": 0.70,
            "avg_score": 0.72,
            "avg_latency_s": 4.0,
            "by_category": [
                {"category": "multi-hop", "pass_rate": 0.50},
                {"category": "stress", "pass_rate": 0.40},
            ],
        },
        "results": [
            {"id": "R1", "passed": True},
            {"id": "M1", "passed": True},
            {"id": "M6", "passed": True},
            {"id": "STR4", "passed": True},
        ],
    }

    candidate = {
        "run_label": "candidate_hybrid_graphrag",
        "summary": {
            "pass_rate": 0.88,
            "avg_score": 0.89,
            "avg_latency_s": 6.5,
            "by_category": [
                {"category": "multi-hop", "pass_rate": 0.85},
                {"category": "stress", "pass_rate": 0.80},
            ],
        },
        "results": [
            {"id": "R1", "passed": True},
            {"id": "M1", "passed": True},
            {"id": "M6", "passed": True},
            {"id": "STR4", "passed": True},
        ],
    }
    return baseline, candidate


def test_release_gate_approved(passing_baseline_and_candidate):
    baseline, candidate = passing_baseline_and_candidate
    res = evaluate_release_gate(baseline, candidate)
    assert res.status == "APPROVED"
    assert res.passed_checks == res.total_checks
    assert len(res.regressions) == 0
    assert not res.rollback_recommended


def test_release_gate_rejected_on_regression(passing_baseline_and_candidate):
    baseline, candidate = passing_baseline_and_candidate
    # Introduce a regression: question R1 was True in baseline, now False in candidate
    candidate["results"] = [
        {"id": "R1", "passed": False},
        {"id": "M1", "passed": True},
        {"id": "M6", "passed": True},
        {"id": "STR4", "passed": True},
    ]

    res = evaluate_release_gate(baseline, candidate)
    assert res.status == "REJECTED"
    assert "R1" in res.regressions
    assert res.rollback_recommended


def test_release_gate_rejected_on_safety_failure(passing_baseline_and_candidate):
    baseline, candidate = passing_baseline_and_candidate
    # Introduce a Class A safety defect failure (M6)
    candidate["results"] = [
        {"id": "R1", "passed": True},
        {"id": "M1", "passed": True},
        {"id": "M6", "passed": False},  # Class A safety recall failure
        {"id": "STR4", "passed": True},
    ]

    res = evaluate_release_gate(baseline, candidate)
    assert res.status == "REJECTED"
    assert any("Class A" in reason for reason in res.reasons)
    assert res.rollback_recommended


def test_release_gate_evaluates_tcer_target(passing_baseline_and_candidate):
    baseline, candidate = passing_baseline_and_candidate
    candidate["token_summary"] = {"avg_tcer": 0.38}
    res = evaluate_release_gate(baseline, candidate)
    assert res.status == "APPROVED"
    assert "token_efficiency_tcer" in res.criteria_results
    assert res.criteria_results["token_efficiency_tcer"]["passed"] is True


def test_release_gate_rejects_on_tcer_exceeded(passing_baseline_and_candidate):
    baseline, candidate = passing_baseline_and_candidate
    candidate["token_summary"] = {"avg_tcer": 0.65}
    res = evaluate_release_gate(baseline, candidate)
    assert res.status == "REJECTED"
    assert "token_efficiency_tcer" in res.criteria_results
    assert res.criteria_results["token_efficiency_tcer"]["passed"] is False
    assert any("TCER" in r for r in res.reasons)
