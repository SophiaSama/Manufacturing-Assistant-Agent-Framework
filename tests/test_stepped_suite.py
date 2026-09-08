"""Tests for the 4-Tier Stepped Evaluation Test Suite."""

import pytest
from nga.evaluation.stepped_suite import (
    TIER_NAMES,
    TIER_TARGETS,
    compute_stepped_summary,
    load_stepped_questions,
)


def test_question_tier_completeness():
    questions = load_stepped_questions("eval-questions/questions.json")
    assert len(questions) == 85

    # Every question must have an unambiguous tier assigned
    for q in questions:
        assert "tier" in q, f"Question {q.get('id')} missing tier"
        assert q["tier"] in ["L1", "L2", "L3", "L4"], f"Invalid tier {q.get('tier')} on {q.get('id')}"
        assert "hop_count" in q, f"Missing hop_count on {q.get('id')}"
        assert q["hop_count"] in [1, 2, 3, 4]


def test_tier_filtering():
    l1_qs = load_stepped_questions(tier="L1")
    l2_qs = load_stepped_questions(tier="L2")
    l3_qs = load_stepped_questions(tier="L3")
    l4_qs = load_stepped_questions(tier="L4")

    assert len(l1_qs) > 0
    assert len(l2_qs) > 0
    assert len(l3_qs) > 0
    assert len(l4_qs) > 0
    assert len(l1_qs) + len(l2_qs) + len(l3_qs) + len(l4_qs) == 85


def test_stepped_summary_calculation():
    mock_results = [
        {"id": "R1", "tier": "L1", "passed": True, "overall_score": 1.0, "latency_s": 1.5},
        {"id": "R2", "tier": "L1", "passed": True, "overall_score": 0.95, "latency_s": 1.8},
        {"id": "M1", "tier": "L2", "passed": True, "overall_score": 0.90, "latency_s": 3.2},
        {"id": "M5", "tier": "L3", "passed": True, "overall_score": 0.85, "latency_s": 6.5},
        {"id": "STR1", "tier": "L4", "passed": False, "overall_score": 0.65, "latency_s": 11.2},
    ]

    summary = compute_stepped_summary(mock_results)
    assert len(summary["tiers"]) == 4
    assert "degradation_slope" in summary
    assert "pass_rates" in summary

    # L1 should have 100% pass rate in this mock
    assert summary["pass_rates"]["L1"] == 1.0
    # L4 should have 0% pass rate
    assert summary["pass_rates"]["L4"] == 0.0
    # Degradation slope = (1.0 - 0.0) / 3 ≈ 0.333
    assert summary["degradation_slope"] > 0.3
