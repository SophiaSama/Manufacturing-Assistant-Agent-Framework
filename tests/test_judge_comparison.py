"""Comparison test and benchmark between Jev Judge and classical LLM Judge."""

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

import pytest
from nga.evaluation.judge import evaluate_with_jev, make_jev_judge, make_llm_judge


@pytest.mark.skipif(not os.getenv("TYPESAFE_API_KEY"), reason="Requires TYPESAFE_API_KEY")
def test_jev_judge_exact_match():
    golden = "105 Nm ± 5% (99.75–110.25 Nm) per SOP-OPR-101"
    candidate = "The target torque for Station 144 is 105 Nm ± 5% (99.75–110.25 Nm) according to SOP-OPR-101."
    
    jev_judge = make_jev_judge()
    score = jev_judge(candidate, golden)
    assert score >= 4, f"Expected high score for exact match, got {score}"


@pytest.mark.skipif(not os.getenv("TYPESAFE_API_KEY"), reason="Requires TYPESAFE_API_KEY")
def test_jev_judge_hallucinated_values():
    golden = "105 Nm ± 5% (99.75–110.25 Nm) per SOP-OPR-101"
    candidate = "The target torque is 175 Nm ± 15% per SOP-ENG-999."
    
    jev_judge = make_jev_judge()
    score = jev_judge(candidate, golden)
    assert score <= 1, f"Expected near-zero score for severe hallucination, got {score}"


@pytest.mark.skipif(not os.getenv("TYPESAFE_API_KEY"), reason="Requires TYPESAFE_API_KEY")
def test_jev_rich_evaluation():
    golden = "Star tightening sequence: 1 -> 3 -> 5 -> 2 -> 4"
    candidate = "The lug nuts should be tightened following the star pattern 1-3-5-2-4."
    
    details = evaluate_with_jev(candidate, golden)
    assert details["score_int"] >= 4
    assert details["is_faithful"] > 0.8
    assert details["covers_core_intent"] > 0.8
    assert details["latency_ms"] < 2000
