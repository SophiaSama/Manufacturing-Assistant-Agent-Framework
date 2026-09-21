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


@pytest.mark.skipif(not os.getenv("TYPESAFE_API_KEY"), reason="Requires TYPESAFE_API_KEY")
def test_jev_judge_contradiction_properly_flagged():
    """Verify Jev rewards answers that detect and flag cross-document collisions."""
    golden = (
        "There is a document contradiction: SOP-OPR-101 specifies 105 Nm ± 5%, "
        "whereas variant document specifies 108 Nm. Both values exist and must be reconciled."
    )
    candidate = (
        "Discrepancy detected: canonical SOP-OPR-101 specifies 105 Nm, but the variant document "
        "lists 108 Nm. Engineering review is required to reconcile."
    )
    jev_judge = make_jev_judge()
    score = jev_judge(candidate, golden)
    assert score >= 4, f"Expected high score for correctly flagging contradiction, got {score}"


@pytest.mark.skipif(not os.getenv("TYPESAFE_API_KEY"), reason="Requires TYPESAFE_API_KEY")
def test_jev_judge_contradiction_uncritical_variant_penalized():
    """Verify Jev penalizes answers that blindly pick one side of a document collision."""
    golden = (
        "There is a document contradiction: SOP-OPR-101 specifies 105 Nm ± 5%, "
        "whereas variant document specifies 108 Nm. Both values exist and must be reconciled."
    )
    candidate = "The correct torque at Station 144 is 108 Nm per the variant specification."
    jev_judge = make_jev_judge()
    score = jev_judge(candidate, golden)
    assert score <= 2, f"Expected low score for failing to catch document conflict, got {score}"


@pytest.mark.skipif(not os.getenv("TYPESAFE_API_KEY"), reason="Requires TYPESAFE_API_KEY")
def test_jev_judge_contradiction_hallucinated_average_penalized():
    """Verify Jev strictly penalizes fabricated compromises between conflicting specs."""
    golden = (
        "Contradiction: Class A safety defects require Quality Director notification within "
        "1 hour per QCR-501, but variant notes 4 hours."
    )
    candidate = "Notification is required within 2.5 hours based on averaging both policies."
    jev_judge = make_jev_judge()
    score = jev_judge(candidate, golden)
    assert score <= 1, f"Expected near-zero score for hallucinated middle ground, got {score}"
