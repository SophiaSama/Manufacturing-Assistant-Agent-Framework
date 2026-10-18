"""Unit tests for Side-by-Side comparison runner and report generator (side_by_side.py)."""

from __future__ import annotations

import json

from nga.evaluation.deepeval_metrics import DeepEvalCaseResult
from nga.evaluation.judge import JevEvaluationResult
from nga.evaluation.side_by_side import (
    CriterionAgreement,
    SideBySideSummary,
    compute_criterion_agreement,
    format_side_by_side_markdown,
    perturb_manufacturing_answer,
    run_side_by_side_comparison,
    save_side_by_side_report,
)
from tests.unit.test_deepeval_metrics import MockDeterministicSystemOneModel


def test_perturb_manufacturing_answer_corrupts_number():
    orig = "The torque requirement is 105 Nm per SOP-OPR-101."
    corrupted = perturb_manufacturing_answer(orig)
    assert corrupted != orig
    assert "105" not in corrupted


def test_compute_criterion_agreement():
    legacy = JevEvaluationResult(
        score_int=5,
        score_continuous=4.95,
        confidence=0.95,
        probabilities={"5": 0.95},
        is_faithful=0.98,
        specs_accurate=0.99,
        completeness=0.92,
        covers_core_intent=0.95,
        failure_mode="none",
        failure_mode_probabilities={"none": 0.98},
        latency_ms=45.0,
    )
    deepeval_res = DeepEvalCaseResult(
        question_id="Q-001",
        category="sop",
        eval_mode="system_one",
        system_one_model="jev-1.13.0",
        faithfulness=0.95,
        answer_relevancy=0.92,
        contextual_precision=0.90,
        contextual_recall=0.91,
        manufacturing_jev_score=0.96,
        specs_accurate=0.98,
        completeness=0.95,
        is_faithful=0.97,
        rubric_score_normalized=0.98,
        rubric_score_0_to_5=5,
        failure_mode="none",
        min_confidence=0.92,
        overall_score=0.96,
        passed=True,
    )

    agree = compute_criterion_agreement(legacy, deepeval_res)
    assert isinstance(agree, CriterionAgreement)
    assert agree.is_faithful_agreement is True
    assert agree.specs_accurate_agreement is True
    assert agree.completeness_agreement is True
    assert agree.rubric_within_1_agreement is True
    assert agree.overall_pass_agreement is True
    assert abs(agree.score_shift) < 0.10


def test_run_side_by_side_comparison_with_mock_model(tmp_path):
    mock_model = MockDeterministicSystemOneModel(
        model_name="jev-1.13.0",
        noul_prob=0.95,
        score_level=5,
        failure_mode="none",
    )

    def mock_legacy_evaluator(cand: str, gold: str) -> JevEvaluationResult:
        is_perfect = "105 Nm" in cand
        score = 5 if is_perfect else 1
        return JevEvaluationResult(
            score_int=score,
            score_continuous=float(score),
            confidence=0.92,
            probabilities={str(score): 0.92},
            is_faithful=0.95 if is_perfect else 0.10,
            specs_accurate=0.95 if is_perfect else 0.10,
            completeness=0.90 if is_perfect else 0.20,
            covers_core_intent=0.90 if is_perfect else 0.20,
            failure_mode="none" if is_perfect else "inaccurate_specs",
            failure_mode_probabilities={"none": 0.90},
            latency_ms=35.0,
        )

    sample_questions = [
        {
            "id": "SAMPLE-1",
            "category": "sop_lookup",
            "tier": "L1",
            "question": "What is the torque?",
            "expected_answer": "105 Nm",
            "source_docs": ["SOP-OPR-101"],
        }
    ]

    sample_conflict = [
        {
            "id": "CONFLICT-TEST",
            "name": "Torque discrepancy",
            "golden": "Conflict between 105 Nm and 108 Nm.",
            "answers": {
                "conflict_flagged": "Conflict between 105 Nm and 108 Nm.",
                "blind_variant": "108 Nm.",
            },
        }
    ]

    summary = run_side_by_side_comparison(
        questions=sample_questions,
        conflict_scenarios=sample_conflict,
        eval_mode="system_one",
        system_one_model=mock_model,
        legacy_evaluator=mock_legacy_evaluator,
        include_perturbed_controls=True,
    )

    assert isinstance(summary, SideBySideSummary)
    assert summary.total_cases == 4  # 1 positive + 1 perturbed + 2 conflict answers
    assert summary.positive_cases == 1
    assert summary.perturbed_cases == 1
    assert summary.contradiction_cases == 2
    assert summary.system_one_model == "jev-1.13.0"
    assert summary.eval_mode == "system_one"

    # Test report formatting and saving
    md = format_side_by_side_markdown(summary)
    assert "# Side-by-Side Evaluation Report" in md
    assert "jev-1.13.0" in md

    md_path, json_path = save_side_by_side_report(summary, output_dir=tmp_path)
    assert md_path.exists()
    assert json_path.exists()

    saved_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved_data["total_cases"] == 4
    assert len(saved_data["cases"]) == 4
