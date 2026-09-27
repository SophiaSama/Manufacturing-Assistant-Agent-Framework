"""Comparison test and benchmark between Jev Judge and classical LLM Judge."""

from __future__ import annotations

import os
from typing import Any

import pytest
from unittest.mock import MagicMock

from dotenv import load_dotenv

from nga.evaluation.judge import (
    JevJudgeAPIError,
    JudgeAPIError,
    LLMJudgeAPIError,
    evaluate_with_jev,
    is_api_error,
    make_jev_judge,
    make_llm_judge,
)

load_dotenv()


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
    assert details["specs_accurate"] > 0.8
    assert details["failure_mode"] == "none"
    assert details["latency_ms"] < 2000


def test_jev_parallel_questions_mock():
    """Unit test verifying the 5-question parallel fan-out structure without live API."""
    from typesafe_sdk import Choice, Noul, Score

    class MockAnswer:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockClient:
        def __init__(self):
            self.last_questions = None

        def system_one(self, state: Any, questions: dict[str, Any]):
            self.last_questions = questions
            # Ensure all 5 official questions are batched in the single call
            assert "score" in questions and isinstance(questions["score"], Score)
            assert "is_faithful" in questions and isinstance(questions["is_faithful"], Noul)
            assert "specs_accurate" in questions and isinstance(questions["specs_accurate"], Noul)
            assert "completeness" in questions and isinstance(questions["completeness"], Noul)
            assert "failure_mode" in questions and isinstance(questions["failure_mode"], Choice)

            class MockResponse:
                answers = {
                    "score": MockAnswer(score=4.8, confidence=0.95),
                    "is_faithful": MockAnswer(noul=0.98),
                    "specs_accurate": MockAnswer(noul=0.95),
                    "completeness": MockAnswer(noul=0.92),
                    "failure_mode": MockAnswer(choice="none", probabilities={"none": 0.95}),
                }
                model = "mock-jev"

            return MockResponse()

    client = MockClient()
    judge = make_jev_judge(client=client)
    score = judge("Candidate answer", "Golden answer")

    assert int(score) == 5
    assert hasattr(score, "diagnostics")
    assert score.diagnostics is not None
    assert score.diagnostics.specs_accurate == 0.95
    assert score.diagnostics.failure_mode == "none"
    assert score.diagnostics.is_faithful == 0.98
    assert score.diagnostics.gated_reason is None


def test_jev_safety_gating_mock():
    """Verify safety gating clamps scores when hallucinations or bad specs are detected."""
    class MockAnswer:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockHallucinationClient:
        def system_one(self, state: Any, questions: dict[str, Any]):
            class MockResponse:
                answers = {
                    "score": MockAnswer(score=5.0, confidence=0.9),
                    "is_faithful": MockAnswer(noul=0.15),  # Severe hallucination!
                    "specs_accurate": MockAnswer(noul=0.9),
                    "completeness": MockAnswer(noul=0.9),
                    "failure_mode": MockAnswer(choice="hallucinated_reference"),
                }
                model = "mock-jev"

            return MockResponse()

    client = MockHallucinationClient()
    judge = make_jev_judge(client=client, apply_safety_gating=True)
    score = judge("Fake SOP-999 cited", "Real golden")

    # Score must be clamped to 1 due to severe hallucination
    assert score == 1
    assert score.diagnostics.gated_reason is not None
    assert "severe hallucination" in score.diagnostics.gated_reason


def test_score_answer_captures_judge_diagnostics():
    """Verify scoring.py score_answer correctly captures Jev Judge diagnostics."""
    from nga.evaluation.judge import JevEvaluationResult, JevScore
    from nga.evaluation.scoring import score_answer

    diag = JevEvaluationResult(
        score_int=4,
        score_continuous=4.2,
        confidence=0.9,
        probabilities={},
        is_faithful=0.96,
        specs_accurate=0.94,
        completeness=0.90,
        covers_core_intent=0.90,
        failure_mode="none",
        failure_mode_probabilities={"none": 0.9},
        latency_ms=35.0,
        model="jev",
    )
    mock_score = JevScore(4, diagnostics=diag)

    def mock_judge(cand: str, gold: str):
        return mock_score

    result = score_answer(
        answer="Valid answer exceeding twenty characters for non empty check.",
        golden="Golden answer",
        expected_tools=set(),
        tools_called=[],
        source_docs=[],
        question_id="test_q1",
        category="general",
        judge=mock_judge,
    )

    assert result.judge_score == 4
    assert result.judge_diagnostics is not None
    assert result.judge_diagnostics["is_faithful"] == 0.96
    assert result.judge_diagnostics["specs_accurate"] == 0.94
    assert result.judge_diagnostics["failure_mode"] == "none"


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


def test_is_api_error_detection():
    """Verify is_api_error correctly identifies 400, 403, quota, and client errors."""
    # 403 key limit exceeded
    err_403 = Exception("Error code: 403 - {'error': {'message': 'Key limit exceeded (weekly limit).', 'code': 403}}")
    is_err, code, msg = is_api_error(err_403)
    assert is_err is True
    assert code == 403

    # 400 bad request
    err_400 = Exception("Error code: 400 - deepseek is not a valid model ID")
    is_err, code, msg = is_api_error(err_400)
    assert is_err is True
    assert code == 400

    # Custom object with status_code attribute
    class CustomHTTPError(Exception):
        status_code = 429

    err_429 = CustomHTTPError("Rate limit reached")
    is_err, code, msg = is_api_error(err_429)
    assert is_err is True
    assert code == 429

    # LLMJudgeAPIError instance
    llm_err = LLMJudgeAPIError("Permission Denied", status_code=403)
    is_err, code, msg = is_api_error(llm_err)
    assert is_err is True
    assert code == 403

    # Non-API errors (e.g. ValueError or parsing errors)
    val_err = ValueError("invalid literal for int() with base 10: 'Answer'")
    is_err, code, msg = is_api_error(val_err)
    assert is_err is False
    assert code is None


def test_make_llm_judge_raises_api_error():
    """Verify make_llm_judge raises LLMJudgeAPIError on 403 / API errors instead of returning 0."""
    class FailingLLM:
        def invoke(self, messages):
            raise Exception("Error code: 403 - Key limit exceeded (weekly limit)")

    judge = make_llm_judge(llm=FailingLLM(), raise_on_api_error=True)
    with pytest.raises(LLMJudgeAPIError) as exc_info:
        judge("Candidate answer", "Golden answer")

    assert exc_info.value.status_code == 403
    assert "403" in str(exc_info.value)


def test_make_llm_judge_returns_zero_on_malformed_text():
    """Verify make_llm_judge returns 0 when output is unparseable (quality failure, not API failure)."""
    class MalformedLLM:
        def invoke(self, messages):
            class Response:
                content = "I cannot provide a rating for this question."
            return Response()

    judge = make_llm_judge(llm=MalformedLLM(), raise_on_api_error=True)
    score = judge("Candidate answer", "Golden answer")
    assert score == 0


def test_full_suite_judge_comparison_skips_on_preflight_api_error():
    """Verify full_suite_judge_comparison skips execution and NEVER calls Jev when LLM judge fails with API error."""
    import pytest
    from scripts.full_suite_judge_comparison import main

    mock_llm_judge = MagicMock(side_effect=LLMJudgeAPIError("Key limit exceeded (weekly limit)", status_code=403))
    mock_jev_evaluator = MagicMock()

    with pytest.raises(pytest.skip.Exception) as exc_info:
        main(limit=2, llm_judge=mock_llm_judge, jev_evaluator=mock_jev_evaluator)

    assert "LLM judge unavailable" in str(exc_info.value)
    # Crucial guarantee: Jev evaluations MUST NOT be called!
    assert mock_jev_evaluator.call_count == 0


def test_full_suite_judge_comparison_aborts_mid_suite_on_api_error():
    """Verify full_suite_judge_comparison aborts immediately and does not call Jev when LLM judge fails during run."""
    import pytest
    from scripts.full_suite_judge_comparison import main

    calls = 0

    def mock_llm_judge(candidate: str, golden: str) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            # Preflight probe succeeds
            return 5
        # First question in loop fails with 403
        raise LLMJudgeAPIError("Error code: 403 - Key limit exceeded", status_code=403)

    mock_jev_evaluator = MagicMock()

    with pytest.raises(pytest.skip.Exception) as exc_info:
        main(limit=2, llm_judge=mock_llm_judge, jev_evaluator=mock_jev_evaluator)

    assert "LLM judge failed" in str(exc_info.value)
    # Exactly 1 call during initial preflight probe; ZERO calls for questions in the benchmark loop!
    assert mock_jev_evaluator.call_count == 1



def test_make_jev_judge_raises_api_error():
    """Verify make_jev_judge raises JevJudgeAPIError on 403 / API errors instead of returning 0."""
    class FailingTypeSafeClient:
        def system_one(self, state, questions):
            raise Exception("Error code: 403 - Forbidden: Invalid TypeSafe API key or quota exceeded")

    judge = make_jev_judge(client=FailingTypeSafeClient(), raise_on_api_error=True)
    with pytest.raises(JevJudgeAPIError) as exc_info:
        judge("Candidate answer", "Golden answer")

    assert exc_info.value.status_code == 403
    assert "403" in str(exc_info.value)


def test_full_suite_judge_comparison_skips_on_jev_preflight_api_error():
    """Verify full_suite_judge_comparison skips execution when Jev judge fails during preflight probe."""
    import pytest
    from scripts.full_suite_judge_comparison import main

    mock_llm_judge = MagicMock(return_value=5)
    mock_jev_evaluator = MagicMock(side_effect=JevJudgeAPIError("403 Forbidden: TypeSafe quota exceeded", status_code=403))

    with pytest.raises(pytest.skip.Exception) as exc_info:
        main(limit=2, llm_judge=mock_llm_judge, jev_evaluator=mock_jev_evaluator)

    assert "Jev judge unavailable" in str(exc_info.value)


def test_full_suite_judge_comparison_skips_when_api_keys_missing(monkeypatch):
    """Verify full_suite_judge_comparison skips when required API keys are missing from environment."""
    import pytest
    from scripts.full_suite_judge_comparison import main

    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(pytest.skip.Exception) as exc_info:
        main()

    assert "Missing required API key(s)" in str(exc_info.value)


def test_llm_judge_model_name_dynamic():
    """Verify make_llm_judge extracts model name dynamically from model or settings rather than hardcoded."""
    class DummyLLM:
        model_name = "deepseek/deepseek-v4.1-flash"

        def invoke(self, messages):
            class Response:
                content = "5"
            return Response()

    judge = make_llm_judge(llm=DummyLLM())
    assert getattr(judge, "model_name", None) == "deepseek/deepseek-v4.1-flash"



