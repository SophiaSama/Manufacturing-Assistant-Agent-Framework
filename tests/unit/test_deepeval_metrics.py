"""Unit tests for DeepEval RAG metrics, JevEval metrics, and execution engine (deepeval_metrics.py)."""

from __future__ import annotations

import pytest
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    JevEval,
)
from deepeval.models import DeepEvalBaseSystemOneModel
from deepeval.models.system_one.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneAnswers,
)
from deepeval.test_case import LLMTestCase

from nga.evaluation.deepeval_metrics import (
    PINNED_JEV_MODEL,
    DeepEvalCaseResult,
    evaluate_test_case_with_deepeval,
    make_built_in_rag_metrics,
    make_contradiction_jev_eval_metric,
    make_manufacturing_jev_eval_metric,
    resolve_deepeval_llm_model,
)


class MockDeterministicSystemOneModel(DeepEvalBaseSystemOneModel):
    """Deterministic mock of TypeSafe System One model for unit testing without API keys."""

    def __init__(
        self,
        model_name: str = "jev-1.13.0",
        noul_prob: float = 0.95,
        score_level: int = 5,
        failure_mode: str = "none",
        fail: bool = False,
    ):
        super().__init__(model_name)
        self._name = model_name
        self.noul_prob = noul_prob
        self.score_level = score_level
        self.failure_mode = failure_mode
        self.fail = fail

    def load_model(self):
        return None

    def get_model_name(self) -> str:
        return self._name

    def decide(self, state, questions):
        if self.fail:
            raise RuntimeError("TypeSafe 401 Unauthorized: Invalid API key")
        nouls = {}
        choices = {}
        scores = {}
        items = questions.items() if isinstance(questions, dict) else enumerate(questions)
        for key, q in items:
            q_key = str(key)
            if isinstance(q, NoulQuestion):
                nouls[q_key] = NoulAnswer(probability=self.noul_prob)
            elif isinstance(q, ChoiceQuestion):
                opts = getattr(q, "options", {}) or {}
                fm = self.failure_mode if self.failure_mode in opts else (list(opts.keys())[0] if opts else "none")
                choices[q_key] = ChoiceAnswer(
                    choice=fm,
                    probabilities={opt: (1.0 if opt == fm else 0.0) for opt in opts},
                    confidence=0.92,
                )
            elif isinstance(q, ScoreQuestion):
                levels = getattr(q, "levels", []) or []
                lvl = min(self.score_level, max(0, len(levels) - 1))
                scores[q_key] = ScoreAnswer(
                    score=float(lvl),
                    probabilities={i: (1.0 if i == lvl else 0.0) for i in range(max(1, len(levels)))},
                    confidence=0.94,
                )
        return SystemOneAnswers(nouls=nouls, choices=choices, scores=scores), 0.93

    async def a_decide(self, state, questions):
        return self.decide(state, questions)


def test_pinned_jev_model_constant():
    assert PINNED_JEV_MODEL == "jev-1.13.0"


def test_resolve_deepeval_llm_model_system_one():
    # Under system_one, no LLM is created or contacted
    assert resolve_deepeval_llm_model(eval_mode="system_one") is None


def test_make_built_in_rag_metrics_system_one():
    metrics = make_built_in_rag_metrics(
        eval_mode="system_one",
        system_one_model=PINNED_JEV_MODEL,
    )
    assert set(metrics.keys()) == {
        "faithfulness",
        "answer_relevancy",
        "contextual_precision",
        "contextual_recall",
    }
    assert isinstance(metrics["faithfulness"], FaithfulnessMetric)
    assert isinstance(metrics["answer_relevancy"], AnswerRelevancyMetric)
    assert isinstance(metrics["contextual_precision"], ContextualPrecisionMetric)
    assert isinstance(metrics["contextual_recall"], ContextualRecallMetric)

    for m in metrics.values():
        assert m.eval_mode == "system_one"
        model_name = (
            m.system_one_model.get_model_name()
            if hasattr(m.system_one_model, "get_model_name")
            else str(m.system_one_model)
        )
        assert "jev-1.13.0" in model_name


def test_make_manufacturing_jev_eval_metric():
    metric = make_manufacturing_jev_eval_metric(system_one_model=PINNED_JEV_MODEL)
    assert isinstance(metric, JevEval)
    assert metric.name == "Manufacturing Quality"
    assert len(metric.questions) == 5
    assert metric.threshold == 0.70


def test_make_contradiction_jev_eval_metric():
    metric = make_contradiction_jev_eval_metric(system_one_model=PINNED_JEV_MODEL)
    assert isinstance(metric, JevEval)
    assert metric.name == "Contradiction Handling"
    assert len(metric.questions) == 2


def test_evaluate_test_case_with_deepeval_mock_pass():
    mock_model = MockDeterministicSystemOneModel(
        model_name="jev-1.13.0",
        noul_prob=0.95,
        score_level=5,
        failure_mode="none",
    )
    tc = LLMTestCase(
        input="What is the torque specification at Station 144?",
        actual_output="105 Nm per SOP-OPR-101",
        expected_output="105 Nm per SOP-OPR-101",
        retrieval_context=["[SOP-OPR-101] Station 144 torque is 105 Nm."],
        additional_metadata={"question_id": "TEST-001", "category": "sop_lookup"},
    )

    result = evaluate_test_case_with_deepeval(
        test_case=tc,
        eval_mode="system_one",
        system_one_model=mock_model,
        pass_threshold=0.70,
    )

    assert isinstance(result, DeepEvalCaseResult)
    assert result.question_id == "TEST-001"
    assert result.system_one_model == "jev-1.13.0"
    assert result.eval_mode == "system_one"
    assert result.passed is True
    assert result.rubric_score_0_to_5 == 5
    assert result.is_faithful >= 0.90
    assert result.specs_accurate >= 0.90
    assert result.failure_mode == "none"
    assert result.min_confidence is not None


def test_evaluate_test_case_safety_gating():
    # Model returns severe hallucination / fake document
    mock_model = MockDeterministicSystemOneModel(
        model_name="jev-1.13.0",
        noul_prob=0.10,
        score_level=4,
        failure_mode="hallucinated_reference",
    )
    tc = LLMTestCase(
        input="What is the torque specification?",
        actual_output="999 Nm per FAKE-SOP",
        expected_output="105 Nm per SOP-OPR-101",
        retrieval_context=["[SOP-OPR-101] 105 Nm"],
    )

    result = evaluate_test_case_with_deepeval(
        test_case=tc,
        eval_mode="system_one",
        system_one_model=mock_model,
        pass_threshold=0.70,
    )

    # Manufacturing safety policy gates hallucinated answer down to <= 0.20 and rubric <= 1
    assert result.passed is False
    assert result.manufacturing_jev_score <= 0.20
    assert result.rubric_score_0_to_5 <= 1


def test_evaluate_test_case_api_error_handling():
    failing_model = MockDeterministicSystemOneModel(fail=True)
    tc = LLMTestCase(
        input="test",
        actual_output="test",
        expected_output="test",
        retrieval_context=["test"],
    )

    with pytest.raises(Exception) as exc_info:
        evaluate_test_case_with_deepeval(
            test_case=tc,
            eval_mode="system_one",
            system_one_model=failing_model,
            raise_on_api_error=True,
        )
    assert "401" in str(exc_info.value) or "Unauthorized" in str(exc_info.value)
