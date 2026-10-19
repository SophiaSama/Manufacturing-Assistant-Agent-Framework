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
    a_evaluate_test_case_with_deepeval,
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


@pytest.mark.asyncio
async def test_safe_openrouter_model_lifecycle():
    from unittest.mock import AsyncMock, MagicMock, patch

    from nga.evaluation.deepeval_metrics import SafeOpenRouterModel

    model = SafeOpenRouterModel(
        model="openrouter/test-model",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
    )
    mock_client = AsyncMock()
    mock_resp = AsyncMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="test response"))]
    mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=15)
    mock_client.chat.completions.create.return_value = mock_resp

    with patch.object(model, "load_model", return_value=mock_client):
        res, cost = await model._a_generate("test prompt")
        assert res == "test response"
        assert cost is not None
        assert mock_client.close.await_count == 1
        assert getattr(mock_client._client, "is_closed", False) is True


@pytest.mark.asyncio
async def test_safe_openrouter_model_exception_teardown():
    from unittest.mock import AsyncMock, patch

    from nga.evaluation.deepeval_metrics import SafeOpenRouterModel

    model = SafeOpenRouterModel(
        model="openrouter/test-model",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
    )
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("OpenRouter 500 error")

    with patch.object(model, "load_model", return_value=mock_client):
        with pytest.raises(RuntimeError, match="OpenRouter 500 error"):
            await model._a_generate("test prompt")
        assert mock_client.close.await_count == 1
        assert getattr(mock_client._client, "is_closed", False) is True


@pytest.mark.asyncio
async def test_safe_openrouter_model_raw_response_lifecycle():
    from unittest.mock import AsyncMock, MagicMock, patch

    from nga.evaluation.deepeval_metrics import SafeOpenRouterModel

    model = SafeOpenRouterModel(
        model="openrouter/test-model",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
    )
    mock_client = AsyncMock()
    mock_resp = AsyncMock()
    mock_resp.usage = MagicMock(prompt_tokens=8, completion_tokens=12)
    mock_client.chat.completions.create.return_value = mock_resp

    with patch.object(model, "load_model", return_value=mock_client):
        raw_res, cost = await model._a_generate_raw_response("test prompt")
        assert raw_res is mock_resp
        assert mock_client.close.await_count == 1
        assert getattr(mock_client._client, "is_closed", False) is True


@pytest.mark.asyncio
async def test_safe_openrouter_model_with_real_httpx_client():
    """Verify that a real httpx.AsyncClient has is_closed == True without AttributeError."""
    from unittest.mock import AsyncMock, MagicMock, patch

    import httpx

    from nga.evaluation.deepeval_metrics import SafeOpenRouterModel

    model = SafeOpenRouterModel(
        model="openrouter/test-model",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
    )
    real_httpx_client = httpx.AsyncClient()
    mock_client = AsyncMock()
    mock_client._client = real_httpx_client
    mock_resp = AsyncMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="real client response"))]
    mock_resp.usage = MagicMock(prompt_tokens=5, completion_tokens=10)
    mock_client.chat.completions.create.return_value = mock_resp

    with patch.object(model, "load_model", return_value=mock_client):
        res, cost = await model._a_generate("test prompt")
        assert res == "real client response"
        assert mock_client._client.is_closed is True


def test_close_active_chat_models_with_real_httpx_client():
    """Verify close_active_chat_models cleans up real httpx clients and clears registry."""
    from unittest.mock import MagicMock

    import httpx

    from nga.providers.factory import (
        close_active_chat_models,
        get_active_chat_models,
        register_chat_model,
    )

    real_httpx_client = httpx.AsyncClient()
    mock_model = MagicMock()
    mock_async_openai = MagicMock()
    mock_async_openai._client = real_httpx_client
    mock_model.root_async_client = mock_async_openai

    register_chat_model(mock_model)
    assert mock_model in get_active_chat_models()

    close_active_chat_models()
    assert real_httpx_client.is_closed is True
    assert get_active_chat_models() == []


def test_close_active_chat_models_with_ollama_client():
    """Verify close_active_chat_models cleans up ChatOllama _async_client."""
    from unittest.mock import AsyncMock, MagicMock

    import httpx

    from nga.providers.factory import (
        close_active_chat_models,
        get_active_chat_models,
        register_chat_model,
    )

    real_httpx_client = httpx.AsyncClient()
    mock_model = MagicMock(spec=["_async_client"])
    mock_ollama_async = MagicMock()
    mock_ollama_async._client = real_httpx_client
    mock_ollama_async.close = AsyncMock()
    mock_model._async_client = mock_ollama_async

    register_chat_model(mock_model)
    assert mock_model in get_active_chat_models()

    close_active_chat_models()
    assert real_httpx_client.is_closed is True
    assert mock_ollama_async.close.await_count == 1
    assert get_active_chat_models() == []


@pytest.mark.asyncio
async def test_close_active_chat_models_when_loop_is_running():
    """Verify close_active_chat_models works safely even inside a running event loop."""
    from unittest.mock import MagicMock

    import httpx

    from nga.providers.factory import (
        close_active_chat_models,
        get_active_chat_models,
        register_chat_model,
    )

    real_httpx_client = httpx.AsyncClient()
    mock_model = MagicMock()
    mock_async_openai = MagicMock()
    mock_async_openai._client = real_httpx_client
    mock_model.root_async_client = mock_async_openai

    register_chat_model(mock_model)
    assert mock_model in get_active_chat_models()

    # Calling synchronous close_active_chat_models from inside an async test
    close_active_chat_models()
    assert real_httpx_client.is_closed is True
    assert get_active_chat_models() == []


@pytest.mark.asyncio
async def test_aclose_active_chat_models():
    """Verify aclose_active_chat_models asynchronously closes all registered models."""
    from unittest.mock import MagicMock

    import httpx

    from nga.providers.factory import (
        aclose_active_chat_models,
        get_active_chat_models,
        register_chat_model,
    )

    real_httpx_client = httpx.AsyncClient()
    mock_model = MagicMock()
    mock_async_openai = MagicMock()
    mock_async_openai._client = real_httpx_client
    mock_model.root_async_client = mock_async_openai

    register_chat_model(mock_model)
    assert mock_model in get_active_chat_models()

    await aclose_active_chat_models()
    assert real_httpx_client.is_closed is True
    assert get_active_chat_models() == []


def test_close_active_chat_models_with_direct_httpx_client():
    """Verify close_active_chat_models cleans up direct httpx.AsyncClient attributes."""
    from unittest.mock import MagicMock

    import httpx

    from nga.providers.factory import (
        close_active_chat_models,
        get_active_chat_models,
        register_chat_model,
    )

    real_httpx_client = httpx.AsyncClient()
    mock_model = MagicMock(spec=["async_client"])
    mock_model.async_client = real_httpx_client

    register_chat_model(mock_model)
    assert mock_model in get_active_chat_models()

    close_active_chat_models()
    assert real_httpx_client.is_closed is True
    assert get_active_chat_models() == []


def test_conftest_cleanup_hooks_execute_cleanly():
    """Verify cleanup_async_resources and pytest_sessionfinish execute without errors."""
    from unittest.mock import MagicMock

    import httpx

    from nga.memory.headroom_memory import NgaEpisodicMemory
    from nga.providers.factory import get_active_chat_models, register_chat_model
    from tests.conftest import cleanup_async_resources, pytest_sessionfinish

    real_httpx_client = httpx.AsyncClient()
    mock_model = MagicMock()
    mock_async = MagicMock()
    mock_async._client = real_httpx_client
    mock_model.root_async_client = mock_async
    register_chat_model(mock_model)

    mem = NgaEpisodicMemory(db_path="/tmp/test_conftest_cleanup.db")

    # Run the generator fixture teardown via its underlying function
    fn = getattr(cleanup_async_resources, "__wrapped__", cleanup_async_resources)
    gen = fn()
    next(gen)  # run up to yield
    try:
        next(gen)
    except StopIteration:
        pass

    assert real_httpx_client.is_closed is True
    assert get_active_chat_models() == []
    assert mem._closed is True

    # Test pytest_sessionfinish
    pytest_sessionfinish(None, 0)


def test_evaluate_test_case_with_deepeval_concurrent_metrics():
    """Verify concurrent metric fan-out delivers identical results and passes cleanly."""
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
        additional_metadata={"question_id": "TEST-CONCURRENT-001", "category": "sop_lookup"},
    )

    # 1. Concurrent execution (default)
    res_concurrent = evaluate_test_case_with_deepeval(
        test_case=tc,
        eval_mode="system_one",
        system_one_model=mock_model,
        pass_threshold=0.70,
        concurrent_metrics=True,
    )

    # 2. Sequential execution
    res_sequential = evaluate_test_case_with_deepeval(
        test_case=tc,
        eval_mode="system_one",
        system_one_model=mock_model,
        pass_threshold=0.70,
        concurrent_metrics=False,
    )

    assert res_concurrent.passed is True
    assert res_sequential.passed is True
    assert res_concurrent.overall_score == res_sequential.overall_score
    assert res_concurrent.rubric_score_0_to_5 == res_sequential.rubric_score_0_to_5
    assert res_concurrent.faithfulness == res_sequential.faithfulness
    assert res_concurrent.answer_relevancy == res_sequential.answer_relevancy


@pytest.mark.asyncio
async def test_a_evaluate_test_case_with_deepeval():
    """Verify direct async a_evaluate_test_case_with_deepeval invocation."""
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
        additional_metadata={"question_id": "TEST-ASYNC-001", "category": "sop_lookup"},
    )

    res = await a_evaluate_test_case_with_deepeval(
        test_case=tc,
        eval_mode="system_one",
        system_one_model=mock_model,
        pass_threshold=0.70,
        concurrent_metrics=True,
    )
    assert res.passed is True
    assert res.overall_score >= 0.70
    assert res.question_id == "TEST-ASYNC-001"



