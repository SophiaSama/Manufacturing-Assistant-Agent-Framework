"""DeepEval RAG metrics and custom JevEval manufacturing criteria for NGA evaluation.

Pins `system_one_model="jev-1.13.0"` by default across all built-in RAG metrics
and custom `JevEval` metrics to ensure reproducible benchmark baselines.

Eval modes:
- `"system_one"` (default): 100% Jev-only execution in a single request per metric,
  zero LLM calls, zero output token billing, and deterministic reasons.
- `"hybrid"` (opt-in): LLM extracts claims/statements and writes narrative reasons
  while Jev decides every verdict in between. Note: because the LLM still runs in
  `hybrid` mode, latency and token cost are not Jev-only.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    JevEval,
)
from deepeval.metrics.jev_eval import Choice, Noul, Score
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel, OpenRouterModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

from nga.evaluation.judge import (
    MANUFACTURING_JUDGE_CRITERIA,
    JevJudgeAPIError,
    LLMJudgeAPIError,
    is_api_error,
)
from nga.tracing import attach_trace_metadata, attach_trace_tags, traceable_if_enabled

PINNED_JEV_MODEL = "jev-1.13.0"
VALID_EVAL_MODES = ("system_one", "hybrid", "llm")


class SafeOpenRouterModel(OpenRouterModel):
    """OpenRouterModel subclass that explicitly closes AsyncOpenAI clients after generation.

    Prevents `RuntimeError: Event loop is closed` during httpx.AsyncClient.aclose()
    when safe_asyncio_run closes its temporary event loop.
    """

    async def _a_generate(
        self,
        prompt: str,
        schema: Any = None,
        audio: Any = None,
    ) -> tuple[Any, float | None]:
        client = self.load_model(async_mode=True)
        try:
            messages = self._build_messages(prompt, audio)
            if schema:
                try:
                    completion = await client.chat.completions.create(
                        model=self.name,
                        messages=messages,
                        response_format=self._schema_response_format(schema),
                        temperature=self.temperature,
                        **self.generation_kwargs,
                    )
                    from deepeval.models.llms.utils import trim_and_load_json

                    json_output = trim_and_load_json(
                        completion.choices[0].message.content
                    )
                    cost = self.calculate_cost(
                        completion.usage.prompt_tokens,
                        completion.usage.completion_tokens,
                        response=completion,
                    )
                    return schema.model_validate(json_output), cost
                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"Structured outputs not supported for model '{self.name}'. "
                        f"Falling back to regular generation with JSON parsing. "
                        f"Error: {str(e)}",
                        UserWarning,
                        stacklevel=3,
                    )

            completion = await client.chat.completions.create(
                model=self.name,
                messages=messages,
                temperature=self.temperature,
                **self.generation_kwargs,
            )
            output = completion.choices[0].message.content
            cost = self.calculate_cost(
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
                response=completion,
            )
            if schema:
                from deepeval.models.llms.utils import trim_and_load_json

                return schema.model_validate(trim_and_load_json(output)), cost
            return output, cost
        finally:
            try:
                import inspect

                if hasattr(client, "close"):
                    res = client.close()
                    if inspect.isawaitable(res):
                        await res
                elif hasattr(client, "aclose"):
                    res = client.aclose()
                    if inspect.isawaitable(res):
                        await res
            except Exception:
                pass

            targets = [client]
            raw_client = getattr(client, "_client", None)
            if raw_client is not None and raw_client is not client:
                targets.append(raw_client)

            for target in targets:
                if hasattr(target, "_state"):
                    try:
                        from httpx._client import ClientState

                        target._state = ClientState.CLOSED
                    except Exception:
                        pass
                try:
                    setattr(target, "is_closed", True)
                except Exception:
                    pass

    async def _a_generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
        audio: Any = None,
    ) -> tuple[object, float | None]:
        client = self.load_model(async_mode=True)
        try:
            completion = await client.chat.completions.create(
                model=self.name,
                messages=self._build_messages(prompt, audio),
                temperature=self.temperature,
                logprobs=True,
                top_logprobs=top_logprobs,
                **self.generation_kwargs,
            )
            cost = self.calculate_cost(
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
                response=completion,
            )
            return completion, cost
        finally:
            try:
                import inspect

                if hasattr(client, "close"):
                    res = client.close()
                    if inspect.isawaitable(res):
                        await res
                elif hasattr(client, "aclose"):
                    res = client.aclose()
                    if inspect.isawaitable(res):
                        await res
            except Exception:
                pass

            targets = [client]
            raw_client = getattr(client, "_client", None)
            if raw_client is not None and raw_client is not client:
                targets.append(raw_client)

            for target in targets:
                if hasattr(target, "_state"):
                    try:
                        from httpx._client import ClientState

                        target._state = ClientState.CLOSED
                    except Exception:
                        pass
                try:
                    setattr(target, "is_closed", True)
                except Exception:
                    pass


def resolve_deepeval_llm_model(
    llm_model: str | DeepEvalBaseLLM | None = None,
    eval_mode: str = "system_one",
) -> str | DeepEvalBaseLLM | None:
    """Resolve the LLM judge model for `hybrid` or `llm` eval modes.

    Under `system_one` mode, returns `None` because no LLM is built or called.
    Under `hybrid` or `llm` mode, uses the provided `llm_model`, or constructs
    a `SafeOpenRouterModel` if `OPENROUTER_API_KEY` is configured, else falls back
    to OpenAI model string.
    """
    if eval_mode == "system_one":
        return None
    if isinstance(llm_model, DeepEvalBaseLLM):
        return llm_model

    openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model_name = (
        llm_model
        if isinstance(llm_model, str) and llm_model.strip()
        else os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash")
    )

    if openrouter_key and not openrouter_key.lower().startswith("your-"):
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return SafeOpenRouterModel(
            model=model_name,
            api_key=openrouter_key,
            base_url=base_url,
        )
    if openai_key:
        return llm_model if isinstance(llm_model, str) else "gpt-4.1"
    # If neither key is set, return OpenRouterModel if OPENROUTER_MODEL is set,
    # or let DeepEval raise its missing key error when building the LLM.
    return model_name


def make_built_in_rag_metrics(
    eval_mode: str = "system_one",
    system_one_model: str | DeepEvalBaseSystemOneModel = PINNED_JEV_MODEL,
    llm_model: str | DeepEvalBaseLLM | None = None,
    threshold: float = 0.5,
    include_reason: bool = True,
    async_mode: bool = False,
) -> dict[str, BaseMetric]:
    """Create the 4 standard DeepEval RAG metrics judged by Jev (`jev-1.13.0`).

    Metrics returned:
    - `faithfulness`: `FaithfulnessMetric`
    - `answer_relevancy`: `AnswerRelevancyMetric`
    - `contextual_precision`: `ContextualPrecisionMetric`
    - `contextual_recall`: `ContextualRecallMetric`

    Args:
        eval_mode: `"system_one"` (default, pure Jev with zero LLM calls) or
                   `"hybrid"` (opt-in, LLM extracts claims & writes reasons while Jev decides verdicts).
        system_one_model: Pinned Jev model name (`"jev-1.13.0"` by default) or custom `DeepEvalBaseSystemOneModel`.
        llm_model: Optional LLM model used only when `eval_mode` is `"hybrid"` or `"llm"`.
        threshold: Passing score threshold in `[0, 1]`.
        include_reason: Whether to include reasons (deterministic and free in `system_one` mode).
        async_mode: Whether each metric runs async internally inside `measure()`.
    """
    if eval_mode not in VALID_EVAL_MODES:
        raise ValueError(f"eval_mode must be one of {VALID_EVAL_MODES}, got {eval_mode!r}")

    resolved_llm = resolve_deepeval_llm_model(llm_model=llm_model, eval_mode=eval_mode)

    common_kwargs: dict[str, Any] = {
        "threshold": threshold,
        "eval_mode": eval_mode,
        "system_one_model": system_one_model,
        "include_reason": include_reason,
        "async_mode": async_mode,
    }
    if resolved_llm is not None:
        common_kwargs["model"] = resolved_llm

    return {
        "faithfulness": FaithfulnessMetric(**common_kwargs),
        "answer_relevancy": AnswerRelevancyMetric(**common_kwargs),
        "contextual_precision": ContextualPrecisionMetric(**common_kwargs),
        "contextual_recall": ContextualRecallMetric(**common_kwargs),
    }


def make_manufacturing_jev_eval_metric(
    system_one_model: str | DeepEvalBaseSystemOneModel = PINNED_JEV_MODEL,
    threshold: float = 0.70,
    include_reason: bool = True,
    strict_mode: bool = False,
    async_mode: bool = False,
) -> JevEval:
    """Create the custom manufacturing quality `JevEval` metric pinned to `jev-1.13.0`.

    Evaluates 5 bounded questions over `SingleTurnParams` (`INPUT`, `ACTUAL_OUTPUT`,
    `EXPECTED_OUTPUT`, `RETRIEVAL_CONTEXT`) in a single Jev `decide()` call:
    1. `overall_quality` (`Score`, weight=2.0): 6-level manufacturing rubric (0-5 mapped to [0, 1])
    2. `is_faithful` (`Noul`, weight=1.5): Zero hallucinated specs, values, or fake document references
    3. `specs_accurate` (`Noul`, weight=1.5): All numerical figures, torque specs, pressures, and tolerances match `expected_output`
    4. `completeness` (`Noul`, weight=1.0): Covers all core questions, steps, and prerequisites in `expected_output`
    5. `failure_mode` (`Choice`, weight=1.0): Primary defect classification (`none`: 1.0, minor/missing: 0.25, inaccurate/hallucinated: 0.0)
    """
    evaluation_params = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ]

    questions = [
        Score(
            "How accurately, faithfully, and completely does actual_output answer input compared to expected_output and retrieval_context?",
            levels=list(MANUFACTURING_JUDGE_CRITERIA),
            weight=2.0,
        ),
        Noul(
            "actual_output contains zero hallucinated specifications, numerical values, or fabricated document references not supported by expected_output or retrieval_context.",
            weight=1.5,
        ),
        Noul(
            "Every numerical figure, torque specification, pressure, dimension, and tolerance range in actual_output is strictly accurate according to expected_output.",
            weight=1.5,
        ),
        Noul(
            "actual_output addresses all core questions, procedures, safety prerequisites, and verification steps covered in expected_output.",
            weight=1.0,
        ),
        Choice(
            "What is the defect status or primary failure mode of actual_output when compared against expected_output?",
            options={
                "none": 1.0,
                "intent_mismatch": 0.2,
                "missing_procedure": 0.25,
                "inaccurate_specs": 0.0,
                "hallucinated_reference": 0.0,
            },
            weight=1.0,
        ),
    ]

    return JevEval(
        name="Manufacturing Quality",
        evaluation_params=evaluation_params,
        questions=questions,
        system_one_model=system_one_model,
        threshold=threshold,
        include_reason=include_reason,
        strict_mode=strict_mode,
        async_mode=async_mode,
    )


def make_contradiction_jev_eval_metric(
    system_one_model: str | DeepEvalBaseSystemOneModel = PINNED_JEV_MODEL,
    threshold: float = 0.60,
    include_reason: bool = True,
    async_mode: bool = False,
) -> JevEval:
    """Create a Jev-native `JevEval` metric for cross-document contradiction handling.

    Uses `Choice` with `no_contradiction_in_expected: None` so the question automatically
    drops out (`applicable=False`) on normal non-contradiction test cases, and scores
    contradiction scenarios strictly when `expected_output` describes a document conflict.
    """
    evaluation_params = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ]

    questions = [
        Noul(
            "When expected_output describes a contradiction or discrepancy between canonical and variant documents, actual_output explicitly flags the conflict and cites both values instead of picking one side blindly or inventing an average.",
            weight=2.0,
        ),
        Choice(
            "How does actual_output handle the cross-document specification conflict described in expected_output?",
            options={
                "flagged_conflict_with_both_values": 1.0,
                "hedged_without_both_values": 0.4,
                "reported_single_variant_blindly": 0.0,
                "invented_compromise_or_average": 0.0,
                "no_contradiction_in_expected": None,
            },
            weight=2.0,
        ),
    ]

    return JevEval(
        name="Contradiction Handling",
        evaluation_params=evaluation_params,
        questions=questions,
        system_one_model=system_one_model,
        threshold=threshold,
        include_reason=include_reason,
        async_mode=async_mode,
    )


@dataclass
class DeepEvalCaseResult:
    """Structured result of running DeepEval RAG + JevEval metrics on a single `LLMTestCase`."""

    question_id: str
    category: str
    eval_mode: str
    system_one_model: str
    # RAG metric scores in [0, 1]
    faithfulness: float
    answer_relevancy: float
    contextual_precision: float
    contextual_recall: float
    # Custom JevEval metric score in [0, 1] and extracted sub-dimensions
    manufacturing_jev_score: float
    specs_accurate: float
    completeness: float
    is_faithful: float
    rubric_score_normalized: float
    rubric_score_0_to_5: int
    failure_mode: str
    failure_mode_probabilities: dict[str, float] = field(default_factory=dict)
    # Contradiction metric (optional)
    contradiction_score: float | None = None
    # Confidence & reasons
    min_confidence: float | None = None
    metric_confidences: dict[str, float | None] = field(default_factory=dict)
    reasons: dict[str, str | None] = field(default_factory=dict)
    fallback_reasons: dict[str, str | None] = field(default_factory=dict)
    score_breakdown: list[dict[str, Any]] = field(default_factory=list)
    # Deterministic checks & composite score
    deterministic_checks: dict[str, bool] = field(default_factory=dict)
    deterministic_score: float = 1.0
    overall_score: float = 0.0
    passed: bool = False
    # Telemetry
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    evaluation_cost_usd: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_manufacturing_breakdown(
    jev_metric: JevEval,
) -> dict[str, Any]:
    """Extract sub-criterion values (`rubric_score`, `is_faithful`, `specs_accurate`,
    `completeness`, `failure_mode`) from `jev_metric.score_breakdown`.
    """
    breakdown = jev_metric.score_breakdown or []
    rubric_norm = float(jev_metric.score or 0.0)
    is_faithful = rubric_norm
    specs_accurate = rubric_norm
    completeness = rubric_norm
    failure_mode = "none"
    failure_probs: dict[str, float] = {}

    for idx, entry in enumerate(breakdown):
        q_type = entry.get("type")
        val = entry.get("value")
        probs = entry.get("probabilities") or {}
        if idx == 0 and q_type == "score" and val is not None:
            rubric_norm = float(val)
        elif idx == 1 and q_type == "noul" and val is not None:
            is_faithful = float(val)
        elif idx == 2 and q_type == "noul" and val is not None:
            specs_accurate = float(val)
        elif idx == 3 and q_type == "noul" and val is not None:
            completeness = float(val)
        elif idx == 4 and q_type == "choice":
            failure_probs = {str(k): float(v) for k, v in probs.items()}
            if failure_probs:
                failure_mode = max(failure_probs.items(), key=lambda kv: kv[1])[0]

    rubric_0_to_5 = max(0, min(5, round(rubric_norm * 5.0)))
    return {
        "rubric_score_normalized": round(rubric_norm, 4),
        "rubric_score_0_to_5": rubric_0_to_5,
        "is_faithful": round(is_faithful, 4),
        "specs_accurate": round(specs_accurate, 4),
        "completeness": round(completeness, 4),
        "failure_mode": failure_mode,
        "failure_mode_probabilities": failure_probs,
    }


@traceable_if_enabled(run_type="chain", name="deepeval.a_evaluate_test_case")
async def a_evaluate_test_case_with_deepeval(
    test_case: LLMTestCase,
    eval_mode: str = "system_one",
    system_one_model: str | DeepEvalBaseSystemOneModel = PINNED_JEV_MODEL,
    llm_model: str | DeepEvalBaseLLM | None = None,
    rag_metrics: dict[str, BaseMetric] | None = None,
    manufacturing_metric: JevEval | None = None,
    contradiction_metric: JevEval | None = None,
    deterministic_checks_map: dict[str, bool] | None = None,
    pass_threshold: float = 0.70,
    raise_on_api_error: bool = True,
    concurrent_metrics: bool = True,
) -> DeepEvalCaseResult:
    """Async variant: Run built-in RAG metrics and custom `JevEval` manufacturing metric on an `LLMTestCase`."""
    if rag_metrics is None:
        rag_metrics = make_built_in_rag_metrics(
            eval_mode=eval_mode,
            system_one_model=system_one_model,
            llm_model=llm_model,
        )
    if manufacturing_metric is None:
        manufacturing_metric = make_manufacturing_jev_eval_metric(
            system_one_model=system_one_model,
            threshold=pass_threshold,
        )

    all_metrics: dict[str, BaseMetric] = {
        **rag_metrics,
        "manufacturing_quality": manufacturing_metric,
    }
    if contradiction_metric is not None:
        all_metrics["contradiction_handling"] = contradiction_metric

    t0 = time.perf_counter()
    confidences: dict[str, float | None] = {}
    reasons: dict[str, str | None] = {}
    fallback_reasons: dict[str, str | None] = {}
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0

    async def _measure_single(m_name: str, metric: BaseMetric) -> None:
        try:
            if hasattr(metric, "a_measure") and callable(metric.a_measure):
                await metric.a_measure(test_case, _show_indicator=False)
            else:
                metric.measure(test_case, _show_indicator=False)
        except Exception as exc:
            is_err, status_code, msg = is_api_error(exc)
            if is_err and raise_on_api_error:
                if eval_mode in ("hybrid", "llm") and "typesafe" not in msg.lower():
                    raise LLMJudgeAPIError(
                        f"DeepEval LLM API error in {m_name} (status={status_code}): {msg}",
                        status_code=status_code,
                        original_error=exc,
                    ) from exc
                raise JevJudgeAPIError(
                    f"DeepEval Jev API error in {m_name} (status={status_code}): {msg}",
                    status_code=status_code,
                    original_error=exc,
                ) from exc
            raise

    if concurrent_metrics:
        await asyncio.gather(
            *[_measure_single(m_name, metric) for m_name, metric in all_metrics.items()]
        )
    else:
        for m_name, metric in all_metrics.items():
            try:
                metric.measure(test_case, _show_indicator=False)
            except Exception as exc:
                is_err, status_code, msg = is_api_error(exc)
                if is_err and raise_on_api_error:
                    if eval_mode in ("hybrid", "llm") and "typesafe" not in msg.lower():
                        raise LLMJudgeAPIError(
                            f"DeepEval LLM API error in {m_name} (status={status_code}): {msg}",
                            status_code=status_code,
                            original_error=exc,
                        ) from exc
                    raise JevJudgeAPIError(
                        f"DeepEval Jev API error in {m_name} (status={status_code}): {msg}",
                        status_code=status_code,
                        original_error=exc,
                    ) from exc
                raise

    for m_name, metric in all_metrics.items():
        conf = getattr(metric, "confidence", None)
        confidences[m_name] = round(float(conf), 4) if conf is not None else None
        reasons[m_name] = getattr(metric, "reason", None)
        fb = getattr(metric, "system_one_fallback_reason", None)
        if fb:
            fallback_reasons[m_name] = str(fb)

        total_input_tokens += int(getattr(metric, "input_tokens", 0) or 0)
        total_output_tokens += int(getattr(metric, "output_tokens", 0) or 0)
        total_cost_usd += float(getattr(metric, "evaluation_cost", 0.0) or 0.0)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    faith_score = float(rag_metrics["faithfulness"].score or 0.0)
    rel_score = float(rag_metrics["answer_relevancy"].score or 0.0)
    prec_score = float(rag_metrics["contextual_precision"].score or 0.0)
    rec_score = float(rag_metrics["contextual_recall"].score or 0.0)
    mfg_score = float(manufacturing_metric.score or 0.0)
    contra_score = (
        float(contradiction_metric.score)
        if contradiction_metric is not None and contradiction_metric.score is not None
        else None
    )

    extracted = extract_manufacturing_breakdown(manufacturing_metric)

    # Apply safety gating consistent with NGA manufacturing safety policy:
    # Severe hallucination or inaccurate specs clamps the manufacturing score
    effective_mfg_score = mfg_score
    if extracted["is_faithful"] < 0.30 or extracted["failure_mode"] == "hallucinated_reference":
        effective_mfg_score = min(effective_mfg_score, 0.20)
        extracted["rubric_score_0_to_5"] = min(extracted["rubric_score_0_to_5"], 1)
    elif extracted["specs_accurate"] < 0.30 or extracted["failure_mode"] == "inaccurate_specs":
        effective_mfg_score = min(effective_mfg_score, 0.40)
        extracted["rubric_score_0_to_5"] = min(extracted["rubric_score_0_to_5"], 2)

    checks = deterministic_checks_map or {}
    det_score = (
        sum(1 for v in checks.values() if v) / max(len(checks), 1)
        if checks
        else 1.0
    )

    # Composite score combines deterministic checks with JevEval manufacturing score (or pure JevEval when no tool checks apply)
    if checks:
        overall_score = round((det_score + effective_mfg_score) / 2.0, 4)
    else:
        overall_score = round(effective_mfg_score, 4)

    non_null_confs = [c for c in confidences.values() if c is not None]
    min_conf = min(non_null_confs) if non_null_confs else None

    meta = getattr(test_case, "metadata", None)
    if not meta:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            meta = getattr(test_case, "additional_metadata", None) or {}
    q_id = str(meta.get("question_id") or test_case.name or "UNKNOWN")
    category = str(meta.get("category") or "general")
    model_str = (
        system_one_model
        if isinstance(system_one_model, str)
        else system_one_model.get_model_name()
    )
    passed = overall_score >= pass_threshold

    attach_trace_metadata(
        question_id=q_id,
        category=category,
        eval_mode=eval_mode,
        system_one_model=model_str,
        faithfulness=round(faith_score, 4),
        answer_relevancy=round(rel_score, 4),
        contextual_precision=round(prec_score, 4),
        contextual_recall=round(rec_score, 4),
        manufacturing_jev_score=round(effective_mfg_score, 4),
        rubric_score_0_to_5=extracted["rubric_score_0_to_5"],
        is_faithful=extracted["is_faithful"],
        specs_accurate=extracted["specs_accurate"],
        completeness=extracted["completeness"],
        failure_mode=extracted["failure_mode"],
        min_confidence=min_conf,
        overall_score=overall_score,
        passed=passed,
        latency_ms=round(latency_ms, 2),
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        evaluation_cost_usd=round(total_cost_usd, 6),
    )
    attach_trace_tags(
        f"eval_mode:{eval_mode}",
        f"model:{model_str}",
        "deepeval-pass" if passed else "deepeval-fail",
    )

    return DeepEvalCaseResult(
        question_id=q_id,
        category=category,
        eval_mode=eval_mode,
        system_one_model=model_str,
        faithfulness=round(faith_score, 4),
        answer_relevancy=round(rel_score, 4),
        contextual_precision=round(prec_score, 4),
        contextual_recall=round(rec_score, 4),
        manufacturing_jev_score=round(effective_mfg_score, 4),
        specs_accurate=extracted["specs_accurate"],
        completeness=extracted["completeness"],
        is_faithful=extracted["is_faithful"],
        rubric_score_normalized=extracted["rubric_score_normalized"],
        rubric_score_0_to_5=extracted["rubric_score_0_to_5"],
        failure_mode=extracted["failure_mode"],
        failure_mode_probabilities=extracted["failure_mode_probabilities"],
        contradiction_score=round(contra_score, 4) if contra_score is not None else None,
        min_confidence=min_conf,
        metric_confidences=confidences,
        reasons=reasons,
        fallback_reasons=fallback_reasons,
        score_breakdown=list(manufacturing_metric.score_breakdown or []),
        deterministic_checks=checks,
        deterministic_score=round(det_score, 4),
        overall_score=overall_score,
        passed=passed,
        latency_ms=round(latency_ms, 2),
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        evaluation_cost_usd=round(total_cost_usd, 6),
    )


@traceable_if_enabled(run_type="chain", name="deepeval.evaluate_test_case")
def evaluate_test_case_with_deepeval(
    test_case: LLMTestCase,
    eval_mode: str = "system_one",
    system_one_model: str | DeepEvalBaseSystemOneModel = PINNED_JEV_MODEL,
    llm_model: str | DeepEvalBaseLLM | None = None,
    rag_metrics: dict[str, BaseMetric] | None = None,
    manufacturing_metric: JevEval | None = None,
    contradiction_metric: JevEval | None = None,
    deterministic_checks_map: dict[str, bool] | None = None,
    pass_threshold: float = 0.70,
    raise_on_api_error: bool = True,
    concurrent_metrics: bool = True,
) -> DeepEvalCaseResult:
    """Run built-in RAG metrics and custom `JevEval` manufacturing metric on an `LLMTestCase`."""
    from deepeval.utils import get_or_create_event_loop

    loop = get_or_create_event_loop()
    return loop.run_until_complete(
        a_evaluate_test_case_with_deepeval(
            test_case=test_case,
            eval_mode=eval_mode,
            system_one_model=system_one_model,
            llm_model=llm_model,
            rag_metrics=rag_metrics,
            manufacturing_metric=manufacturing_metric,
            contradiction_metric=contradiction_metric,
            deterministic_checks_map=deterministic_checks_map,
            pass_threshold=pass_threshold,
            raise_on_api_error=raise_on_api_error,
            concurrent_metrics=concurrent_metrics,
        )
    )
