"""Side-by-side comparison harness: Legacy NGA Jev harness vs DeepEval (RAG + JevEval).

Runs both the legacy evaluation harness (`evaluate_with_jev` / `score_answer`) and
the new DeepEval harness (`evaluate_test_case_with_deepeval`) on the same benchmark
cases and cross-document contradiction scenarios, computing per-criterion verdict
agreement, score shift distributions, confidence, latency, and cost before retiring
the legacy harness.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel

from nga.evaluation.deepeval_adapter import (
    conflict_scenario_to_test_cases,
    to_deep_eval_test_case,
)
from nga.evaluation.deepeval_metrics import (
    PINNED_JEV_MODEL,
    DeepEvalCaseResult,
    evaluate_test_case_with_deepeval,
    make_built_in_rag_metrics,
    make_contradiction_jev_eval_metric,
    make_manufacturing_jev_eval_metric,
)
from nga.evaluation.judge import (
    JevEvaluationResult,
    evaluate_with_jev,
)
from nga.tracing import attach_trace_metadata, attach_trace_tags, traceable_if_enabled

logger = logging.getLogger("nga.evaluation.side_by_side")


DEFAULT_CONFLICT_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "CONFLICT-1",
        "name": "Wheel torque (105 Nm canonical vs 108 Nm variant)",
        "golden": "There is a document contradiction: SOP-OPR-101 specifies 105 Nm ± 5%, whereas variant document specifies 108 Nm. Both values exist and must be reconciled.",
        "answers": {
            "conflict_flagged": "Discrepancy detected: canonical SOP-OPR-101 specifies 105 Nm, but the variant document lists 108 Nm. Engineering review is required.",
            "blind_variant": "The correct torque at Station 144 is 108 Nm per the variant specification.",
            "hallucinated_middle": "The torque target is 106.5 Nm based on averaging the station documents.",
        },
    },
    {
        "id": "CONFLICT-2",
        "name": "Electrode cap life (4,000 vs 5,000 welds)",
        "golden": "Inconsistency detected: Canonical SOP-WLD-002 sets cap life at 4,000 welds, while variant documentation states 5,000 welds.",
        "answers": {
            "conflict_flagged": "There is a conflict between 4,000 welds in standard SOP and 5,000 welds in the variant document.",
            "blind_variant": "The electrode cap life is 5,000 welds.",
            "hallucinated_middle": "The cap life is 4,500 welds.",
        },
    },
    {
        "id": "CONFLICT-3",
        "name": "Calibration interval (3 months vs 6 months)",
        "golden": "Contradiction found: TorqMaster TF-6000 calibration interval is 3 months in canonical SOP-OPR-158 but 6 months in variant records.",
        "answers": {
            "conflict_flagged": "A discrepancy exists: canonical SOP requires 3 months or 100k cycles, whereas variant states 6 months.",
            "blind_variant": "The calibration interval is 6 months.",
            "hallucinated_middle": "Calibration is performed every 4.5 months.",
        },
    },
    {
        "id": "CONFLICT-4",
        "name": "Class A defect notification (1 hour vs 4 hours)",
        "golden": "Conflict: Class A safety defects require Quality Director notification within 1 hour per QCR-501, but variant notes 4 hours.",
        "answers": {
            "conflict_flagged": "Contradiction identified: QCR-501 dictates 1 hour for Class A defects, while variant document allows 4 hours.",
            "blind_variant": "The Quality Director must be notified within 4 hours.",
            "hallucinated_middle": "Notification is required within 2.5 hours.",
        },
    },
    {
        "id": "CONFLICT-5",
        "name": "Safety defect rate threshold (0.5% vs 1.0%)",
        "golden": "Contradiction: Recall criteria triggers at 0.5% safety defect rate in QCR-501, but variant document states 1.0%.",
        "answers": {
            "conflict_flagged": "Conflict detected: Standard policy triggers recall evaluation at 0.5%, whereas variant document specifies 1.0%.",
            "blind_variant": "The defect rate threshold triggering recall is 1.0%.",
            "hallucinated_middle": "The threshold is 0.75% based on combined standards.",
        },
    },
    {
        "id": "CONFLICT-6",
        "name": "Rear mount torque (45 Nm vs 50 Nm)",
        "golden": "Conflict: Rear transmission mount bolt torque is 45 Nm in SOP-MEC-305, but 50 Nm in variant specs.",
        "answers": {
            "conflict_flagged": "Inconsistency found: SOP-MEC-305 specifies 45 Nm, but variant documentation lists 50 Nm.",
            "blind_variant": "The rear transmission mount bolt torque is 50 Nm.",
            "hallucinated_middle": "The torque specification is 47.5 Nm.",
        },
    },
    {
        "id": "CONFLICT-7",
        "name": "Brake bleed sequence (RR-RL-FR-FL vs FR-FL-RR-RL)",
        "golden": "Contradiction: Canonical bleed sequence is furthest-to-closest (RR→RL→FR→FL), but variant states closest-first (FR→FL→RR→RL).",
        "answers": {
            "conflict_flagged": "There is a direct contradiction in bleed sequences: canonical is RR→RL→FR→FL, whereas variant lists FR→FL→RR→RL.",
            "blind_variant": "Bleed brakes starting from front right: FR → FL → RR → RL.",
            "hallucinated_middle": "Bleed in diagonal sequence FL → RR → FR → RL.",
        },
    },
    {
        "id": "CONFLICT-8",
        "name": "Windshield adhesive cure time (4 hours vs 2 hours)",
        "golden": "Contradiction: SOP-GLS-102 sets minimum cure time at 4 hours (6 h winter), but variant document claims 2 hours.",
        "answers": {
            "conflict_flagged": "Discrepancy: Canonical SOP-GLS-102 requires 4 hours cure time, while variant claims 2 hours. 4 hours must be adhered to.",
            "blind_variant": "The minimum cure time is 2 hours.",
            "hallucinated_middle": "The adhesive cure time is 3 hours.",
        },
    },
]


def perturb_manufacturing_answer(expected: str) -> str:
    """Create a realistic manufacturing spec/procedure defect for negative control testing."""
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", expected)
    if numbers:
        target = numbers[0]
        try:
            val = float(target)
            corrupted = f"{val * 1.5:.1f}" if "." in target else str(int(val * 1.5 + 10))
            return expected.replace(target, corrupted, 1)
        except ValueError:
            pass

    if "star" in expected.lower():
        return "Tighten bolts in circular order clockwise starting at top bolt."
    if "every" in expected.lower():
        return "Inspection occurs at end of month by maintenance."
    return f"Status confirmed normal according to legacy plant manual: {expected[:20]}."


@dataclass
class CriterionAgreement:
    """Per-criterion verdict agreement between Legacy Jev harness and DeepEval harness."""

    is_faithful_agreement: bool
    rag_faithfulness_agreement: bool
    specs_accurate_agreement: bool
    completeness_agreement: bool
    rag_recall_agreement: bool
    rubric_within_1_agreement: bool
    overall_pass_agreement: bool
    score_shift: float  # deepeval_overall_score - legacy_normalized_score

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SideBySideCaseComparison:
    """Comparison record for a single benchmark or contradiction test case."""

    case_id: str
    category: str
    condition: str  # "positive", "perturbed", "conflict_flagged", "blind_variant", "hallucinated_middle"
    legacy_score_int: int
    legacy_score_normalized: float
    legacy_passed: bool
    legacy_is_faithful: float
    legacy_specs_accurate: float
    legacy_completeness: float
    legacy_failure_mode: str
    legacy_confidence: float | None
    legacy_latency_ms: float
    deepeval_result: DeepEvalCaseResult
    agreement: CriterionAgreement

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["deepeval_result"] = self.deepeval_result.as_dict()
        data["agreement"] = self.agreement.as_dict()
        return data


@dataclass
class SideBySideSummary:
    """Aggregate summary of running legacy and DeepEval harnesses side-by-side."""

    eval_mode: str
    system_one_model: str
    total_cases: int
    positive_cases: int
    perturbed_cases: int
    contradiction_cases: int
    # Per-criterion verdict agreement rates in [0, 1]
    is_faithful_agreement_rate: float
    rag_faithfulness_agreement_rate: float
    specs_accurate_agreement_rate: float
    completeness_agreement_rate: float
    rag_recall_agreement_rate: float
    rubric_within_1_agreement_rate: float
    overall_pass_agreement_rate: float
    # Pass / catch rates
    legacy_positive_pass_rate: float
    deepeval_positive_pass_rate: float
    legacy_perturbed_catch_rate: float
    deepeval_perturbed_catch_rate: float
    legacy_conflict_flagged_pass_rate: float
    deepeval_conflict_flagged_pass_rate: float
    legacy_conflict_defect_catch_rate: float
    deepeval_conflict_defect_catch_rate: float
    # Score shifts & confidence
    mean_score_shift: float
    mean_abs_score_shift: float
    max_abs_score_shift: float
    legacy_mean_confidence: float | None
    deepeval_mean_min_confidence: float | None
    # Speed & cost
    legacy_avg_latency_ms: float
    deepeval_avg_latency_ms: float
    deepeval_total_input_tokens: int
    deepeval_total_output_tokens: int
    deepeval_total_cost_usd: float
    cases: list[SideBySideCaseComparison] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cases"] = [c.as_dict() for c in self.cases]
        return d


def compute_criterion_agreement(
    legacy: JevEvaluationResult,
    deepeval_res: DeepEvalCaseResult,
    pass_threshold: float = 0.70,
    criterion_threshold: float = 0.50,
) -> CriterionAgreement:
    """Compute per-criterion binary verdict agreement and continuous score shift."""
    leg_faithful = legacy.is_faithful >= criterion_threshold
    de_jev_faithful = deepeval_res.is_faithful >= criterion_threshold
    de_rag_faithful = deepeval_res.faithfulness >= criterion_threshold

    leg_specs = legacy.specs_accurate >= criterion_threshold
    de_specs = deepeval_res.specs_accurate >= criterion_threshold

    leg_comp = legacy.completeness >= criterion_threshold
    de_jev_comp = deepeval_res.completeness >= criterion_threshold
    de_rag_recall = deepeval_res.contextual_recall >= criterion_threshold

    rubric_within_1 = abs(legacy.score_int - deepeval_res.rubric_score_0_to_5) <= 1

    leg_norm = legacy.score_int / 5.0
    leg_passed = leg_norm >= pass_threshold
    overall_pass_agree = leg_passed == deepeval_res.passed
    score_shift = round(deepeval_res.overall_score - leg_norm, 4)

    return CriterionAgreement(
        is_faithful_agreement=(leg_faithful == de_jev_faithful),
        rag_faithfulness_agreement=(leg_faithful == de_rag_faithful),
        specs_accurate_agreement=(leg_specs == de_specs),
        completeness_agreement=(leg_comp == de_jev_comp),
        rag_recall_agreement=(leg_comp == de_rag_recall),
        rubric_within_1_agreement=rubric_within_1,
        overall_pass_agreement=overall_pass_agree,
        score_shift=score_shift,
    )


@traceable_if_enabled(run_type="chain", name="side_by_side.run_comparison")
def run_side_by_side_comparison(
    questions: list[dict[str, Any]],
    conflict_scenarios: list[dict[str, Any]] | None = None,
    eval_mode: str = "system_one",
    system_one_model: str | DeepEvalBaseSystemOneModel = PINNED_JEV_MODEL,
    llm_model: str | DeepEvalBaseLLM | None = None,
    legacy_evaluator: Callable[[str, str], JevEvaluationResult] | None = None,
    include_perturbed_controls: bool = True,
    pass_threshold: float = 0.70,
) -> SideBySideSummary:
    """Run legacy Jev harness and DeepEval harness side-by-side on benchmark & conflict cases."""
    if legacy_evaluator is None:
        def _default_legacy(cand: str, gold: str) -> JevEvaluationResult:
            return evaluate_with_jev(cand, gold)
        legacy_evaluator = _default_legacy

    rag_metrics = make_built_in_rag_metrics(
        eval_mode=eval_mode,
        system_one_model=system_one_model,
        llm_model=llm_model,
    )
    mfg_metric = make_manufacturing_jev_eval_metric(
        system_one_model=system_one_model,
        threshold=pass_threshold,
    )
    contra_metric = make_contradiction_jev_eval_metric(
        system_one_model=system_one_model,
    )

    comparisons: list[SideBySideCaseComparison] = []

    # 1. Evaluate benchmark questions (positive controls + optional perturbed negative controls)
    for q in questions:
        qid = str(q.get("id", "UNKNOWN"))
        category = str(q.get("category", "general"))
        expected = str(q.get("expected_answer") or q.get("golden") or "")
        pos_candidate = str(
            q.get("actual_output")
            or f"Based on plant specifications and records, the verified answer is: {expected}"
        )

        # Positive control
        pos_row = {**q, "id": f"{qid}:positive", "condition": "positive"}
        pos_tc = to_deep_eval_test_case(question_data=pos_row, actual_output=pos_candidate)
        leg_pos = legacy_evaluator(pos_candidate, expected)
        de_pos = evaluate_test_case_with_deepeval(
            test_case=pos_tc,
            eval_mode=eval_mode,
            system_one_model=system_one_model,
            llm_model=llm_model,
            rag_metrics=rag_metrics,
            manufacturing_metric=mfg_metric,
            pass_threshold=pass_threshold,
        )
        agree_pos = compute_criterion_agreement(leg_pos, de_pos, pass_threshold=pass_threshold)
        comparisons.append(
            SideBySideCaseComparison(
                case_id=f"{qid}:positive",
                category=category,
                condition="positive",
                legacy_score_int=leg_pos.score_int,
                legacy_score_normalized=round(leg_pos.score_int / 5.0, 4),
                legacy_passed=(leg_pos.score_int / 5.0) >= pass_threshold,
                legacy_is_faithful=round(leg_pos.is_faithful, 4),
                legacy_specs_accurate=round(leg_pos.specs_accurate, 4),
                legacy_completeness=round(leg_pos.completeness, 4),
                legacy_failure_mode=leg_pos.failure_mode,
                legacy_confidence=leg_pos.confidence,
                legacy_latency_ms=round(leg_pos.latency_ms, 2),
                deepeval_result=de_pos,
                agreement=agree_pos,
            )
        )

        # Perturbed negative control
        if include_perturbed_controls:
            neg_candidate = perturb_manufacturing_answer(expected)
            neg_row = {**q, "id": f"{qid}:perturbed", "condition": "perturbed"}
            neg_tc = to_deep_eval_test_case(question_data=neg_row, actual_output=neg_candidate)
            leg_neg = legacy_evaluator(neg_candidate, expected)
            de_neg = evaluate_test_case_with_deepeval(
                test_case=neg_tc,
                eval_mode=eval_mode,
                system_one_model=system_one_model,
                llm_model=llm_model,
                rag_metrics=rag_metrics,
                manufacturing_metric=mfg_metric,
                pass_threshold=pass_threshold,
            )
            agree_neg = compute_criterion_agreement(leg_neg, de_neg, pass_threshold=pass_threshold)
            comparisons.append(
                SideBySideCaseComparison(
                    case_id=f"{qid}:perturbed",
                    category=category,
                    condition="perturbed",
                    legacy_score_int=leg_neg.score_int,
                    legacy_score_normalized=round(leg_neg.score_int / 5.0, 4),
                    legacy_passed=(leg_neg.score_int / 5.0) >= pass_threshold,
                    legacy_is_faithful=round(leg_neg.is_faithful, 4),
                    legacy_specs_accurate=round(leg_neg.specs_accurate, 4),
                    legacy_completeness=round(leg_neg.completeness, 4),
                    legacy_failure_mode=leg_neg.failure_mode,
                    legacy_confidence=leg_neg.confidence,
                    legacy_latency_ms=round(leg_neg.latency_ms, 2),
                    deepeval_result=de_neg,
                    agreement=agree_neg,
                )
            )

    # 2. Evaluate cross-document contradiction scenarios
    scenarios = conflict_scenarios if conflict_scenarios is not None else DEFAULT_CONFLICT_SCENARIOS
    for scenario in scenarios:
        golden = str(scenario.get("golden", ""))
        tc_map = conflict_scenario_to_test_cases(scenario)
        for condition, tc in tc_map.items():
            cand = tc.actual_output or ""
            leg_c = legacy_evaluator(cand, golden)
            de_c = evaluate_test_case_with_deepeval(
                test_case=tc,
                eval_mode=eval_mode,
                system_one_model=system_one_model,
                llm_model=llm_model,
                rag_metrics=rag_metrics,
                manufacturing_metric=mfg_metric,
                contradiction_metric=contra_metric,
                pass_threshold=pass_threshold,
            )
            agree_c = compute_criterion_agreement(leg_c, de_c, pass_threshold=pass_threshold)
            comparisons.append(
                SideBySideCaseComparison(
                    case_id=str(tc.name or f"{scenario.get('id')}:{condition}"),
                    category="contradiction",
                    condition=condition,
                    legacy_score_int=leg_c.score_int,
                    legacy_score_normalized=round(leg_c.score_int / 5.0, 4),
                    legacy_passed=(leg_c.score_int / 5.0) >= pass_threshold,
                    legacy_is_faithful=round(leg_c.is_faithful, 4),
                    legacy_specs_accurate=round(leg_c.specs_accurate, 4),
                    legacy_completeness=round(leg_c.completeness, 4),
                    legacy_failure_mode=leg_c.failure_mode,
                    legacy_confidence=leg_c.confidence,
                    legacy_latency_ms=round(leg_c.latency_ms, 2),
                    deepeval_result=de_c,
                    agreement=agree_c,
                )
            )

    total = max(len(comparisons), 1)
    pos_cases = [c for c in comparisons if c.condition == "positive"]
    neg_cases = [c for c in comparisons if c.condition == "perturbed"]
    contra_flagged = [c for c in comparisons if c.condition == "conflict_flagged"]
    contra_defects = [
        c for c in comparisons if c.condition in ("blind_variant", "hallucinated_middle")
    ]

    def _rate(items: list[SideBySideCaseComparison], pred: Callable[[SideBySideCaseComparison], bool]) -> float:
        if not items:
            return 0.0
        return round(sum(1 for x in items if pred(x)) / len(items), 4)

    shifts = [c.agreement.score_shift for c in comparisons]
    abs_shifts = [abs(s) for s in shifts]

    leg_confs = [c.legacy_confidence for c in comparisons if c.legacy_confidence is not None]
    de_confs = [
        c.deepeval_result.min_confidence
        for c in comparisons
        if c.deepeval_result.min_confidence is not None
    ]

    model_str = (
        system_one_model
        if isinstance(system_one_model, str)
        else system_one_model.get_model_name()
    )

    summary = SideBySideSummary(
        eval_mode=eval_mode,
        system_one_model=model_str,
        total_cases=len(comparisons),
        positive_cases=len(pos_cases),
        perturbed_cases=len(neg_cases),
        contradiction_cases=len(contra_flagged) + len(contra_defects),
        is_faithful_agreement_rate=_rate(comparisons, lambda c: c.agreement.is_faithful_agreement),
        rag_faithfulness_agreement_rate=_rate(comparisons, lambda c: c.agreement.rag_faithfulness_agreement),
        specs_accurate_agreement_rate=_rate(comparisons, lambda c: c.agreement.specs_accurate_agreement),
        completeness_agreement_rate=_rate(comparisons, lambda c: c.agreement.completeness_agreement),
        rag_recall_agreement_rate=_rate(comparisons, lambda c: c.agreement.rag_recall_agreement),
        rubric_within_1_agreement_rate=_rate(comparisons, lambda c: c.agreement.rubric_within_1_agreement),
        overall_pass_agreement_rate=_rate(comparisons, lambda c: c.agreement.overall_pass_agreement),
        legacy_positive_pass_rate=_rate(pos_cases, lambda c: c.legacy_passed),
        deepeval_positive_pass_rate=_rate(pos_cases, lambda c: c.deepeval_result.passed),
        legacy_perturbed_catch_rate=_rate(neg_cases, lambda c: not c.legacy_passed),
        deepeval_perturbed_catch_rate=_rate(neg_cases, lambda c: not c.deepeval_result.passed),
        legacy_conflict_flagged_pass_rate=_rate(contra_flagged, lambda c: c.legacy_passed),
        deepeval_conflict_flagged_pass_rate=_rate(contra_flagged, lambda c: c.deepeval_result.passed),
        legacy_conflict_defect_catch_rate=_rate(contra_defects, lambda c: not c.legacy_passed),
        deepeval_conflict_defect_catch_rate=_rate(contra_defects, lambda c: not c.deepeval_result.passed),
        mean_score_shift=round(sum(shifts) / total, 4) if comparisons else 0.0,
        mean_abs_score_shift=round(sum(abs_shifts) / total, 4) if comparisons else 0.0,
        max_abs_score_shift=round(max(abs_shifts), 4) if abs_shifts else 0.0,
        legacy_mean_confidence=round(sum(leg_confs) / len(leg_confs), 4) if leg_confs else None,
        deepeval_mean_min_confidence=round(sum(de_confs) / len(de_confs), 4) if de_confs else None,
        legacy_avg_latency_ms=round(sum(c.legacy_latency_ms for c in comparisons) / total, 2) if comparisons else 0.0,
        deepeval_avg_latency_ms=round(sum(c.deepeval_result.latency_ms for c in comparisons) / total, 2) if comparisons else 0.0,
        deepeval_total_input_tokens=sum(c.deepeval_result.input_tokens for c in comparisons),
        deepeval_total_output_tokens=sum(c.deepeval_result.output_tokens for c in comparisons),
        deepeval_total_cost_usd=round(sum(c.deepeval_result.evaluation_cost_usd for c in comparisons), 6),
        cases=comparisons,
    )

    attach_trace_metadata(
        eval_mode=summary.eval_mode,
        system_one_model=summary.system_one_model,
        total_cases=summary.total_cases,
        overall_pass_agreement_rate=summary.overall_pass_agreement_rate,
        is_faithful_agreement_rate=summary.is_faithful_agreement_rate,
        specs_accurate_agreement_rate=summary.specs_accurate_agreement_rate,
        completeness_agreement_rate=summary.completeness_agreement_rate,
        mean_score_shift=summary.mean_score_shift,
        deepeval_total_input_tokens=summary.deepeval_total_input_tokens,
        deepeval_total_output_tokens=summary.deepeval_total_output_tokens,
        deepeval_total_cost_usd=summary.deepeval_total_cost_usd,
    )
    attach_trace_tags(
        "side-by-side-eval",
        f"eval_mode:{summary.eval_mode}",
        f"model:{summary.system_one_model}",
    )

    return summary


def format_side_by_side_markdown(summary: SideBySideSummary) -> str:
    """Render a comprehensive Markdown report of the side-by-side comparison."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Side-by-Side Evaluation Report: Legacy NGA Jev Harness vs DeepEval (`JevEval` + RAG Metrics)",
        "",
        f"- **Generated:** {ts}",
        f"- **Pinned System One Model:** `{summary.system_one_model}`",
        f"- **DeepEval Mode:** `{summary.eval_mode}` "
        + (
            "(100% Jev-only execution, zero LLM calls, zero output tokens)"
            if summary.eval_mode == "system_one"
            else "(Hybrid: LLM extracts claims & writes reasons; Jev decides verdicts)"
        ),
        f"- **Total Cases Evaluated:** {summary.total_cases} "
        f"({summary.positive_cases} positive, {summary.perturbed_cases} perturbed negative, {summary.contradiction_cases} contradiction)",
        "",
        "## 1. Per-Criterion Verdict Agreement",
        "",
        "| Criterion Comparison | Verdict Agreement Rate |",
        "|---|---|",
        f"| `is_faithful` (Legacy Jev vs DeepEval `JevEval` Noul) | {summary.is_faithful_agreement_rate * 100:.1f}% |",
        f"| `faithfulness` (Legacy Jev vs DeepEval `FaithfulnessMetric`) | {summary.rag_faithfulness_agreement_rate * 100:.1f}% |",
        f"| `specs_accurate` (Legacy Jev vs DeepEval `JevEval` Noul) | {summary.specs_accurate_agreement_rate * 100:.1f}% |",
        f"| `completeness` (Legacy Jev vs DeepEval `JevEval` Noul) | {summary.completeness_agreement_rate * 100:.1f}% |",
        f"| `contextual_recall` (Legacy Jev vs DeepEval `ContextualRecallMetric`) | {summary.rag_recall_agreement_rate * 100:.1f}% |",
        f"| Rubric 0–5 Agreement (Within ±1 Level) | {summary.rubric_within_1_agreement_rate * 100:.1f}% |",
        f"| **Overall Pass/Fail Verdict Agreement** | **{summary.overall_pass_agreement_rate * 100:.1f}%** |",
        "",
        "## 2. Pass Rates, Defect Catch Rates & Score Shifts",
        "",
        "| Metric | Legacy Harness (`jev-1.13.0`) | DeepEval Harness (`jev-1.13.0`) |",
        "|---|---|---|",
        f"| Positive Control Pass Rate | {summary.legacy_positive_pass_rate * 100:.1f}% | {summary.deepeval_positive_pass_rate * 100:.1f}% |",
        f"| Perturbed Defect Catch Rate | {summary.legacy_perturbed_catch_rate * 100:.1f}% | {summary.deepeval_perturbed_catch_rate * 100:.1f}% |",
        f"| Contradiction Flagged Pass Rate | {summary.legacy_conflict_flagged_pass_rate * 100:.1f}% | {summary.deepeval_conflict_flagged_pass_rate * 100:.1f}% |",
        f"| Contradiction Blind/Hallucinated Catch Rate | {summary.legacy_conflict_defect_catch_rate * 100:.1f}% | {summary.deepeval_conflict_defect_catch_rate * 100:.1f}% |",
        f"| Mean Confidence | {summary.legacy_mean_confidence if summary.legacy_mean_confidence is not None else 'N/A'} | {summary.deepeval_mean_min_confidence if summary.deepeval_mean_min_confidence is not None else 'N/A'} |",
        f"| Average Latency per Case (ms) | {summary.legacy_avg_latency_ms:.1f} ms | {summary.deepeval_avg_latency_ms:.1f} ms |",
        "",
        "### Score Shift Summary (`deepeval_overall_score - legacy_normalized_score`)",
        f"- **Mean Score Shift:** `{summary.mean_score_shift:+.4f}`",
        f"- **Mean Absolute Score Shift:** `{summary.mean_abs_score_shift:.4f}`",
        f"- **Max Absolute Score Shift:** `{summary.max_abs_score_shift:.4f}`",
        f"- **DeepEval Token Usage:** `{summary.deepeval_total_input_tokens}` input tokens, `{summary.deepeval_total_output_tokens}` output tokens (`${summary.deepeval_total_cost_usd:.6f}` USD)",
        "",
        "## 3. Case-by-Case Breakdown",
        "",
        "| Case ID | Condition | Legacy Score | DeepEval Score | Faithfulness | Rel | Prec | Recall | Agree Pass? | Shift |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for c in summary.cases:
        de = c.deepeval_result
        agree_mark = "YES" if c.agreement.overall_pass_agreement else "NO"
        lines.append(
            f"| `{c.case_id}` | `{c.condition}` | {c.legacy_score_int}/5 ({c.legacy_score_normalized:.2f}) "
            f"| {de.rubric_score_0_to_5}/5 ({de.overall_score:.2f}) "
            f"| {de.faithfulness:.2f} | {de.answer_relevancy:.2f} | {de.contextual_precision:.2f} | {de.contextual_recall:.2f} "
            f"| {agree_mark} | {c.agreement.score_shift:+.2f} |"
        )

    lines.append("")
    return "\n".join(lines)


def save_side_by_side_report(
    summary: SideBySideSummary,
    output_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Save the side-by-side comparison Markdown and JSON artifacts."""
    root_dir = Path(__file__).resolve().parents[3]
    eval_dir = Path(output_dir) if output_dir is not None else (root_dir / "reports" / "eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    md_path = eval_dir / "side_by_side_deepeval_comparison.md"
    json_path = eval_dir / "side_by_side_deepeval_comparison.json"

    md_content = format_side_by_side_markdown(summary)
    md_path.write_text(md_content, encoding="utf-8")
    json_path.write_text(
        json.dumps(summary.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if output_dir is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        judge_dir = root_dir / "reports" / "judge" / date_str
        judge_dir.mkdir(parents=True, exist_ok=True)
        (judge_dir / "side_by_side_deepeval.md").write_text(md_content, encoding="utf-8")
        (judge_dir / "side_by_side_deepeval.json").write_text(
            json.dumps(summary.as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return md_path, json_path
