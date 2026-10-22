"""Standardized DeepEval & JevEval Evaluation Subsystem for Enterprise Agent."""

from enterprise_agent.evaluation.ci_tracker import (
    generate_ci_markdown_summary,
    generate_svg_trend_chart,
    load_history,
    record_eval_run,
    save_history,
)
from enterprise_agent.evaluation.deepeval_adapter import (
    DeepEvalAdapter,
    conflict_scenario_to_test_cases,
    to_deep_eval_test_case,
)
from enterprise_agent.evaluation.deepeval_metrics import (
    PINNED_JEV_MODEL,
    DeepEvalCaseResult,
    SafeOpenRouterModel,
    evaluate_test_case_with_deepeval,
    make_contradiction_jev_eval_metric,
    make_domain_jev_eval_metric,
    make_manufacturing_jev_eval_metric,
)
from enterprise_agent.evaluation.grounding import (
    GroundingResult,
    build_corpus_manifest,
    evaluate_grounding,
)
from enterprise_agent.evaluation.judge import (
    JevEvaluationResult,
    evaluate_with_jev,
)
from enterprise_agent.evaluation.release_gate import (
    ReleaseGateResult,
    evaluate_release_gate,
)
from enterprise_agent.evaluation.report import (
    build_summary,
    generate_markdown_report,
    save_report,
)
from enterprise_agent.evaluation.scoring import (
    ScoreResult,
    deterministic_checks,
    score_answer,
)
from enterprise_agent.evaluation.side_by_side import (
    CriterionAgreement,
    SideBySideCaseComparison,
    SideBySideSummary,
    compute_criterion_agreement,
    format_side_by_side_markdown,
    perturb_answer,
    perturb_manufacturing_answer,
    run_side_by_side_comparison,
    save_side_by_side_report,
)
from enterprise_agent.evaluation.stepped_suite import (
    compute_stepped_summary,
    load_stepped_questions,
)

__all__ = [
    # DeepEval & Adapters
    "DeepEvalAdapter",
    "to_deep_eval_test_case",
    "conflict_scenario_to_test_cases",
    "SafeOpenRouterModel",
    "DeepEvalCaseResult",
    "evaluate_test_case_with_deepeval",
    "make_domain_jev_eval_metric",
    "make_manufacturing_jev_eval_metric",
    "make_contradiction_jev_eval_metric",
    "PINNED_JEV_MODEL",
    # Judges
    "JevEvaluationResult",
    "evaluate_with_jev",
    # Side-by-Side
    "CriterionAgreement",
    "SideBySideCaseComparison",
    "SideBySideSummary",
    "compute_criterion_agreement",
    "format_side_by_side_markdown",
    "save_side_by_side_report",
    "run_side_by_side_comparison",
    "perturb_answer",
    "perturb_manufacturing_answer",
    # Grounding
    "GroundingResult",
    "build_corpus_manifest",
    "evaluate_grounding",
    # Stepped Suite
    "load_stepped_questions",
    "compute_stepped_summary",
    # Release Gate
    "ReleaseGateResult",
    "evaluate_release_gate",
    # CI Tracker
    "record_eval_run",
    "load_history",
    "save_history",
    "generate_svg_trend_chart",
    "generate_ci_markdown_summary",
    # Scoring & Report
    "ScoreResult",
    "deterministic_checks",
    "score_answer",
    "build_summary",
    "generate_markdown_report",
    "save_report",
]
