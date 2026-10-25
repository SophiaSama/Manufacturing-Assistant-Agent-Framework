"""Side-by-side comparison harness: Legacy NGA Jev harness vs DeepEval (RAG + JevEval).

Backward-compatibility shim re-exporting from enterprise_agent.evaluation.side_by_side.
"""

from enterprise_agent.evaluation.side_by_side import (
    DEFAULT_CONFLICT_SCENARIOS,
    CriterionAgreement,
    SideBySideCaseComparison,
    SideBySideSummary,
    compute_criterion_agreement,
    format_side_by_side_markdown,
    perturb_manufacturing_answer,
    run_side_by_side_comparison,
    save_side_by_side_report,
)

__all__ = [
    "DEFAULT_CONFLICT_SCENARIOS",
    "CriterionAgreement",
    "SideBySideCaseComparison",
    "SideBySideSummary",
    "compute_criterion_agreement",
    "format_side_by_side_markdown",
    "perturb_manufacturing_answer",
    "run_side_by_side_comparison",
    "save_side_by_side_report",
]
