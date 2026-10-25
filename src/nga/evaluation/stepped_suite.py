"""The 4-Tier Stepped Evaluation Test Suite (阶梯式评测集).

Backward-compatibility shim re-exporting from enterprise_agent.evaluation.stepped_suite.
"""

from enterprise_agent.evaluation.stepped_suite import (
    TIER_NAMES,
    TIER_TARGETS,
    compute_stepped_summary,
    load_stepped_questions,
)

__all__ = [
    "TIER_NAMES",
    "TIER_TARGETS",
    "load_stepped_questions",
    "compute_stepped_summary",
]
