"""Automated Hard Release Criteria and Regression Safeguards.

Backward-compatibility shim re-exporting from enterprise_agent.evaluation.release_gate.
"""

from enterprise_agent.evaluation.release_gate import (
    DEFAULT_RELEASE_CRITERIA,
    ReleaseGateResult,
    evaluate_release_gate,
)

__all__ = [
    "DEFAULT_RELEASE_CRITERIA",
    "ReleaseGateResult",
    "evaluate_release_gate",
]
