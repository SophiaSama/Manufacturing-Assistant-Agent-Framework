"""Hybrid deterministic + LLM-as-judge scoring for NGA evaluation.

Backward-compatibility shim re-exporting from enterprise_agent.evaluation.scoring.
"""

from enterprise_agent.evaluation.scoring import (
    PASS_THRESHOLD,
    ScoreResult,
    deterministic_checks,
    score_answer,
)

__all__ = [
    "PASS_THRESHOLD",
    "ScoreResult",
    "deterministic_checks",
    "score_answer",
]
