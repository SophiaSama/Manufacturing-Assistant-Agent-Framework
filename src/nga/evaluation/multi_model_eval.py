"""Backward compatibility shim for nga.evaluation.multi_model_eval.

Re-exports all functionality from enterprise_agent.evaluation.multi_model_eval.
"""

from enterprise_agent.evaluation.multi_model_eval import (
    MODEL_REGISTRY,
    calculate_query_cost,
    compute_pareto_frontier,
    generate_multi_model_analysis,
)

__all__ = [
    "MODEL_REGISTRY",
    "calculate_query_cost",
    "compute_pareto_frontier",
    "generate_multi_model_analysis",
]
