"""DeepEval RAG metrics and custom JevEval manufacturing criteria for NGA evaluation.

Backward-compatibility shim re-exporting from enterprise_agent.evaluation.deepeval_metrics.
"""

from enterprise_agent.evaluation.deepeval_metrics import (
    PINNED_JEV_MODEL,
    VALID_EVAL_MODES,
    DeepEvalCaseResult,
    SafeOpenRouterModel,
    a_evaluate_test_case_with_deepeval,
    build_rag_metrics,
    evaluate_test_case_with_deepeval,
    extract_manufacturing_breakdown,
    make_built_in_rag_metrics,
    make_contradiction_jev_eval_metric,
    make_domain_jev_eval_metric,
    make_manufacturing_jev_eval_metric,
    resolve_deepeval_llm_model,
)

__all__ = [
    "PINNED_JEV_MODEL",
    "VALID_EVAL_MODES",
    "SafeOpenRouterModel",
    "DeepEvalCaseResult",
    "make_manufacturing_jev_eval_metric",
    "make_domain_jev_eval_metric",
    "make_contradiction_jev_eval_metric",
    "make_built_in_rag_metrics",
    "build_rag_metrics",
    "resolve_deepeval_llm_model",
    "extract_manufacturing_breakdown",
    "evaluate_test_case_with_deepeval",
    "a_evaluate_test_case_with_deepeval",
]
