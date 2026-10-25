"""DeepEval test case conversion layer for NGA Manufacturing Assistant benchmarks.

Backward-compatibility shim re-exporting from enterprise_agent.evaluation.deepeval_adapter.
"""

from enterprise_agent.evaluation.deepeval_adapter import (
    DeepEvalAdapter,
    build_deepeval_tool_calls,
    build_expected_tool_calls,
    conflict_scenario_to_test_cases,
    extract_retrieval_context,
    to_deep_eval_test_case,
)

__all__ = [
    "DeepEvalAdapter",
    "build_deepeval_tool_calls",
    "build_expected_tool_calls",
    "conflict_scenario_to_test_cases",
    "extract_retrieval_context",
    "to_deep_eval_test_case",
]
