"""CLI script to run the Legacy NGA Jev Harness and DeepEval Harness side-by-side.

Usage:
    # Default: pure Jev-only execution (`eval_mode="system_one"`, `model="jev-1.13.0"`)
    uv run python scripts/side_by_side_eval.py --limit 6

    # Opt-in hybrid mode (LLM extracts claims & writes reasons; Jev decides verdicts)
    uv run python scripts/side_by_side_eval.py --eval-mode hybrid --limit 6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from nga.evaluation.deepeval_adapter import to_deep_eval_test_case
from nga.evaluation.deepeval_metrics import (
    PINNED_JEV_MODEL,
    VALID_EVAL_MODES,
    evaluate_test_case_with_deepeval,
)
from nga.evaluation.judge import (
    JevEvaluationResult,
    evaluate_with_jev,
    is_api_error,
)
from nga.evaluation.side_by_side import (
    DEFAULT_CONFLICT_SCENARIOS,
    format_side_by_side_markdown,
    run_side_by_side_comparison,
    save_side_by_side_report,
)
from nga.tracing import configure_tracing

load_dotenv()

logger = logging.getLogger("nga.side_by_side_eval")


def main(
    limit: int | None = None,
    eval_mode: str = "system_one",
    system_one_model: Any = PINNED_JEV_MODEL,
    llm_model: Any = None,
    legacy_evaluator: Callable[[str, str], JevEvaluationResult] | None = None,
    include_contradictions: bool = True,
    output_dir: Path | str | None = None,
) -> bool:
    parser = argparse.ArgumentParser(
        description="Run Legacy NGA Jev Harness and DeepEval Harness side-by-side."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of benchmark questions to evaluate (default: all 85).",
    )
    parser.add_argument(
        "--eval-mode",
        choices=VALID_EVAL_MODES,
        default="system_one",
        help="DeepEval RAG metric execution mode: 'system_one' (default, 100%% Jev-only) or 'hybrid' (opt-in LLM extraction + Jev verdicts).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=PINNED_JEV_MODEL,
        help=f"Pinned System One Jev model name (default: {PINNED_JEV_MODEL}).",
    )
    parser.add_argument(
        "--no-contradictions",
        action="store_true",
        help="Skip cross-document contradiction scenarios.",
    )
    args, _ = parser.parse_known_args()

    eff_limit = limit if limit is not None else args.limit
    eff_eval_mode = eval_mode if eval_mode != "system_one" else args.eval_mode
    eff_model = system_one_model if system_one_model != PINNED_JEV_MODEL else args.model
    eff_contradictions = include_contradictions and not args.no_contradictions

    configure_tracing()

    print("=" * 80)
    print("NGA MANUFACTURING ASSISTANT — SIDE-BY-SIDE HARNESS COMPARISON (LEGACY vs DEEPEVAL)")
    print(f"Pinned Model: {eff_model} | Eval Mode: {eff_eval_mode}")
    print("=" * 80)

    questions_path = Path(__file__).resolve().parents[1] / "eval-questions" / "questions.json"
    with open(questions_path, encoding="utf-8") as f:
        data = json.load(f)

    questions: list[dict[str, Any]] = data["questions"]
    if eff_limit and eff_limit > 0:
        questions = questions[:eff_limit]

    # 1. Validate required API keys unless custom mock evaluators/models are injected
    custom_models_injected = (
        legacy_evaluator is not None and not isinstance(eff_model, str)
    )
    if not custom_models_injected:
        missing_keys: list[str] = []
        typesafe_key = (os.getenv("TYPESAFE_API_KEY") or "").strip()
        if not typesafe_key or typesafe_key.lower().startswith("your-"):
            missing_keys.append("TYPESAFE_API_KEY")

        if eff_eval_mode in ("hybrid", "llm") and llm_model is None:
            openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
            openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
            if not openrouter_key and not openai_key:
                missing_keys.append("OPENROUTER_API_KEY (or OPENAI_API_KEY) for hybrid/llm mode")

        if missing_keys:
            skip_msg = (
                f"Missing required API key(s) for side-by-side comparison: {', '.join(missing_keys)}.\n"
                f"Skipping side-by-side live evaluation."
            )
            print(skip_msg)
            logger.warning(skip_msg)
            if "pytest" in sys.modules:
                import pytest

                pytest.skip(skip_msg)
            return False

    # 2. Preflight probe on a single test case
    probe_q = {
        "id": "PROBE-1",
        "category": "sop_lookup",
        "tier": "L1",
        "question": "What is the lug nut torque at Station 144?",
        "expected_answer": "105 Nm per SOP-OPR-101",
        "source_docs": ["SOP-OPR-101"],
    }
    probe_cand = "Based on SOP-OPR-101, the lug nut torque at Station 144 is 105 Nm."
    leg_fn = legacy_evaluator or (lambda c, g: evaluate_with_jev(c, g))

    try:
        _ = leg_fn(probe_cand, probe_q["expected_answer"])
        probe_tc = to_deep_eval_test_case(probe_q, actual_output=probe_cand)
        _ = evaluate_test_case_with_deepeval(
            test_case=probe_tc,
            eval_mode=eff_eval_mode,
            system_one_model=eff_model,
            llm_model=llm_model,
        )
    except Exception as exc:
        is_err, status_code, msg = is_api_error(exc)
        if is_err:
            skip_msg = (
                f"Side-by-side preflight probe failed due to upstream API error (status={status_code}): {msg}\n"
                f"Skipping side-by-side comparison."
            )
            print(skip_msg)
            logger.warning(skip_msg)
            if "pytest" in sys.modules:
                import pytest

                pytest.skip(skip_msg)
            return False
        raise

    summary = run_side_by_side_comparison(
        questions=questions,
        conflict_scenarios=DEFAULT_CONFLICT_SCENARIOS if eff_contradictions else [],
        eval_mode=eff_eval_mode,
        system_one_model=eff_model,
        llm_model=llm_model,
        legacy_evaluator=legacy_evaluator,
    )

    md_path, json_path = save_side_by_side_report(summary, output_dir=output_dir)
    print(format_side_by_side_markdown(summary))
    print(f"Saved Markdown report to: {md_path}")
    print(f"Saved JSON report to:     {json_path}")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 0)
