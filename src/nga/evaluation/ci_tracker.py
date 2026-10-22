"""CI evaluation tracker, historical ledger, and trend visualization engine.

Backward-compatibility shim re-exporting from enterprise_agent.evaluation.ci_tracker.
"""

from enterprise_agent.evaluation.ci_tracker import (
    DEFAULT_HISTORY_PATH,
    DEFAULT_PASS_THRESHOLD,
    DEFAULT_REPORTS_DIR,
    generate_ci_markdown_summary,
    generate_svg_trend_chart,
    get_git_metadata,
    get_trend_series,
    load_history,
    main,
    record_eval_run,
    save_history,
)

__all__ = [
    "DEFAULT_HISTORY_PATH",
    "DEFAULT_PASS_THRESHOLD",
    "DEFAULT_REPORTS_DIR",
    "generate_ci_markdown_summary",
    "generate_svg_trend_chart",
    "get_git_metadata",
    "get_trend_series",
    "load_history",
    "main",
    "record_eval_run",
    "save_history",
]

if __name__ == "__main__":
    main()
