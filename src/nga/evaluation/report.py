"""Markdown and HTML report generation for NGA evaluation runs.

Backward-compatibility shim re-exporting from enterprise_agent.evaluation.report.
"""

from enterprise_agent.evaluation.report import (
    build_summary,
    format_report_label,
    generate_markdown_report,
    sanitize_model_name,
    save_report,
)

__all__ = [
    "sanitize_model_name",
    "format_report_label",
    "build_summary",
    "generate_markdown_report",
    "save_report",
]
