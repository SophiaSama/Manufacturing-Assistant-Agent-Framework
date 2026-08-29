"""CI evaluation tracker, historical ledger, and trend visualization engine.

Tracks evaluation results over commits/time, detects performance regressions in CI,
and generates trend visualizations (SVG / HTML / JSON) and GitHub Actions summaries.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("nga.evaluation.ci_tracker")

DEFAULT_HISTORY_PATH = "reports/eval/history.json"
DEFAULT_REPORTS_DIR = "reports/eval"
DEFAULT_PASS_THRESHOLD = 0.70


def get_git_metadata() -> dict[str, str]:
    """Retrieve git metadata (commit hash, branch, author, commit message, timestamp).

    Supports standard local git repositories as well as CI environment variables
    (GitHub Actions, GitLab CI, CircleCI, etc.).
    """
    # 1. Check CI environment variables first
    commit_sha = (
        os.getenv("GITHUB_SHA")
        or os.getenv("CI_COMMIT_SHA")
        or os.getenv("GIT_COMMIT")
        or ""
    )
    branch = (
        os.getenv("GITHUB_REF_NAME")
        or os.getenv("CI_COMMIT_REF_NAME")
        or os.getenv("GIT_BRANCH")
        or ""
    )
    author = (
        os.getenv("GITHUB_ACTOR")
        or os.getenv("CI_COMMIT_AUTHOR")
        or os.getenv("GIT_AUTHOR_NAME")
        or ""
    )

    # 2. Try git CLI if in a git repo
    try:
        if not commit_sha:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            commit_sha = res.stdout.strip()

        if not branch:
            res = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            branch = res.stdout.strip()

        commit_msg = ""
        try:
            res = subprocess.run(
                ["git", "log", "-1", "--pretty=%B"],
                capture_output=True,
                text=True,
                check=True,
            )
            commit_msg = res.stdout.strip().split("\n")[0]
        except Exception:
            commit_msg = "Automated evaluation run"

        if not author:
            try:
                res = subprocess.run(
                    ["git", "log", "-1", "--pretty=%an"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                author = res.stdout.strip()
            except Exception:
                author = "NGA Assistant CI"

    except Exception as exc:
        logger.debug("Git metadata extraction fallback: %s", exc)
        commit_sha = commit_sha or "local-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        branch = branch or "main"
        author = author or "Local User"
        commit_msg = "Evaluation run"

    short_sha = commit_sha[:7] if len(commit_sha) >= 7 else commit_sha
    return {
        "commit_sha": commit_sha,
        "short_sha": short_sha,
        "branch": branch,
        "author": author,
        "commit_message": commit_msg,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def load_history(history_file: str = DEFAULT_HISTORY_PATH) -> list[dict[str, Any]]:
    """Load historical evaluation ledger."""
    path = Path(history_file)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "runs" in data:
                return data["runs"]
            return []
    except Exception as exc:
        logger.warning("Failed to parse history ledger %s: %s", path, exc)
        return []


def save_history(history: list[dict[str, Any]], history_file: str = DEFAULT_HISTORY_PATH) -> Path:
    """Persist historical evaluation ledger."""
    path = Path(history_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return path


def record_eval_run(
    report_data: dict[str, Any],
    history_file: str = DEFAULT_HISTORY_PATH,
    custom_git_meta: dict[str, str] | None = None,
    threshold: float = DEFAULT_PASS_THRESHOLD,
) -> dict[str, Any]:
    """Record an evaluation run into the persistent history ledger with commit tracking and deltas."""
    history = load_history(history_file)
    git_meta = custom_git_meta or get_git_metadata()

    summary = report_data.get("summary", {})
    run_label = report_data.get("run_label") or git_meta.get("short_sha")

    pass_rate = float(summary.get("pass_rate", 0.0))
    avg_score = float(summary.get("avg_score", 0.0))
    avg_latency = float(summary.get("avg_latency_s", 0.0))
    total = int(summary.get("total", len(report_data.get("results", []))))
    passed = int(summary.get("passed", 0))

    # Calculate delta against the previous recorded run
    delta: dict[str, Any] = {
        "pass_rate_delta": 0.0,
        "score_delta": 0.0,
        "latency_delta": 0.0,
        "previous_commit": None,
    }
    if history:
        prev = history[-1]
        prev_sum = prev.get("summary", {})
        prev_pr = float(prev_sum.get("pass_rate", 0.0))
        prev_score = float(prev_sum.get("avg_score", 0.0))
        prev_lat = float(prev_sum.get("avg_latency_s", 0.0))

        delta = {
            "pass_rate_delta": round(pass_rate - prev_pr, 3),
            "pass_rate_pct_delta": round((pass_rate - prev_pr) * 100, 1),
            "score_delta": round(avg_score - prev_score, 3),
            "latency_delta": round(avg_latency - prev_lat, 2),
            "previous_commit": prev.get("commit_sha", "")[:7],
            "previous_run_label": prev.get("run_label", ""),
        }

    entry = {
        "id": f"eval_{git_meta['short_sha']}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "run_label": run_label,
        "timestamp": git_meta["timestamp"],
        "commit_sha": git_meta["commit_sha"],
        "short_sha": git_meta["short_sha"],
        "branch": git_meta["branch"],
        "author": git_meta["author"],
        "commit_message": git_meta["commit_message"],
        "ci_passed": pass_rate >= threshold,
        "threshold": threshold,
        "summary": {
            "total": total,
            "passed": passed,
            "pass_rate": pass_rate,
            "avg_score": avg_score,
            "avg_latency_s": avg_latency,
            "by_category": summary.get("by_category", []),
        },
        "delta_from_previous": delta,
    }

    history.append(entry)
    save_history(history, history_file)
    logger.info("Recorded evaluation run %s (Commit: %s, Pass Rate: %.1f%%)", run_label, git_meta['short_sha'], pass_rate * 100)
    return entry


def get_trend_series(history_file: str = DEFAULT_HISTORY_PATH) -> dict[str, Any]:
    """Format historical ledger into time-series data suitable for visualization charts."""
    history = load_history(history_file)
    if not history:
        return {
            "commits": [],
            "labels": [],
            "timestamps": [],
            "pass_rates": [],
            "scores": [],
            "latencies": [],
            "categories": {},
            "runs": [],
        }

    commits: list[str] = []
    labels: list[str] = []
    timestamps: list[str] = []
    pass_rates: list[float] = []
    scores: list[float] = []
    latencies: list[float] = []
    category_trends: dict[str, list[dict[str, Any]]] = {}

    for run in history:
        short_sha = run.get("short_sha") or run.get("commit_sha", "")[:7]
        label = run.get("run_label", short_sha)
        ts = run.get("timestamp", "")
        sum_data = run.get("summary", {})

        commits.append(short_sha)
        labels.append(label)
        timestamps.append(ts)
        pass_rates.append(round(float(sum_data.get("pass_rate", 0.0)) * 100, 1))
        scores.append(round(float(sum_data.get("avg_score", 0.0)), 3))
        latencies.append(round(float(sum_data.get("avg_latency_s", 0.0)), 2))

        for cat in sum_data.get("by_category", []):
            cat_name = cat.get("category", "general")
            if cat_name not in category_trends:
                category_trends[cat_name] = []
            category_trends[cat_name].append({
                "commit": short_sha,
                "pass_rate": round(float(cat.get("pass_rate", 0.0)) * 100, 1),
                "score": round(float(cat.get("avg_score", 0.0)), 3),
            })

    return {
        "commits": commits,
        "labels": labels,
        "timestamps": timestamps,
        "pass_rates": pass_rates,
        "scores": scores,
        "latencies": latencies,
        "categories": category_trends,
        "runs": history,
    }


def generate_svg_trend_chart(history_file: str = DEFAULT_HISTORY_PATH, width: int = 760, height: int = 280) -> str:
    """Generate a clean, standalone SVG line chart of Pass Rate (%) across commits."""
    trends = get_trend_series(history_file)
    rates = trends.get("pass_rates", [])
    commits = trends.get("commits", [])

    if not rates:
        return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><text x="50%" y="50%" fill="#94a3b8" text-anchor="middle">No evaluation history available yet</text></svg>'

    padding_left = 60
    padding_right = 30
    padding_top = 40
    padding_bottom = 50

    chart_w = width - padding_left - padding_right
    chart_h = height - padding_top - padding_bottom

    n = len(rates)
    points: list[tuple[float, float]] = []
    for i, r in enumerate(rates):
        x = padding_left + (i * chart_w / max(n - 1, 1))
        # Y axis: 0% to 100%
        y = padding_top + chart_h - ((r / 100.0) * chart_h)
        points.append((round(x, 1), round(y, 1)))

    path_d = " ".join([f"{'M' if i == 0 else 'L'} {pt[0]} {pt[1]}" for i, pt in enumerate(points)])
    area_d = f"{path_d} L {points[-1][0]} {padding_top + chart_h} L {points[0][0]} {padding_top + chart_h} Z"

    # Threshold Y (70%)
    threshold_y = round(padding_top + chart_h - (0.70 * chart_h), 1)

    svg_lines = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:#0f172a; border-radius:10px; font-family: -apple-system, Inter, sans-serif;">',
        '<!-- Grid & Axes -->',
        f'<line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{padding_top + chart_h}" stroke="#334155" stroke-width="1"/>',
        f'<line x1="{padding_left}" y1="{padding_top + chart_h}" x2="{width - padding_right}" y2="{padding_top + chart_h}" stroke="#334155" stroke-width="1"/>',
        '<!-- Y Grid Lines -->',
        f'<line x1="{padding_left}" y1="{padding_top}" x2="{width - padding_right}" y2="{padding_top}" stroke="#1e293b" stroke-dasharray="3,3"/>',
        f'<text x="{padding_left - 10}" y="{padding_top + 4}" fill="#64748b" font-size="11" text-anchor="end">100%</text>',
        f'<line x1="{padding_left}" y1="{threshold_y}" x2="{width - padding_right}" y2="{threshold_y}" stroke="#f59e0b" stroke-dasharray="4,4" stroke-width="1.2"/>',
        f'<text x="{width - padding_right}" y="{threshold_y - 6}" fill="#f59e0b" font-size="10" text-anchor="end">Target (70%)</text>',
        f'<text x="{padding_left - 10}" y="{threshold_y + 4}" fill="#f59e0b" font-size="11" text-anchor="end">70%</text>',
        f'<line x1="{padding_left}" y1="{padding_top + chart_h}" x2="{width - padding_right}" y2="{padding_top + chart_h}" stroke="#1e293b"/>',
        f'<text x="{padding_left - 10}" y="{padding_top + chart_h + 4}" fill="#64748b" font-size="11" text-anchor="end">0%</text>',
        '<!-- Gradient & Area -->',
        '<defs>',
        '  <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">',
        '    <stop offset="0%" stop-color="#0284c7" stop-opacity="0.35"/>',
        '    <stop offset="100%" stop-color="#0284c7" stop-opacity="0.0"/>',
        '  </linearGradient>',
        '</defs>',
        f'<path d="{area_d}" fill="url(#areaGrad)"/>',
        f'<path d="{path_d}" fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>',
        '<!-- Data Points & Labels -->',
    ]

    for i, (pt, r, c) in enumerate(zip(points, rates, commits)):
        color = "#10b981" if r >= 70 else "#ef4444"
        svg_lines.append(f'<circle cx="{pt[0]}" cy="{pt[1]}" r="4.5" fill="{color}" stroke="#0f172a" stroke-width="2"/>')
        svg_lines.append(f'<text x="{pt[0]}" y="{pt[1] - 8}" fill="#f8fafc" font-size="11" font-weight="bold" text-anchor="middle">{r}%</text>')
        svg_lines.append(f'<text x="{pt[0]}" y="{padding_top + chart_h + 20}" fill="#94a3b8" font-size="10" font-family="monospace" text-anchor="middle">{c}</text>')

    svg_lines.append(f'<text x="{padding_left}" y="22" fill="#f8fafc" font-size="13" font-weight="bold">📈 Evaluation Pass Rate Across Commits</text>')
    svg_lines.append('</svg>')
    return "\n".join(svg_lines)


def generate_ci_markdown_summary(
    current_entry: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a GitHub Step Summary Markdown report for CI runs."""
    summary = current_entry.get("summary", {})
    delta = current_entry.get("delta_from_previous", {})
    pr_pct = float(summary.get("pass_rate", 0.0)) * 100
    avg_score = float(summary.get("avg_score", 0.0))
    avg_lat = float(summary.get("avg_latency_s", 0.0))
    threshold = float(current_entry.get("threshold", 0.70)) * 100
    ci_passed = current_entry.get("ci_passed", pr_pct >= threshold)

    status_badge = "✅ **PASSED**" if ci_passed else "❌ **FAILED (Below Threshold)**"
    pr_delta_str = f" ({delta.get('pass_rate_pct_delta', 0):+0.1f}%)" if delta.get("pass_rate_pct_delta") is not None else ""

    lines = [
        "## 🏭 NGA Manufacturing Assistant — CI Evaluation Report",
        "",
        f"| Metric | Value | Delta vs Previous ({delta.get('previous_commit') or 'Initial'}) | Status |",
        "|---|---|---|---|",
        f"| **Overall Pass Rate** | **{pr_pct:.1f}%** | `{pr_delta_str or '—'}` | {status_badge} |",
        f"| **Target Threshold** | **{threshold:.1f}%** | — | {'✅ Met' if pr_pct >= threshold else '❌ Unmet'} |",
        f"| **Average Score** | **{avg_score:.3f}** | `{delta.get('score_delta', 0):+0.3f}` | — |",
        f"| **Average Latency** | **{avg_lat:.2f}s** | `{delta.get('latency_delta', 0):+0.2f}s` | — |",
        f"| **Questions Evaluated** | **{summary.get('total', 0)}** ({summary.get('passed', 0)} passed) | — | — |",
        "",
        "### 📦 Commit & Build Context",
        f"- **Commit**: `{current_entry.get('short_sha')}` ({current_entry.get('commit_message')})",
        f"- **Branch**: `{current_entry.get('branch')}`",
        f"- **Author**: {current_entry.get('author')}",
        f"- **Timestamp**: {current_entry.get('timestamp')}",
        "",
        "### 📊 Breakdown by Test Category",
        "| Category | Total | Passed | Pass Rate | Avg Score | Avg Latency |",
        "|---|---|---|---|---|---|",
    ]

    for cat in summary.get("by_category", []):
        cat_pr = float(cat.get("pass_rate", 0.0)) * 100
        cat_badge = "✅" if cat_pr >= threshold else "⚠️"
        lines.append(
            f"| **{cat.get('category')}** | {cat.get('total')} | {cat.get('passed')} | {cat_badge} {cat_pr:.1f}% | {cat.get('avg_score', 0):.3f} | {cat.get('avg_latency_s', 0):.2f}s |"
        )

    lines.append("")
    lines.append("> *Note: Evaluation data recorded into `reports/eval/history.json` for visual tracking over time/commits.*")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="NGA CI Evaluation Tracker")
    parser.add_argument("--report-file", help="Path to evaluation JSON report file to record")
    parser.add_argument("--run-label", help="Run label to find in reports/eval/")
    parser.add_argument("--history-file", default=DEFAULT_HISTORY_PATH, help="Path to history.json ledger")
    parser.add_argument("--threshold", type=float, default=DEFAULT_PASS_THRESHOLD, help="Pass rate threshold (0.0 to 1.0)")
    parser.add_argument("--generate-svg", help="Output path for SVG trend chart")
    parser.add_argument("--github-summary", action="store_true", help="Write summary to $GITHUB_STEP_SUMMARY")
    parser.add_argument("--fail-on-regression", action="store_true", help="Exit code 1 if run fails threshold")

    args = parser.parse_args(argv)

    # 1. Determine report JSON
    report_data = None
    if args.report_file:
        path = Path(args.report_file)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                report_data = json.load(f)
    elif args.run_label:
        path = Path(DEFAULT_REPORTS_DIR) / f"{args.run_label}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                report_data = json.load(f)

    # 2. Record if report provided
    entry = None
    if report_data:
        entry = record_eval_run(
            report_data=report_data,
            history_file=args.history_file,
            threshold=args.threshold,
        )
    else:
        # Load latest from history
        hist = load_history(args.history_file)
        if hist:
            entry = hist[-1]

    if not entry:
        print("No evaluation report found to track.")
        sys.exit(0)

    # 3. Generate SVG if requested
    if args.generate_svg:
        svg_content = generate_svg_trend_chart(args.history_file)
        out_svg = Path(args.generate_svg)
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        out_svg.write_text(svg_content, encoding="utf-8")
        print(f"Generated SVG trend chart: {out_svg}")

    # 4. Generate GitHub Actions summary
    summary_md = generate_ci_markdown_summary(entry)
    step_summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if args.github_summary or step_summary_file:
        if step_summary_file:
            with open(step_summary_file, "a", encoding="utf-8") as f:
                f.write(summary_md + "\n\n")
            print("Wrote evaluation report to GITHUB_STEP_SUMMARY")
        else:
            print("\n" + summary_md + "\n")

    # 5. Check regression / threshold
    if args.fail_on_regression and not entry.get("ci_passed", True):
        pr = float(entry.get("summary", {}).get("pass_rate", 0.0)) * 100
        print(f"\n❌ CI FAILED: Evaluation pass rate {pr:.1f}% is below threshold {args.threshold * 100:.1f}%")
        sys.exit(1)


if __name__ == "__main__":
    main()
