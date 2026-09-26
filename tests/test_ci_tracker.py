"""Tests for CI Evaluation Tracker, Ledger, and Trend Visualizations."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nga.evaluation.ci_tracker import (
    generate_ci_markdown_summary,
    generate_svg_trend_chart,
    get_git_metadata,
    get_trend_series,
    load_history,
    record_eval_run,
    save_history,
)
from nga.ui.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_get_git_metadata(monkeypatch):
    """Test git metadata extraction and CI environment variable fallbacks."""
    monkeypatch.setenv("GITHUB_SHA", "abcdef1234567890abcdef1234567890abcdef12")
    monkeypatch.setenv("GITHUB_REF_NAME", "feature/ci-eval")
    monkeypatch.setenv("GITHUB_ACTOR", "ci-bot")

    meta = get_git_metadata()
    assert meta["commit_sha"] == "abcdef1234567890abcdef1234567890abcdef12"
    assert meta["short_sha"] == "abcdef1"
    assert meta["branch"] == "feature/ci-eval"
    assert meta["author"] == "ci-bot"
    assert "timestamp" in meta


def test_record_eval_run_and_delta_tracking():
    """Test recording evaluation runs and calculating deltas against previous commits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hist_file = str(Path(tmpdir) / "history.json")

        report_1 = {
            "run_label": "commit_1_run",
            "summary": {
                "total": 10,
                "passed": 7,
                "pass_rate": 0.70,
                "avg_score": 0.75,
                "avg_latency_s": 2.0,
                "by_category": [{"category": "retrieval", "total": 5, "passed": 4, "pass_rate": 0.8, "avg_score": 0.8, "avg_latency_s": 1.5}],
            },
            "results": [],
        }
        git_meta_1 = {
            "commit_sha": "1111111111111111111111111111111111111111",
            "short_sha": "1111111",
            "branch": "main",
            "author": "Alice",
            "commit_message": "feat: initial commit",
            "timestamp": "2026-08-20T10:00:00Z",
        }

        # 1. Record first run
        entry_1 = record_eval_run(report_1, history_file=hist_file, custom_git_meta=git_meta_1)
        assert entry_1["short_sha"] == "1111111"
        assert entry_1["ci_passed"] is True
        assert entry_1["delta_from_previous"]["pass_rate_delta"] == 0.0

        # 2. Record second improved run
        report_2 = {
            "run_label": "commit_2_run",
            "summary": {
                "total": 10,
                "passed": 9,
                "pass_rate": 0.90,
                "avg_score": 0.92,
                "avg_latency_s": 1.6,
                "by_category": [{"category": "retrieval", "total": 5, "passed": 5, "pass_rate": 1.0, "avg_score": 0.95, "avg_latency_s": 1.2}],
            },
            "results": [],
        }
        git_meta_2 = {
            "commit_sha": "2222222222222222222222222222222222222222",
            "short_sha": "2222222",
            "branch": "main",
            "author": "Bob",
            "commit_message": "fix: improve prompt grounding",
            "timestamp": "2026-08-21T12:00:00Z",
        }

        entry_2 = record_eval_run(report_2, history_file=hist_file, custom_git_meta=git_meta_2)
        assert entry_2["short_sha"] == "2222222"
        assert entry_2["delta_from_previous"]["pass_rate_pct_delta"] == 20.0
        assert entry_2["delta_from_previous"]["score_delta"] == 0.17
        assert entry_2["delta_from_previous"]["previous_commit"] == "1111111"

        # Verify ledger has both
        history = load_history(hist_file)
        assert len(history) == 2


def test_trend_series_and_svg_generation():
    """Test generating time-series arrays and SVG visual chart markup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hist_file = str(Path(tmpdir) / "history.json")

        runs = [
            {
                "commit_sha": "abc0001",
                "short_sha": "abc0001",
                "run_label": "run_1",
                "timestamp": "2026-08-20T10:00:00Z",
                "summary": {"pass_rate": 0.65, "avg_score": 0.70, "avg_latency_s": 2.2, "by_category": []},
            },
            {
                "commit_sha": "abc0002",
                "short_sha": "abc0002",
                "run_label": "run_2",
                "timestamp": "2026-08-21T10:00:00Z",
                "summary": {"pass_rate": 0.85, "avg_score": 0.88, "avg_latency_s": 1.8, "by_category": []},
            },
        ]
        save_history(runs, hist_file)

        # 1. Test trend series
        series = get_trend_series(hist_file)
        assert series["commits"] == ["abc0001", "abc0002"]
        assert series["pass_rates"] == [65.0, 85.0]
        assert series["scores"] == [0.70, 0.88]

        # 2. Test SVG generation
        svg = generate_svg_trend_chart(hist_file)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "abc0001" in svg
        assert "abc0002" in svg
        assert "65.0%" in svg
        assert "85.0%" in svg
        assert "Target (70%)" in svg


def test_generate_ci_markdown_summary():
    """Test generation of GitHub Step Summary Markdown report."""
    entry = {
        "short_sha": "a1b2c3d",
        "branch": "main",
        "author": "Marcus",
        "commit_message": "feat: add stress test suite",
        "timestamp": "2026-08-23T22:00:00Z",
        "ci_passed": True,
        "threshold": 0.70,
        "summary": {
            "total": 85,
            "passed": 72,
            "pass_rate": 0.847,
            "avg_score": 0.892,
            "avg_latency_s": 1.45,
            "by_category": [
                {"category": "retrieval", "total": 30, "passed": 28, "pass_rate": 0.933, "avg_score": 0.95, "avg_latency_s": 1.1},
                {"category": "stress", "total": 10, "passed": 8, "pass_rate": 0.800, "avg_score": 0.85, "avg_latency_s": 1.9},
            ],
        },
        "delta_from_previous": {
            "pass_rate_pct_delta": 4.5,
            "score_delta": 0.05,
            "latency_delta": -0.2,
            "previous_commit": "e5f6g7h",
        },
    }

    md = generate_ci_markdown_summary(entry)
    assert "## 🏭 NGA Manufacturing Assistant — CI Evaluation Report" in md
    assert "84.7%" in md
    assert "+4.5%" in md
    assert "PASSED" in md
    assert "retrieval" in md
    assert "stress" in md
    assert "a1b2c3d" in md


def test_api_trends_and_history_endpoints(client, monkeypatch):
    """Test REST API endpoints for history, trends, and SVG charts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hist_file = str(Path(tmpdir) / "history.json")

        runs = [
            {
                "commit_sha": "commit_111",
                "short_sha": "c111",
                "run_label": "run_a",
                "branch": "main",
                "author": "Alice",
                "timestamp": "2026-08-23T10:00:00Z",
                "ci_passed": True,
                "summary": {"total": 5, "passed": 4, "pass_rate": 0.80, "avg_score": 0.85, "avg_latency_s": 1.5, "by_category": []},
            }
        ]
        save_history(runs, hist_file)

        import nga.ui.server as server_mod
        monkeypatch.setattr(server_mod, "load_history", lambda: load_history(hist_file))
        monkeypatch.setattr(server_mod, "get_trend_series", lambda: get_trend_series(hist_file))
        monkeypatch.setattr(server_mod, "generate_svg_trend_chart", lambda: generate_svg_trend_chart(hist_file))

        # 1. Test GET /api/eval/history
        res_hist = client.get("/api/eval/history")
        assert res_hist.status_code == 200
        data_hist = res_hist.json()
        assert data_hist["total_recorded"] == 1
        assert data_hist["history"][0]["short_sha"] == "c111"

        # 2. Test GET /api/eval/trends
        res_trends = client.get("/api/eval/trends")
        assert res_trends.status_code == 200
        data_trends = res_trends.json()
        assert data_trends["commits"] == ["c111"]
        assert data_trends["pass_rates"] == [80.0]

        # 3. Test GET /api/eval/trends.svg
        res_svg = client.get("/api/eval/trends.svg")
        assert res_svg.status_code == 200
        assert res_svg.headers["content-type"].startswith("image/svg+xml")
        assert "<svg" in res_svg.text


def test_list_evaluation_reports_ignores_history_and_non_dict(caplog):
    """Ensure list_evaluation_reports safely ignores history.json and non-dict JSON files without warnings."""
    import json
    import logging

    from nga.evaluation.eval_runner import (
        get_evaluation_report,
        list_evaluation_reports,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        reports_dir = Path(tmpdir)
        # Create history.json which is a list
        (reports_dir / "history.json").write_text(json.dumps([{"commit_sha": "abc"}]), encoding="utf-8")
        # Create a valid report
        valid_report = {
            "run_label": "run_valid",
            "summary": {"total": 5, "passed": 5, "pass_rate": 1.0, "avg_score": 0.9, "avg_latency_s": 1.0, "by_category": []},
            "results": [],
        }
        (reports_dir / "run_valid.json").write_text(json.dumps(valid_report), encoding="utf-8")
        # Create another non-dict json file (e.g. array)
        (reports_dir / "other_list.json").write_text(json.dumps(["item1", "item2"]), encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            reports = list_evaluation_reports(str(reports_dir))

        # Only run_valid should be parsed
        assert len(reports) == 1
        assert reports[0]["run_label"] == "run_valid"
        # No warning should have been logged for history.json or other_list.json
        assert "Failed to parse eval report" not in caplog.text

        # get_evaluation_report should safely return None for non-dict files
        assert get_evaluation_report("history", str(reports_dir)) is None
        assert get_evaluation_report("run_valid", str(reports_dir)) is not None


def test_record_eval_run_tracks_model_and_token_summary():
    """Verify record_eval_run records model_name, token_summary, and computes tcer_delta."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hist_file = str(Path(tmpdir) / "history.json")

        report_1 = {
            "run_label": "run_model_1",
            "model_name": "anthropic/claude-haiku-4-5",
            "token_summary": {
                "total_tokens": 15000,
                "avg_tcer": 0.42,
                "avg_cost_per_query_usd": 0.0025,
            },
            "summary": {
                "total": 5,
                "passed": 4,
                "pass_rate": 0.80,
                "avg_score": 0.85,
                "avg_latency_s": 2.1,
            },
        }
        git_meta_1 = {
            "commit_sha": "aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111",
            "short_sha": "aaaa111",
            "branch": "main",
            "author": "Alice",
            "commit_message": "baseline model",
            "timestamp": "2026-08-20T10:00:00Z",
        }
        entry_1 = record_eval_run(report_1, history_file=hist_file, custom_git_meta=git_meta_1)
        assert entry_1["model_name"] == "anthropic/claude-haiku-4-5"
        assert entry_1["token_summary"]["avg_tcer"] == 0.42

        report_2 = {
            "run_label": "run_model_2",
            "model_name": "anthropic/claude-sonnet-4-5",
            "token_summary": {
                "total_tokens": 12000,
                "avg_tcer": 0.35,
                "avg_cost_per_query_usd": 0.0018,
            },
            "summary": {
                "total": 5,
                "passed": 5,
                "pass_rate": 1.0,
                "avg_score": 0.95,
                "avg_latency_s": 1.8,
            },
        }
        git_meta_2 = {
            "commit_sha": "bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222",
            "short_sha": "bbbb222",
            "branch": "main",
            "author": "Bob",
            "commit_message": "candidate model with Jev",
            "timestamp": "2026-08-21T10:00:00Z",
        }
        entry_2 = record_eval_run(report_2, history_file=hist_file, custom_git_meta=git_meta_2)
        assert entry_2["model_name"] == "anthropic/claude-sonnet-4-5"
        assert entry_2["token_summary"]["avg_tcer"] == 0.35
        # TCER delta should be 0.35 - 0.42 = -0.07
        assert entry_2["delta_from_previous"]["tcer_delta"] == -0.07
        assert entry_2["delta_from_previous"]["cost_per_query_delta"] == -0.0007

