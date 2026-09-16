"""Tests for Multi-Model Evaluation Arena & Pareto calculations."""

from nga.evaluation.multi_model_eval import (
    calculate_query_cost,
    compute_pareto_frontier,
    generate_multi_model_analysis,
)


def test_calculate_query_cost():
    # Test Claude 3.5 Sonnet: $3.00/M in, $15.00/M out
    # 1,000 prompt tokens = $0.003, 500 completion tokens = $0.0075 -> $0.0105
    cost = calculate_query_cost("anthropic/claude-3.5-sonnet", 1000, 500)
    assert 0.010 <= cost <= 0.011

    # Test DeepSeek V3: $0.14/M in, $0.28/M out
    ds_cost = calculate_query_cost("deepseek/deepseek-chat", 1000, 500)
    assert ds_cost < 0.001


def test_pareto_frontier_identification():
    models = [
        {"model_slug": "cheap-fast", "avg_score": 0.85, "cost_per_1k": 0.50},
        {"model_slug": "expensive-accurate", "avg_score": 0.95, "cost_per_1k": 10.00},
        {"model_slug": "dominated-bad", "avg_score": 0.80, "cost_per_1k": 12.00},  # worse score, higher cost!
    ]

    frontier = compute_pareto_frontier(models)
    assert "cheap-fast" in frontier
    assert "expensive-accurate" in frontier
    assert "dominated-bad" not in frontier


def test_multi_model_analysis_generation(tmp_path):
    mock_runs = {
        "anthropic/claude-3.5-sonnet": {
            "summary": {"avg_score": 0.92, "avg_latency_s": 5.0, "avg_prompt_tokens": 1200, "avg_completion_tokens": 300},
            "results": [{"id": "R1", "question": "What is torque?", "answer": "105 Nm", "passed": True, "overall_score": 1.0, "latency_s": 4.5}],
        },
        "deepseek/deepseek-chat": {
            "summary": {"avg_score": 0.88, "avg_latency_s": 6.5, "avg_prompt_tokens": 1300, "avg_completion_tokens": 320},
            "results": [{"id": "R1", "question": "What is torque?", "answer": "105 Nm", "passed": True, "overall_score": 0.9, "latency_s": 6.0}],
        },
    }

    analysis = generate_multi_model_analysis(mock_runs, reports_dir=str(tmp_path))
    assert len(analysis["models"]) == 2
    assert len(analysis["pareto_efficient_models"]) > 0
    assert len(analysis["question_matrix"]) == 1
    assert (tmp_path / "multi_model_benchmark.json").exists()
