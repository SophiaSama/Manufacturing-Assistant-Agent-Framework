"""Unit tests for Jev Choice-based reranking."""

from unittest.mock import MagicMock, patch

import pytest

from enterprise_agent.retrieval.reranker import (
    get_typesafe_api_key,
    is_rerank_enabled,
    rerank_documents,
)


def test_typesafe_api_key_resolution(monkeypatch):
    """Test resolution of TYPESAFE_API_KEY."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    assert get_typesafe_api_key() is None
    assert is_rerank_enabled() is False

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_test_key_123")
    assert get_typesafe_api_key() == "ts_test_key_123"
    assert is_rerank_enabled() is True

    # Placeholder values are rejected
    monkeypatch.setenv("TYPESAFE_API_KEY", "your-key-here")
    assert get_typesafe_api_key() is None


def test_rerank_empty_or_single():
    """Test handling of empty and single-element document lists."""
    assert rerank_documents("query", []) == []
    single_doc = [{"text": "single passage", "doc_id": "D1"}]
    assert rerank_documents("query", single_doc) == single_doc


def test_rerank_without_api_key(monkeypatch):
    """Test graceful fallback when no TypeSafe API key is configured."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("RERANK_ENABLED", raising=False)

    docs = [
        {"text": "passage 1", "doc_id": "D1"},
        {"text": "passage 2", "doc_id": "D2"},
        {"text": "passage 3", "doc_id": "D3"},
    ]
    result = rerank_documents("query", docs, top_n=2)
    assert len(result) == 2
    assert result[0]["doc_id"] == "D1"
    assert result[1]["doc_id"] == "D2"


def test_rerank_with_mocked_jev(monkeypatch):
    """Test rerank execution, score attachment, and ordering with mock Jev."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_mock_key")
    monkeypatch.delenv("RERANK_MODEL", raising=False)

    docs = [
        {"text": "Low relevance about fruit", "doc_id": "D1"},
        {"text": "Exact match for wheel torque specifications", "doc_id": "D2"},
        {"text": "Moderate relevance about automotive tools", "doc_id": "D3"},
    ]

    # Mock TypeSafe response with probability distribution
    mock_answer = MagicMock()
    mock_answer.probabilities = {
        "doc_0": 0.05,
        "doc_1": 0.72,
        "doc_2": 0.23,
    }

    mock_response = MagicMock()
    mock_response.answers = {"best_match": mock_answer}

    with patch("enterprise_agent.retrieval.reranker.TypeSafeClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.system_one.return_value = mock_response
        MockClient.return_value = mock_instance

        reranked = rerank_documents(
            query="wheel torque",
            documents=docs,
            top_n=2,
        )

        # Verify TypeSafeClient was created with the key
        MockClient.assert_called_once_with(api_key="ts_mock_key")

        # Verify system_one was called with Choice
        mock_instance.system_one.assert_called_once()
        call_kwargs = mock_instance.system_one.call_args[1]
        assert call_kwargs["state"]["query"] == "wheel torque"
        assert "candidates" in call_kwargs["state"]
        assert "best_match" in call_kwargs["questions"]
        assert call_kwargs["model"] == "jev-1.12"

        assert len(reranked) == 2
        # Best match (highest probability) should be first
        assert reranked[0]["doc_id"] == "D2"
        assert reranked[0]["rerank_score"] == pytest.approx(0.72)

        assert reranked[1]["doc_id"] == "D3"
        assert reranked[1]["rerank_score"] == pytest.approx(0.23)


def test_rerank_failure_fallback(monkeypatch):
    """Test that exceptions from Jev fall back gracefully to raw top-n."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_mock_key")

    docs = [
        {"text": "doc 1", "doc_id": "D1"},
        {"text": "doc 2", "doc_id": "D2"},
        {"text": "doc 3", "doc_id": "D3"},
    ]

    with patch("enterprise_agent.retrieval.reranker.TypeSafeClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.system_one.side_effect = RuntimeError("TypeSafe API Timeout")
        MockClient.return_value = mock_instance

        result = rerank_documents("query", docs, top_n=2)
        assert len(result) == 2
        assert result[0]["doc_id"] == "D1"
        assert result[1]["doc_id"] == "D2"
        assert "rerank_score" not in result[0]
