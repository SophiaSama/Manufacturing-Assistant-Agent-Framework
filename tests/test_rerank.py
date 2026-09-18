"""Unit tests for LiteLLM cross-encoder reranking with Hugging Face."""

import os
from unittest.mock import MagicMock, patch
import pytest

from enterprise_agent.retrieval.reranker import (
    DEFAULT_RERANK_MODEL,
    get_hf_token,
    get_rerank_model,
    is_rerank_enabled,
    rerank_documents,
)


def test_token_and_model_resolution(monkeypatch):
    """Test resolution of HF_TOKEN, HUGGINGFACE_API_KEY, and RERANK_MODEL."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    assert get_hf_token() is None
    assert is_rerank_enabled() is False

    monkeypatch.setenv("HF_TOKEN", "hf_test_token_123")
    assert get_hf_token() == "hf_test_token_123"
    assert is_rerank_enabled() is True

    # Test fallback to HUGGINGFACE_API_KEY
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_alt_key_456")
    assert get_hf_token() == "hf_alt_key_456"

    # Test model prefix
    monkeypatch.setenv("RERANK_MODEL", "BAAI/bge-reranker-large")
    assert get_rerank_model() == "huggingface/BAAI/bge-reranker-large"


def test_rerank_empty_or_single():
    """Test handling of empty and single-element document lists."""
    assert rerank_documents("query", []) == []
    single_doc = [{"text": "single passage", "doc_id": "D1"}]
    assert rerank_documents("query", single_doc) == single_doc


def test_rerank_without_token(monkeypatch):
    """Test graceful fallback when no HF token is configured."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)

    docs = [
        {"text": "passage 1", "doc_id": "D1"},
        {"text": "passage 2", "doc_id": "D2"},
        {"text": "passage 3", "doc_id": "D3"},
    ]
    result = rerank_documents("query", docs, top_n=2)
    assert len(result) == 2
    assert result[0]["doc_id"] == "D1"
    assert result[1]["doc_id"] == "D2"


def test_rerank_with_mocked_litellm(monkeypatch):
    """Test rerank execution, score attachment, and ordering with mock litellm."""
    monkeypatch.setenv("HF_TOKEN", "hf_mock_token")
    monkeypatch.setenv("RERANK_MODEL", "huggingface/BAAI/bge-reranker-base")

    docs = [
        {"text": "Low relevance about fruit", "doc_id": "D1"},
        {"text": "Exact match for wheel torque specifications", "doc_id": "D2"},
        {"text": "Moderate relevance about automotive tools", "doc_id": "D3"},
    ]

    # Mock LiteLLM rerank response
    mock_item_0 = MagicMock()
    mock_item_0.index = 1
    mock_item_0.relevance_score = 0.98

    mock_item_1 = MagicMock()
    mock_item_1.index = 2
    mock_item_1.relevance_score = 0.75

    mock_response = MagicMock()
    mock_response.results = [mock_item_0, mock_item_1]

    with patch("litellm.rerank", return_value=mock_response) as mock_rerank:
        reranked = rerank_documents(
            query="wheel torque",
            documents=docs,
            top_n=2,
        )

        mock_rerank.assert_called_once()
        args, kwargs = mock_rerank.call_args
        assert kwargs["query"] == "wheel torque"
        assert kwargs["model"] == "huggingface/BAAI/bge-reranker-base"
        assert kwargs["api_key"] == "hf_mock_token"
        assert kwargs["documents"] == [
            "Low relevance about fruit",
            "Exact match for wheel torque specifications",
            "Moderate relevance about automotive tools",
        ]

        assert len(reranked) == 2
        # Best match should now be first
        assert reranked[0]["doc_id"] == "D2"
        assert reranked[0]["rerank_score"] == pytest.approx(0.98)

        assert reranked[1]["doc_id"] == "D3"
        assert reranked[1]["rerank_score"] == pytest.approx(0.75)


def test_rerank_failure_fallback(monkeypatch):
    """Test that exceptions from litellm (e.g. timeout) fall back gracefully to raw top-n."""
    monkeypatch.setenv("HF_TOKEN", "hf_mock_token")

    docs = [
        {"text": "doc 1", "doc_id": "D1"},
        {"text": "doc 2", "doc_id": "D2"},
        {"text": "doc 3", "doc_id": "D3"},
    ]

    with patch("litellm.rerank", side_effect=RuntimeError("Hugging Face API Timeout")):
        result = rerank_documents("query", docs, top_n=2)
        assert len(result) == 2
        assert result[0]["doc_id"] == "D1"
        assert result[1]["doc_id"] == "D2"
        assert "rerank_score" not in result[0]
