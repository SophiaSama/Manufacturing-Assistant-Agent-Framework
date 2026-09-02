"""Unit tests for model capability tiers (docs/model-capability-gating-design.md)."""

from __future__ import annotations

import pytest

from nga.evaluation.capability import (
    cloud_allowlist_tier,
    parse_local_params,
    require_tier,
    resolve_tier,
    tier_meets,
)


class TestParseLocalParams:
    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("llama3.2:3b", 3.0),
            ("qwen2.5:7b-instruct", 7.0),
            ("llama3.1:70b", 70.0),
            ("mistral:7b-q4_K_M", 7.0),
            ("llama3.2:1b", 1.0),
        ],
    )
    def test_sizes(self, model_id, expected):
        assert parse_local_params(model_id) == expected

    @pytest.mark.parametrize(
        "model_id",
        ["llama3.2:latest", "gpt-4o", "b3:chat", "qwen3", ""],
    )
    def test_no_size(self, model_id):
        assert parse_local_params(model_id) is None


class TestCloudAllowlist:
    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("anthropic/claude-haiku-4-5", "standard"),
            ("anthropic/claude-sonnet-4-5", "strong"),
            ("anthropic/claude-opus-4-5", "strong"),
            ("openai/gpt-4o", "strong"),
            ("deepseek/deepseek-v4", "strong"),
            ("qwen/qwen3-embedding-4b", None),  # embedding model — not chat
        ],
    )
    def test_allowlist(self, model_id, expected):
        assert cloud_allowlist_tier(model_id) == expected


class TestResolveTier:
    def test_local_3b_is_basic(self):
        assert resolve_tier("llama3.2:3b", "ollama") == "basic"

    def test_local_7b_is_strong(self):
        assert resolve_tier("llama3.1:8b", "ollama") == "strong"
        assert resolve_tier("qwen2.5:7b", "ollama") == "strong"

    def test_unknown_local_is_basic(self):
        assert resolve_tier("some-model:latest", "ollama") == "basic"

    def test_cloud_haiku_standard(self):
        assert resolve_tier("anthropic/claude-haiku-4-5", "openrouter") == "standard"

    def test_cloud_sonnet_strong(self):
        assert resolve_tier("anthropic/claude-sonnet-4-5", "openrouter") == "strong"

    def test_unknown_cloud_fail_open(self):
        assert resolve_tier("mystery-provider/model-x", "openrouter") == "standard"

    def test_override_wins(self):
        assert (
            resolve_tier("llama3.2:3b", "ollama", override="strong")
            == "strong"
        )
        assert (
            resolve_tier("anthropic/claude-haiku-4-5", "openrouter", override="basic")
            == "basic"
        )

    def test_empty_model_local_default(self):
        assert resolve_tier(None, "ollama") == "basic"
        assert resolve_tier("", "ollama") == "basic"


class TestTierMeets:
    @pytest.mark.parametrize(
        ("active", "required", "expected"),
        [
            ("basic", "basic", True),
            ("basic", "standard", False),
            ("standard", "standard", True),
            ("standard", "strong", False),
            ("strong", "strong", True),
            ("strong", "basic", True),
        ],
    )
    def test_meets(self, active, required, expected):
        assert tier_meets(active, required) is expected


class TestMarker:
    def test_require_tier_unknown_rejected(self):
        with pytest.raises(ValueError):
            require_tier("ultra")

    def test_require_tier_returns_marker(self):
        marker = require_tier("strong")
        assert getattr(marker, "name", None) == "required_tier"
        assert marker.args == ("strong",)
