"""Pytest configuration for NGA integration tests."""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--judge",
        action="store_true",
        default=False,
        help="Enable LLM-as-judge scoring (slower, requires API key)",
    )


@pytest.fixture(scope="session")
def use_judge(request):
    return request.config.getoption("--judge")
