"""Retrieval Augmented Generation (RAG) components for Enterprise Agent."""

from __future__ import annotations

from enterprise_agent.rag.retrieval import (
    build_or_open_vector_store,
    make_generic_retrieval_tool,
)

__all__ = [
    "build_or_open_vector_store",
    "make_generic_retrieval_tool",
]
