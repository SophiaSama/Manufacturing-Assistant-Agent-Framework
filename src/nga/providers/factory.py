"""Resolve chat + embedding providers from the execution environment."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from nga.config import Settings


def make_embeddings(settings: Settings) -> Embeddings:
    if settings.provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    from nga.providers.openrouter_embeddings import (
        BatchOpenRouterEmbeddings,
        OpenRouterEmbeddings,
    )

    if settings.embedding_strategy == "batch":
        return BatchOpenRouterEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            poll_interval=settings.embedding_poll_interval,
            max_wait=settings.embedding_max_wait,
        )

    # "sequential" — one synchronous request per text
    return OpenRouterEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


def make_chat_model(settings: Settings, model_override: str | None = None) -> BaseChatModel:
    """Create a chat model, optionally overriding the model slug (for tier routing)."""
    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_override or settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
        )

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model_override or settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        # Bound request time so a single stalled upstream call cannot freeze
        # the eval/agent loop (the OpenAI SDK default is 600s).
        timeout=60.0,
        max_retries=2,
    )
