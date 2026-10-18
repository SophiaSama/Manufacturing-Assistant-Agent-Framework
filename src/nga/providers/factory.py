"""Resolve chat + embedding providers from the execution environment."""

from __future__ import annotations

import threading
import weakref

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from nga.config import Settings
from nga.tracing import traceable_if_enabled

_active_chat_models: weakref.WeakSet[BaseChatModel] = weakref.WeakSet()
_active_chat_models_lock = threading.Lock()


def register_chat_model(model: BaseChatModel) -> BaseChatModel:
    """Track instantiated chat models to facilitate session teardown."""
    try:
        with _active_chat_models_lock:
            _active_chat_models.add(model)
    except Exception:
        pass
    return model


def get_active_chat_models() -> list[BaseChatModel]:
    with _active_chat_models_lock:
        return list(_active_chat_models)


async def aclose_chat_model(model: BaseChatModel) -> None:
    """Safely close underlying async HTTP clients on a chat model."""
    import inspect

    try:
        candidates = []
        for attr in ("root_async_client", "_async_client", "async_client"):
            candidate = getattr(model, attr, None)
            if candidate is not None:
                candidates.append(candidate)

        for candidate in candidates:
            try:
                if hasattr(candidate, "close"):
                    res = candidate.close()
                    if inspect.isawaitable(res):
                        await res
                elif hasattr(candidate, "aclose"):
                    res = candidate.aclose()
                    if inspect.isawaitable(res):
                        await res
            except Exception:
                pass

            targets = [candidate]
            raw_client = getattr(candidate, "_client", None)
            if raw_client is not None and raw_client is not candidate:
                targets.append(raw_client)

            for target in targets:
                if hasattr(target, "_state"):
                    try:
                        from httpx._client import ClientState

                        target._state = ClientState.CLOSED
                    except Exception:
                        pass
                try:
                    setattr(target, "is_closed", True)
                except Exception:
                    pass
    except Exception:
        pass


async def aclose_active_chat_models() -> None:
    """Asynchronously close any active async clients across all registered models."""
    with _active_chat_models_lock:
        models = list(_active_chat_models)
        _active_chat_models.clear()

    for model in models:
        await aclose_chat_model(model)


def close_active_chat_models() -> None:
    """Synchronously close any active async clients across all registered models."""
    import asyncio
    import concurrent.futures

    with _active_chat_models_lock:
        models = list(_active_chat_models)
        _active_chat_models.clear()

    if not models:
        return

    async def _close_all_coro() -> None:
        for model in models:
            await aclose_chat_model(model)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        try:
            asyncio.run(_close_all_coro())
        except Exception:
            pass
    else:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(asyncio.run, _close_all_coro())
                fut.result(timeout=5.0)
        except Exception:
            pass


@traceable_if_enabled(run_type="chain", name="providers.make_embeddings")
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


@traceable_if_enabled(run_type="chain", name="providers.make_chat_model")
def make_chat_model(settings: Settings, model_override: str | None = None) -> BaseChatModel:
    """Create a chat model, optionally overriding the model slug (for tier routing)."""
    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama
        model = ChatOllama(
            model=model_override or settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
        )
        return register_chat_model(model)

    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(
        model=model_override or settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        # Bound request time so a single stalled upstream call cannot freeze
        # the eval/agent loop (the OpenAI SDK default is 600s).
        timeout=60.0,
        max_retries=2,
    )
    return register_chat_model(model)

