"""Resolve chat + embedding providers from the execution environment.

Backward-compatibility shim re-exporting from enterprise_agent.providers.factory.
"""

from enterprise_agent.providers.factory import (
    aclose_active_chat_models,
    aclose_chat_model,
    close_active_chat_models,
    get_active_chat_models,
    make_chat_model,
    make_embeddings,
    register_chat_model,
)

__all__ = [
    "register_chat_model",
    "get_active_chat_models",
    "aclose_chat_model",
    "aclose_active_chat_models",
    "close_active_chat_models",
    "make_embeddings",
    "make_chat_model",
]
