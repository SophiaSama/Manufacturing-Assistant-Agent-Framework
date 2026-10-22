"""Headroom compression manager compatibility re-export."""

from enterprise_agent.compression.manager import (
    HeadroomManager,
    SessionRawContentStore,
)

__all__ = [
    "HeadroomManager",
    "SessionRawContentStore",
]
