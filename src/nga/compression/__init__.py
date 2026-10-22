"""Headroom context compression and session storage package.

Re-exports from enterprise_agent.compression for backward compatibility.
"""

from enterprise_agent.compression.manager import (
    HeadroomManager,
    SessionRawContentStore,
)

__all__ = [
    "HeadroomManager",
    "SessionRawContentStore",
]
