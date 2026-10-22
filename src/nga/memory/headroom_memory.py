"""Backward-compatibility re-export for NgaEpisodicMemory."""

from enterprise_agent.memory.episodic import (
    EpisodicMemoryStore,
    NgaEpisodicMemory,
    _BackgroundLoopRunner,
)

__all__ = ["EpisodicMemoryStore", "NgaEpisodicMemory", "_BackgroundLoopRunner"]
