"""Cache hit/miss metrics with per-layer timing.

The eval runner snapshots these counters around each question to compute
per-layer latency and hit-rate contributions for the `cache_improvement`
block (docs/cache-design.md §7.4.1).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

VALID_LAYERS = ("emb", "retr", "sql")


@dataclass
class _LayerStats:
    hits: int = 0
    misses: int = 0
    total_ms: float = 0.0

    def snapshot(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / max(self.hits + self.misses, 1), 3),
            "total_ms": round(self.total_ms, 2),
        }


class CacheMetrics:
    """Thread-safe per-layer counters."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._layers: dict[str, _LayerStats] = {
            layer: _LayerStats() for layer in VALID_LAYERS
        }

    def record_hit(self, layer: str, elapsed_ms: float) -> None:
        self._record(layer, hit=True, elapsed_ms=elapsed_ms)

    def record_miss(self, layer: str, elapsed_ms: float) -> None:
        self._record(layer, hit=False, elapsed_ms=elapsed_ms)

    def _record(self, layer: str, *, hit: bool, elapsed_ms: float) -> None:
        if layer not in self._layers:
            return
        with self._lock:
            stat = self._layers[layer]
            if hit:
                stat.hits += 1
            else:
                stat.misses += 1
            stat.total_ms += elapsed_ms

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {layer: stat.snapshot() for layer, stat in self._layers.items()}

    def reset(self) -> None:
        with self._lock:
            for stat in self._layers.values():
                stat.hits = 0
                stat.misses = 0
                stat.total_ms = 0.0

    def delta(self, before: dict[str, dict], after: dict[str, dict]) -> dict[str, dict]:
        """Per-layer delta between two snapshots (for per-question timing)."""
        result: dict[str, dict] = {}
        for layer in before:
            b, a = before[layer], after[layer]
            result[layer] = {
                "hits": a["hits"] - b["hits"],
                "misses": a["misses"] - b["misses"],
                "ms": round(a["total_ms"] - b["total_ms"], 2),
            }
        return result
