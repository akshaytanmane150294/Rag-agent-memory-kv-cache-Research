"""
Tracks which memory items are currently "warm" (i.e. their KV cache is
still resident / likely resident) in the serving engine.

This is a lightweight approximation you can start with on top of any
serving engine. If your serving engine (vLLM + LMCache) exposes real
cache-residency introspection, swap `is_warm` to query that directly
instead of using a TTL heuristic.
"""
import time


class CacheResidencyTracker:
    def __init__(self, ttl_seconds: float = 60.0):
        self.resident: dict[int, float] = {}  # memory_id -> last_used_ts
        self.ttl = ttl_seconds

    def mark_used(self, memory_id: int):
        self.resident[memory_id] = time.time()

    def is_warm(self, memory_id: int) -> bool:
        ts = self.resident.get(memory_id)
        return ts is not None and (time.time() - ts) < self.ttl

    def warm_set(self) -> set[int]:
        now = time.time()
        return {mid for mid, ts in self.resident.items() if now - ts < self.ttl}

    def evict_expired(self):
        now = time.time()
        self.resident = {
            mid: ts for mid, ts in self.resident.items() if now - ts < self.ttl
        }
