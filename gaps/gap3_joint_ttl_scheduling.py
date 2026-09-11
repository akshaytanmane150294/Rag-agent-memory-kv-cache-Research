"""
Gap 3 — Multi-turn, tool-interleaved scheduling for memory-augmented agents.
Continuum ([30]) introduced tool-aware KV-cache time-to-live (TTL)
scheduling but without an agent-memory subsystem. Here, TTL decisions also
factor in memory "warmth" -- turns whose context includes more warm
memories get pinned a bit longer, since they're more likely to be reused
on the very next turn.

This module is scheduler-logic-only; wire `compute_ttl()` into your actual
serving engine's cache-pinning API (vLLM + LMCache offload hooks, or a
custom eviction-delay wrapper if that's not exposed).
"""
import time

from memory.cache_tracker import CacheResidencyTracker


class ToolLatencyEstimator:
    """Learns expected duration of each tool call from observed history."""

    def __init__(self):
        self.history: dict[str, list[float]] = {}

    def record(self, tool_name: str, duration: float):
        self.history.setdefault(tool_name, []).append(duration)

    def expected(self, tool_name: str, default: float = 1.0) -> float:
        vals = self.history.get(tool_name, [])
        return sum(vals) / len(vals) if vals else default


def compute_ttl(
    tool_name: str,
    memory_ids_in_context: list[int],
    estimator: ToolLatencyEstimator,
    tracker: CacheResidencyTracker,
    base_ttl: float = 5.0,
    warmth_weight: float = 0.5,
) -> float:
    """Returns how long (seconds) this turn's KV cache should stay pinned."""
    tool_wait = estimator.expected(tool_name)
    warmth_boost = sum(1 for m in memory_ids_in_context if tracker.is_warm(m))
    return base_ttl + tool_wait + warmth_weight * warmth_boost


class PinningScheduler:
    """
    Minimal reference scheduler: call `on_turn_start`/`on_tool_call`/
    `on_turn_end` around your existing agent loop; it decides how long to
    keep each turn's cache pinned before allowing eviction.
    Swap `_pin` / `_evict` bodies for your serving engine's real API.
    """

    def __init__(self, estimator: ToolLatencyEstimator, tracker: CacheResidencyTracker):
        self.estimator = estimator
        self.tracker = tracker
        self.pinned_until: dict[int, float] = {}  # turn_id -> unpin timestamp

    def on_tool_call(self, turn_id: int, tool_name: str, memory_ids_in_context: list[int]):
        ttl = compute_ttl(tool_name, memory_ids_in_context, self.estimator, self.tracker)
        self.pinned_until[turn_id] = time.time() + ttl
        self._pin(turn_id, ttl)

    def maybe_evict(self, turn_id: int):
        deadline = self.pinned_until.get(turn_id)
        if deadline is not None and time.time() >= deadline:
            self._evict(turn_id)
            del self.pinned_until[turn_id]

    def _pin(self, turn_id: int, ttl: float):
        # TODO: call into vLLM/LMCache to keep this request's KV cache
        # resident in GPU memory for `ttl` seconds.
        pass

    def _evict(self, turn_id: int):
        # TODO: call into vLLM/LMCache to release this request's KV cache.
        pass


if __name__ == "__main__":
    estimator = ToolLatencyEstimator()
    tracker = CacheResidencyTracker(ttl_seconds=60.0)
    scheduler = PinningScheduler(estimator, tracker)

    # simulate a tool call and TTL decision
    estimator.record("web_search", 1.8)
    tracker.mark_used(memory_id=0)
    ttl = compute_ttl("web_search", memory_ids_in_context=[0], estimator=estimator, tracker=tracker)
    print(f"Computed TTL for this turn: {ttl:.2f}s")
