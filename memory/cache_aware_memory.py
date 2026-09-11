"""
Gap 1: memory-aware KV-cache reuse.

Extends SimpleMemory's retrieval so that items whose KV cache is still
"warm" (per CacheResidencyTracker) get a similarity boost — preferring
to re-surface content that is cheap to reuse, without sacrificing recall.
"""
from memory.simple_memory import SimpleMemory
from memory.cache_tracker import CacheResidencyTracker


def cache_aware_retrieve(
    query: str,
    memory: SimpleMemory,
    tracker: CacheResidencyTracker,
    k: int = 5,
    warm_bonus: float = 0.15,
    candidate_multiplier: int = 3,
) -> list[str]:
    if not memory.items:
        return []

    qvec = memory.embedder.encode([query]).astype("float32")
    n_candidates = min(len(memory.items), k * candidate_multiplier)
    distances, indices = memory.index.search(qvec, n_candidates)

    scored = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(memory.items):
            continue
        similarity = 1.0 / (1.0 + float(dist))
        if tracker.is_warm(int(idx)):
            similarity += warm_bonus
        scored.append((similarity, int(idx)))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]

    for _, idx in top:
        tracker.mark_used(idx)

    return [memory.items[idx] for _, idx in top]
