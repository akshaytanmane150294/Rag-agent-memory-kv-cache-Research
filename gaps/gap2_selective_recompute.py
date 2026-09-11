"""
Gap 2 — Quality-preserving reuse under memory reordering
(CacheBlend-style selective recompute, applied specifically to agent
MEMORY items rather than static document chunks -- this per-memory-item
setting is what the original CacheBlend paper does NOT evaluate, and is
your novelty claim per the literature review's Gap #2).

Approach: precompute KV for each memory item once; when a set of memory
items is retrieved (in whatever order), only recompute a small "boundary"
window of tokens at each chunk edge (where cross-chunk attention matters
most) and reuse the rest -- instead of naive full concatenation (A5) or
full recomputation (A1, quality upper bound but slow).

Requires LMCache (https://github.com/LMCache/LMCache) or an equivalent
selective-recompute fusion backend; the functions below are the glue you
need to call at the right points in your generation loop.
"""
import time

# TODO: pip install lmcache
# from lmcache.integration.vllm.utils import init_lmcache_engine

from memory.simple_memory import SimpleMemory
from common.metrics import RunMetrics
from common.prompt_utils import build_prompt

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"  # TODO
BOUNDARY_WINDOW = 8  # tokens recomputed at each chunk edge; ablate as % of chunk length


def precompute_memory_kv(memory_items: list[str], llm) -> dict[int, dict]:
    kv_store = {}
    for i, text in enumerate(memory_items):
        # TODO: real precompute call via LMCache, e.g. llm.precompute_kv(text)
        kv_store[i] = {"text": text, "kv": None}
    return kv_store


def selective_recompute_fuse(retrieved_ids: list[int], kv_store: dict,
                              boundary_window: int = BOUNDARY_WINDOW) -> str:
    """
    Placeholder for the real fusion call. In a full implementation this
    calls into LMCache's blend/fusion API: reuse cached KV for the bulk of
    each chunk, recompute only `boundary_window` tokens at each edge so
    cross-chunk attention is approximately restored.
    Returns the (conceptually fused) context text to feed into the prompt;
    swap this for the actual fused-KV call once LMCache is wired in.
    """
    return "\n".join(kv_store[i]["text"] for i in retrieved_ids)


def run(conversation: list[dict], run_name: str = "gap2_selective_recompute",
        boundary_window: int = BOUNDARY_WINDOW):
    from sentence_transformers import SentenceTransformer
    from vllm import LLM, SamplingParams

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    memory = SimpleMemory(embedder)
    llm = LLM(model=MODEL_PATH, enable_prefix_caching=True)
    kv_store: dict[int, dict] = {}

    metrics = RunMetrics(run_name)
    history: list[str] = []

    for turn_id, turn in enumerate(conversation):
        query = turn["query"]
        retrieved_texts = memory.read(query, k=5)
        retrieved_ids = [memory.items.index(t) for t in retrieved_texts]

        fused_context = selective_recompute_fuse(retrieved_ids, kv_store, boundary_window)
        prompt = build_prompt(history, retrieved=[], query=query,
                               memories=[fused_context] if fused_context else None)

        t0 = time.time()
        outputs = llm.generate([prompt], SamplingParams(max_tokens=256, temperature=0.0, stop=["<END>", "[End of answer]", "\n\nQuestion:", "\n\n[Current question]"]))
        t1 = time.time()

        answer = outputs[0].outputs[0].text
        prompt_tokens = len(prompt.split())
        recomputed_est = boundary_window * max(1, len(retrieved_ids))  # rough proxy
        reused_est = max(0, prompt_tokens - recomputed_est)

        metrics.record(
            turn_id=turn_id,
            ttft=t1 - t0,
            total_latency=t1 - t0,
            tokens_generated=len(outputs[0].outputs[0].token_ids),
            kv_tokens_reused=reused_est,
            kv_tokens_recomputed=recomputed_est,
            answer_quality=None,  # TODO: score vs gold; compare to A5 (should be higher)
                                   # and A1 full-recompute (should be close)
        )

        history.append(f"Q: {query}\nA: {answer}")
        if turn.get("fact_to_store"):
            memory.write(turn["fact_to_store"])
            kv_store = precompute_memory_kv(memory.items, llm)  # re-precompute after writes

    return metrics


if __name__ == "__main__":
    demo_conv = [
        {"query": "My favorite color is blue.", "fact_to_store": "User's favorite color is blue."},
        {"query": "What's my favorite color?"},
    ]
    m = run(demo_conv)
    print(m.summary())
    m.save()
