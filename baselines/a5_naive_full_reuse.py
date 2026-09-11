"""
A5 — Naive full KV-cache reuse baseline (no cross-chunk fusion).
Precompute each retrieved chunk's KV cache independently, then simply
CONCATENATE whichever chunks are retrieved for a given query, with no
recomputation of cross-chunk attention. The literature (CacheBlend, [37])
reports this degrades multi-hop QA F1 by up to ~55% -- implementing this
baseline lets you reproduce that number as a sanity check before building
Gap 2's selective-recompute fix.

Recommended: use LMCache (https://github.com/LMCache/LMCache), which
already exposes a "naive concat" mode alongside selective recompute, so
you don't have to hand-roll low-level KV tensor manipulation in vLLM.
"""
import time

# TODO: pip install lmcache, then something like:
# from lmcache.integration.vllm.utils import init_lmcache_engine

from common.metrics import RunMetrics
from common.prompt_utils import build_prompt

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"  # Colab T4-friendly default; swap for Llama-3.1-8B-Instruct on bigger GPU


def precompute_chunk_kv(chunks: list[str], llm) -> dict[int, object]:
    """Precompute and store KV cache for each chunk independently (offline)."""
    kv_store = {}
    for i, chunk in enumerate(chunks):
        # TODO: replace with real LMCache / vLLM precompute call
        kv_store[i] = {"text": chunk, "kv": None}
    return kv_store


def naive_concat(retrieved_ids: list[int], kv_store: dict) -> str:
    """Just concatenate cached chunk texts/KV with no cross-chunk fusion."""
    return "\n".join(kv_store[i]["text"] for i in retrieved_ids)


def run(docs: list[str], conversation: list[dict], run_name: str = "a5_naive_full_reuse"):
    from sentence_transformers import SentenceTransformer
    from baselines.a1_naive_rag import build_doc_index, retrieve
    from vllm import LLM, SamplingParams

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index = build_doc_index(docs, embedder)
    llm = LLM(model=MODEL_PATH, enable_prefix_caching=True)
    kv_store = precompute_chunk_kv(docs, llm)

    metrics = RunMetrics(run_name)
    history: list[str] = []

    for turn_id, turn in enumerate(conversation):
        query = turn["query"]
        retrieved_texts = retrieve(query, embedder, index, docs)
        retrieved_ids = [docs.index(t) for t in retrieved_texts]

        fused_context = naive_concat(retrieved_ids, kv_store)
        prompt = build_prompt(history, retrieved=[fused_context], query=query)

        t0 = time.time()
        outputs = llm.generate([prompt], SamplingParams(max_tokens=256, temperature=0.0, stop=["<END>", "[End of answer]", "\n\nQuestion:", "\n\n[Current question]"]))
        t1 = time.time()

        answer = outputs[0].outputs[0].text
        prompt_tokens = len(prompt.split())

        metrics.record(
            turn_id=turn_id,
            ttft=t1 - t0,
            total_latency=t1 - t0,
            tokens_generated=len(outputs[0].outputs[0].token_ids),
            kv_tokens_reused=prompt_tokens,  # everything "reused", none fused/recomputed
            kv_tokens_recomputed=0,
            answer_quality=None,  # TODO: score vs gold — expect a notable drop here
        )
        history.append(f"Q: {query}\nA: {answer}")

    return metrics


if __name__ == "__main__":
    demo_docs = ["Doc A about topic 1", "Doc B about topic 2"]
    demo_conv = [{"query": "What is topic 1?"}]
    m = run(demo_docs, demo_conv)
    print(m.summary())
    m.save()
