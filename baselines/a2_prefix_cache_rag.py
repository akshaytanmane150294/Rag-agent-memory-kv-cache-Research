"""
A2 — Prefix-caching RAG baseline (RAGCache / PagedAttention style, via vLLM's
built-in `enable_prefix_caching=True`). Only the SHARED, EXACT prefix
(system prompt, static instructions) gets reused; retrieved chunks that
change order/content across turns will NOT benefit from reuse.

This is meant to expose the exact limitation described in the literature
review Section 7: brittle-to-reordering caching.
"""
import time

from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer

from baselines.a1_naive_rag import build_doc_index, retrieve
from common.metrics import RunMetrics
from common.prompt_utils import build_prompt

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"  # Colab T4-friendly default; swap for Llama-3.1-8B-Instruct on bigger GPU


def run(docs: list[str], conversation: list[dict], run_name: str = "a2_prefix_cache_rag"):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index = build_doc_index(docs, embedder)
    llm = LLM(model=MODEL_PATH, enable_prefix_caching=True)  # cache ON

    metrics = RunMetrics(run_name)
    history: list[str] = []

    for turn_id, turn in enumerate(conversation):
        query = turn["query"]
        retrieved = retrieve(query, embedder, index, docs)
        prompt = build_prompt(history, retrieved, query)

        print()
        print("=" * 70)
        print(f"Turn {turn_id}")
        print(f"Query: {query}")
        print(f"Retrieved documents: {retrieved}")
        print("=" * 70)

        t0 = time.time()
        outputs = llm.generate([prompt], SamplingParams(max_tokens=256, temperature=0.0, stop=["<END>", "[End of answer]", "\n\nQuestion:", "\n\n[Current question]"]))
        t1 = time.time()

        answer = outputs[0].outputs[0].text
        n_tokens = len(outputs[0].outputs[0].token_ids)

        # Real cache-hit stats from vLLM's RequestOutput (verified via diagnostic.py)
        prompt_tokens = len(outputs[0].prompt_token_ids)
        reused_est = getattr(outputs[0], "num_cached_tokens", 0) or 0

        print()
        print("Answer:")
        print(answer)
        gold = turn.get("gold_answer")
        if gold:
            print(f"(gold answer: {gold})")
        print()
        print(f"Generated tokens : {n_tokens}")
        print(f"Latency          : {t1 - t0:.4f} sec")
        print(f"Cached tokens    : {reused_est} / {prompt_tokens}")

        metrics.record(
            turn_id=turn_id,
            ttft=t1 - t0,
            total_latency=t1 - t0,
            tokens_generated=n_tokens,
            kv_tokens_reused=reused_est,
            kv_tokens_recomputed=max(0, prompt_tokens - reused_est),
            answer_quality=None,  # TODO: score vs gold answer
        )
        history.append(f"Q: {query}\nA: {answer}")

    return metrics


if __name__ == "__main__":
    demo_docs = ["Doc A about topic 1", "Doc B about topic 2"]
    demo_conv = [{"query": "What is topic 1?"}, {"query": "And topic 2?"}]
    m = run(demo_docs, demo_conv)
    print(m.summary())
    m.save()
