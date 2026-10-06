"""
Gap 1 — Memory-aware cache reuse policy.
Same as A3 (agent memory), except retrieval is now cache-aware: items whose
KV cache is still "warm" get a similarity boost, so the memory subsystem
naturally prefers content that's cheap to reuse.

Compare directly against baselines/a3_agent_memory.py using the same
conversation/dataset — the ONLY difference should be `cache_aware_retrieve`
vs `memory.read`.
"""
import time

from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer

from memory.simple_memory import SimpleMemory
from memory.cache_tracker import CacheResidencyTracker
from memory.cache_aware_memory import cache_aware_retrieve
from common.metrics import RunMetrics
from common.prompt_utils import build_prompt

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"  # Colab/Kaggle T4-friendly default; matches baselines
WARM_BONUS = 0.15  # ablate this: 0, 0.05, 0.15, 0.3 ...


def run(conversation: list[dict], run_name: str = "gap1_cache_aware_retrieval",
        warm_bonus: float = WARM_BONUS):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    memory = SimpleMemory(embedder)
    tracker = CacheResidencyTracker(ttl_seconds=60.0)
    llm = LLM(
        model=MODEL_PATH,
        enable_prefix_caching=True,
        dtype="half",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enforce_eager=True,
    )

    metrics = RunMetrics(run_name)
    history: list[str] = []

    for turn_id, turn in enumerate(conversation):
        query = turn["query"]
        retrieved_memories = cache_aware_retrieve(query, memory, tracker, k=5, warm_bonus=warm_bonus)
        prompt = build_prompt(history, retrieved=[], query=query, memories=retrieved_memories)

        print()
        print("=" * 70)
        print(f"Turn {turn_id}")
        print(f"Query: {query}")
        print(f"Retrieved memories: {retrieved_memories}")
        print("=" * 70)

        t0 = time.time()
        outputs = llm.generate([prompt], SamplingParams(max_tokens=256, temperature=0.0, stop=["<END>", "[End of answer]", "\n\nQuestion:", "\n\n[Current question]"]))
        t1 = time.time()

        answer = outputs[0].outputs[0].text
        n_tokens = len(outputs[0].outputs[0].token_ids)
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
            kv_tokens_recomputed=prompt_tokens - reused_est,
            answer_quality=None,
        )

        history.append(f"Q: {query}\nA: {answer}")
        if turn.get("fact_to_store"):
            memory.write(turn["fact_to_store"])

    return metrics


if __name__ == "__main__":
    demo_conv = [
        {"query": "My favorite color is blue.", "fact_to_store": "User's favorite color is blue."},
        {"query": "What's my favorite color?"},
    ]
    m = run(demo_conv)
    print(m.summary())
    m.save()
