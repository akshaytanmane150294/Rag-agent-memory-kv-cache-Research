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

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"  # TODO
WARM_BONUS = 0.15  # ablate this: 0, 0.05, 0.15, 0.3 ...


def run(conversation: list[dict], run_name: str = "gap1_cache_aware_retrieval",
        warm_bonus: float = WARM_BONUS):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    memory = SimpleMemory(embedder)
    tracker = CacheResidencyTracker(ttl_seconds=60.0)
    llm = LLM(model=MODEL_PATH, enable_prefix_caching=True)

    metrics = RunMetrics(run_name)
    history: list[str] = []

    for turn_id, turn in enumerate(conversation):
        query = turn["query"]
        retrieved_memories = cache_aware_retrieve(query, memory, tracker, k=5, warm_bonus=warm_bonus)
        prompt = build_prompt(history, retrieved=[], query=query, memories=retrieved_memories)

        t0 = time.time()
        outputs = llm.generate([prompt], SamplingParams(max_tokens=256, temperature=0.0, stop=["<END>", "[End of answer]", "\n\nQuestion:", "\n\n[Current question]"]))
        t1 = time.time()

        answer = outputs[0].outputs[0].text
        prompt_tokens = len(prompt.split())

        # TODO: pull real reuse stats; for now approximate reuse by how many
        # retrieved memories were already warm before this call
        warm_before = sum(1 for m in retrieved_memories if m in history)  # placeholder proxy
        reused_est = int(prompt_tokens * (warm_before / max(1, len(retrieved_memories))))

        metrics.record(
            turn_id=turn_id,
            ttft=t1 - t0,
            total_latency=t1 - t0,
            tokens_generated=len(outputs[0].outputs[0].token_ids),
            kv_tokens_reused=reused_est,
            kv_tokens_recomputed=prompt_tokens - reused_est,
            answer_quality=None,  # TODO: score vs gold, must stay >= A3's quality
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
