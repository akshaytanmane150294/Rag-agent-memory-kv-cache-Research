"""
A3 — Simple agent-memory baseline (MemGPT/Mem0-style), cache-UNAWARE.
Every turn: write a memory of what happened, read top-k similar memories
back in, with no notion of what's warm in the KV cache. Same
enable_prefix_caching=True setting as A2, so the ONLY difference vs A2
is the presence of a memory subsystem.
"""
import time

from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer

from memory.simple_memory import SimpleMemory
from common.metrics import RunMetrics
from common.prompt_utils import build_prompt

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"  # Colab T4-friendly default; swap for Llama-3.1-8B-Instruct on bigger GPU


def run(conversation: list[dict], run_name: str = "a3_agent_memory"):
    """
    conversation: list of {"query": str, "fact_to_store": str (optional), "gold_answer": str (optional)}
    Use a multi-session dataset (e.g. LongMemEval) where facts stated early
    need to be recalled several turns/sessions later.
    """
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    memory = SimpleMemory(embedder)
    llm = LLM(model=MODEL_PATH, enable_prefix_caching=True)

    metrics = RunMetrics(run_name)
    history: list[str] = []

    for turn_id, turn in enumerate(conversation):
        query = turn["query"]
        retrieved_memories = memory.read(query, k=5)
        prompt = build_prompt(history, retrieved=[], query=query, memories=retrieved_memories)

        t0 = time.time()
        outputs = llm.generate([prompt], SamplingParams(max_tokens=256, temperature=0.0, stop=["<END>", "[End of answer]", "\n\nQuestion:", "\n\n[Current question]"]))
        t1 = time.time()

        answer = outputs[0].outputs[0].text
        n_tokens = len(outputs[0].outputs[0].token_ids)

        prompt_tokens = len(prompt.split())
        # NOTE: because retrieved memory order/content changes every turn,
        # expect a LOWER reuse ratio here than A2 — this is the effect
        # Gap 1 / Gap 2 are meant to fix.
        reused_est = 0

        metrics.record(
            turn_id=turn_id,
            ttft=t1 - t0,
            total_latency=t1 - t0,
            tokens_generated=n_tokens,
            kv_tokens_reused=reused_est,
            kv_tokens_recomputed=prompt_tokens,
            answer_quality=None,  # TODO: score recall accuracy vs turn.get("gold_answer")
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
