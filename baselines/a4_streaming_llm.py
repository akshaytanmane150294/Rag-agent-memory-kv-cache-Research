"""
A4 — KV-cache EVICTION baseline (StreamingLLM-style attention sinks + sliding
recent window). Requires the external streaming-llm repo:

    git clone https://github.com/mit-han-lab/streaming-llm

Use this to see where quality degrades on long conversations when
middle-context content is evicted but is still needed for the answer.
"""
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

# TODO: `pip install` / vendor the streaming-llm repo, then:
# from streaming_llm.utils import enable_streaming_llm

from common.metrics import RunMetrics
from common.prompt_utils import build_prompt

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"  # Colab T4-friendly default; swap for Llama-3.1-8B-Instruct on bigger GPU
START_SIZE = 4        # number of "attention sink" tokens to always keep
RECENT_SIZE = 2000     # sliding window of most recent tokens to keep


def run(conversation: list[dict], run_name: str = "a4_streaming_llm"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    # TODO: wire in the real streaming-llm wrapper, e.g.:
    # model, tokenizer = enable_streaming_llm(model, tokenizer,
    #                                          start_size=START_SIZE,
    #                                          recent_size=RECENT_SIZE)

    metrics = RunMetrics(run_name)
    history: list[str] = []

    for turn_id, turn in enumerate(conversation):
        query = turn["query"]
        prompt = build_prompt(history, retrieved=[], query=query)
        inputs = tokenizer(prompt, return_tensors="pt")

        t0 = time.time()
        gen_ids = model.generate(**inputs, max_new_tokens=256)
        t1 = time.time()

        answer = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)

        prompt_tokens = inputs["input_ids"].shape[1]
        kept = min(prompt_tokens, START_SIZE + RECENT_SIZE)
        evicted = max(0, prompt_tokens - kept)

        metrics.record(
            turn_id=turn_id,
            ttft=t1 - t0,          # TODO: real TTFT via streaming generation
            total_latency=t1 - t0,
            tokens_generated=gen_ids.shape[1] - prompt_tokens,
            kv_tokens_reused=kept,     # tokens whose KV state is retained
            kv_tokens_recomputed=evicted,  # tokens dropped -> unavailable if needed again
            answer_quality=None,  # TODO: score vs gold answer, esp. for mid-context recall
        )
        history.append(f"Q: {query}\nA: {answer}")

    return metrics


if __name__ == "__main__":
    demo_conv = [{"query": "Summarize what we discussed."}]
    m = run(demo_conv)
    print(m.summary())
    m.save()
