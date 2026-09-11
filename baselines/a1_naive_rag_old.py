"""
A1 — Naive RAG baseline.
Fixed retrieve-once-then-generate, NO KV-cache reuse (prefix caching OFF).
This is the "floor" every other system should beat.
"""
import time

from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from common.metrics import RunMetrics
from common.prompt_utils import build_prompt

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"  # TODO: point at your local/HF model


def build_doc_index(docs: list[str], embedder):
    vecs = embedder.encode(docs).astype("float32")
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)
    return index


def retrieve(query: str, embedder, index, docs: list[str], k: int = 3) -> list[str]:
    qvec = embedder.encode([query]).astype("float32")
    k = min(k, len(docs))
    _, idx = index.search(qvec, k)
    return [docs[i] for i in idx[0] if 0 <= i < len(docs)]


def run(docs: list[str], conversation: list[dict], run_name: str = "a1_naive_rag"):
    """
    conversation: list of {"query": str, "gold_answer": str (optional)}
    """
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index = build_doc_index(docs, embedder)
    llm = LLM(model=MODEL_PATH, enable_prefix_caching=False)  # cache OFF on purpose

    metrics = RunMetrics(run_name)
    history: list[str] = []

    for turn_id, turn in enumerate(conversation):
        query = turn["query"]
        retrieved = retrieve(query, embedder, index, docs)
        prompt = build_prompt(history, retrieved, query)

        t0 = time.time()
        outputs = llm.generate([prompt], SamplingParams(max_tokens=256))
        t1 = time.time()

        answer = outputs[0].outputs[0].text
        n_tokens = len(outputs[0].outputs[0].token_ids)

        metrics.record(
            turn_id=turn_id,
            ttft=t1 - t0,          # TODO: use vLLM per-token timing for a true TTFT
            total_latency=t1 - t0,
            tokens_generated=n_tokens,
            kv_tokens_reused=0,                       # no reuse by design
            kv_tokens_recomputed=len(prompt.split()),  # rough token proxy
            answer_quality=None,   # TODO: score vs turn.get("gold_answer") with F1/EM
        )
        history.append(f"Q: {query}\nA: {answer}")

    return metrics


if __name__ == "__main__":
    demo_docs = ["Doc A about topic 1", "Doc B about topic 2"]
    demo_conv = [{"query": "What is topic 1?"}, {"query": "And topic 2?"}]
    m = run(demo_docs, demo_conv)
    print(m.summary())
    m.save()
