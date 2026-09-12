"""
A1 — Naive RAG baseline.

Fixed retrieve-once-then-generate:
- NO KV-cache reuse
- Prefix caching OFF
- T4/Colab-friendly configuration

This is the baseline/floor that other systems should beat.
"""

import time

from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
import faiss

from common.metrics import RunMetrics
from common.prompt_utils import build_prompt


# ============================================================
# Model configuration
# ============================================================

# Colab T4-friendly model.
# Qwen2.5-1.5B-Instruct is much lighter than Llama-3.1-8B.
MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"


# ============================================================
# Retrieval
# ============================================================

def build_doc_index(docs: list[str], embedder):
    """
    Build a FAISS index from document embeddings.
    """

    if not docs:
        raise ValueError("docs cannot be empty")

    vecs = embedder.encode(
        docs,
        convert_to_numpy=True
    ).astype("float32")

    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)

    return index


def retrieve(
    query: str,
    embedder,
    index,
    docs: list[str],
    k: int = 3
) -> list[str]:
    """
    Retrieve top-k relevant documents using FAISS.
    """

    if not query.strip():
        return []

    if not docs:
        return []

    qvec = embedder.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    k = min(k, len(docs))

    _, idx = index.search(qvec, k)

    return [
        docs[i]
        for i in idx[0]
        if 0 <= i < len(docs)
    ]


# ============================================================
# LLM initialization
# ============================================================

def create_llm():
    """
    Create a Colab T4-friendly vLLM instance.

    Important:
    - Prefix caching is intentionally OFF for A1.
    - FP16 is used for T4 compatibility.
    - max_model_len is limited to reduce KV-cache memory usage.
    """

    llm = LLM(
        model=MODEL_PATH,

        # A1 baseline must NOT use prefix caching.
        enable_prefix_caching=False,

        # NVIDIA T4 is better suited to FP16.
        dtype="half",

        # Leave some GPU memory headroom in Colab.
        gpu_memory_utilization=0.85,

        # Keep KV-cache allocation manageable.
        max_model_len=4096,

        # Avoid CUDA graph compilation issues/startup overhead.
        enforce_eager=True,
    )

    return llm


# ============================================================
# Main baseline
# ============================================================

def run(
    docs: list[str],
    conversation: list[dict],
    run_name: str = "a1_naive_rag"
):
    """
    Run the Naive RAG baseline.

    Parameters
    ----------
    docs:
        List of documents available to the retriever.

    conversation:
        List of dictionaries:

        {
            "query": str,
            "gold_answer": str  # optional
        }

    Returns
    -------
    RunMetrics
    """

    if not docs:
        raise ValueError("docs cannot be empty")

    if not conversation:
        raise ValueError("conversation cannot be empty")

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    print("Loading embedding model...")

    embedder = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # --------------------------------------------------------
    # FAISS document index
    # --------------------------------------------------------

    print("Building FAISS document index...")

    index = build_doc_index(
        docs,
        embedder
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    print(f"Loading LLM: {MODEL_PATH}")

    llm = create_llm()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = RunMetrics(run_name)

    # Conversation history.
    history: list[str] = []

    # --------------------------------------------------------
    # Conversation loop
    # --------------------------------------------------------

    for turn_id, turn in enumerate(conversation):

        query = turn["query"]

        print()
        print("=" * 70)
        print(f"Turn {turn_id}")
        print(f"Query: {query}")
        print("=" * 70)

        # ----------------------------------------------------
        # Retrieve documents
        # ----------------------------------------------------

        retrieved = retrieve(
            query=query,
            embedder=embedder,
            index=index,
            docs=docs,
            k=3
        )

        print(f"Retrieved documents: {len(retrieved)}")

        # ----------------------------------------------------
        # Build prompt
        # ----------------------------------------------------

        prompt = build_prompt(
            history,
            retrieved,
            query
        )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        sampling_params = SamplingParams(
            max_tokens=256,

            # Deterministic baseline.
            temperature=0.0,
            stop=["<END>", "[End of answer]", "\n\nQuestion:", "\n\n[Current question]"],
        )

        t0 = time.time()

        outputs = llm.generate(
            [prompt],
            sampling_params
        )

        t1 = time.time()

        # ----------------------------------------------------
        # Extract answer
        # ----------------------------------------------------

        output = outputs[0]

        generated = output.outputs[0]

        answer = generated.text

        # vLLM provides generated token IDs.
        token_ids = getattr(
            generated,
            "token_ids",
            []
        )

        n_tokens = len(token_ids)

        total_latency = t1 - t0

        # Real cache-hit stats from vLLM's RequestOutput (verified via diagnostic.py)
        real_prompt_tokens = len(outputs[0].prompt_token_ids)
        real_cached_tokens = getattr(outputs[0], "num_cached_tokens", 0) or 0

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print()
        print("Answer:")
        print(answer)

        print()
        print(f"Generated tokens : {n_tokens}")
        print(f"Latency          : {total_latency:.4f} sec")

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        # Real vLLM cache-hit count (should be 0, caching intentionally OFF for A1)
        kv_tokens_reused = real_cached_tokens

        kv_tokens_recomputed = real_prompt_tokens - real_cached_tokens

        # Gold-answer scoring is not implemented here.
        # Keep None unless your metrics implementation supports
        # F1/EM scoring.
        answer_quality = None

        metrics.record(
            turn_id=turn_id,

            # NOTE:
            # This is total generation time, not true TTFT.
            # True TTFT requires streaming/per-token timing.
            ttft=total_latency,

            total_latency=total_latency,

            tokens_generated=n_tokens,

            kv_tokens_reused=kv_tokens_reused,

            kv_tokens_recomputed=kv_tokens_recomputed,

            answer_quality=answer_quality,
        )

        # ----------------------------------------------------
        # Update conversation history
        # ----------------------------------------------------

        history.append(
            f"Q: {query}\n"
            f"A: {answer}"
        )

    return metrics


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":

    demo_docs = [
        (
            "Topic 1 is an introduction to artificial intelligence. "
            "Artificial intelligence allows computers to perform "
            "tasks that normally require human intelligence."
        ),

        (
            "Topic 2 describes machine learning. "
            "Machine learning is a branch of artificial intelligence "
            "where models learn patterns from data."
        ),

        (
            "Topic 3 describes natural language processing. "
            "NLP allows computers to process and understand human language."
        ),
    ]

    demo_conversation = [
        {
            "query": "What is topic 1?"
        },
        {
            "query": "What is machine learning?"
        },
        {
            "query": "How is NLP related to artificial intelligence?"
        },
    ]

    print("=" * 70)
    print("A1 — NAIVE RAG BASELINE")
    print("=" * 70)

    metrics = run(
        docs=demo_docs,
        conversation=demo_conversation
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(metrics.summary())

    # Save results.
    metrics.save()

    print()
    print("Results saved successfully.")