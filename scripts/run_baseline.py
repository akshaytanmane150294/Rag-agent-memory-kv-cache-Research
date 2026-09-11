"""
Entry point: run a chosen baseline (A1-A5) on a dataset and save results.

Usage:
    python scripts/run_baseline.py --baseline a1
    python scripts/run_baseline.py --baseline a3 --data data/longmemeval_sample.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASELINES = {
    "a1": "baselines.a1_naive_rag",
    "a2": "baselines.a2_prefix_cache_rag",
    "a3": "baselines.a3_agent_memory",
    "a4": "baselines.a4_streaming_llm",
    "a5": "baselines.a5_naive_full_reuse",
}


def load_conversation(data_path: str | None) -> list[dict]:
    if not data_path:
        return [
            {"query": "My favorite color is blue.", "fact_to_store": "User's favorite color is blue."},
            {"query": "What's my favorite color?"},
        ]
    return json.loads(Path(data_path).read_text())


def load_docs(docs_path: str | None) -> list[str]:
    if not docs_path:
        return ["Doc A about topic 1", "Doc B about topic 2"]
    return json.loads(Path(docs_path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=list(BASELINES.keys()))
    parser.add_argument("--data", default=None, help="path to conversation JSON")
    parser.add_argument("--docs", default=None, help="path to doc-corpus JSON (for A1/A2/A5)")
    args = parser.parse_args()

    module_name = BASELINES[args.baseline]
    module = __import__(module_name, fromlist=["run"])

    conversation = load_conversation(args.data)

    if args.baseline in ("a1", "a2", "a5"):
        docs = load_docs(args.docs)
        metrics = module.run(docs, conversation)
    else:
        metrics = module.run(conversation)

    print(json.dumps(metrics.summary(), indent=2))
    out_path = metrics.save()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
