"""
Loads LongMemEval (long-term conversational memory benchmark) and converts
it into the project's turn-list format.

*** IMPORTANT — VERIFY BEFORE RELYING ON THIS ***
I am not fully confident of the exact HuggingFace dataset repo id / schema
for LongMemEval from memory (this is a specific external resource, and
getting the id wrong is a real risk). Before running this:

  1. Enable web search / go to https://huggingface.co/datasets and search
     "LongMemEval", OR check the paper's official repo (search
     "LongMemEval benchmarking chat assistants long-term interactive memory"
     GitHub) for the canonical download link / HF dataset id.
  2. Update DATASET_ID below to whatever you confirm.
  3. Inspect one raw example (this script prints one) and adjust
     `convert_example()` to match the ACTUAL field names -- LongMemEval's
     schema (session lists, questions, evidence turns, answers) may not
     exactly match the placeholder field names guessed below.

Until verified, use scripts/generate_synthetic_dataset.py instead -- it
needs no external download and matches the exact same evaluation pattern
(facts stated early, recalled later).
"""
import argparse
import json
from pathlib import Path

DATASET_ID = "TODO-verify-exact-huggingface-dataset-id-for-LongMemEval"


def load_raw():
    from datasets import load_dataset
    return load_dataset(DATASET_ID)


def convert_example(example: dict) -> list[dict]:
    """
    TODO: adjust field names once you've confirmed the real schema by
    printing a raw example (see main() below). LongMemEval conversations
    are organized as multiple sessions with a final question that requires
    recalling information from earlier sessions -- map that structure to:

        [{"query": ..., "fact_to_store": ..., "gold_answer": ...}, ...]

    A rough sketch (field names are placeholders, NOT confirmed):
    """
    turns = []
    for session in example.get("sessions", []):          # TODO: confirm key name
        for msg in session.get("messages", []):           # TODO: confirm key name
            turns.append({
                "query": msg.get("content", ""),           # TODO: confirm key name
                "fact_to_store": msg.get("content", "") if msg.get("role") == "user" else None,
                "gold_answer": None,
            })
    # final recall question
    turns.append({
        "query": example.get("question", ""),              # TODO: confirm key name
        "fact_to_store": None,
        "gold_answer": example.get("answer", ""),           # TODO: confirm key name
    })
    return turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/longmemeval_converted.json")
    parser.add_argument("--limit", type=int, default=20, help="how many examples to convert")
    parser.add_argument("--inspect-only", action="store_true",
                         help="just print one raw example's keys/schema and exit, don't convert")
    args = parser.parse_args()

    ds = load_raw()
    split = ds[list(ds.keys())[0]]

    if args.inspect_only:
        example = split[0]
        print("Raw example keys:", list(example.keys()))
        print(json.dumps(example, indent=2)[:3000])
        print("\n^ use this to fix convert_example() field names above before converting.")
        return

    conversations = []
    for i, example in enumerate(split):
        if i >= args.limit:
            break
        conversations.append({"conversation_id": i, "turns": convert_example(example)})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(conversations, indent=2))
    print(f"Wrote {len(conversations)} conversations -> {out_path}")


if __name__ == "__main__":
    main()
