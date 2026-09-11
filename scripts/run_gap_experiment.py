"""
Entry point: run a chosen research-gap experiment (Gap 1-3) on a dataset.

Usage:
    python scripts/run_gap_experiment.py --gap gap1
    python scripts/run_gap_experiment.py --gap gap1 --warm-bonus 0.3
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GAPS = {
    "gap1": "gaps.gap1_cache_aware_retrieval",
    "gap2": "gaps.gap2_selective_recompute",
    "gap3": "gaps.gap3_joint_ttl_scheduling",
}


def load_conversation(data_path: str | None) -> list[dict]:
    if not data_path:
        return [
            {"query": "My favorite color is blue.", "fact_to_store": "User's favorite color is blue."},
            {"query": "What's my favorite color?"},
        ]
    return json.loads(Path(data_path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap", required=True, choices=list(GAPS.keys()))
    parser.add_argument("--data", default=None, help="path to conversation JSON")
    parser.add_argument("--warm-bonus", type=float, default=None, help="Gap 1 ablation param")
    parser.add_argument("--boundary-window", type=int, default=None, help="Gap 2 ablation param")
    args = parser.parse_args()

    module_name = GAPS[args.gap]
    module = __import__(module_name, fromlist=["run"])
    conversation = load_conversation(args.data)

    kwargs = {}
    if args.gap == "gap1" and args.warm_bonus is not None:
        kwargs["warm_bonus"] = args.warm_bonus
    if args.gap == "gap2" and args.boundary_window is not None:
        kwargs["boundary_window"] = args.boundary_window

    if args.gap == "gap3":
        print("gap3 is scheduler-logic only — run gaps/gap3_joint_ttl_scheduling.py directly, "
              "or wire PinningScheduler into your agent loop.")
        return

    metrics = module.run(conversation, **kwargs)
    print(json.dumps(metrics.summary(), indent=2))
    out_path = metrics.save()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
