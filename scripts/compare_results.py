"""
Aggregates every results/*.json (produced by run_baseline.py /
run_gap_experiment.py) into one comparison table.

Usage:
    python scripts/compare_results.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

COLUMNS = ["run_name", "n_turns", "avg_ttft", "p95_ttft",
           "avg_total_latency", "avg_reuse_ratio", "avg_answer_quality"]


def main():
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        payload = json.loads(f.read_text())
        summary = payload.get("summary", {})
        if summary:
            rows.append(summary)

    if not rows:
        print(f"No results found in {RESULTS_DIR}. Run a baseline or gap experiment first.")
        return

    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in COLUMNS}
    header = " | ".join(c.ljust(widths[c]) for c in COLUMNS)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in COLUMNS))


if __name__ == "__main__":
    main()
