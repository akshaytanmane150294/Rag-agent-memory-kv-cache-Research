"""
Shared measurement harness used by every baseline (Part A) and every
research-gap experiment (Part B), so results stay directly comparable.

Metrics tracked per turn:
  - ttft               : time to first token (seconds)
  - total_latency       : full generation latency (seconds)
  - tokens_generated    : number of output tokens
  - kv_tokens_reused    : how many context tokens were served from cache
  - kv_tokens_recomputed: how many context tokens had to be recomputed
  - answer_quality      : task-specific score (F1 / EM / ROUGE), optional
"""
import json
import statistics as st
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class TurnRecord:
    turn_id: int
    ttft: float
    total_latency: float
    tokens_generated: int
    kv_tokens_reused: int
    kv_tokens_recomputed: int
    answer_quality: Optional[float] = None


class RunMetrics:
    def __init__(self, run_name: str):
        self.run_name = run_name
        self.records: list[TurnRecord] = []

    def record(
        self,
        turn_id: int,
        ttft: float,
        total_latency: float,
        tokens_generated: int,
        kv_tokens_reused: int,
        kv_tokens_recomputed: int,
        answer_quality: Optional[float] = None,
    ):
        self.records.append(
            TurnRecord(
                turn_id=turn_id,
                ttft=ttft,
                total_latency=total_latency,
                tokens_generated=tokens_generated,
                kv_tokens_reused=kv_tokens_reused,
                kv_tokens_recomputed=kv_tokens_recomputed,
                answer_quality=answer_quality,
            )
        )

    def summary(self) -> dict:
        if not self.records:
            return {}
        ttfts = [r.ttft for r in self.records]
        latencies = [r.total_latency for r in self.records]
        reuse_ratios = [
            r.kv_tokens_reused / max(1, r.kv_tokens_reused + r.kv_tokens_recomputed)
            for r in self.records
        ]
        qualities = [r.answer_quality for r in self.records if r.answer_quality is not None]

        def p95(vals):
            s = sorted(vals)
            return s[max(0, int(0.95 * len(s)) - 1)]

        return {
            "run_name": self.run_name,
            "n_turns": len(self.records),
            "avg_ttft": st.mean(ttfts),
            "p95_ttft": p95(ttfts),
            "avg_total_latency": st.mean(latencies),
            "avg_reuse_ratio": st.mean(reuse_ratios),
            "avg_answer_quality": st.mean(qualities) if qualities else None,
        }

    def save(self, out_dir: str = "results"):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(out_dir) / f"{self.run_name}.json"
        payload = {
            "run_name": self.run_name,
            "summary": self.summary(),
            "records": [asdict(r) for r in self.records],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        return str(out_path)


class Timer:
    """Small context manager to time a code block."""

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
