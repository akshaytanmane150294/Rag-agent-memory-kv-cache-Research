# Improving RAG in Agents via Agent Memory + KV-Cache Optimization

Yeh repo tumhare literature review (RAG + Agent Memory + KV-Cache) ko implement/test karne ke liye
scaffold hai. Full step-by-step plan ke liye `RESEARCH_PLAN.md` dekho.

## Structure

```
rag-agent-memory-kv-cache/
├── RESEARCH_PLAN.md        <- full Hinglish step-by-step plan (baselines + gaps)
├── requirements.txt
├── common/                 <- shared utilities used by every experiment
│   ├── metrics.py           (RunMetrics harness: TTFT, latency, reuse ratio, quality)
│   └── prompt_utils.py      (prompt building helpers)
├── memory/                 <- agent memory system + cache tracking
│   ├── simple_memory.py     (vector-store based memory, A3 baseline)
│   ├── cache_tracker.py     (tracks which memory items are "warm" in KV cache)
│   └── cache_aware_memory.py (Gap 1: warm-aware retrieval)
├── baselines/               <- Part A: baselines to compare against
│   ├── a1_naive_rag.py
│   ├── a2_prefix_cache_rag.py
│   ├── a3_agent_memory.py
│   ├── a4_streaming_llm.py
│   └── a5_naive_full_reuse.py
├── gaps/                    <- Part B: research-gap implementations (your contribution)
│   ├── gap1_cache_aware_retrieval.py
│   ├── gap2_selective_recompute.py
│   └── gap3_joint_ttl_scheduling.py
├── scripts/                 <- entry points to run experiments
│   ├── run_baseline.py
│   ├── run_gap_experiment.py
│   └── compare_results.py
├── data/                    <- put your dataset(s) here (LongMemEval, HotpotQA, custom, etc.)
└── results/                 <- experiment outputs (json/csv) get written here
```

## Quick start

```bash
pip install -r requirements.txt

# run a baseline
python scripts/run_baseline.py --baseline a1

# run a gap/improvement experiment
python scripts/run_gap_experiment.py --gap gap1

# compare all logged results
python scripts/compare_results.py
```

## Notes

- Code files are **runnable skeletons**, not finished production code — sections marked `# TODO`
  need your model path, dataset path, and (for Gap 2 / A4 / A5) external repo integration
  (LMCache, StreamingLLM) filled in, as described in `RESEARCH_PLAN.md`.
- All experiments log through `common/metrics.py::RunMetrics` so results are directly comparable
  across baselines and gaps.
