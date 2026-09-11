# Data

Put your datasets here, e.g.:

- `longmemeval_sample.json` — multi-session memory-recall conversations
  (list of `{"query": str, "fact_to_store": str?, "gold_answer": str?}`)
- `hotpotqa_sample.json` — multi-hop QA, used for A5 / Gap 2 quality tests
- `docs_corpus.json` — flat list of document chunks used by A1 / A2 / A5

Format expected by `scripts/run_baseline.py` and `scripts/run_gap_experiment.py`:

```json
[
  {"query": "My favorite color is blue.", "fact_to_store": "User's favorite color is blue.", "gold_answer": null},
  {"query": "What's my favorite color?", "gold_answer": "blue"}
]
```

See `RESEARCH_PLAN.md` Part 0, Step 0.2 for dataset recommendations.
