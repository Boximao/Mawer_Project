# Agent loop (Jason)

Bounded for-loop (max 4 / 40k tokens / 45s). Spec §5. No LangGraph.

```text
python -m agent
```

Serves FastAPI on `127.0.0.1:8000` (demo UI + `POST /runs` + `POST /hitl/approve`).
Without `DATABASE_URL`, retrieve uses `data/processed/**/records.jsonl` (and `.npz` if a query embedder is available).
