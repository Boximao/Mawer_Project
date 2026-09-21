# Mawer Apple agentic RAG MVP

Allowlisted Apple-filing QA with hybrid retrieval, a bounded deterministic
agent loop, claim-level citations, numeric validation, and mandatory HITL.
The binding contract is [`docs/AGENTIC_RAG_SPEC.md`](docs/AGENTIC_RAG_SPEC.md).
Demonstration diagrams (product view and technical structure) are in
[`docs/diagrams/`](docs/diagrams/README.md).

## Authoritative corpus

Max's parsed documents live in `data/processed/`:

- 3 Apple 10-Q filings
- 204 chunks in `records.jsonl`
- source tables and exact citable `raw_text`
- local BGE 384-d vectors in `embeddings_bge.npz`

`data/corpus_manifest.json` is the query-time allowlist. The agent never
downloads source URLs and never exposes unallowlisted chunks.

## Run the complete demo

Windows PowerShell:

```powershell
C:\Users\GuestUser04\AppData\Local\Programs\Python\Python312\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
C:\Users\GuestUser04\AppData\Local\Programs\Python\Python312\python.exe run_demo.py
```

Open <http://127.0.0.1:8000/>. Workshop PIN: `2026`.

With `DATABASE_URL` and `HITL_SIGNING_KEY`, the app uses canonical Neon
`documents`, `chunks`, `runs`, `agent_steps`, and `approvals` tables. If Neon
is unavailable, it automatically uses the committed processed corpus and an
in-memory run store. `answer` remains null until signed human approval in
both modes.

## Corpus and database setup

The vectors are already present. To regenerate them from `data/processed`
without reparsing PDFs:

```powershell
$env:PYTHONPATH = "src"
python -m finembed embed-bge
```

To apply migrations and upload the canonical BGE corpus to Neon:

```powershell
python scripts/init_cloud_db.py
python -m finembed embed-neon
```

No OpenAI key is needed for ingest, retrieval, or the extractive fallback.
An optional OpenAI/Gemini key enables JSON drafting; Cohere/Jina enables
cloud reranking. Missing or failed vendor services fall back to RRF and
extractive claims.

## Runtime guarantees

- Query tools are exactly `search_chunks`, `get_chunk`,
  `get_document_meta`.
- Loop limits: 4 steps, 40,000 tokens, 45 seconds, and no-new-chunk stop.
- Vector 20 + keyword 20 → weighted RRF (`k=60`, cap 30) → cloud rerank
  top 6 or RRF-only.
- Explicit filing periods scope retrieval; otherwise the latest filing is
  used.
- Quotes are verified against allowlisted chunks.
- Every number must occur in the cited quote itself.
- Model-supplied citation metadata is discarded and rebuilt by Python.
- `RELEASED`/`ABSTAINED` require a fresh server HMAC approval and legal SQL
  state transition.

## Verification

```powershell
python -m pytest -q
python src/eval/run_eval.py
python src/eval/run_agent_eval.py
```

The first evaluation checks all 18 retrieval/abstain questions. The second
runs the complete BGE → hybrid retrieve → draft → validate loop and reports
citation hit, claim support, numeric exactness, fact hit, and abstain quality.
