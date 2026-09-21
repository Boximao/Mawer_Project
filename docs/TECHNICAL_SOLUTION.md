# High-level technical solution

Status: **local binding architecture** (saved on disk under `docs/`)  
Parent contract: [`AGENTIC_RAG_SPEC.md`](./AGENTIC_RAG_SPEC.md)  
Runtime: [`RUNNING_ENVIRONMENT.md`](./RUNNING_ENVIRONMENT.md)  
Presentation diagrams: [`diagrams/product-design.png`](./diagrams/product-design.png) (product view) and [`diagrams/technical-architecture.png`](./diagrams/technical-architecture.png) (this document as one picture) — sources and re-render script in [`diagrams/`](./diagrams/README.md)

If this file and the spec disagree, **the spec wins**. This file is the map of *how* we implement the spec, including the repo tree Max and Jason share.

---

## 1. Problem and solution in one paragraph

Investment research cannot trust a single-shot RAG answer. The MVP is an **allowlisted, Apple-only, agentic retrieve–rerank–draft–validate loop** with **claim-level citations** and a **non-bypassable HITL state machine**. Max's authoritative parsed corpus lives in `data/processed`; canonical 384-d BGE vectors are available locally and in Neon pgvector (`dawn-moon-44249230` / `production`). The **scored 3:00 demo** is `demo/index.html` on the company TV (HDMI), and the same page automatically uses live FastAPI when served from `127.0.0.1:8000`.

---

## 2. System planes

```
┌────────────── talk (required) ──────────────┐
│  TV ← HDMI ← Edge/Chrome F11                │
│  demo/index.html  (fixture loop + PIN 2026) │
└─────────────────────────────────────────────┘
                      │ stretch after script
┌────────────── control (Python) ─────────────┐
│  FastAPI :8000  1 worker                    │
│  src/agent/loop.py  max 4 / 40k tok / 45s   │
│  tools: search_chunks | get_chunk | get_document_meta only │
│  src/hitl  HMAC-SHA256 + SQL trigger        │
└──────────┬──────────────┬──────────┬────────┘
           ▼              ▼          ▼
     bge-small 384-d   Neon       Cohere/Jina
     local CPU         pgvector   rerank ≤30
                       + tsvector RRF if no key
                       + HITL rows
```

The **LLM does not own control**. Postgres `runs.state` and HMAC do.

---

## 3. Project structure (as-built)

```
Mawer_Project/
├── docs/
│   ├── AGENTIC_RAG_SPEC.md          # contract (wins)
│   ├── RUNNING_ENVIRONMENT.md       # 3:00 topology
│   └── TECHNICAL_SOLUTION.md        # this file
├── demo/                            # scored presentation UI
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── data/
│   ├── raw/AAPL/                    # Max — source filings
│   ├── processed/                   # Max — chunks, tables, local npz
│   └── corpus_manifest.json         # allowlist (Git)
├── data_filing/                     # original GitHub Desktop PDFs (legacy)
├── src/
│   ├── finembed/                    # Max — ingest, chunk, embed → Neon
│   ├── retrieve/                    # hybrid + RRF + cloud rerank
│   ├── agent/                       # bounded loop, stops, JSON draft
│   ├── hitl/                        # HMAC, PIN, state transitions
│   └── eval/                        # citation-hit / numeric / abstain
├── prompts/
├── eval/questions.jsonl
├── requirements.txt
├── neon.ts                          # defineConfig({}); linked production
├── package.json                     # @neon/config only
├── docker-compose.yml               # unused on GUEST-LAP-04
└── .env / .env.local                # gitignored DATABASE_URL
```

| Path | Owner | Spec role |
|---|---|---|
| `src/finembed/` | Max | Ingest/chunk/embed. Must write **allowlisted** rows only. Spec default vector dim is **384** (`bge-small-en-v1.5`). If OpenAI 1536-d is used, **full re-embed** and one dim for the whole team. |
| `src/retrieve/` | Shared | `search_chunks`: vector 20 + keyword 20 → RRF cap 30 → cloud rerank top 6 (or RRF-only). Discard fusion &lt; 0.40. |
| `src/agent/` | Jason | Plain for-loop, 4 steps, no LangGraph. |
| `src/hitl/` | Jason | `AWAITING_HUMAN` → `RELEASED` / `ABSTAINED` only with HMAC. |
| `demo/` | Jason | Fixture = talk path. PIN stand-in; say so if asked. |
| Neon `filing_chunks` (or equivalent) | Shared | Source of truth for retrieval, not `embeddings.npz` on disk. |

---

## 4. Data flow

### 4.1 Ingest (Max, once)

1. PDF/HTML/MD/CSV under `data/raw/` (Apple only).
2. Parse + **table-aware chunk** (spec: ~800 tokens, 120 overlap; never split a numeric row; repeat table header).
3. Embed with **one** model (spec: local BGE 384-d unless the team jointly switches).
4. Upsert to Neon `filing_chunks` (`vector(dim)`, HNSW cosine, `doc_id` index, tsvector on text).
5. Append/update `data/corpus_manifest.json` (`document_id`, sha256, form, period, `ingest_status=ready`).

Query path **never** fetches URLs.

### 4.2 Query (agent)

1. `RECEIVED` → embed question → hybrid retrieve (allowlist filter) → `RETRIEVING`.
2. Cloud rerank or RRF → `RERANKING`.
3. Generator (if a chat key exists) JSON claims+quotes, temp **0**, top_p **1** → `DRAFTING`.
4. Python: quote ⊆ chunk text; numbers verbatim → `VALIDATING`.
5. Always `AWAITING_HUMAN`. `answer` stays **null**.
6. Human PIN/HMAC → `RELEASED` or ack `ABSTAINED`. Empty retrieve must not invent Tesla/Apple facts.

Stops: 4 steps, 40k tokens, 45s, no new `chunk_id`.

---

## 5. HITL (cannot escape)

States: `RECEIVED → RETRIEVING → RERANKING → DRAFTING → VALIDATING → AWAITING_HUMAN → RELEASED | ABSTAINED`.

Illegal: any skip to `RELEASED` from draft/validate.

Production signature:

```
HMAC-SHA256(HITL_SIGNING_KEY, v1|{run_id}|{from}|{to}|{step_id}|{payload_sha256}|{approver_id}|{unix_ts})
```

Key never in the prompt or tools. Demo `app.js` WebCrypto is **talk-only**.

---

## 6. Citation and confidence

Not one float.

- Retrieval: fusion ≥ 0.40 to enter the loop.
- Claim: entailment ≥ 0.90 **and** citation (`document_id`, period, page, quote).
- Numbers: digit string in the claim appears in the cited span (normalize `$`, commas).
- No 85% “answer with warning.” Fail → `human_review: true`, `answer: null`.

---

## 7. What is not in this architecture

- Vercel as app/DB of record  
- Local Postgres on GUEST-LAP-04  
- Local 7B / Ollama / RyzenAI NPU LLM  
- Local 278M reranker on the talk path  
- LangChain/LangGraph as the state machine  
- Live web search at query time  
- Mixing 384-d and 1536-d in one Neon table  

---

## 8. Workshop definition of done (same as spec §10)

1. In-corpus 10-Q → cited draft (doc, period, page, quote).  
2. Tesla / out-of-corpus → abstain.  
3. No release without HITL.  
4. TV shows the full loop via `demo/index.html`.  
5. Eval JSONL if Python is available.

---

## 9. File location

This document is stored at:

`docs/TECHNICAL_SOLUTION.md`

on the local workspace `C:\Users\GuestUser04\Documents\GitHub\Mawer_Project`. Commit it with the spec so Max has the same map; secrets stay in `.env.local` only.
