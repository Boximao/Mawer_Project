# Agentic RAG MVP spec (Apple filings)

Status: **binding for Jason / Max / Yupo (Jason)**  
Horizon: **2–3 hour workshop MVP**  
Demo issuer: **Apple Inc. only**  
Primary goals: **accuracy, reliability, non-bypassable data control, non-bypassable HITL**

This file is the shared contract. Implementation may be incomplete; behavior must not contradict it.

---

## 1. Product boundary

Build a **question-answering agent over an ingested Apple corpus**, not a general research copilot.

In scope:

- Ingest a **small** Apple document set (10-Q/10-K and similar filings Max collects). Multiple file types are allowed (PDF, HTML, Markdown, CSV) if they are in the manifest.
- Hybrid retrieve → BGE rerank → bounded agent loop → **claim-level citations**.
- If evidence is insufficient, **abstain** and force human review.

Out of scope for this MVP:

- Live web search or URL fetch **during a query**.
- Non-Apple issuers.
- Local generation LLMs.
- Bloomberg/paid terminals.
- Production multi-tenant SaaS.

Official workshop knowledge-base vision (chats, meetings, news) is **architecture-compatible later**. It is not today’s corpus.

---

## 2. Data control (agent cannot ignore)

The model is untrusted. Control is **code + database**, not the system prompt.

1. **Allowlist only.** At query time the retriever filters `document_id IN corpus_manifest` where `ingest_status = 'ready'`.
2. **No web tool.** The agent tool list is exactly: `search_chunks`, `get_chunk`, `get_document_meta`. Anything else is a spec violation.
3. **Prompt may only contain retrieved chunk text** plus the user question. Raw files and secrets never go to the vendor API except those chunks.
4. **Manifest in Git** (`data/corpus_manifest.json`): `document_id`, source URI used at ingest time, sha256, mime type, filing period, page count, ingest git SHA.
5. If retrieval returns nothing usable, the loop **must** go to `ABSTAINED` / `AWAITING_HUMAN`. The generator is not allowed to “recall” Apple facts from pretraining.

Max owns ingest. Query path never re-downloads.

---

## 3. Infrastructure (this device + Cursor)

Accepted demonstration topology (talk laptop + fixture + Neon pgvector + cloud rerank) is preserved in [`RUNNING_ENVIRONMENT.md`](./RUNNING_ENVIRONMENT.md). This section remains the binding contract.

### 3.1 Hardware this spec is sized for

Profiled machine: **GUEST-LAP-04** — AMD **Ryzen 7 PRO 8840U** (Hawk Point, 8c/16t), **32 GB RAM**, **Radeon 780M** iGPU (Task Manager “1 GB”; actually shared DDR5), ~400 GB free SSD, Windows 64-bit, Cursor as the only IDE.

What this chip is good at:

- **Local query/ingest embeddings** (`bge-small-en-v1.5`, 384-d) for a tiny Apple corpus.
- Driving a **browser fixture** and HDMI output. Do **not** size the 3:00 path around a local 278M reranker.

What it is **not** good at (measured class of this APU, not a guess):

- **Local chat LLMs.** Hawk Point **NPU is not supported** for RyzenAI hybrid LLM. 780M is not a CUDA card. A 7B–8B Q4 will steal RAM from Postgres + reranker and blow the 45s budget.
- **Docker-as-the-only-path on this guest image.** This laptop currently has **no working `python` (Store stub only) and no Docker**. Compose remains valid on Max’s machine *if* Docker exists there. It is not the default for GUEST-LAP-04.

### 3.2 Topology (binding)

**Shared database (no laptop-to-laptop corpus sync):**

- **Vercel App is not a vector database.** Vercel Postgres was retired; Vercel only *provisions* marketplace Postgres (usually **Neon**). Do not deploy the FastAPI/HITL/reranker process to Vercel as the system of record.
- **Use one Neon project** (free plan, `pgvector` included). Jason and Max both put the **same pooled `DATABASE_URL`** in local `.env` (never Git). Ingest once; both laptops query the same chunks/vectors/HITL rows.
- Optional: create that Neon DB from the **Vercel Marketplace “Neon”** tile if someone already has a Vercel login (`vc i neon`). Functionally identical to neon.tech signup.
- **Supabase** is a fine backup (also free pgvector) but can **pause after ~7 days idle**; Neon auto-wakes after 5 min sleep (first query ~1s cold start — retry once in the demo).
- Git still holds **source PDFs + `corpus_manifest.json`**. Do not copy embedding caches between machines.

| Choice | Verdict |
|---|---|
| **Cloud pgvector (Neon free)** | **Use this.** Shared with Max, SSL, no Windows Postgres. |
| Vercel as *host* for the agent | **No.** Use Vercel only as an optional Neon signup shortcut. |
| **Local pgvector** | **Do not use on this laptop.** |

```
Browser  demo/index.html  ──always ready──► boardroom (fixture)
                │
                ▼ live
           FastAPI (localhost)  — orchestration + HITL only
                │
     ┌──────────┼──────────────┐
     ▼          ▼              ▼
  Embed query Neon pgvector  Cloud rerank
  (tiny CPU     chunks+      (Cohere or Jina)
   or same      vectors      30 pairs / query
   cloud API)
```

**Embed vs rerank (this APU is limited — do not treat them the same):**

| Step | Load on GUEST-LAP-04 | Binding |
|---|---|---|
| Ingest embeddings (3 Apple PDFs, once) | Light (minutes on CPU) | **Local `bge-small-en-v1.5`** is OK. Write **once** to Neon so Max does not re-embed. |
| Query embedding (one sentence) | Trivial | Same model as ingest (local 384-d). |
| Rerank (up to 30 chunk pairs, 512 tokens) | **Heavy** (278M+ cross-encoder, RAM + 45s budget) | **Cloud rerank API**, not local. |
| Local `bge-reranker-base` | First download + RAM fight with browser/Cursor | **Fallback only** if the rerank API key is missing. Prefer **RRF-only** (skip cross-encoder) over loading 278M mid-demo. |

| Layer | Binding choice | Settings |
|---|---|---|
| Coding | Cursor **local**; optional **Cloud Agents** | Cloud = Ubuntu **dev VM**, not the vector DB. |
| App process | **One FastAPI + uvicorn** on `127.0.0.1:8000` | Workers **1**. |
| Python | **3.12 x64** (`winget install Python.Python.3.12`) | Never Store `python.exe`. |
| Vector / state DB | **Neon Postgres 16 + pgvector** | Region closest to Calgary. Pooled URI + `sslmode=require`. |
| Index | Cosine **`vector(384)`**; HNSW `m=16, ef_construction=64`; `ef_search=40` | Tiny corpus; seq scan OK if n &lt; 200. |
| Embeddings | **Local `BAAI/bge-small-en-v1.5` (CPU)**; optional later OpenAI 1536-d only after full re-embed | Query = one vector. Never mix dimensions. |
| Rerank | **Cloud: Cohere `rerank-v3.5` or Jina `jina-reranker-v2-base-en`** | `top_n=6`, timeout 8s. Env: `COHERE_API_KEY` or `JINA_API_KEY`. |
| Rerank fallback | **RRF only** (no local cross-encoder on the talk machine) | Still show a rerank *panel* in `demo/index.html`. |
| Generation | **No OpenAI on this machine today.** 3:00 = `demo/index.html`. Live chat only if someone adds `OPENAI_API_KEY` or `GEMINI_API_KEY` | Temp **0**, top_p **1**, max_tokens **800**. |
| Presentation | **§3.6** — company TV + HDMI + Wi‑Fi; **Chrome/Edge full screen on `demo/index.html`** | Live Neon is optional stretch after the fixture script. |
| Secrets | `.env` gitignored | `DATABASE_URL`, `HITL_SIGNING_KEY`, `HITL_PIN`; OpenAI/Cohere/Jina optional. |
| Neon project | `dawn-moon-44249230` (production branch) | Shared by Jason/Max. CLI link still requires `neon login`. |

**Do not** deploy this MVP to Vercel as system of record (no durable pgvector + HMAC process). **Do not** treat a Cursor Cloud Agent disk as production data.

### 3.3 Main workflow control (who is in charge)

The LLM is a **drafting subroutine**. Control lives in **deterministic Python + Postgres**.

1. HTTP `POST /runs` creates `runs` row `state=RECEIVED`.
2. Orchestrator (`src/agent/loop.py`) is a **plain for-loop**, max **4** iterations, not an unbounded ReAct agent.
3. Each iteration: **write `agent_steps` first** → retrieve → rerank → generate JSON → validate (numeric verbatim + entailment flag from the same JSON, temp 0).
4. Tool allowlist enforced in code: only `search_chunks`, `get_chunk`, `get_document_meta`. No `requests.get`, no browser, no shell.
5. Stop: passing validate **or** max steps **or** 40k tokens **or** 45s **or** no new chunk ids → `AWAITING_HUMAN`.
6. `RELEASED` / human-acked `ABSTAINED` only via `POST /hitl/approve` HMAC (see §6).
7. FastAPI dependency `require_not_released_by_model()`: response builder **zeros `answer`** unless DB state is `RELEASED` with verified signature.

**Framework rule:** use **Pydantic models + our loop**; `openai` SDK only if a key appears. Do **not** use LangGraph/LangChain/LlamaIndex as the state machine. Those frameworks’ “HITL interrupt” is easy to mis-wire so the model resumes past approval. HMAC + SQL trigger remain the source of truth.

Entailment for this MVP: the generator must emit `claims[].supported: bool` and `claims[].quote`; Python checks the quote is a substring of an allowlisted chunk **and** numbers match. No second NLI model (RAM/time).

### 3.4 Python dependencies (agent harness)

Pin in `requirements.txt`. No extras.

| Package | Role |
|---|---|
| `fastapi`, `uvicorn[standard]` | App + static `/demo` |
| `pydantic>=2`, `pydantic-settings` | Claim/envelope schema, env |
| `openai` | Optional chat JSON if a key appears later |
| `sentence-transformers` | Local `bge-small-en-v1.5` embeddings (CPU, no API key) |
| `tiktoken` | 800/120 chunking with `cl100k_base` |
| `psycopg[binary]`, `pgvector` | Postgres + vectors |
| `pypdf` | Filing ingest (Max) |
| `numpy` | Score helpers |
| `onnxruntime` / `huggingface_hub` | **Not on the talk path.** Do not install mid-demo. |
| `python-dotenv`, `httpx` | Env + timeouts |
| `pytest` | Eval hooks |

**Do not add:** `torch`+CUDA, `langchain*`, `langgraph`, `llama-index`, `chromadb`, `faiss-gpu`, `ollama`, `vllm`, `unstructured` (too heavy), `flagembedding` if it pulls a full GPU torch stack — prefer a **small ONNX export** of the reranker. If ONNX export costs too much time, **CPU `sentence-transformers` + torch CPU** is the only allowed fallback (still no CUDA extra index).

Install:

```text
winget install Python.Python.3.12
py -3.12 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

### 3.5 Defects to avoid

| Defect | Why it kills this MVP |
|---|---|
| Local 7B+ generator / Ollama / RyzenAI NPU | Unsupported or too slow; wrecks 45s stop; 780M is not VRAM. |
| Docker-only demo on GUEST-LAP-04 | Docker is missing here; presentation dies. |
| Windows Store `python.exe` | Silent “Python was not found” during the talk. |
| LangChain default agent | Hidden web tools, unbounded steps, HITL skip. |
| Single “confidence” float | Cannot eval citations vs retrieval. |
| Live web at query time | Breaks allowlist; judges will ask. |
| Vercel as DB | Reranker + HMAC + pgvector will not live there. |
| Sharing `.env` in Git or screenshots | Key leak. |
| Multi-worker uvicorn | Two copies of BGE → RAM spike / lock. |
| HNSW + huge `ef` on 50 chunks | Cargo-cult; keep defaults above. |
| Chunking through tables | Wrong EPS/sales; numeric gate fails. |
| Letting the model set `human_review=false` | Spec violation; UI must ignore it. |
| Mixing local 384-d with OpenAI 1536-d | Dimension mismatch. This workshop stays **384-d BGE** unless both machines re-embed together. |
| Installing local Postgres on GUEST-LAP-04 | No Docker; Windows service fight. Use **Neon**. |
| Blocking the talk on OpenAI | This machine has **no key**. Fixture demo is valid. |
| Downloading any local reranker mid-demo | Misses the talk. Cloud rerank or **RRF-only**. |
| Presenting from Cursor Simple Browser | IDE chrome on the TV. Use **full-screen Edge/Chrome**. |
| Depending on Wi‑Fi for the five-minute script | Room Wi‑Fi exists; fixture still must run if Neon is cold or login failed. |

### 3.6 Boardroom running environment (binding)

Company provides **TV, HDMI, Wi‑Fi**. That does **not** change the primary process.

| Item | Binding |
|---|---|
| Display | Presenting laptop → HDMI → company TV. Duplicate desktop. |
| Browser | **Edge or Chrome, F11.** Not Cursor. |
| URL | Open `demo/index.html` (or `npx serve demo` if `file://` is blocked). |
| Network | Wi‑Fi is available. Use it to wake Neon (`SELECT 1`) **before** the talk if showing a stretch live query. The **scored script does not require Wi‑Fi**. |
| Script | §9 five minutes. PIN `2026`. |
| Stretch (optional, after script) | If Max has `DATABASE_URL` and Python: one hybrid retrieve screenshot or query against Neon. No live LLM required. |
| Forbidden during Q&A | Docker, local Postgres, Ollama, Vercel deploy, `pip install` of torch/rerankers. |

---

## 4. Retrieval and generation settings

### 4.1 Chunking (ingest)

SEC PDFs mix prose and tables. Naive token windows destroy numbers.

| Parameter | Value |
|---|---|
| Target chunk size | **800 tokens** (tiktoken `cl100k_base`, ≈ OpenAI embed tokenizer) |
| Overlap | **120 tokens** |
| Hard max | **1,000 tokens**; if a table still overflows, split by **row groups** and **repeat table header** on each piece |
| Min chunk | **50 tokens** (drop running headers/footers) |
| Unit of cite | `(document_id, page, section, chunk_id)` |
| Table rule | Never split a numeric row across chunks. Header row travels with every table fragment. |

Metadata on every chunk: `document_id`, `filing_period`, `form_type`, `page`, `section_path`, `chunk_type` (`prose` \| `table` \| `note`), `sha256`.

### 4.2 Hybrid retrieve + rerank

| Parameter | Value |
|---|---|
| Vector | **`bge-small-en-v1.5`, 384-d**, cosine, pgvector `hnsw` (OpenAI 1536-d only after a key + full re-embed) |
| Keyword | Postgres **tsvector** (simple English config) on `chunk_text` |
| Fusion | Reciprocal Rank Fusion, `k = 60` |
| `top_k_vector` | **20** |
| `top_k_keyword` | **20** |
| RRF candidate cap | **30** unique chunks |
| Rerank | Cloud Cohere/Jina on ≤30 pairs; **RRF-only** if no rerank key |
| `top_k_context` | **6** chunks into the generator (8 if all remaining are `table`) |
| Retrieval discard | RRF/fusion normalized score **&lt; 0.40** dropped before rerank |
| Rerank keep | Keep top 6; drop rerank score below **0.20** even if in top 6 |

### 4.3 Claim / confidence gate (financial)

Two scores. No single opaque “confidence.”

| Gate | Rule |
|---|---|
| Chunk may enter agent | fusion ≥ **0.40**; then cloud rerank top 6, or **RRF top 6** if no rerank key |
| Claim may be shown | **entailment ≥ 0.90** vs cited span **and** citation present |
| Numbers | Digit string in the claim **must appear verbatim** in the cited chunk (normalize commas/`$`/nbsp). Mismatch → fail |
| Auto-answer to user | **Forbidden without HITL signature** (see §6) |
| Soft / medium answers | **None.** No 85% “answer with warning.” |

If any claim fails: `answer = null`, `human_review = true`, `reason_codes` includes `INSUFFICIENT_EVIDENCE` and/or `NUMERIC_MISMATCH` / `UNCITED_CLAIM`.

### 4.4 Generator decoding

Financial QA is extraction, not style.

| Parameter | Value |
|---|---|
| Temperature | **0** |
| Top-p | **1.0** (ignored in practice at temp 0; do not set 0.9) |
| Top-k (API) | unset / full |
| Max output tokens | **800** |
| JSON mode | on; envelope in §6.4 plus `claims[].supported`, `claims[].quote` |
| Seed | `42` if the API supports it (eval stability) |

---

## 5. Agent loop and stop signals

Max steps **4**. Token budget **40,000** (prompt + completion, all calls). Wall clock **45s**.

Stop on the **first** of:

1. Validator accepts every claim (still cannot **release** without HITL).
2. Step count = 4.
3. Token budget exceeded.
4. Timeout.
5. No new unique `chunk_id` after a search.

Outcomes 2–5 without a passing draft → `ABSTAINED`. Outcome 1 → `AWAITING_HUMAN` with a draft payload. **Never** `RELEASED` from the model.

Each step writes an append-only `agent_step` row **before** the next tool call. The LLM cannot update that table.

---

## 6. HITL state machine (cannot be escaped)

### 6.1 States

```
RECEIVED
  → RETRIEVING
  → RERANKING
  → DRAFTING
  → VALIDATING
  → AWAITING_HUMAN
       → RELEASED          # only with valid human signature
       → ABSTAINED         # only with valid human signature (ack of failure)
       → RETURN_TO_DRAFT   # human requested another retrieve/generate cycle (counts against max steps)
  → FAILED                 # timeout / crash; still requires human ack to close
```

Illegal: `VALIDATING → RELEASED`, `DRAFTING → RELEASED`, any jump that skips `AWAITING_HUMAN`.

`human_review` on the API response is **true** for every run until `RELEASED` with a verified signature.

### 6.2 Why the model cannot forge approval

- Signing key `HITL_SIGNING_KEY` lives only in the **HITL service** process (env). It is **not** in the model context, not in tool schemas, not in client JS.
- The generator has **no** `approve_run` tool.
- Postgres trigger: `UPDATE runs SET state = 'RELEASED'` is rejected unless `approval_signature` verifies.
- UI “Approve” calls `POST /hitl/approve` with human identity from a **workshop PIN or logged-in user**, not a boolean the model emitted.

### 6.3 Signature (explicit, non-fabricable)

Canonical string (UTF-8, no extra spaces):

```
v1|{run_id}|{from_state}|{to_state}|{step_id}|{payload_sha256}|{approver_id}|{unix_ts}
```

`payload_sha256` = SHA-256 of the exact JSON draft (answer, claims, citations) bytes that will be released.

`approval_signature` = **HMAC-SHA256**(`HITL_SIGNING_KEY`, canonical string), hex.

Store: `run_id`, `from_state`, `to_state`, `step_id`, `payload_sha256`, `approver_id`, `unix_ts`, `signature`, `prev_audit_hash`.

`prev_audit_hash` chains the audit log (hash of previous row). Tampering a step breaks the chain.

Verifier: recompute HMAC; compare with `hmac.compare_digest`; reject if `|now - unix_ts| > 300s`; reject if `payload_sha256` ≠ current draft hash; reject if `from_state` ≠ DB state.

A model outputting `"approved": true` or a fake hex string **does nothing**. Only the HITL endpoint can write a signature.

### 6.4 API envelope (always)

```json
{
  "run_id": "uuid",
  "state": "AWAITING_HUMAN",
  "human_review": true,
  "answer": null,
  "draft_answer": "optional string if VALIDATING passed",
  "claims": [],
  "citations": [],
  "reason_codes": [],
  "stop_reason": "awaiting_human | max_steps | timeout | token_budget | empty_retrieve",
  "usage": { "steps": 0, "tokens": 0, "latency_ms": 0 }
}
```

After HITL approve, a **second** fetch may return `state=RELEASED`, `human_review=false`, `answer` copied from signed draft. Until then `answer` stays `null`.

---

## 7. Evaluation (Git-controlled)

Commit:

- `eval/questions.jsonl` — question, expected facts, expected `document_id` + page
- `eval/metrics.md` — definitions
- `data/corpus_manifest.json`
- this spec
- prompts under `prompts/`

Each eval run records `git_commit_sha`, model names, chunk/top-k/temp settings.

Metrics:

1. **Citation hit**: gold page in `top_k_context` after rerank.
2. **Claim support**: human or checklist: each claim span-entailed.
3. **Numeric exact**: extracted numbers match filing.
4. **Abstain quality**: unanswerable questions must not `RELEASED` without HITL; draft must be null or flagged.

Do not commit: traces, `.env`, `eval/runs/*`, embedding caches, Postgres volumes.

---

## 8. Repo layout (owners)

```
docs/AGENTIC_RAG_SPEC.md          # this file (Jason) — committed
docs/RUNNING_ENVIRONMENT.md       # accepted demo topology (Jason)
docs/TECHNICAL_SOLUTION.md        # high-level architecture + repo map (Jason)
prompts/                          # Jason
src/finembed/                # Max — ingest/chunk/embed (was src/ingest in earlier draft)
src/retrieve/                # shared hybrid + rerank
src/agent/                   # Jason — loop, stops
src/hitl/                    # Jason — HMAC + state machine
src/eval/                    # Jason
data/raw + data/processed    # Max — Apple corpus
data_filing/                 # legacy Desktop copies
data/corpus_manifest.json    # Max
eval/questions.jsonl         # Jason
requirements.txt             # pinned harness
demo/index.html              # 3:00 primary UI
docker-compose.yml           # optional; not used on GUEST-LAP-04
neon.ts                      # Neon project config (after CLI login)
```

Python for the service. **Presentation UI** is `demo/index.html` so the loop can be shown even when Cloud/API is slow.

---

## 9. Presentation (required, not optional)

The 3:00 boardroom demo must **show the loop**, not a chatbot that hides retrieval.

**Required on screen at all times**

| Panel | What judges must see |
|---|---|
| Corpus allowlist | Ingested Apple docs only (id, form, period). No live web. |
| State machine | Current state highlighted; illegal skip to `RELEASED` impossible. |
| Retrieve | Hybrid hits (vector + keyword) with scores. |
| Rerank | Cloud Cohere/Jina scores **or RRF ranks** and the **6** chunks kept. |
| Draft | Claims with `human_review: true` and `answer: null`. |
| Validate | Entailment, numeric verbatim pass/fail. |
| HITL | Approve control; signature preview (`v1\|run\|…`). Released answer appears **only after** approve. |
| Abstain path | A second scripted question **not** in corpus → no invented Apple/Tesla facts. |

**Five-minute live script** (fixture is primary even with room Wi‑Fi)

1. HDMI to TV. Full-screen browser on `demo/index.html`. Show allowlist (three 10-Qs).
2. **Script 1 · Diluted EPS** → Run agent. Pause on `AWAITING_HUMAN` — point at `answer: null`.
3. PIN `2026` → **Sign & release**. Point at the page-4 quote.
4. **Script 2 · Out of corpus** → Tesla → `ABSTAINED`, human ack. No invented number.
5. One sentence: “Production index is Neon pgvector; this is the same state machine. Query-time tools cannot hit the web.”

Optional stretch **only after** that script: Max shows Neon chunks if the DB is linked. Do not fake a live web search.

Demo HMAC in `demo/app.js` is a **presentation stand-in** (PIN + WebCrypto). Production HMAC (`HITL_SIGNING_KEY` in FastAPI + Postgres trigger) is the non-forgeable gate. Say this if judges ask.

---

## 10. Definition of done (workshop)

1. Answerable 10-Q question → draft with claim-level citations (doc, period, page, quote).
2. Out-of-corpus question → abstain, `human_review=true`, no pretraining.
3. UI or DB: run stays `AWAITING_HUMAN` until HMAC/PIN approve; JSON `approved: true` does not release.
4. Live or fixture walkthrough of retrieve → rerank → draft → validate → HITL.
5. Eval file run prints commit SHA + citation-hit for at least 3 questions **if** Python is available.

Time-box: **no local reranker download** during the talk. Cloud rerank or RRF-only.
