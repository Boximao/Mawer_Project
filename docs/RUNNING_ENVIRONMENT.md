# Running environment (accepted)

Status: **accepted for the workshop demonstration**  
Parent contract: [`AGENTIC_RAG_SPEC.md`](./AGENTIC_RAG_SPEC.md) §3  
Architecture map: [`TECHNICAL_SOLUTION.md`](./TECHNICAL_SOLUTION.md)  
Does not change product behavior. If this file and the spec disagree, **the spec wins**.

Verdict: **talk laptop + fixture + Neon pgvector + cloud rerank**. Sentence-Transformers is the embedding library, not a database. Object storage is a later ingest URI, not a third cluster.

---

## 1. Topology

```
HDMI TV  ←  Edge/Chrome F11  ←  demo/index.html   (scored 3:00 path; offline-capable)
                                │
                                ▼ stretch only
                         FastAPI 127.0.0.1:8000  (1 uvicorn worker)
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
     Local BGE 384-d      Neon pgvector     Cloud rerank
     (CPU, query/ingest)  + tsvector        Cohere or Jina
                          + HITL tables     ≤30 pairs / query
                          + HMAC trigger
```

Three planes:

| Plane | What runs | Binding |
|---|---|---|
| Talk | Presenting laptop → HDMI → company TV. `demo/index.html` in Edge/Chrome full screen. PIN `2026`. | Fixture is the scored five-minute script. Not Cursor Simple Browser. |
| Data | Neon Postgres 16 + pgvector, project `dawn-moon-44249230` (production branch). | Jason and Max share one pooled `DATABASE_URL`. Ingest once. |
| Compute | Cloud rerank API. Optional cloud LLM if a key exists. Local CPU only for `bge-small-en-v1.5`. | No local 7B generator. No local 278M reranker on the talk path. |

Wake Neon with `SELECT 1` before HDMI if a stretch live query is planned. First query after sleep is ~1s. The scored script does not require Wi-Fi.

---

## 2. Talk machine (GUEST-LAP-04)

Profiled: AMD Ryzen 7 PRO 8840U (8c/16t), 32 GB RAM, Radeon 780M shared DDR5, ~400 GB free SSD, Windows 64-bit. Hawk Point NPU is **not** supported for RyzenAI hybrid LLM. 780M is not CUDA.

| Layer | Setting |
|---|---|
| Python | **3.12 x64** via `winget install Python.Python.3.12`. Never Store `python.exe`. |
| venv | `py -3.12 -m venv .venv` then `.\.venv\Scripts\pip install -r requirements.txt` **before** the room. |
| App | One FastAPI + uvicorn on `127.0.0.1:8000`, workers **1**. |
| IDE | Cursor local. Cloud Agents = Ubuntu **dev VM**, not the vector DB or corpus. |
| Docker | Compose in-repo is optional for Max **if** Docker exists there. **Not** the default on this guest image (Docker missing). |

---

## 3. Vector / state store

Sentence-Transformers (`BAAI/bge-small-en-v1.5`) produces **384-d** vectors. The store is **Postgres**, not a second vector product.

| Item | Binding |
|---|---|
| Product | Neon free, `pgvector` included. Region closest to Calgary. |
| URI | Pooled `DATABASE_URL` with `sslmode=require` in **local `.env` only**. |
| Index | Cosine `vector(384)`; HNSW `m=16`, `ef_construction=64`, `ef_search=40`. Seq scan OK if n &lt; 200. |
| Keyword | `tsvector` (simple English) on `chunk_text`; RRF `k=60`. |
| Same DB | `runs`, `agent_steps`, `approval_signature` / HMAC trigger. Do not split HITL out. |
| Backup | Supabase pgvector only if Neon is unavailable (pauses ~7 days idle). |
| Forbidden | Local Postgres on GUEST-LAP-04; Docker pgvector as the talk path; Pinecone / Weaviate / Qdrant / Chroma / FAISS; Vercel as system of record; Cursor Cloud Agent disk as corpus. |

Vercel may only be used as an optional Neon marketplace signup shortcut (`vc i neon`). Do not deploy FastAPI / HITL / rerank there.

---

## 4. Where compute lives

| Step | Load | Put it here |
|---|---|---|
| Ingest embeddings (3 Apple PDFs, once) | Light CPU, minutes | Max’s machine → write **once** to Neon |
| Query embedding | One 384-d vector | Same BGE, local CPU |
| Hybrid retrieve | Tiny n | Neon HNSW + tsvector + RRF |
| Rerank ≤30 pairs | Heavy if local 278M | Cohere `rerank-v3.5` or Jina `jina-reranker-v2-base-en`; `top_n=6`, timeout 8s |
| Rerank if no API key | — | **RRF-only**. Still show the rerank panel. Do not download `bge-reranker` mid-demo. |
| Generate JSON | Vendor GPU | Optional `OPENAI_API_KEY` / `GEMINI_API_KEY`; temp **0**, top_p **1**, max_tokens **800**. This talk machine has **no** OpenAI key. |
| HITL / HMAC | Trivial | FastAPI process + Postgres trigger |

Never mix local 384-d with OpenAI 1536-d unless both machines re-embed together.

---

## 5. Document bytes (Git now; object storage later)

Query tools are exactly `search_chunks`, `get_chunk`, `get_document_meta`. The agent never fetches URLs.

| Horizon | Where bytes live | Agent sees |
|---|---|---|
| Workshop MVP (**this demo**) | Git `data/` (or `data_filing/`) + `data/corpus_manifest.json` (`document_id`, ingest URI, sha256, mime, period) | Allowlisted `chunk_text` only |
| Optional narrative | One private S3-compatible bucket (Cloudflare R2 or AWS S3, Calgary-adjacent). Max fills once; URI + sha256 stay in the committed manifest. | Same chunks. No `get_object` tool. |
| Later knowledge base | S3/R2/Azure Blob + lifecycle | Still chunks; raw files never go to the vendor API |

Do **not** run MinIO or any object-store sidecar on GUEST-LAP-04. Do not copy embedding caches between laptops.

---

## 6. Secrets (`.env`, gitignored)

Copy from `.env.example`. Never commit or screenshot.

| Variable | Required for scored script | Notes |
|---|---|---|
| `DATABASE_URL` | Stretch live retrieve only | Neon pooled URI |
| `HITL_SIGNING_KEY` | Live FastAPI HITL | Not in model context or client JS |
| `HITL_PIN` | Demo PIN | `2026` |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | If live embed | `BAAI/bge-small-en-v1.5` / `384` |
| `COHERE_API_KEY` or `JINA_API_KEY` | Preferred for live rerank | Else RRF-only |
| `RERANK_PROVIDER` / `RERANK_MODEL` | If rerank key set | Default Cohere `rerank-v3.5` |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` | No | Optional live draft only |
| `CHAT_MODEL` | If generation key set | e.g. `gpt-4.1-mini` |

---

## 7. Before the room / skip list

Must already exist:

1. Python 3.12 via winget (not Store stub).
2. `.venv` + `requirements.txt` installed (no `pip install` of torch/rerankers in the room).
3. `demo/index.html` fixture rehearsed **offline**.
4. Shared Neon `.env` on Jason and Max (`neon login` before `neon link` / `neon deploy`).
5. `SELECT 1` wake ~10 minutes before HDMI if stretch is planned.
6. Rerank key **or** an explicit RRF-only plan.
7. `HITL_SIGNING_KEY` + PIN never in Git.

Do not provision for 3:00: GPU VM, Ollama, vLLM, RyzenAI, Docker-as-the-only-path, a second vector DB, local object store, LangChain/LangGraph as the state machine, multi-worker uvicorn, live web search at query time.

Stretch **after** the scored script only: one hybrid retrieve against a woken Neon; optional live LLM; optional S3 URI shown in the allowlist panel. Failure of any stretch must not kill the five-minute path.
