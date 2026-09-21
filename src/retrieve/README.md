# Retrieve (Jason + Max)

Hybrid search over allowlisted chunks: vector 20 + keyword 20 → RRF `k=60` cap 30 → cloud rerank top 6, or RRF-only.

Query-time tools: `search_chunks`, `get_chunk`, `get_document_meta`. Allowlist: `data/corpus_manifest.json` (`ingest_status=ready`).
