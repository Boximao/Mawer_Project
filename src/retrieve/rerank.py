"""Cloud rerank (Cohere or Jina). No local 278M cross-encoder."""
from __future__ import annotations

import os
from typing import Sequence


def rerank_cloud(
    query: str,
    texts: Sequence[str],
    *,
    top_n: int = 6,
    timeout_s: float = 8.0,
) -> list[tuple[int, float]] | None:
    """Return (original_index, score) for kept docs, or None to use RRF-only."""
    if not texts:
        return []
    provider = (os.environ.get("RERANK_PROVIDER") or "").strip().lower()
    cohere = os.environ.get("COHERE_API_KEY", "").strip()
    jina = os.environ.get("JINA_API_KEY", "").strip()
    if provider == "jina" or (jina and not cohere):
        if not jina:
            return None
        return _jina(query, texts, jina, top_n, timeout_s)
    if not cohere:
        return None
    return _cohere(query, texts, cohere, top_n, timeout_s)


def _cohere(query, texts, key, top_n, timeout_s) -> list[tuple[int, float]]:
    import httpx

    model = os.environ.get("RERANK_MODEL") or "rerank-v3.5"
    r = httpx.post(
        "https://api.cohere.com/v2/rerank",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "query": query, "documents": list(texts), "top_n": min(top_n, len(texts))},
        timeout=timeout_s,
    )
    r.raise_for_status()
    data = r.json()
    out = []
    for row in data.get("results", []):
        out.append((int(row["index"]), float(row.get("relevance_score", 0.0))))
    return out


def _jina(query, texts, key, top_n, timeout_s) -> list[tuple[int, float]]:
    import httpx

    model = os.environ.get("RERANK_MODEL") or "jina-reranker-v2-base-en"
    r = httpx.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "query": query, "documents": list(texts), "top_n": min(top_n, len(texts))},
        timeout=timeout_s,
    )
    r.raise_for_status()
    data = r.json()
    out = []
    for row in data.get("results", []):
        out.append((int(row["index"]), float(row.get("relevance_score", 0.0))))
    return out
