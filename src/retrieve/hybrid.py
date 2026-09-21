"""Hybrid retrieve: vector + keyword → RRF → optional cloud rerank. Allowlisted docs only."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .allowlist import load_manifest, out_of_corpus_issuer, ready_document_ids
from .rerank import rerank_cloud
from .rrf import fuse

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ChunkHit:
    chunk_id: str
    document_id: str
    page: int
    section: str
    chunk_type: str
    text: str
    filing_period: str
    fusion: float
    rerank: Optional[float] = None
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    search_text: str = field(default="", repr=False)


@dataclass
class RetrieveResult:
    hits: list[ChunkHit]
    mode: str  # cloud_rerank | rrf_only | empty | out_of_corpus
    vector_ids: list[str] = field(default_factory=list)
    keyword_ids: list[str] = field(default_factory=list)


def _section(path) -> str:
    if path is None:
        return ""
    if isinstance(path, str):
        try:
            path = json.loads(path)
        except json.JSONDecodeError:
            return path
    if isinstance(path, list):
        tail = path[2:] if len(path) > 2 else path
        return " > ".join(str(x) for x in tail) or (path[-1] if path else "")
    return str(path)


def _period(value) -> str:
    if value is None:
        return ""
    return str(value)[:10]


_TOKEN = re.compile(r"[a-z0-9]{3,}")
_STOP = frozenset({
    "and", "are", "did", "does", "for", "from", "had", "has", "have", "how",
    "inc", "into", "its", "the", "their", "this", "three", "was", "were",
    "what", "when", "where", "which", "why", "with", "apple", "aapl",
})
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _query_tokens(question: str) -> list[str]:
    month_names = set(_MONTHS)
    return [
        term for term in _TOKEN.findall(question.lower())
        if term not in _STOP
        and term not in month_names
        and term not in {"fiscal", "quarter"}
        and not re.fullmatch(r"20\d{2}", term)
    ]


class HybridRetriever:
    def __init__(
        self,
        *,
        dsn: Optional[str] = None,
        table: str = "filing_chunks",
        processed_dir: Optional[Path] = None,
        embedder=None,
        ready_ids: Optional[frozenset[str]] = None,
    ):
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", table):
            raise ValueError(f"invalid table name {table!r}")
        self.dsn = dsn
        self.table = table
        self.processed_dir = processed_dir or (ROOT / "data" / "processed")
        self.embedder = embedder
        self.ready_ids = ready_ids if ready_ids is not None else ready_document_ids()
        self._local_cache: dict[str, ChunkHit] | None = None

    def get_chunk(self, chunk_id: str) -> Optional[ChunkHit]:
        rows = self._load_local_index()
        rec = rows.get(chunk_id)
        if rec and rec.document_id in self.ready_ids:
            return rec
        if self.dsn:
            hit = self._pg_get(chunk_id)
            if hit and hit.document_id in self.ready_ids:
                return hit
        return None

    def get_document_meta(self, document_id: str) -> Optional[dict]:
        from .allowlist import load_manifest

        if document_id not in self.ready_ids:
            return None
        for d in load_manifest():
            if d.document_id == document_id:
                return {
                    "document_id": d.document_id,
                    "form_type": d.form_type,
                    "filing_period": d.filing_period,
                    "page_count": d.page_count,
                    "ingest_status": d.ingest_status,
                    "mime_type": d.mime_type,
                }
        return None

    def search(self, question: str, top_k_context: int = 6) -> RetrieveResult:
        if out_of_corpus_issuer(question):
            return RetrieveResult(hits=[], mode="out_of_corpus")
        if not self.ready_ids:
            return RetrieveResult(hits=[], mode="empty")

        by_id: dict[str, ChunkHit] = {}
        vector_ids: list[str] = []
        keyword_ids: list[str] = []
        scoped_ids = self._document_scope(question)

        q_vec = self._embed_query(question)
        if self.dsn:
            try:
                vector_ids, keyword_ids, by_id = self._search_pg(
                    question, q_vec, scoped_ids
                )
            except Exception:
                vector_ids, keyword_ids, by_id = self._search_local(
                    question, q_vec, scoped_ids
                )
            if not vector_ids and not keyword_ids:
                # A provisioned but not-yet-ingested cloud table must not hide
                # the committed workshop corpus.
                vector_ids, keyword_ids, by_id = self._search_local(
                    question, q_vec, scoped_ids
                )
        else:
            vector_ids, keyword_ids, by_id = self._search_local(
                question, q_vec, scoped_ids
            )

        # Financial labels and exact filing language are unusually valuable;
        # give the lexical list extra weight while retaining semantic recall.
        fused = fuse(
            [vector_ids, keyword_ids],
            k=60,
            cap=30,
            discard_below=0.40,
            weights=[1.0, 2.0],
        )
        if not fused:
            return RetrieveResult(hits=[], mode="empty", vector_ids=vector_ids, keyword_ids=keyword_ids)

        ordered: list[ChunkHit] = []
        for cid, score in fused:
            hit = by_id.get(cid) or self.get_chunk(cid)
            if not hit:
                continue
            hit.fusion = score
            hit.vector_rank = vector_ids.index(cid) + 1 if cid in vector_ids else None
            hit.keyword_rank = keyword_ids.index(cid) + 1 if cid in keyword_ids else None
            ordered.append(hit)

        mode = "rrf_only"
        try:
            rr = rerank_cloud(question, [h.text[:4000] for h in ordered], top_n=min(top_k_context, len(ordered)))
        except Exception:
            rr = None
        if rr:
            mode = "cloud_rerank"
            reranked: list[ChunkHit] = []
            for idx, score in rr:
                if score < 0.20:
                    continue
                h = ordered[idx]
                h.rerank = score
                reranked.append(h)
            ordered = reranked
        else:
            table_bonus = [h for h in ordered if h.chunk_type in ("table", "table_summary")]
            cap = 8 if table_bonus and len(table_bonus) >= 4 else top_k_context
            ordered = ordered[:cap]

        cap = top_k_context
        if mode != "cloud_rerank" and ordered:
            head = ordered[:8]
            if head and all(h.chunk_type in ("table", "table_summary") for h in head):
                cap = 8
        return RetrieveResult(hits=ordered[:cap], mode=mode, vector_ids=vector_ids, keyword_ids=keyword_ids)

    def _document_scope(self, question: str) -> frozenset[str]:
        """Resolve explicit periods; otherwise use the latest ready filing."""
        docs = sorted(
            (
                d for d in load_manifest()
                if d.document_id in self.ready_ids and d.filing_period
            ),
            key=lambda d: d.filing_period or "",
        )
        if not docs:
            return self.ready_ids
        lowered = question.lower()
        quarter = re.search(r"\b(?:fiscal\s+)?q([1-4])\s+(20\d{2})\b", lowered)
        if quarter:
            fiscal_quarter, fiscal_year = int(quarter.group(1)), int(quarter.group(2))
            month = {1: 12, 2: 3, 3: 6, 4: 9}[fiscal_quarter]
            calendar_year = fiscal_year - 1 if fiscal_quarter == 1 else fiscal_year
            matches = {
                d.document_id for d in docs
                if d.filing_period.startswith(f"{calendar_year:04d}-{month:02d}-")
            }
            if matches:
                return frozenset(matches)
        for name, month in _MONTHS.items():
            match = re.search(
                rf"\b{name}(?:\s+\d{{1,2}},?)?\s+(20\d{{2}})\b",
                lowered,
            )
            if match:
                year = int(match.group(1))
                matches = {
                    d.document_id for d in docs
                    if d.filing_period.startswith(f"{year:04d}-{month:02d}-")
                }
                if matches:
                    return frozenset(matches)
        iso = re.search(r"\b(20\d{2})-(\d{2})-\d{2}\b", lowered)
        if iso:
            prefix = f"{iso.group(1)}-{iso.group(2)}-"
            matches = {
                d.document_id for d in docs if d.filing_period.startswith(prefix)
            }
            if matches:
                return frozenset(matches)
        return frozenset({docs[-1].document_id})

    def _embed_query(self, question: str) -> Optional[np.ndarray]:
        if self.embedder is None:
            return None
        try:
            v = self.embedder.embed([question])[0]
        except Exception:
            # RRF keyword retrieval is the required fail-safe when an embedding
            # provider/model is unavailable on the talk machine.
            self.embedder = None
            return None
        vector = np.asarray(v, dtype=np.float32)
        expected = {
            d.embedding_dim
            for d in load_manifest()
            if d.document_id in self.ready_ids and d.embedding_dim is not None
        }
        if len(expected) == 1 and vector.shape != (next(iter(expected)),):
            return None
        return vector

    def _search_pg(
        self,
        question: str,
        q_vec: Optional[np.ndarray],
        document_ids: frozenset[str],
    ):
        import psycopg
        from pgvector.psycopg import register_vector

        ids = list(document_ids)
        by_id: dict[str, ChunkHit] = {}
        vector_ids: list[str] = []
        keyword_ids: list[str] = []
        canonical = self.table == "chunks"
        if canonical:
            cols = (
                "c.chunk_id::text, c.document_id, c.chunk_text, c.section_path, "
                "c.page, c.page::text, c.chunk_type, d.filing_period"
            )
            source = "chunks c JOIN documents d ON d.document_id = c.document_id"
            doc_col = "c.document_id"
            vector_col = "c.embedding"
            search_col = "c.search_vector"
            ready_clause = " AND d.ingest_status = 'ready'"
        else:
            cols = (
                "id, doc_id, raw_text, section_path, pdf_page_start, "
                "page_label, content_type, period_end"
            )
            source = self.table
            doc_col = "doc_id"
            vector_col = "embedding"
            search_col = "search_vector"
            ready_clause = ""
        with psycopg.connect(self.dsn) as conn:
            register_vector(conn)
            if q_vec is not None:
                rows = conn.execute(
                    f"SELECT {cols} FROM {source} WHERE {doc_col} = ANY(%s)"
                    f"{ready_clause} ORDER BY {vector_col} <=> %s LIMIT 20",
                    (ids, q_vec),
                ).fetchall()
                for i, row in enumerate(rows, start=1):
                    hit = self._row_to_hit(row)
                    by_id[hit.chunk_id] = hit
                    vector_ids.append(hit.chunk_id)
            terms = _query_tokens(question)
            ts_query = " | ".join(dict.fromkeys(terms))
            try:
                if not ts_query:
                    krows = []
                else:
                    krows = conn.execute(
                        f"SELECT {cols} FROM {source}, to_tsquery('simple', %s) q "
                        f"WHERE {doc_col} = ANY(%s){ready_clause} AND {search_col} @@ q "
                        f"ORDER BY ts_rank_cd({search_col}, q) DESC LIMIT 20",
                        (ts_query, ids),
                    ).fetchall()
            except Exception:
                conn.rollback()
                text_col = "c.chunk_text" if canonical else "raw_text"
                krows = conn.execute(
                    f"SELECT {cols} FROM {source} WHERE {doc_col} = ANY(%s)"
                    f"{ready_clause} AND to_tsvector('simple', coalesce({text_col},'')) "
                    f"@@ to_tsquery('simple', %s) "
                    f"LIMIT 20",
                    (ids, ts_query),
                ).fetchall()
            for row in krows:
                hit = self._row_to_hit(row)
                by_id.setdefault(hit.chunk_id, hit)
                keyword_ids.append(hit.chunk_id)
        return vector_ids, keyword_ids, by_id

    def _pg_get(self, chunk_id: str) -> Optional[ChunkHit]:
        import psycopg

        canonical = self.table == "chunks"
        if canonical:
            cols = (
                "c.chunk_id::text, c.document_id, c.chunk_text, c.section_path, "
                "c.page, c.page::text, c.chunk_type, d.filing_period"
            )
            source = "chunks c JOIN documents d ON d.document_id = c.document_id"
            id_col, doc_col = "c.chunk_id::text", "c.document_id"
            ready_clause = " AND d.ingest_status = 'ready'"
        else:
            cols = (
                "id, doc_id, raw_text, section_path, pdf_page_start, "
                "page_label, content_type, period_end"
            )
            source = self.table
            id_col, doc_col = "id", "doc_id"
            ready_clause = ""
        with psycopg.connect(self.dsn) as conn:
            row = conn.execute(
                f"SELECT {cols} FROM {source} WHERE {id_col} = %s "
                f"AND {doc_col} = ANY(%s){ready_clause}",
                (chunk_id, list(self.ready_ids)),
            ).fetchone()
        return self._row_to_hit(row) if row else None

    def _row_to_hit(self, row) -> ChunkHit:
        cid, doc_id, text, section_path, page, page_label, ctype, period = row
        try:
            page_i = int(page or 1)
        except (TypeError, ValueError):
            page_i = 1
        return ChunkHit(
            chunk_id=cid,
            document_id=doc_id,
            page=page_i,
            section=_section(section_path),
            chunk_type=str(ctype or "prose"),
            text=text or "",
            filing_period=_period(period),
            fusion=0.0,
        )

    def _load_local_index(self) -> dict[str, ChunkHit]:
        if self._local_cache is not None:
            return self._local_cache
        out: dict[str, ChunkHit] = {}
        if not self.processed_dir.exists():
            return out
        for rec_path in self.processed_dir.glob("*/records.jsonl"):
            doc_id = rec_path.parent.name
            if doc_id not in self.ready_ids:
                continue
            for line in rec_path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("doc_id") not in self.ready_ids:
                    continue
                out[r["id"]] = ChunkHit(
                    chunk_id=r["id"],
                    document_id=r["doc_id"],
                    page=int(r.get("pdf_page_start") or 1),
                    section=_section(r.get("section_path")),
                    chunk_type=r.get("content_type") or "narrative",
                    text=r.get("raw_text") or "",
                    filing_period=_period(r.get("period_end")),
                    fusion=0.0,
                    search_text=(r.get("embed_text") or "") + "\n" + (r.get("raw_text") or ""),
                )
        self._local_cache = out
        return self._local_cache

    def _search_local(
        self,
        question: str,
        q_vec: Optional[np.ndarray],
        document_ids: frozenset[str],
    ):
        by_id = {
            cid: hit for cid, hit in self._load_local_index().items()
            if hit.document_id in document_ids
        }
        q_tokens = _query_tokens(question)
        q_terms = set(q_tokens)
        tokenized = {
            cid: _TOKEN.findall((hit.search_text or hit.text).lower())
            for cid, hit in by_id.items()
        }
        document_frequency = Counter(
            term for terms in tokenized.values() for term in set(terms) if term in q_terms
        )
        n_docs = max(len(tokenized), 1)
        lexical: list[tuple[float, str]] = []
        for cid, terms in tokenized.items():
            counts = Counter(terms)
            score = 0.0
            for term in q_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1.0 + (n_docs - document_frequency[term] + 0.5) /
                               (document_frequency[term] + 0.5))
                score += idf * (tf * 2.2 / (tf + 1.2))
            searchable = " ".join(terms)
            for left, right in zip(q_tokens, q_tokens[1:]):
                if f"{left} {right}" in searchable:
                    score += 1.5
            if score > 0:
                lexical.append((score, cid))
        lexical.sort(key=lambda item: (-item[0], item[1]))
        keyword_ids = [cid for _, cid in lexical[:20]]

        vector_ids: list[str] = []
        if q_vec is not None:
            scored: list[tuple[float, str]] = []
            manifest = {d.document_id: d for d in load_manifest()}
            for rec_path in self.processed_dir.glob("*/embeddings.npz"):
                doc_id = rec_path.parent.name
                if doc_id not in document_ids:
                    continue
                doc = manifest.get(doc_id)
                vector_path = rec_path
                if doc and doc.embedding_model and not doc.embedding_model.startswith("openai:"):
                    vector_path = rec_path.parent / "embeddings_bge.npz"
                if not vector_path.exists():
                    continue
                z = np.load(vector_path, allow_pickle=False)
                V = z["vectors"]
                if V.ndim != 2 or V.shape[1] != q_vec.shape[0]:
                    z.close()
                    continue
                ids = z["ids"].tolist()
                denom = np.linalg.norm(V, axis=1) * (np.linalg.norm(q_vec) + 1e-9) + 1e-9
                sims = V @ q_vec / denom
                for i, sim in zip(ids, sims):
                    if i in by_id:
                        scored.append((float(sim), i))
                z.close()
            scored.sort(reverse=True)
            vector_ids = [i for _, i in scored[:20]]
        return vector_ids, keyword_ids, by_id
