"""Two stages, runnable separately:

  build_records  PDF > chunks > embed_text   (needs PyMuPDF, no API)   writes data/processed/<doc_id>/
  embed_records  records > OpenAI vectors    (no PyMuPDF needed)       writes embeddings.npz, then pgvector
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from pathlib import Path
from typing import Optional

from .config import Settings
from .models import FilingMeta, Record
from .text_utils import sha256

log = logging.getLogger(__name__)


def build_records(f: FilingMeta, s: Settings, embedding_model: str):
    """Parse and chunk one filing, write inspection files, return (records, summary)."""
    from .chunk import chunk_document  # imported here so the embed stage never loads PyMuPDF
    from .embed_text import build_embed_text
    from .parse import parse_filing

    out = s.processed_dir / f.doc_id
    (out / "tables").mkdir(parents=True, exist_ok=True)

    parsed = parse_filing(f.pdf_path, f)
    result = chunk_document(parsed, s.chunking)
    tables = {t.table_id: t for t in parsed.tables}

    records = []
    for c in result.chunks:
        table = tables.get(c.table_id) if c.content_type == "table_summary" else None
        text = build_embed_text(f, c, table)
        records.append(Record(
            id=c.chunk_id, doc_id=f.doc_id, ticker=f.ticker, cik=f.cik, company_name=f.company_name,
            form_type=f.form_type, accession_no=f.accession_no, filing_date=f.filing_date,
            fiscal_year=f.fiscal_year, fiscal_quarter=f.fiscal_quarter, period_end=f.period_end,
            calendar_quarter=f.calendar_quarter, source_url=f.source_url, section_path=c.section_path,
            item_code=c.item_code, content_type=c.content_type, parent_id=c.parent_id, table_id=c.table_id,
            units=c.units, durations=c.durations, char_start=c.char_start, char_end=c.char_end,
            pdf_page_start=c.pdf_page_start, pdf_page_end=c.pdf_page_end, page_label=c.page_label,
            n_tokens=c.n_tokens, embed_text=text, raw_text=c.raw_text,
            content_hash=sha256(embedding_model + "\x00" + text), embedding_model=embedding_model,
            pipeline_version=s.pipeline_version,
        ))

    # files for inspection and citation checks: clean.txt[char_start:char_end] == raw_text
    (out / "clean.txt").write_text(parsed.clean_text, encoding="utf-8")
    for t in tables.values():
        (out / "tables" / f"{t.table_id}.json").write_text(t.model_dump_json(indent=2), encoding="utf-8")
    (out / "sections.json").write_text(json.dumps(result.sections, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "skipped.json").write_text(json.dumps(result.skipped, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "records.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.model_dump_json(exclude={"embedding"}) + "\n")

    summary = {"doc_id": f.doc_id, "chunks": len(records), "tables": len(tables),
               "narrative": sum(r.content_type == "narrative" for r in records),
               "skipped_blocks": len(result.skipped)}
    return records, summary


def load_records(s: Settings, doc_id: str) -> list[Record]:
    path = s.processed_dir / doc_id / "records.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `ingest --dry-run` first")
    return [Record.model_validate_json(line) for line in path.open(encoding="utf-8") if line.strip()]


def processed_doc_ids(s: Settings) -> list[str]:
    return sorted(p.parent.name for p in s.processed_dir.glob("*/records.jsonl"))


def embed_records(doc_id: str, s: Settings, embedder, store=None, npz_name: str = "embeddings.npz") -> dict:
    """Embed one parsed filing. Vectors go to embeddings.npz (always) and pgvector (if store is given).

    Chunks whose embed_text is unchanged reuse the vectors already in embeddings.npz, so re-runs are free.
    """
    records = load_records(s, doc_id)
    npz_path = s.processed_dir / doc_id / npz_name
    for r in records:  # the hash depends on the model actually used
        r.embedding_model = embedder.name
        r.content_hash = sha256(embedder.name + "\x00" + r.embed_text)

    cached: dict[str, list[float]] = {}
    if npz_path.exists():
        z = np.load(npz_path, allow_pickle=False)
        cached = {h: v for h, v in zip(z["hashes"].tolist(), z["vectors"])}

    todo = [r for r in records if r.content_hash not in cached]
    if todo:
        vectors = embedder.embed([r.embed_text for r in todo])
        for r, v in zip(todo, vectors):
            cached[r.content_hash] = np.asarray(v, dtype=np.float32)
    for r in records:
        r.embedding = np.asarray(cached[r.content_hash], dtype=np.float32).tolist()

    np.savez(npz_path, ids=np.array([r.id for r in records]), hashes=np.array([r.content_hash for r in records]),
             vectors=np.array([r.embedding for r in records], dtype=np.float32))
    summary = {"doc_id": doc_id, "chunks": len(records), "embedded": len(todo),
               "reused": len(records) - len(todo), "saved": str(npz_path)}
    if store is not None:
        if hasattr(store, "replace_chunks"):
            store.replace_chunks(doc_id, records)
        else:
            store.replace_document(doc_id, records)
        summary["stored_in_pgvector"] = len(records)
    return summary


def ingest_filing(f: FilingMeta, s: Settings, embedder=None, store=None, dry_run: bool = False) -> dict:
    model_name = embedder.name if embedder else f"openai:{s.embedding_model}:{s.embedding_dimensions}"
    _, summary = build_records(f, s, model_name)
    if dry_run:
        return summary
    summary.update(embed_records(f.doc_id, s, embedder, store))
    return summary


def embed_processed_bge(s: Settings, embedder, doc_ids: list[str], store=None) -> list[dict]:
    """BGE-embed Max's processed records; optionally upsert canonical Neon tables."""
    npz_name = "embeddings_bge.npz"
    summaries = []
    manifest_path = s.root / "data" / "corpus_manifest.json"
    manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_by_id = {
        row["document_id"]: row for row in manifest_raw.get("documents", [])
    }
    for doc_id in doc_ids:
        summary = embed_records(doc_id, s, embedder, store=None, npz_name=npz_name)
        records = load_records(s, doc_id)
        if not records:
            raise FileNotFoundError(f"no processed records for {doc_id}")
        z = np.load(s.processed_dir / doc_id / npz_name, allow_pickle=False)
        by_id = {i: v for i, v in zip(z["ids"].tolist(), z["vectors"])}
        for r in records:
            r.embedding = np.asarray(by_id[r.id], dtype=np.float32).tolist()
            r.embedding_model = embedder.name
        page_count = max((r.pdf_page_end for r in records), default=1)
        old = manifest_by_id.get(doc_id, {})
        first = records[0]
        source_uri = old.get("source_uri") or first.source_url or f"processed://{doc_id}"
        digest = old.get("sha256")
        if not digest:
            digest = sha256(
                (s.processed_dir / doc_id / "records.jsonl").read_text(encoding="utf-8")
            )
        if store is not None:
            store.upsert_document(
                document_id=doc_id,
                source_uri=source_uri,
                sha256=digest,
                mime_type=old.get("mime_type", "application/pdf"),
                form_type=first.form_type,
                filing_period=first.period_end,
                page_count=page_count,
                ingest_status="ready",
            )
            store.replace_chunks(doc_id, records)
            summary["stored_in_pgvector"] = len(records)
        summaries.append(summary)
        manifest_by_id[doc_id] = {
            "document_id": doc_id,
            "source_uri": source_uri,
            "sha256": digest,
            "mime_type": old.get("mime_type", "application/pdf"),
            "form_type": first.form_type,
            "filing_period": first.period_end.isoformat(),
            "page_count": page_count,
            "ingest_status": "ready",
            "embedding_model": embedder.name,
            "embedding_dim": int(embedder.dimensions),
            "chunk_count": len(records),
        }
    manifest_path.write_text(
        json.dumps(
            {"documents": [manifest_by_id[k] for k in sorted(manifest_by_id)]},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return summaries


def embed_to_neon(s: Settings, embedder, store, doc_ids: list[str], filings=None) -> list[dict]:
    """Backward-compatible wrapper for the canonical BGE/Neon path."""
    return embed_processed_bge(s, embedder, doc_ids, store=store)
