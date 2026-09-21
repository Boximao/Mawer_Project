"""Neon `documents` + `chunks` store (migrations/001_initial.sql, vector(384))."""
from __future__ import annotations

import hashlib
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .models import Record

CHUNK_NS = uuid.uuid5(uuid.NAMESPACE_URL, "https://mawer.local/finembed/chunk")
TYPE_MAP = {"narrative": "prose", "table_summary": "table", "footnote": "note"}
COPY_COLS = (
    "chunk_id",
    "document_id",
    "page",
    "section_path",
    "chunk_type",
    "chunk_text",
    "sha256",
    "embedding",
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chunk_uuid(record_id: str) -> uuid.UUID:
    return uuid.uuid5(CHUNK_NS, record_id)


class NeonChunkStore:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.dsn, connect_timeout=30)
        register_vector(conn)
        return conn

    def upsert_document(
        self,
        *,
        document_id: str,
        source_uri: str,
        sha256: str,
        mime_type: str,
        form_type: str | None,
        filing_period: date | None,
        page_count: int | None,
        ingest_status: str = "ready",
    ) -> None:
        sql = """
        INSERT INTO documents (
            document_id, source_uri, sha256, mime_type, form_type, filing_period,
            page_count, ingest_status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (document_id) DO UPDATE SET
            source_uri = EXCLUDED.source_uri,
            sha256 = EXCLUDED.sha256,
            mime_type = EXCLUDED.mime_type,
            form_type = EXCLUDED.form_type,
            filing_period = EXCLUDED.filing_period,
            page_count = EXCLUDED.page_count,
            ingest_status = EXCLUDED.ingest_status
        """
        with self.connect() as conn:
            conn.execute(
                sql,
                (
                    document_id,
                    source_uri,
                    sha256,
                    mime_type,
                    form_type,
                    filing_period,
                    page_count,
                    ingest_status,
                ),
            )
            conn.commit()

    def replace_chunks(self, document_id: str, records: list[Record]) -> int:
        with self.connect() as conn, conn.transaction():
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            with conn.cursor().copy(
                f"COPY chunks ({', '.join(COPY_COLS)}) FROM STDIN"
            ) as copy:
                for r in records:
                    copy.write_row(_row(r))
        return len(records)

    def counts(self) -> tuple[int, int]:
        with self.connect() as conn:
            docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
            chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return docs, chunks


def _row(r: Record) -> list:
    page = max(int(r.pdf_page_start or 1), 1)
    section = " > ".join(r.section_path) if r.section_path else None
    chunk_type = TYPE_MAP.get(r.content_type, "prose")
    text = r.raw_text or r.embed_text
    digest = hashlib.sha256(f"{r.id}\n{text}".encode("utf-8")).hexdigest()
    vec = np.asarray(r.embedding, dtype=np.float32)
    return [chunk_uuid(r.id), r.doc_id, page, section, chunk_type, text, digest, vec]
