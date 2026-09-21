"""pgvector storage (PostgreSQL + the vector extension)."""
from __future__ import annotations

import logging
import re

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .models import Record

log = logging.getLogger(__name__)

COLUMNS = ["id", "doc_id", "ticker", "cik", "company_name", "form_type", "accession_no", "filing_date",
           "fiscal_year", "fiscal_quarter", "period_end", "calendar_quarter", "source_url", "section_path",
           "item_code", "content_type", "parent_id", "table_id", "units", "durations", "char_start", "char_end",
           "pdf_page_start", "pdf_page_end", "page_label", "n_tokens", "embed_text", "raw_text", "content_hash",
           "embedding_model", "pipeline_version", "embedding"]

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS {t} (
    id                TEXT PRIMARY KEY,
    doc_id            TEXT NOT NULL,
    ticker            TEXT NOT NULL,
    cik               TEXT NOT NULL,
    company_name      TEXT NOT NULL,
    form_type         TEXT NOT NULL,
    accession_no      TEXT,
    filing_date       DATE NOT NULL,
    fiscal_year       INT NOT NULL,
    fiscal_quarter    INT,
    period_end        DATE NOT NULL,
    calendar_quarter  TEXT NOT NULL,
    source_url        TEXT,
    section_path      TEXT[] NOT NULL,
    item_code         TEXT NOT NULL,
    content_type      TEXT NOT NULL,
    parent_id         TEXT NOT NULL,
    table_id          TEXT,
    units             TEXT,
    durations         TEXT[],
    char_start        INT NOT NULL,
    char_end          INT NOT NULL,
    pdf_page_start    INT NOT NULL,
    pdf_page_end      INT NOT NULL,
    page_label        TEXT,
    n_tokens          INT NOT NULL,
    embed_text        TEXT NOT NULL,
    raw_text          TEXT NOT NULL,
    content_hash      TEXT NOT NULL,
    embedding_model   TEXT NOT NULL,
    pipeline_version  TEXT NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding         vector({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS {t}_embedding_hnsw ON {t} USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS {t}_filters ON {t} (ticker, form_type, fiscal_year, fiscal_quarter);
CREATE INDEX IF NOT EXISTS {t}_doc ON {t} (doc_id);
"""


class PgVectorStore:
    def __init__(self, dsn: str, table: str = "filing_chunks", dim: int = 1536):
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", table):
            raise ValueError(f"invalid table name {table!r}")
        if dim > 2000:
            log.warning("vector(%d) cannot use an HNSW index (limit 2000); set embedding.dimensions <= 2000", dim)
        self.dsn, self.table, self.dim = dsn, table, dim

    def connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.dsn)
        register_vector(conn)
        return conn

    def init_schema(self) -> None:
        with psycopg.connect(self.dsn) as conn:  # register_vector needs the extension to exist first
            sql = SCHEMA_SQL.format(t=self.table, dim=self.dim)
            if self.dim > 2000:
                sql = "\n".join(line for line in sql.splitlines() if "hnsw" not in line)
            conn.execute(sql)
            conn.commit()

    def existing_vectors(self, doc_id: str, embedding_model: str) -> dict[str, list[float]]:
        """content_hash -> embedding for rows already stored, so unchanged chunks are not re-embedded."""
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT content_hash, embedding FROM {self.table} WHERE doc_id = %s AND embedding_model = %s",
                (doc_id, embedding_model)).fetchall()
        return {h: (e.to_list() if hasattr(e, "to_list") else list(e)) for h, e in rows}

    def replace_document(self, doc_id: str, records: list[Record]) -> int:
        """Atomically swap all rows of one filing (idempotent re-ingest)."""
        cols = ", ".join(COLUMNS)
        with self.connect() as conn, conn.transaction():
            conn.execute(f"DELETE FROM {self.table} WHERE doc_id = %s", (doc_id,))
            with conn.cursor().copy(f"COPY {self.table} ({cols}) FROM STDIN") as copy:
                for r in records:
                    d = r.model_dump()
                    d["embedding"] = np.asarray(d["embedding"], dtype=np.float32)
                    copy.write_row([d[c] for c in COLUMNS])
        return len(records)

    def count(self, doc_id: str | None = None) -> int:
        with self.connect() as conn:
            if doc_id:
                return conn.execute(f"SELECT count(*) FROM {self.table} WHERE doc_id = %s", (doc_id,)).fetchone()[0]
            return conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]
