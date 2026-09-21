"""Core acceptance tests on Apple's 10-Q for the quarter ended 2026-06-27.

The pgvector test runs only when TEST_DATABASE_URL points at a Postgres with the vector extension.
It uses a fake embedder, so no OpenAI key is needed.
"""
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pytest

from finembed.chunk import chunk_document
from finembed.config import load_manifest, load_settings
from finembed.embed_text import HEADER_REGEX
from finembed.parse import parse_filing
from finembed.pipeline import build_records, ingest_filing

ROOT = Path(__file__).resolve().parents[1]
DOC_ID = "AAPL_10-Q_2026-06-27"


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    s = load_settings(ROOT / "config.yaml")
    s.processed_dir = tmp_path_factory.mktemp("processed")
    return s


@pytest.fixture(scope="module")
def filing(settings):
    return next(f for f in load_manifest(settings.manifest, settings.root) if f.doc_id == DOC_ID)


@pytest.fixture(scope="module")
def parsed(filing):
    return parse_filing(filing.pdf_path, filing)


@pytest.fixture(scope="module")
def chunks(parsed, settings):
    return chunk_document(parsed, settings.chunking).chunks


def test_no_chunk_crosses_item_boundary(parsed, chunks):
    spans = parsed.item_spans()
    for c in chunks:
        s, e = spans[c.item_code]
        assert s <= c.char_start and c.char_end <= e, c.chunk_id


def test_raw_text_is_exact_span_of_clean_text(parsed, chunks):
    for c in chunks:
        assert c.section_path, c.chunk_id
        assert parsed.clean_text[c.char_start:c.char_end] == c.raw_text


def test_each_table_is_one_chunk_and_never_inside_narrative(parsed, chunks):
    table_ids = [c.table_id for c in chunks if c.content_type == "table_summary"]
    assert sorted(table_ids) == sorted(t.table_id for t in parsed.tables)
    for c in chunks:
        if c.content_type != "table_summary":
            assert "[Table]" not in c.raw_text, c.chunk_id


def test_boilerplate_produces_no_chunks(chunks):
    text = "\n".join(c.raw_text for c in chunks)
    for marker in ["TABLE OF CONTENTS", "Commission File Number", "/s/", "CERTIFICATION", "Yes ☒"]:
        assert marker not in text, marker


def test_key_sections_detected(chunks):
    items = {c.item_code for c in chunks}
    assert {"10Q-P1-I1", "10Q-P1-I2", "10Q-P2-I1", "10Q-P2-I1A"} <= items
    assert any("Macroeconomic Conditions" in c.section_path for c in chunks)


def test_table_values_extracted(parsed):
    t = next(t for t in parsed.tables if t.title.startswith("Net sales by category"))
    iphone = next(r for r in t.rows if r.label == "iPhone")
    assert iphone.values[0].num == 54252 and "Three Months Ended June 27, 2026" in t.columns[0]
    assert t.units == "USD millions" and t.durations == ["3M", "9M"]


def test_records_have_consistent_header(filing, settings):
    records, _ = build_records(filing, settings, "fake:test")
    assert records
    for r in records:
        assert HEADER_REGEX.match(r.embed_text.split("\n", 1)[0]), r.id
        assert r.calendar_quarter == "2026-Q2" and r.fiscal_quarter == 3


class FakeEmbedder:
    name = "fake:16"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            v = np.frombuffer(hashlib.sha256(t.encode()).digest()[:16], dtype=np.uint8).astype(np.float32)
            out.append((v / np.linalg.norm(v)).tolist())
        return out


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
def test_pgvector_roundtrip_and_rerun_is_free(filing, settings):
    from finembed.store import PgVectorStore

    import psycopg

    store = PgVectorStore(os.environ["TEST_DATABASE_URL"], "test_filing_chunks", dim=16)
    with psycopg.connect(store.dsn) as conn:
        conn.execute("DROP TABLE IF EXISTS test_filing_chunks")
        conn.commit()
    store.init_schema()

    emb = FakeEmbedder()
    first = ingest_filing(filing, settings, emb, store)
    assert first["embedded"] == first["chunks"] and store.count(DOC_ID) == first["chunks"]
    calls = emb.calls
    second = ingest_filing(filing, settings, emb, store)
    assert emb.calls == calls and second["reused"] == second["chunks"]  # zero new embedding calls
    assert store.count(DOC_ID) == first["chunks"]  # replaced, not duplicated

    with store.connect() as conn:  # filtered similarity search works
        q = np.asarray(emb.embed(["x"])[0], dtype=np.float32)
        rows = conn.execute(
            "SELECT id, page_label FROM test_filing_chunks WHERE ticker = %s AND fiscal_quarter = %s "
            "ORDER BY embedding <=> %s LIMIT 3", ("AAPL", 3, q)).fetchall()
    assert len(rows) == 3
