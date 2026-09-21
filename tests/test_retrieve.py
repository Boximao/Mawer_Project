"""Hybrid retrieve + RRF + allowlist."""
from retrieve.allowlist import (
    out_of_corpus_issuer,
    ready_document_ids,
    validate_processed_corpus,
)
from retrieve.hybrid import HybridRetriever
from retrieve.rrf import fuse, rrf_scores


def test_rrf_prefers_agreement():
    a = ["c1", "c2", "c3"]
    b = ["c2", "c1", "c4"]
    scores = rrf_scores([a, b], k=60)
    assert scores["c1"] > scores["c3"]
    assert scores["c2"] >= scores["c1"]


def test_fusion_drops_below_threshold():
    kept = fuse([["only"]], k=60, cap=30, discard_below=0.40)
    assert kept and kept[0][0] == "only" and kept[0][1] == 1.0


def test_tesla_is_out_of_corpus():
    assert out_of_corpus_issuer("What was Tesla FY2025 total revenue?")
    assert out_of_corpus_issuer("What was Ford's quarterly revenue?")
    assert not out_of_corpus_issuer("What was Apple diluted EPS?")


def test_manifest_ready_ids():
    ids = ready_document_ids()
    assert "AAPL_10-Q_2025-12-27" in ids
    assert len(ids) == 3
    assert validate_processed_corpus() == []


def test_local_hybrid_tesla_empty():
    r = HybridRetriever(dsn=None, embedder=None)
    res = r.search("What was Tesla FY2025 total revenue?")
    assert res.mode == "out_of_corpus"
    assert res.hits == []


def test_local_hybrid_apple_keyword_hits():
    r = HybridRetriever(dsn=None, embedder=None)
    res = r.search("What was Apple diluted earnings per share for the quarter ended December 27, 2025?")
    assert res.hits, "expected keyword hits from processed 10-Q chunks"
    assert all(h.document_id in ready_document_ids() for h in res.hits)


def test_get_chunk_allowlisted():
    r = HybridRetriever(dsn=None, embedder=None)
    res = r.search("iPhone net sales June 27 2026")
    assert res.hits
    got = r.get_chunk(res.hits[0].chunk_id)
    assert got is not None
    assert r.get_document_meta(got.document_id)["ingest_status"] == "ready"
    assert r.get_document_meta("TSLA_10-K_2025") is None
