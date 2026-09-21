"""Bounded loop, claim gate, tool allowlist, HITL envelope."""
import pytest

from agent.loop import require_not_released_by_model, run_loop
from agent.tools import ToolDispatcher, ToolNotAllowed
from agent.validate import validate_draft
from hitl.approve import HitlError, approve
from hitl.store import MemoryStore
from retrieve.hybrid import ChunkHit, RetrieveResult


def _hit(**kw):
    base = dict(
        chunk_id="c1",
        document_id="AAPL_10-Q_2025-12-27",
        page=4,
        section="Statements of Operations",
        chunk_type="narrative",
        text="Diluted earnings per share were $2.84 and $2.40.",
        filing_period="2025-12-27",
        fusion=0.9,
    )
    base.update(kw)
    return ChunkHit(**base)


class FakeRetriever:
    def __init__(self, result: RetrieveResult):
        self.result = result
        self.calls = 0

    def search(self, query: str):
        self.calls += 1
        return self.result

    def get_chunk(self, chunk_id: str):
        for h in self.result.hits:
            if h.chunk_id == chunk_id:
                return h
        return None

    def get_document_meta(self, document_id: str):
        return {"document_id": document_id, "ingest_status": "ready"} if document_id.startswith("AAPL") else None


def _gen_ok(question, chunks):
    h = chunks[0]
    return {
        "draft_answer": "Diluted EPS was $2.84.",
        "claims": [{
            "text": "Diluted EPS $2.84",
            "citation_chunk_ids": [h.chunk_id],
            "quote": "$2.84",
            "supported": True,
        }],
        "citations": [],
    }, 12


def test_validator_rejects_uncited_and_numeric_mismatch():
    h = _hit()
    bad = validate_draft({
        "draft_answer": "EPS was $9.99",
        "claims": [{"text": "EPS $9.99", "citation_chunk_ids": ["nope"], "quote": "nope", "supported": True}],
    }, [h])
    assert not bad["passed"]
    assert "UNCITED_CLAIM" in bad["reason_codes"]

    num = validate_draft({
        "draft_answer": "EPS was $9.99",
        "claims": [{"text": "EPS $9.99", "citation_chunk_ids": ["c1"], "quote": "$2.84", "supported": True}],
    }, [h])
    assert "NUMERIC_MISMATCH" in num["reason_codes"]

    ok = validate_draft({
        "draft_answer": "Diluted EPS was $2.84",
        "claims": [{"text": "Diluted EPS $2.84", "citation_chunk_ids": ["c1"], "quote": "$2.84", "supported": True}],
    }, [h])
    assert ok["passed"]


def test_validator_uses_quote_not_other_chunk_numbers_and_rebuilds_citations():
    h = _hit(text="The requested value was $2.84. An unrelated value was $9.99.")
    result = validate_draft({
        "draft_answer": "The value was $9.99.",
        "claims": [{
            "text": "The value was $9.99.",
            "citation_chunk_ids": ["c1"],
            "quote": "The requested value was $2.84.",
            "supported": True,
        }],
        "citations": [{
            "document_id": "TSLA_FAKE",
            "chunk_id": "fake",
            "quote": "invented",
        }],
    }, [h])
    assert not result["passed"]
    assert "NUMERIC_MISMATCH" in result["reason_codes"]
    assert all(c["document_id"] != "TSLA_FAKE" for c in result["citations"])


def test_validator_requires_explicit_supported_true():
    h = _hit()
    result = validate_draft({
        "draft_answer": "Diluted EPS was $2.84.",
        "claims": [{
            "text": "Diluted EPS $2.84",
            "citation_chunk_ids": ["c1"],
            "quote": "$2.84",
        }],
    }, [h])
    assert not result["passed"]


def test_tool_allowlist():
    tools = ToolDispatcher(FakeRetriever(RetrieveResult(hits=[], mode="empty")))
    with pytest.raises(ToolNotAllowed):
        tools.call("requests.get", url="https://example.com")


def test_loop_abstain_tesla_and_answer_stays_null():
    tools = ToolDispatcher(FakeRetriever(RetrieveResult(hits=[], mode="out_of_corpus")))
    store = MemoryStore()
    env = run_loop("What was Tesla FY2025 total revenue?", tools=tools, store=store, generate=_gen_ok)
    assert env["state"] == "AWAITING_HUMAN"
    assert env["answer"] is None
    assert env["human_review"] is True
    assert "OUT_OF_CORPUS" in env["reason_codes"]
    row = store.get(env["run_id"])
    with pytest.raises(HitlError):
        approve(store, row, to_state="RELEASED", pin="2026", signing_key="k")
    row = store.get(env["run_id"])
    env2, _, _ = approve(store, row, to_state="ABSTAINED", pin="2026", signing_key="k")
    out = require_not_released_by_model(env2)
    assert out["state"] == "ABSTAINED"
    assert out["answer"] is None


def test_loop_valid_draft_still_requires_hitl():
    hits = [_hit()]
    tools = ToolDispatcher(FakeRetriever(RetrieveResult(hits=hits, mode="rrf_only", vector_ids=["c1"])))
    store = MemoryStore()
    env = run_loop("What was Apple diluted EPS?", tools=tools, store=store, generate=_gen_ok)
    assert env["state"] == "AWAITING_HUMAN"
    assert env["answer"] is None
    assert env["draft_answer"]
    assert env["human_review"] is True
    row = store.get(env["run_id"])
    with pytest.raises(HitlError):
        approve(store, row, to_state="RELEASED", pin="0000", signing_key="k")
    env2, _, sig = approve(store, row, to_state="RELEASED", pin="2026", signing_key="k")
    out = require_not_released_by_model(env2)
    assert out["state"] == "RELEASED"
    assert out["answer"] == "Diluted EPS was $2.84."
    assert out["human_review"] is False
    assert len(sig) == 64


def test_loop_stops_when_no_new_chunk_ids():
    hits = [_hit()]
    retriever = FakeRetriever(RetrieveResult(hits=hits, mode="rrf_only", vector_ids=["c1"]))
    tools = ToolDispatcher(retriever)

    def gen_fail(q, chunks):
        return {
            "draft_answer": "nope",
            "claims": [{"text": "x", "citation_chunk_ids": ["missing"], "quote": "x", "supported": True}],
            "next_search_query": q,
        }, 1

    env = run_loop("Apple EPS?", tools=tools, store=MemoryStore(), generate=gen_fail)
    assert env["state"] == "AWAITING_HUMAN"
    assert retriever.calls == 1


def test_loop_timeout_stop():
    hits = [_hit()]
    tools = ToolDispatcher(FakeRetriever(RetrieveResult(hits=hits, mode="rrf_only")))
    t = {"v": 0.0}

    def clock():
        t["v"] += 50.0
        return t["v"]

    store = MemoryStore()
    env = run_loop("Apple EPS?", tools=tools, store=store, generate=_gen_ok, now=clock)
    assert env["stop_reason"] == "timeout"
    assert env["state"] == "FAILED"
    failed = store.get(env["run_id"])
    acknowledged, _, _ = approve(
        store, failed, to_state="ABSTAINED", pin="2026", signing_key="k"
    )
    assert acknowledged.state == "ABSTAINED"
