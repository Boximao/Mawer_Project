"""Workshop happy path: retrieve Apple EPS, keep answer null, then HITL release."""
import pytest

from agent.generate import generate_draft
from agent.loop import require_not_released_by_model, run_loop
from agent.tools import ToolDispatcher
from hitl.approve import HitlError, approve
from hitl.store import MemoryStore
from retrieve.hybrid import HybridRetriever

QUESTION = (
    "What was Apple diluted earnings per share for the quarter ended December 27, 2025?"
)


def test_happy_path_retrieve_validate_hitl_release():
    tools = ToolDispatcher(HybridRetriever(dsn=None, embedder=None))
    store = MemoryStore()
    env = run_loop(QUESTION, tools=tools, store=store, generate=generate_draft)

    assert env["state"] == "AWAITING_HUMAN"
    assert env["human_review"] is True
    assert env["answer"] is None
    assert env["draft_answer"]
    assert "2.84" in env["draft_answer"]
    assert env["citations"]
    assert all(c["document_id"].startswith("AAPL_") for c in env["citations"])
    assert all(c["supported"] for c in env["claims"])
    assert env["trace"]["retrieve_mode"] in {"rrf_only", "cloud_rerank"}
    assert env["trace"]["kept"]

    row = store.get(env["run_id"])
    with pytest.raises(HitlError):
        approve(store, row, to_state="RELEASED", pin="0000", signing_key="k")

    released, _, sig = approve(store, row, to_state="RELEASED", pin="2026", signing_key="k")
    out = require_not_released_by_model(released)
    assert out["state"] == "RELEASED"
    assert out["human_review"] is False
    assert out["answer"] == env["draft_answer"]
    assert "2.84" in out["answer"]
    assert len(sig) == 64


def test_happy_path_tesla_stays_unreleased_until_ack():
    tools = ToolDispatcher(HybridRetriever(dsn=None, embedder=None))
    store = MemoryStore()
    env = run_loop(
        "What was Tesla FY2025 total revenue?",
        tools=tools,
        store=store,
        generate=generate_draft,
    )
    assert env["answer"] is None
    assert env["draft_answer"] is None
    assert env["state"] == "AWAITING_HUMAN"
    assert "OUT_OF_CORPUS" in env["reason_codes"]
    row = store.get(env["run_id"])
    with pytest.raises(HitlError):
        approve(store, row, to_state="RELEASED", pin="2026", signing_key="k")
    acked, _, _ = approve(store, row, to_state="ABSTAINED", pin="2026", signing_key="k")
    out = require_not_released_by_model(acked)
    assert out["state"] == "ABSTAINED"
    assert out["answer"] is None
