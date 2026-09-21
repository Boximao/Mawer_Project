from hitl.crypto import canonical, payload_sha256, sign, verify
from hitl.store import MemoryStore
import pytest


def test_hmac_roundtrip_and_expiry():
    key = "workshop-server-key"
    h = payload_sha256({"draft_answer": "x", "claims": [], "citations": []})
    canon = canonical("rid", "AWAITING_HUMAN", "RELEASED", "s1", h, "presenter", 1_000_000)
    sig = sign(key, canon)
    assert verify(key, canon, sig, 1_000_000, now=1_000_010)
    assert not verify(key, canon, sig, 1_000_000, now=1_000_000 + 301)
    assert not verify("other", canon, sig, 1_000_000, now=1_000_010)


def test_memory_store_rejects_illegal_release_transition():
    store = MemoryStore()
    run = store.create_run("question")
    with pytest.raises(ValueError, match="illegal state transition"):
        store.set_state(run, "RELEASED")
