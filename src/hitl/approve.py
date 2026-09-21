"""Approve a run after PIN check. Server-side HMAC; the model has no approve tool."""
from __future__ import annotations

import os
import time

from .crypto import canonical, payload_sha256, sign, verify
from .store import RunRow


class HitlError(ValueError):
    pass


def approve(
    store,
    run: RunRow,
    *,
    to_state: str,
    pin: str,
    approver_id: str = "presenter",
    signing_key: str | None = None,
    expected_pin: str | None = None,
    unix_ts: int | None = None,
) -> tuple[RunRow, str, str]:
    expected_pin = expected_pin if expected_pin is not None else os.environ.get("HITL_PIN", "2026")
    signing_key = signing_key if signing_key is not None else os.environ.get("HITL_SIGNING_KEY", "")
    if not signing_key:
        raise HitlError("HITL_SIGNING_KEY missing")
    if pin != expected_pin:
        raise HitlError("PIN rejected")
    if run.state not in ("AWAITING_HUMAN", "FAILED"):
        raise HitlError("run is not awaiting human acknowledgement")
    if to_state not in ("RELEASED", "ABSTAINED"):
        raise HitlError("to_state must be RELEASED or ABSTAINED")
    if run.state == "FAILED" and to_state != "ABSTAINED":
        raise HitlError("FAILED runs may only be acknowledged as ABSTAINED")
    if to_state == "RELEASED" and not (run.draft and run.draft.get("draft_answer")):
        raise HitlError("no signed draft to release")

    ts = unix_ts if unix_ts is not None else int(time.time())
    payload_hash = run.draft_sha256 or payload_sha256(run.draft or {})
    step_id = run.last_step_id or "s0"
    from_state = run.state
    canon = canonical(run.run_id, from_state, to_state, step_id, payload_hash, approver_id, ts)
    sig = sign(signing_key, canon)
    if not verify(signing_key, canon, sig, ts):
        raise HitlError("signature expired")

    store.add_approval({
        "run_id": run.run_id,
        "from_state": from_state,
        "to_state": to_state,
        "step_id": step_id,
        "payload_sha256": payload_hash,
        "approver_id": approver_id,
        "unix_ts": ts,
        "signature": sig,
    })
    if to_state == "RELEASED":
        run.answer = run.draft.get("draft_answer")
    else:
        run.answer = None
    store.save(run)
    store.set_state(run, to_state)
    return run, canon, sig
