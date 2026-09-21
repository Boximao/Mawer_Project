"""HMAC-SHA256 HITL signatures. Key never enters the model context or tool schemas."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any


def payload_sha256(draft: dict[str, Any]) -> str:
    body = json.dumps(
        {
            "draft_answer": draft.get("draft_answer"),
            "claims": draft.get("claims") or [],
            "citations": draft.get("citations") or [],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def canonical(
    run_id: str,
    from_state: str,
    to_state: str,
    step_id: str,
    payload_hash: str,
    approver_id: str,
    unix_ts: int,
) -> str:
    return f"v1|{run_id}|{from_state}|{to_state}|{step_id}|{payload_hash}|{approver_id}|{unix_ts}"


def sign(key: str, canon: str) -> str:
    return hmac.new(key.encode("utf-8"), canon.encode("utf-8"), hashlib.sha256).hexdigest()


def verify(
    key: str,
    canon: str,
    signature: str,
    unix_ts: int,
    *,
    now: float | None = None,
    max_skew_s: int = 300,
) -> bool:
    if abs((now if now is not None else time.time()) - unix_ts) > max_skew_s:
        return False
    return hmac.compare_digest(sign(key, canon), signature)


def audit_hash(prev: str | None, signature: str) -> str:
    material = f"{prev or ''}|{signature}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
