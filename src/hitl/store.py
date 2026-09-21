"""Run + approval persistence. Postgres when DATABASE_URL is set; otherwise memory."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


TERMINAL = frozenset({"RELEASED", "ABSTAINED"})
ALLOWED_TRANSITIONS = {
    "RECEIVED": {"RETRIEVING", "FAILED"},
    "RETRIEVING": {"RERANKING", "AWAITING_HUMAN", "FAILED"},
    "RERANKING": {"DRAFTING", "AWAITING_HUMAN", "FAILED"},
    "DRAFTING": {"VALIDATING", "AWAITING_HUMAN", "FAILED"},
    "VALIDATING": {"RETRIEVING", "AWAITING_HUMAN", "FAILED"},
    "AWAITING_HUMAN": {"RELEASED", "ABSTAINED", "RETURN_TO_DRAFT"},
    "RETURN_TO_DRAFT": {"RETRIEVING", "FAILED"},
    "FAILED": {"ABSTAINED"},
    "RELEASED": set(),
    "ABSTAINED": set(),
}


@dataclass
class RunRow:
    run_id: str
    question: str
    state: str = "RECEIVED"
    draft: dict[str, Any] | None = None
    draft_sha256: str | None = None
    answer: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=lambda: {"steps": 0, "tokens": 0, "latency_ms": 0})
    last_step_id: str = "s0"
    trace: dict[str, Any] = field(default_factory=dict)
    visited: list[str] = field(default_factory=lambda: ["RECEIVED"])


class MemoryStore:
    def __init__(self):
        self.runs: dict[str, RunRow] = {}
        self.approvals: list[dict] = []
        self.steps: list[dict] = []
        self._prev_audit: str | None = None

    def create_run(self, question: str) -> RunRow:
        row = RunRow(run_id=str(uuid.uuid4()), question=question)
        self.runs[row.run_id] = row
        return row

    def get(self, run_id: str) -> Optional[RunRow]:
        return self.runs.get(run_id)

    def save(self, row: RunRow) -> None:
        self.runs[row.run_id] = row

    def set_state(self, row: RunRow, state: str) -> None:
        if state == row.state:
            return
        if state not in ALLOWED_TRANSITIONS.get(row.state, set()):
            raise ValueError(f"illegal state transition {row.state} -> {state}")
        if state not in row.visited:
            row.visited.append(state)
        row.state = state
        self.save(row)

    def write_step(self, run_id: str, step_number: int, state: str, event_type: str, payload: dict) -> str:
        step_id = str(uuid.uuid4())
        self.steps.append({
            "step_id": step_id,
            "run_id": run_id,
            "step_number": step_number,
            "state": state,
            "event_type": event_type,
            "payload": payload,
        })
        row = self.runs[run_id]
        row.last_step_id = step_id
        return step_id

    def add_approval(self, row: dict, prev_needed: bool = True) -> str:
        from .crypto import audit_hash

        h = audit_hash(self._prev_audit, row["signature"])
        row = {**row, "prev_audit_hash": self._prev_audit, "audit_hash": h}
        self.approvals.append(row)
        self._prev_audit = h
        return h


class PgStore:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._mem_trace: dict[str, dict] = {}
        self._visited: dict[str, list[str]] = {}
        self._last_step: dict[str, str] = {}

    def _connect(self):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self.dsn, row_factory=dict_row)

    def create_run(self, question: str) -> RunRow:
        with self._connect() as conn:
            rec = conn.execute(
                "INSERT INTO runs (question, state) VALUES (%s, 'RECEIVED') RETURNING run_id",
                (question,),
            ).fetchone()
            conn.commit()
        run_id = str(rec["run_id"])
        row = RunRow(run_id=run_id, question=question)
        self._visited[run_id] = ["RECEIVED"]
        return row

    def get(self, run_id: str) -> Optional[RunRow]:
        with self._connect() as conn:
            rec = conn.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,)).fetchone()
        if not rec:
            return None
        draft = rec["draft"]
        if isinstance(draft, str):
            draft = json.loads(draft)
        codes = rec["reason_codes"] or []
        if isinstance(codes, str):
            codes = json.loads(codes)
        usage = rec["usage"] or {}
        if isinstance(usage, str):
            usage = json.loads(usage)
        row = RunRow(
            run_id=str(rec["run_id"]),
            question=rec["question"],
            state=rec["state"],
            draft=draft,
            draft_sha256=rec["draft_sha256"],
            answer=rec["answer"],
            reason_codes=list(codes),
            stop_reason=rec["stop_reason"],
            usage=dict(usage),
        )
        row.trace = dict(rec.get("trace") or self._mem_trace.get(row.run_id, {}))
        row.visited = list(rec.get("visited") or self._visited.get(row.run_id, [row.state]))
        last = None
        with self._connect() as conn:
            last = conn.execute(
                "SELECT step_id FROM agent_steps WHERE run_id = %s "
                "ORDER BY created_at DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        row.last_step_id = str(last["step_id"]) if last else self._last_step.get(row.run_id, "s0")
        return row

    def save(self, row: RunRow) -> None:
        self._mem_trace[row.run_id] = row.trace
        self._visited[row.run_id] = row.visited
        with self._connect() as conn:
            conn.execute(
                """UPDATE runs SET draft = %s::jsonb, draft_sha256 = %s, answer = %s,
                   reason_codes = %s::jsonb, stop_reason = %s, usage = %s::jsonb,
                   trace = %s::jsonb, visited = %s::jsonb
                   WHERE run_id = %s""",
                (
                    json.dumps(row.draft) if row.draft is not None else None,
                    row.draft_sha256,
                    row.answer,
                    json.dumps(row.reason_codes),
                    row.stop_reason,
                    json.dumps(row.usage),
                    json.dumps(row.trace),
                    json.dumps(row.visited),
                    row.run_id,
                ),
            )
            conn.commit()

    def set_state(self, row: RunRow, state: str) -> None:
        if state == row.state:
            return
        if state not in ALLOWED_TRANSITIONS.get(row.state, set()):
            raise ValueError(f"illegal state transition {row.state} -> {state}")
        if state not in row.visited:
            row.visited.append(state)
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET state = %s, answer = %s, visited = %s::jsonb "
                "WHERE run_id = %s",
                (state, row.answer if state == "RELEASED" else None,
                 json.dumps(row.visited), row.run_id),
            )
            conn.commit()
        row.state = state
        self._visited[row.run_id] = row.visited

    def write_step(self, run_id: str, step_number: int, state: str, event_type: str, payload: dict) -> str:
        with self._connect() as conn:
            rec = conn.execute(
                """INSERT INTO agent_steps (run_id, step_number, state, event_type, payload)
                   VALUES (%s, %s, %s, %s, %s::jsonb)
                   ON CONFLICT (run_id, step_number, event_type) DO UPDATE SET payload = EXCLUDED.payload
                   RETURNING step_id""",
                (run_id, step_number, state, event_type, json.dumps(payload)),
            ).fetchone()
            conn.commit()
        sid = str(rec["step_id"])
        self._last_step[run_id] = sid
        return sid

    def add_approval(self, row: dict, prev_needed: bool = True) -> str:
        from .crypto import audit_hash

        with self._connect() as conn:
            prev = conn.execute(
                "SELECT audit_hash FROM approvals ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            prev_h = prev["audit_hash"] if prev else None
            h = audit_hash(prev_h, row["signature"])
            conn.execute(
                """INSERT INTO approvals (
                    run_id, from_state, to_state, step_id, payload_sha256, approver_id,
                    unix_ts, signature, prev_audit_hash, audit_hash
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    row["run_id"], row["from_state"], row["to_state"], row["step_id"],
                    row["payload_sha256"], row["approver_id"], row["unix_ts"],
                    row["signature"], prev_h, h,
                ),
            )
            conn.commit()
        return h
