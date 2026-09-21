"""Bounded retrieve → rerank → draft → validate loop. The model cannot RELEASE."""
from __future__ import annotations

import time
from typing import Any, Callable

from finembed.text_utils import count_tokens
from hitl.crypto import payload_sha256
from hitl.store import MemoryStore, RunRow

from .generate import generate_draft
from .tools import ToolDispatcher, ToolNotAllowed
from .validate import validate_draft

MAX_STEPS = 4
TOKEN_BUDGET = 40_000
WALL_S = 45.0


def require_not_released_by_model(run: RunRow) -> dict[str, Any]:
    released = run.state == "RELEASED"
    draft = run.draft or {}
    return {
        "run_id": run.run_id,
        "state": run.state,
        "human_review": not released,
        "answer": run.answer if released else None,
        "draft_answer": draft.get("draft_answer"),
        "claims": draft.get("claims") or [],
        "citations": draft.get("citations") or [],
        "reason_codes": run.reason_codes,
        "stop_reason": run.stop_reason,
        "usage": run.usage,
        "trace": run.trace,
        "visited": run.visited,
    }


def run_loop(
    question: str,
    *,
    tools: ToolDispatcher,
    store=None,
    generate: Callable = generate_draft,
    now: Callable[[], float] | None = None,
) -> dict[str, Any]:
    store = store or MemoryStore()
    clock = now or time.monotonic
    t0 = clock()
    row = store.create_run(question)
    tokens = 0
    seen: set[str] = set()
    query = question
    stop = "awaiting_human"
    validated: dict[str, Any] | None = None

    def remaining() -> bool:
        return (clock() - t0) < WALL_S and tokens < TOKEN_BUDGET

    def elapsed_ms() -> int:
        return int((clock() - t0) * 1000)

    def bump(state: str):
        store.set_state(row, state)

    try:
        for step in range(1, MAX_STEPS + 1):
            if not remaining():
                stop = "timeout" if (clock() - t0) >= WALL_S else "token_budget"
                break

            store.write_step(row.run_id, step, "RETRIEVING", "begin", {"query": query})
            bump("RETRIEVING")
            try:
                result = tools.call("search_chunks", query=query)
            except ToolNotAllowed:
                stop = "empty_retrieve"
                row.reason_codes = ["TOOL_DENIED"]
                break

            hits = result.hits
            ids = [h.chunk_id for h in hits]
            store.write_step(row.run_id, step, "RERANKING", "retrieve", {
                "mode": result.mode,
                "chunk_ids": ids,
            })
            bump("RERANKING")
            row.trace = {
                "retrieve_mode": result.mode,
                "vector_ids": result.vector_ids[:8],
                "keyword_ids": result.keyword_ids[:8],
                "kept": [
                    {
                        "chunk_id": h.chunk_id,
                        "document_id": h.document_id,
                        "page": h.page,
                        "section": h.section,
                        "fusion": round(h.fusion, 3),
                        "rerank": h.rerank,
                    }
                    for h in hits
                ],
            }

            if result.mode == "out_of_corpus":
                stop = "empty_retrieve"
                row.reason_codes = ["OUT_OF_CORPUS", "INSUFFICIENT_EVIDENCE"]
                validated = {
                    "passed": False,
                    "draft_answer": None,
                    "claims": [],
                    "citations": [],
                    "reason_codes": row.reason_codes,
                }
                break

            new_ids = [i for i in ids if i not in seen]
            if not new_ids:
                stop = "empty_retrieve" if not seen else "no_new_chunks"
                if not seen:
                    row.reason_codes = ["INSUFFICIENT_EVIDENCE"]
                break
            seen.update(new_ids)

            bump("DRAFTING")
            store.write_step(row.run_id, step, "DRAFTING", "generate", {"n_chunks": len(hits)})
            draft, used = generate(question, hits)
            tokens += used + count_tokens(query) + sum(count_tokens(h.text) for h in hits)
            bump("VALIDATING")
            store.write_step(row.run_id, step, "VALIDATING", "validate", {"claims": len(draft.get("claims") or [])})
            if tokens > TOKEN_BUDGET or (clock() - t0) >= WALL_S:
                stop = "token_budget" if tokens > TOKEN_BUDGET else "timeout"
                validated = {
                    "passed": False,
                    "draft_answer": None,
                    "claims": [],
                    "citations": [],
                    "reason_codes": [stop.upper()],
                }
                row.reason_codes = list(validated["reason_codes"])
                break
            validated = validate_draft(draft, hits)
            row.reason_codes = list(validated["reason_codes"])
            row.draft = {
                "draft_answer": validated["draft_answer"],
                "claims": validated["claims"],
                "citations": validated["citations"],
            }
            row.draft_sha256 = payload_sha256(row.draft)
            row.usage = {"steps": step, "tokens": tokens, "latency_ms": elapsed_ms()}
            store.save(row)

            if validated["passed"]:
                stop = "awaiting_human"
                break
            nxt = (draft or {}).get("next_search_query")
            if nxt and nxt.strip() and nxt.strip() != query:
                query = nxt.strip()
                continue
            # same query would yield same ids
            stop = "awaiting_human"
            break
        else:
            stop = "max_steps"
    except Exception as exc:
        stop = "timeout" if isinstance(exc, TimeoutError) else "failed"
        row.reason_codes = list(dict.fromkeys(row.reason_codes + ["FAILED"]))
        row.trace = {**row.trace, "error": type(exc).__name__}
        bump("FAILED")
        row.stop_reason = stop
        row.usage = {"steps": row.usage.get("steps", 0), "tokens": tokens, "latency_ms": elapsed_ms()}
        store.save(row)
        return require_not_released_by_model(row)

    if validated is None:
        validated = {
            "passed": False,
            "draft_answer": None,
            "claims": [],
            "citations": [],
            "reason_codes": row.reason_codes or ["INSUFFICIENT_EVIDENCE"],
        }
    row.draft = {
        "draft_answer": validated["draft_answer"],
        "claims": validated["claims"],
        "citations": validated["citations"],
    }
    row.draft_sha256 = payload_sha256(row.draft)
    row.reason_codes = list(validated["reason_codes"] or row.reason_codes)

    if not remaining() and stop == "awaiting_human" and not (validated and validated.get("passed")):
        stop = "timeout" if (clock() - t0) >= WALL_S else stop

    row.stop_reason = stop
    row.usage = {
        "steps": row.usage.get("steps") or len(seen) or 1,
        "tokens": tokens,
        "latency_ms": elapsed_ms(),
    }
    store.save(row)
    terminal_fail = stop in {"timeout", "token_budget", "failed"} and not (
        validated and validated.get("passed")
    )
    if row.state == "RECEIVED" or terminal_fail:
        if row.state != "FAILED":
            bump("FAILED")
        store.save(row)
        return require_not_released_by_model(row)
    bump("AWAITING_HUMAN")
    store.save(row)
    return require_not_released_by_model(row)
