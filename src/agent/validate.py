"""Quote-substring + numeric-verbatim gates. No second NLI model."""
from __future__ import annotations

import re
from typing import Any, Iterable

from retrieve.hybrid import ChunkHit

_WS = re.compile(r"\s+")
_NUM = re.compile(r"\d+(?:\.\d+)?")


def squash(text: str) -> str:
    return _WS.sub(" ", (text or "").replace("\u00a0", " ")).strip()


def normalize_money(text: str) -> str:
    return squash(text).replace("$", "").replace(",", "")


def numbers_in(text: str) -> list[str]:
    return _NUM.findall(normalize_money(text))


def quote_in_chunk(quote: str, chunk_text: str) -> bool:
    q, c = squash(quote), squash(chunk_text)
    if not q:
        return False
    if q in c:
        return True
    return normalize_money(q) in normalize_money(c)


def validate_draft(draft: dict[str, Any], chunks: Iterable[ChunkHit]) -> dict[str, Any]:
    by_id = {h.chunk_id: h for h in chunks}
    reasons: list[str] = []
    claims_out = []
    # Citations are rebuilt from validated chunk IDs. Never trust citation
    # metadata supplied by the model.
    citations = []

    raw_claims = draft.get("claims") or []
    if not raw_claims:
        reasons.append("INSUFFICIENT_EVIDENCE")

    for claim in raw_claims:
        text = claim.get("text") or ""
        quote = claim.get("quote") or ""
        ids = claim.get("citation_chunk_ids") or []
        ok = True
        if len(ids) != 1 or ids[0] not in by_id:
            reasons.append("UNCITED_CLAIM")
            ok = False
        else:
            chunk = by_id[ids[0]]
            if not quote_in_chunk(quote, chunk.text):
                reasons.append("INSUFFICIENT_EVIDENCE")
                ok = False
            for n in numbers_in(text):
                # Numeric claims must be supported by the cited span itself,
                # not merely by some unrelated number elsewhere in the chunk.
                if n not in numbers_in(quote):
                    reasons.append("NUMERIC_MISMATCH")
                    ok = False
                    break
            if ok:
                citations.append({
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "filing_period": chunk.filing_period,
                    "page": chunk.page,
                    "section": chunk.section,
                    "quote": quote,
                })
        claims_out.append({
            "text": text,
            "citation_chunk_ids": ids,
            "quote": quote,
            "supported": bool(ok and claim.get("supported") is True),
            "entailment": 1.0 if ok else 0.0,
            "numeric_verbatim": ok,
        })
        if not ok:
            claims_out[-1]["supported"] = False

    # unique reason order
    seen = []
    for r in reasons:
        if r not in seen:
            seen.append(r)

    passed = bool(claims_out) and all(c["supported"] for c in claims_out) and not seen
    draft_answer = draft.get("draft_answer") if passed else None
    # Deduplicate citations by (chunk_id, quote)
    uniq = []
    keys = set()
    for c in citations:
        k = (c.get("chunk_id"), c.get("quote"))
        if k in keys:
            continue
        keys.add(k)
        uniq.append(c)
    return {
        "passed": passed,
        "draft_answer": draft_answer,
        "claims": claims_out,
        "citations": uniq,
        "reason_codes": seen,
    }
