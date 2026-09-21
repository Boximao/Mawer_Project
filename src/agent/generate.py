"""JSON draft from allowlisted chunks only. Temp 0. Extractive fallback if no chat key."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from retrieve.hybrid import ChunkHit

from .validate import quote_in_chunk

ROOT = Path(__file__).resolve().parents[2]
_SENT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]{3,}")
_STOP = frozenset({
    "and", "did", "does", "ended", "for", "from", "how", "the", "three",
    "was", "were", "what", "when", "where", "which", "why", "with", "apple",
})


def load_prompts() -> tuple[str, str]:
    system = (ROOT / "prompts" / "system.txt").read_text(encoding="utf-8")
    user = (ROOT / "prompts" / "user_template.txt").read_text(encoding="utf-8")
    return system, user


def format_context(chunks: Sequence[ChunkHit]) -> str:
    parts = []
    for h in chunks:
        parts.append(
            f"[chunk_id={h.chunk_id} document_id={h.document_id} period={h.filing_period} "
            f"page={h.page} section={h.section}]\n{h.text}"
        )
    return "\n\n".join(parts)


def empty_draft() -> dict[str, Any]:
    return {"draft_answer": None, "claims": [], "citations": [], "next_search_query": None}


def extractive_draft(question: str, chunks: Sequence[ChunkHit]) -> dict[str, Any]:
    q_terms = {t for t in _TOKEN.findall(question.lower()) if t not in _STOP}
    q_lower = question.lower()
    expansions = {
        "buy back": {"repurchase", "repurchased", "shares", "stock"},
        "fall": {"decreased", "decline", "declined", "lower"},
        "grow": {"increased", "growth", "higher"},
        "trading plan": {"10b5", "entered", "plan"},
    }
    for phrase, terms in expansions.items():
        if phrase in q_lower:
            q_terms.update(terms)
    causal_question = q_lower.startswith("why ") or "why did" in q_lower
    narrative_question = causal_question or q_lower.startswith(("which ", "who "))
    numeric_question = q_lower.startswith(("how much ", "how large ", "what was "))
    scored: list[tuple[float, str, ChunkHit]] = []
    for rank, ch in enumerate(chunks, start=1):
        if not ch.text:
            continue
        # Table summaries are line-oriented. Treating the whole table as one
        # sentence can truncate a numeric row mid-token and fail the exact
        # number gate.
        candidates = (
            ch.text.splitlines()
            if ch.chunk_type in ("table", "table_summary")
            else _SENT.split(ch.text.replace("\n", " "))
        )
        for sent in candidates:
            s = sent.strip()
            if len(s) < 12:
                continue
            if s.lower().startswith(("columns:", "[table]")):
                continue
            overlap = len(q_terms & set(_TOKEN.findall(s.lower())))
            if overlap >= 2 or (overlap >= 1 and any(char.isdigit() for char in s)):
                score = overlap * 2.0 + 2.0 / rank
                lower_sentence = s.lower()
                if causal_question and any(
                    marker in lower_sentence
                    for marker in ("due to", "driven by", "because", "primarily", "resulted")
                ):
                    score += 5.0
                is_table = ch.chunk_type in ("table", "table_summary")
                if narrative_question:
                    score += -2.0 if is_table else 2.0
                numeric_tokens = _TOKEN.findall(s)
                value_tokens = [
                    token for token in numeric_tokens
                    if token.isdigit() and not (1900 <= int(token) <= 2100)
                ]
                if numeric_question and ("$" in s or "%" in s or "|" in s):
                    score += 6.0
                elif numeric_question and value_tokens:
                    score += 2.0
                if numeric_question and is_table and overlap >= 1:
                    score += 5.0
                scored.append((score, s, ch))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return empty_draft()
    claims = []
    citations = []
    seen = set()
    for _, sent, ch in scored:
        if ch.chunk_id in seen:
            continue
        quote = sent
        if not quote_in_chunk(quote, ch.text):
            continue
        seen.add(ch.chunk_id)
        claims.append({
            "text": sent,
            "citation_chunk_ids": [ch.chunk_id],
            "quote": quote,
            "supported": True,
        })
        citations.append({
            "document_id": ch.document_id,
            "chunk_id": ch.chunk_id,
            "filing_period": ch.filing_period,
            "page": ch.page,
            "section": ch.section,
            "quote": quote,
        })
        # One exact extractive claim is safer than combining nearby periods.
        if len(claims) >= 1:
            break
    return {
        "draft_answer": claims[0]["text"] if claims else None,
        "claims": claims,
        "citations": citations,
        "next_search_query": None,
    }


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def generate_draft(question: str, chunks: Sequence[ChunkHit]) -> tuple[dict[str, Any], int]:
    """Returns (draft, completion_tokens_estimate)."""
    if not chunks:
        return empty_draft(), 0
    system, user_t = load_prompts()
    user = user_t.replace("{question}", question).replace("{context}", format_context(chunks))
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if openai_key:
        try:
            return _openai(system, user)
        except Exception:
            # Workshop keys may be short-lived. Retrieval and HITL must remain
            # available when the generation vendor rejects or times out.
            pass
    if gemini_key:
        try:
            return _gemini(system, user)
        except Exception:
            pass
    draft = extractive_draft(question, chunks)
    return draft, 0


def _openai(system: str, user: str) -> tuple[dict[str, Any], int]:
    from openai import OpenAI

    model = os.environ.get("CHAT_MODEL") or "gpt-4.1-mini"
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        top_p=1,
        max_tokens=800,
        seed=42,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    tokens = 0
    if resp.usage:
        tokens = int(resp.usage.prompt_tokens or 0) + int(resp.usage.completion_tokens or 0)
    return _parse_json(content), tokens


def _gemini(system: str, user: str) -> tuple[dict[str, Any], int]:
    import httpx

    model = os.environ.get("CHAT_MODEL") or "gemini-2.0-flash"
    key = os.environ["GEMINI_API_KEY"]
    r = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0,
                "topP": 1,
                "maxOutputTokens": 800,
                "responseMimeType": "application/json",
            },
        },
        timeout=20.0,
    )
    r.raise_for_status()
    data = r.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return _parse_json(text), 0
