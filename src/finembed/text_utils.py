"""Small text helpers used across stages."""
from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from typing import Optional

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+|[^\w\s]")


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:  # offline sandboxes cannot download the BPE file
        log.warning("tiktoken unavailable (%s); using regex token estimate", type(e).__name__)
        return None


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    # words + punctuation is a close proxy for BPE counts on filing prose
    return len(_WORD_RE.findall(text))


def slugify(text: str, max_len: int = 40) -> str:
    s = text.lower().replace("&", " and ")
    s = re.sub(r"[’'`]", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:max_len].rstrip("-") or "x"


def short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_SENT_RE = re.compile(r"([.!?][”\"’)]?)(\s+)(?=[A-Z(“\"$\d])")
_ABBREV_RE = re.compile(r"\b(U\.S|Inc|No|Co|Corp|vs|e\.g|i\.e|Mr|Ms|Dr|Jr|Sr)\.$")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences within text (end exclusive)."""
    spans, start = [], 0
    for m in _SENT_RE.finditer(text):
        end = m.end(1)
        if _ABBREV_RE.search(text[max(0, end - 6):end]):
            continue  # "U.S. Supreme Court" is not a sentence break
        spans.append((start, end))
        start = m.end()
    spans.append((start, len(text)))
    return [(a, b) for a, b in spans if b > a]


_NUM_CELL_RE = re.compile(r"^\(?-?\$?\d[\d,]*(\.\d+)?\)?%?\)?$|^\(?\d+(\.\d+)?\)?%$")
DASHES = {"—", "–", "-", "−"}


def is_numeric_cell(text: str) -> bool:
    t = text.replace(" ", "").strip()
    if not t:
        return False
    if t in DASHES or t == "$":
        return True
    return bool(_NUM_CELL_RE.match(t))


def parse_number(text: str) -> Optional[float]:
    """'(1,094)' -> -1094.0, '11%' -> 11.0, a dash cell -> None."""
    t = text.replace(" ", "").replace("$", "").strip()
    if not t or t in DASHES:
        return None
    neg = t.startswith("(") or t.startswith("-") or t.startswith("−")
    t = t.strip("()%-−").replace(",", "").replace(")", "").replace("(", "")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


_NUM_IN_TEXT_RE = re.compile(r"(?<![\w.])\$?\(?\d[\d,]*(?:\.\d+)?\)?%?")


def numbers_in_text(text: str) -> set[str]:
    """Normalized numbers mentioned in free text (used to check LLM table summaries)."""
    out = set()
    for m in _NUM_IN_TEXT_RE.finditer(text):
        v = parse_number(m.group(0))
        if v is not None:
            v = abs(v)
            out.add(f"{v:.2f}".rstrip("0").rstrip("."))
    return out
