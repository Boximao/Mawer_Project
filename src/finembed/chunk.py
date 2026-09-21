"""Section-aware chunking.

Rules (all enforced here and covered by tests):
  * never cross an Item boundary
  * narrative is split on paragraph boundaries (sentence boundaries only for very long paragraphs)
  * tiny sections merge with siblings under the same parent (e.g. the five one-paragraph
    geographic segment notes become one chunk), the merged chunk keeps the parent path
  * tables never enter narrative chunks: each table becomes one table_summary chunk
  * every chunk is one contiguous span of clean.txt, so raw_text is exactly citable
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .config import ChunkingConfig
from .models import Chunk
from .parse import Block, ParsedDoc
from .text_utils import count_tokens, sentence_spans, short_hash, slugify

log = logging.getLogger(__name__)


@dataclass
class _Piece:
    start: int
    end: int
    tokens: int
    block: Block


@dataclass
class _Group:
    item_code: str
    path: list[str]
    kind: str  # narrative | footnote
    blocks: list[Block]
    barrier_before: bool
    tokens: int = 0
    table_id: Optional[str] = None


@dataclass
class ChunkResult:
    chunks: list[Chunk]
    sections: dict[str, dict]
    skipped: list[dict] = field(default_factory=list)


def path_slug(path: list[str]) -> str:
    tail = path[2:]
    if not tail:
        return "item"
    slug = "--".join(slugify(p, 28) for p in tail)
    return slug if len(slug) <= 80 else slug[:71].rstrip("-") + "-" + short_hash(slug)


def make_parent_id(doc_id: str, item_code: str, path: list[str]) -> str:
    return f"{doc_id}#{item_code}#{path_slug(path)}"


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


_NOTE_RE = re.compile(r"^Note\s+\d+\s*[–—-]")


def _is_major(title: str) -> bool:
    """Financial statement titles and Notes are hard boundaries for merging."""
    letters = [ch for ch in title if ch.isalpha()]
    upper = sum(ch.isupper() for ch in letters) / max(len(letters), 1)
    return bool(_NOTE_RE.match(title)) or upper > 0.7


def _related(a: list[str], b: list[str]) -> bool:
    """Siblings or parent/child within the same Item, never across two Notes or statements."""
    c = _common_prefix(a, b)
    if len(c) < 2 or len(c) < min(len(a), len(b)) - 1:
        return False
    for p in (a, b):
        if len(p) > len(c) and _is_major(p[len(c)]):
            return False
    return True


def _is_descendant(child: list[str], parent: list[str]) -> bool:
    return len(child) > len(parent) and child[:len(parent)] == parent


def chunk_document(parsed: ParsedDoc, cfg: ChunkingConfig) -> ChunkResult:
    doc_id = parsed.filing.doc_id
    clean = parsed.clean_text
    skipped: list[dict] = []

    # ---- 1. trivial items ("None.", "Not applicable.")
    item_tokens: dict[str, int] = defaultdict(int)
    item_tables: dict[str, int] = defaultdict(int)
    for b in parsed.blocks:
        if b.skip_reason or b.item_code is None:
            continue
        if b.kind == "table":
            item_tables[b.item_code] += 1
        elif b.kind != "heading":
            item_tokens[b.item_code] += b.n_tokens
    for b in parsed.blocks:
        if b.skip_reason is None and b.item_code is not None and b.kind != "heading" \
                and not item_tables[b.item_code] and item_tokens[b.item_code] < cfg.trivial_item_tokens:
            b.skip_reason = "trivial_item"
    for b in parsed.blocks:
        if b.skip_reason and b.kind != "heading":
            skipped.append({"reason": b.skip_reason, "item_code": b.item_code, "page": b.page_start,
                            "kind": b.kind, "preview": b.text[:80]})

    # ---- 2. units of contiguous narrative with one section path
    units: list[_Group] = []
    table_blocks: list[Block] = []
    last_table_in_item: dict[str, str] = {}
    barrier = True
    for b in parsed.blocks:
        if b.skip_reason:
            barrier = True
            continue
        if b.kind == "heading":
            continue  # headings change the path but keep text contiguous
        if b.kind == "table":
            table_blocks.append(b)
            last_table_in_item[b.item_code] = b.table.table_id
            barrier = True
            continue
        if b.kind == "units_note" or b.consumed_as_caption:
            barrier = True  # these live inside the table chunk's caption/units instead
            continue
        kind = "footnote" if b.kind == "footnote" else "narrative"
        u = units[-1] if units else None
        if u and not barrier and u.item_code == b.item_code and u.path == b.section_path and u.kind == kind:
            u.blocks.append(b)
            u.tokens += b.n_tokens
        else:
            units.append(_Group(item_code=b.item_code, path=list(b.section_path), kind=kind, blocks=[b],
                                barrier_before=barrier, tokens=b.n_tokens,
                                table_id=last_table_in_item.get(b.item_code) if kind == "footnote" else None))
        barrier = False

    # ---- 3. merge tiny sections with related neighbours (no table in between)
    groups: list[_Group] = []
    for i, u in enumerate(units):
        g = groups[-1] if groups else None
        nxt = units[i + 1] if i + 1 < len(units) else None
        # a tiny section intro followed by its own subsections merges forward, not backward
        intro_of_children = (u.tokens < cfg.min_tokens and nxt is not None and not nxt.barrier_before
                             and _is_descendant(nxt.path, u.path))
        if g and intro_of_children and g.tokens >= cfg.min_tokens:
            groups.append(u)
            continue
        if (g and not u.barrier_before and g.item_code == u.item_code and g.kind == u.kind
                and _related(g.path, u.path) and (g.tokens < cfg.min_tokens or u.tokens < cfg.min_tokens)
                and g.tokens + u.tokens <= cfg.max_tokens):
            g.path = _common_prefix(g.path, u.path)
            g.blocks.extend(u.blocks)
            g.tokens += u.tokens
        else:
            groups.append(u)

    # ---- 4. split groups into token-bounded chunks
    chunks: list[Chunk] = []
    counters: dict[str, int] = defaultdict(int)

    def overlap_tail(p: _Piece) -> Optional[_Piece]:
        if cfg.overlap_tokens <= 0:
            return None
        text = clean[p.start:p.end]
        spans = sentence_spans(text)
        tok, start_off = 0, None
        for s, e in reversed(spans):
            t = count_tokens(text[s:e])
            if tok + t > cfg.overlap_tokens:
                break
            tok += t
            start_off = s
        if start_off is None:
            return None
        return _Piece(p.start + start_off, p.end, tok, p.block)

    for g in groups:
        pieces: list[_Piece] = []
        for b in g.blocks:
            t = b.n_tokens
            if t <= cfg.max_tokens:
                pieces.append(_Piece(b.char_start, b.char_end, t, b))
            else:
                for s, e in sentence_spans(b.text):
                    pieces.append(_Piece(b.char_start + s, b.char_start + e, count_tokens(b.text[s:e]), b))
        packs: list[list[_Piece]] = []
        cur: list[_Piece] = []
        cur_tok = 0
        for p in pieces:
            if cur and (cur_tok + p.tokens > cfg.max_tokens or cur_tok >= cfg.target_tokens):
                packs.append(cur)
                ov = overlap_tail(cur[-1])
                cur, cur_tok = ([ov], ov.tokens) if ov else ([], 0)
            cur.append(p)
            cur_tok += p.tokens
        if cur:
            packs.append(cur)

        parent = make_parent_id(doc_id, g.item_code, g.path)
        for pack in packs:
            start, end = pack[0].start, pack[-1].end
            pages = [x.block.page_start for x in pack] + [x.block.page_end for x in pack]
            idx = counters[parent]
            counters[parent] += 1
            raw = clean[start:end]
            chunks.append(Chunk(
                chunk_id=f"{parent}#{idx}", doc_id=doc_id, content_type=g.kind, section_path=g.path,
                item_code=g.item_code, raw_text=raw, char_start=start, char_end=end,
                pdf_page_start=min(pages), pdf_page_end=max(pages),
                page_label=parsed.page_labels.get(min(pages)), parent_id=parent,
                table_id=g.table_id, n_tokens=count_tokens(raw),
            ))

    # ---- 5. one table_summary chunk per table
    for b in table_blocks:
        t = b.table
        parent = make_parent_id(doc_id, b.item_code, b.section_path)
        idx = counters[parent + "#t"]
        counters[parent + "#t"] += 1
        chunks.append(Chunk(
            chunk_id=f"{parent}#t{idx}", doc_id=doc_id, content_type="table_summary",
            section_path=list(b.section_path), item_code=b.item_code, raw_text=clean[b.char_start:b.char_end],
            char_start=b.char_start, char_end=b.char_end, pdf_page_start=b.page_start, pdf_page_end=b.page_end,
            page_label=parsed.page_labels.get(b.page_start), parent_id=parent, table_id=t.table_id,
            units=t.units, durations=t.durations, n_tokens=b.n_tokens,
        ))
    chunks.sort(key=lambda c: (c.char_start, c.chunk_id))

    # ---- 6. section index for parent-document retrieval
    sections: dict[str, dict] = {}
    for b in parsed.blocks:
        if b.skip_reason or b.item_code is None:
            continue
        for d in range(2, len(b.section_path) + 1):
            pid = make_parent_id(doc_id, b.item_code, b.section_path[:d])
            s = sections.setdefault(pid, {"parent_id": pid, "section_path": b.section_path[:d],
                                          "item_code": b.item_code, "char_start": b.char_start,
                                          "char_end": b.char_end, "pdf_page_start": b.page_start,
                                          "pdf_page_end": b.page_end})
            s["char_start"] = min(s["char_start"], b.char_start)
            s["char_end"] = max(s["char_end"], b.char_end)
            s["pdf_page_start"] = min(s["pdf_page_start"], b.page_start)
            s["pdf_page_end"] = max(s["pdf_page_end"], b.page_end)
    for c in chunks:  # merged groups may point at a parent that has only child blocks
        sections.setdefault(c.parent_id, {"parent_id": c.parent_id, "section_path": c.section_path,
                                          "item_code": c.item_code, "char_start": c.char_start,
                                          "char_end": c.char_end, "pdf_page_start": c.pdf_page_start,
                                          "pdf_page_end": c.pdf_page_end})
    log.info("%s: %d chunks (%d tables)", doc_id, len(chunks), len(table_blocks))
    return ChunkResult(chunks=chunks, sections=sections, skipped=skipped)
