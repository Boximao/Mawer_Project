"""Parse an SEC filing PDF into structured blocks.

Pipeline inside this module:
  1. PDF lines -> Frags (text + bbox + font style), superscripts dropped
  2. Frags -> visual Rows (cluster by y). Table cells on one line become one Row.
  3. Drop running footers, company-name page headers, boilerplate lines
  4. State machine over rows: headings update a section stack
     (Part > Item > statement/notes title > Note > bold subheading > italic subheading),
     body rows become paragraphs, cell-like rows become tables.
  5. Everything is written into clean.txt order so every block has char offsets.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import pymupdf

from .models import FilingMeta, TableData
from .tables import build_table, is_header_row, render_table
from .text_utils import count_tokens, is_numeric_cell, normalize_ws

log = logging.getLogger(__name__)

PART_RE = re.compile(r"^PART\s+(IV|III|II|I)\b")
ITEM_RE = re.compile(r"^Item\s+(\d{1,2}[A-C]?)\.\s*(.*)$", re.I)
NOTE_RE = re.compile(r"^Note\s+(\d{1,2})\s*[–—-]\s*(.+)$")
FOOTER_RE = re.compile(r"^.{2,80}\|\s*.{2,60}\|\s*(\d{1,4})$")
FOOTNOTE_RE = re.compile(r"^\((\d{1,2})\)\s+\S")
MARKER_ONLY_RE = re.compile(r"^(\(\d{1,2}\)\s*)+$")
UNITS_LINE_RE = re.compile(r"^\((?:dollars\s+|net income\s+)?in\s+(millions|thousands|billions)", re.I)
BOILERPLATE_RES = [
    re.compile(r"^See accompanying Notes to .*Financial Statements\.?$", re.I),
]
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}

# heading levels
L_PART, L_ITEM, L_TITLE, L_NOTE, L_BOLD, L_SUB = 0, 1, 2, 3, 4, 5


@dataclass
class Frag:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool
    italic: bool
    lead: int = 0  # leading spaces (used for table row indentation)


@dataclass
class Row:
    page: int  # 1-indexed physical page
    frags: list[Frag]

    @property
    def y0(self) -> float:
        return min(f.y0 for f in self.frags)

    @property
    def x0(self) -> float:
        return self.frags[0].x0

    @property
    def size(self) -> float:
        return max(f.size for f in self.frags)

    @property
    def text(self) -> str:
        return normalize_ws(" ".join(f.text for f in self.frags))

    @property
    def single(self) -> bool:
        return len(self.frags) == 1


@dataclass
class Block:
    kind: str  # heading | paragraph | footnote | units_note | table
    text: str
    page_start: int
    page_end: int
    section_path: list[str]
    item_code: Optional[str]
    level: Optional[int] = None
    skip_reason: Optional[str] = None
    table: Optional[TableData] = None
    consumed_as_caption: bool = False
    char_start: int = 0
    char_end: int = 0
    rows: list[Row] = field(default_factory=list, repr=False)

    @property
    def n_tokens(self) -> int:
        return count_tokens(self.text)


@dataclass
class ParsedDoc:
    filing: FilingMeta
    blocks: list[Block]
    clean_text: str
    tables: list[TableData]
    page_labels: dict[int, str]
    body_size: float
    n_pages: int

    def item_spans(self) -> dict[str, tuple[int, int]]:
        spans: dict[str, tuple[int, int]] = {}
        for b in self.blocks:
            if b.item_code is None:
                continue
            s, e = spans.get(b.item_code, (b.char_start, b.char_end))
            spans[b.item_code] = (min(s, b.char_start), max(e, b.char_end))
        return spans


# --------------------------------------------------------------------------- extraction

def _is_bold(span) -> bool:
    return bool(span["flags"] & 16) or "Bold" in span["font"]


def _is_italic(span) -> bool:
    return bool(span["flags"] & 2) or "Italic" in span["font"] or "Oblique" in span["font"]


def _extract_rows(doc: pymupdf.Document, min_size: float = 6.5) -> list[list[Row]]:
    pages: list[list[Row]] = []
    for pno, page in enumerate(doc, start=1):
        frags: list[Frag] = []
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            for line in b["lines"]:
                spans = [s for s in line["spans"] if s["size"] >= min_size]
                if not spans or not "".join(s["text"] for s in spans).strip():
                    continue
                raw = "".join(s["text"] for s in spans)
                text = raw.strip()
                styled = [s for s in spans if s["text"].strip()]
                x0, y0, x1, y1 = line["bbox"]
                frags.append(Frag(
                    text=text, x0=x0, y0=y0, x1=x1, y1=y1,
                    size=max(s["size"] for s in styled),
                    bold=all(_is_bold(s) for s in styled),
                    italic=all(_is_italic(s) for s in styled),
                    lead=len(raw) - len(raw.lstrip(" ")),
                ))
        frags.sort(key=lambda f: (round(f.y0, 1), f.x0))
        rows: list[Row] = []
        for f in frags:
            if rows and abs(f.y0 - rows[-1].frags[0].y0) <= 3.0:
                rows[-1].frags.append(f)
            else:
                rows.append(Row(page=pno, frags=[f]))
        for r in rows:
            r.frags.sort(key=lambda f: f.x0)
        pages.append(rows)
    return pages


def _body_font_size(pages: list[list[Row]]) -> float:
    c: Counter = Counter()
    for rows in pages:
        for r in rows:
            for f in r.frags:
                c[round(f.size, 1)] += len(f.text)
    return c.most_common(1)[0][0] if c else 10.0


def _strip_page_furniture(pages: list[list[Row]], company_name: str) -> tuple[list[list[Row]], dict[int, str]]:
    """Remove running footers/headers, company-name headers and boilerplate lines."""
    labels: dict[int, str] = {}
    norm_counts: Counter = Counter()
    for rows in pages:
        seen = set()
        for r in (rows[:1] + rows[-1:]):
            key = re.sub(r"\d+", "#", r.text)
            if key not in seen:
                norm_counts[key] += 1
                seen.add(key)
    n = max(len(pages), 1)
    repeated = {k for k, v in norm_counts.items() if v / n >= 0.4}

    out = []
    for rows in pages:
        kept = []
        for i, r in enumerate(rows):
            t = r.text
            m = FOOTER_RE.match(t)
            if m:
                labels[r.page] = m.group(1)
                continue
            if (i == 0 or i == len(rows) - 1) and re.sub(r"\d+", "#", t) in repeated:
                continue
            if t == company_name or MARKER_ONLY_RE.match(t) or t in {"®", "™"}:
                continue
            if any(p.match(t) for p in BOILERPLATE_RES):
                continue
            kept.append(r)
        out.append(kept)
    return out, labels


# --------------------------------------------------------------------------- row classification

YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def _is_table_ish(row: Row, body: float) -> bool:
    right = [f for f in row.frags if f.x0 > 150]
    if not right:
        return False
    if any(is_numeric_cell(f.text) for f in right):
        return True
    return _is_header_row(row, body)


def _is_header_row(row: Row, body: float) -> bool:
    return is_header_row(row.frags, body)


def _heading(row: Row, prev: Optional[Row], body: float) -> Optional[tuple[int, str]]:
    if not row.single:
        return None
    f = row.frags[0]
    t = row.text
    body_sized = abs(f.size - body) < 0.5
    # a line tightly below body text is a paragraph continuation, not a heading
    glued = prev is not None and prev.page == row.page and (row.y0 - prev.y0) < 1.35 * body and prev.single \
        and not (prev.frags[0].bold or prev.frags[0].italic)
    if f.bold and not f.italic:
        if PART_RE.match(t):
            return L_PART, t
        if ITEM_RE.match(t) and f.x0 < 40:
            return L_ITEM, t
        letters = [c for c in t if c.isalpha()]
        upper_ratio = sum(c.isupper() for c in letters) / max(len(letters), 1)
        if f.x0 > 100 and body_sized and not t.endswith(":") and len(t.split()) >= 3 \
                and (upper_ratio > 0.7 or t.lower().startswith("notes to")):
            return L_TITLE, t
        if NOTE_RE.match(t):
            return L_NOTE, t
        if f.x0 < 40 and body_sized and len(t.split()) <= 14 and not glued:
            return L_BOLD, t
    if f.italic and not f.bold and f.x0 < 40 and body_sized and not glued:
        if len(t.split()) <= 12 and not t.endswith("."):
            return L_SUB, t
    if f.italic and f.bold and f.x0 < 40 and body_sized:
        return L_SUB, t  # risk factor headline (may continue on the next line)
    return None


def _continues_table(row: Row, nxt: Optional[Row], body: float, prev: Row) -> bool:
    if row.page != prev.page:
        return False
    if _is_table_ish(row, body):
        return True
    t = row.text
    if not row.single:
        return False
    next_is_header = nxt is not None and _is_header_row(nxt, body)
    gap = row.y0 - prev.y0
    # group label such as "Net sales:", "Level 1:", "ASSETS:" (a caption for a new table is
    # followed by column headers instead)
    if t.endswith(":") and len(t) <= 100 and gap < 30 and not next_is_header:
        return True
    if _heading(row, None, body) is not None:
        return False
    if row.x0 < 150:
        if len(t) <= 60 and not t.endswith(".") and gap < 30 and not next_is_header:
            return True
        if nxt is not None and nxt.page == row.page and _is_table_ish(nxt, body) and (nxt.y0 - row.y0) < 14:
            return True  # wrapped row label
    return False


def _join_lines(rows: list[Row]) -> str:
    out = ""
    for r in rows:
        t = r.text
        if not out:
            out = t
        elif t.startswith("•"):
            out += "\n" + t
        elif out.endswith("-") and t[:1].islower():
            out += t
        else:
            out += " " + t
    return out


# --------------------------------------------------------------------------- main parse

def parse_filing(pdf_path: str, filing: FilingMeta) -> ParsedDoc:
    doc = pymupdf.open(pdf_path)
    raw_pages = _extract_rows(doc)
    body = _body_font_size(raw_pages)
    pages, page_labels = _strip_page_furniture(raw_pages, filing.company_name)
    annual = filing.is_annual

    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []
    state = {"started": False, "stopped": False, "part": None, "item": None, "item_title": ""}
    para: list[Row] = []
    table_rows: list[Row] = []
    table_count = 0

    def item_code() -> Optional[str]:
        if state["item"] is None:
            return None
        if annual:
            return f"10K-I{state['item']}"
        return f"10Q-P{state['part'] or 0}-I{state['item']}"

    def skip_reason() -> Optional[str]:
        if not state["started"]:
            return "cover_or_toc"
        if state["stopped"]:
            return "signatures_and_certifications"
        if re.search(r"\bExhibits\b", state["item_title"], re.I):
            return "exhibit_index"
        return None

    def path() -> list[str]:
        return [t for _, t in stack]

    def add_block(kind: str, text: str, rows: list[Row], **kw) -> Block:
        b = Block(kind=kind, text=text, page_start=rows[0].page, page_end=rows[-1].page,
                  section_path=path(), item_code=item_code(), skip_reason=skip_reason(), rows=rows, **kw)
        blocks.append(b)
        return b

    def flush_para():
        nonlocal para
        if not para:
            return
        text = _join_lines(para)
        if FOOTNOTE_RE.match(text):
            kind = "footnote"
        elif UNITS_LINE_RE.match(text) and len(text) < 200:
            kind = "units_note"
        else:
            kind = "paragraph"
        add_block(kind, text, para)
        para = []

    def flush_table():
        nonlocal table_rows, table_count
        if not table_rows:
            return
        rows, table_rows = table_rows, []
        caption, source, units_note = _caption_context(blocks, stack)
        table = build_table(
            rows, body_size=body, table_id=f"{filing.doc_id}-T{table_count + 1:03d}", doc_id=filing.doc_id,
            caption=caption, caption_source=source, units_note=units_note, section_path=path(),
            item_code=item_code() or "none", pdf_page=rows[0].page, page_label=page_labels.get(rows[0].page),
        )
        if table is None:  # no numbers: it was text laid out in columns
            for r in rows:
                add_block("paragraph", r.text, [r])
            return
        table_count += 1
        if source == "preceding_paragraph" and blocks and blocks[-1].kind == "paragraph":
            blocks[-1].consumed_as_caption = True
        add_block("table", render_table(table), rows, table=table)

    def on_heading(level: int, text: str, row: Row):
        if level == L_PART:
            m = PART_RE.match(text)
            state["started"] = True
            state["part"] = ROMAN[m.group(1)]
            state["item"], state["item_title"] = None, ""
            stack.clear()
            stack.append((L_PART, f"Part {m.group(1)}"))
        elif level == L_ITEM:
            m = ITEM_RE.match(text)
            state["item"] = m.group(1).upper()
            state["item_title"] = normalize_ws(m.group(2))
            while stack and stack[-1][0] >= L_ITEM:
                stack.pop()
            stack.append((L_ITEM, f"Item {state['item']}. {state['item_title']}"))
        else:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
        add_block("heading", text, [row], level=level)

    for rows in pages:
        for i, row in enumerate(rows):
            prev = rows[i - 1] if i > 0 else None
            nxt = rows[i + 1] if i + 1 < len(rows) else None
            t = row.text

            if t in {"SIGNATURE", "SIGNATURES"} and state["started"]:
                flush_para(); flush_table()
                state["stopped"] = True
                stack.clear()
                state["item"], state["item_title"] = None, ""

            if table_rows:
                has_data = any(not _is_header_row(r, body) for r in table_rows)
                if has_data and _is_header_row(row, body) and not _is_header_row(table_rows[-1], body):
                    flush_table()  # a new header block after data rows starts a new table
                    table_rows = [row]
                    continue
                if _continues_table(row, nxt, body, table_rows[-1]):
                    table_rows.append(row)
                    continue
                flush_table()

            h = _heading(row, prev, body)
            if h:
                level, htext = h
                last = blocks[-1] if blocks else None
                # multi-line bold-italic risk factor headline: glue continuation lines
                if (level == L_SUB and row.frags[0].bold and row.frags[0].italic and last is not None
                        and last.kind == "heading" and last.level == L_SUB and not para
                        and last.rows[-1].page == row.page and (row.y0 - last.rows[-1].y0) < 1.5 * body
                        and last.rows[-1].frags[0].bold and last.rows[-1].frags[0].italic):
                    last.text = normalize_ws(last.text + " " + htext)
                    last.rows.append(row)
                    stack[-1] = (L_SUB, last.text)
                    last.section_path = path()
                    continue
                flush_para()
                on_heading(level, htext, row)
                continue

            if _is_table_ish(row, body):
                flush_para()
                table_rows = [row]
                continue

            # body text row
            if para:
                last_row = para[-1]
                same_style = last_row.frags[0].italic == row.frags[0].italic
                if last_row.page == row.page:
                    joined = same_style and (row.y0 - last_row.y0) < 1.6 * body and not FOOTNOTE_RE.match(t)
                else:  # paragraph continuing across a page break
                    joined = same_style and not re.search(r"[.:;?!”)]$", last_row.text) and t[:1].islower()
                if not joined:
                    flush_para()
            para.append(row)
        flush_table()  # tables never continue across pages (headers repeat instead)
    flush_para()
    flush_table()

    # write clean text and offsets
    parts, pos = [], 0
    for b in blocks:
        b.char_start = pos
        parts.append(b.text)
        pos += len(b.text)
        b.char_end = pos
        parts.append("\n\n")
        pos += 2
    clean_text = "".join(parts)
    tables = [b.table for b in blocks if b.table is not None and b.skip_reason is None]
    log.info("%s: %d pages, %d blocks, %d tables", filing.doc_id, len(doc), len(blocks), len(tables))
    return ParsedDoc(filing=filing, blocks=blocks, clean_text=clean_text, tables=tables,
                     page_labels=page_labels, body_size=body, n_pages=len(doc))


def _caption_context(blocks: list[Block], stack: list[tuple[int, str]]) -> tuple[str, str, Optional[str]]:
    """Find the caption and units note for a table that is about to be built."""
    units_note = None
    # nearest units parenthetical since the last Note/title/Item heading
    for b in reversed(blocks):
        if b.kind == "heading" and b.level is not None and b.level <= L_NOTE:
            break
        if b.kind in ("paragraph", "units_note"):
            m = re.search(r"\(([^()]*\bin (?:millions|thousands|billions)[^()]*)\)", b.text, re.I)
            if m:
                units_note = m.group(1).strip()
                break
    prev = blocks[-1] if blocks else None
    if prev is not None and prev.kind == "paragraph" and (
            prev.text.rstrip().endswith(":") or "following table" in prev.text.lower()):
        return prev.text, "preceding_paragraph", units_note
    titles = [t for lvl, t in stack if lvl == L_TITLE]
    leaf_level = stack[-1][0] if stack else None
    if titles and leaf_level == L_TITLE:
        return titles[-1], "statement_title", units_note
    if stack:
        return stack[-1][1], "section_heading", units_note
    return "Table", "none", units_note
