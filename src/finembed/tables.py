"""Turn positioned table rows from the PDF into structured TableData.

Financial tables in EDGAR PDFs are laid out as positioned text: every cell is its own
line, numbers are right-aligned, "$" signs are separate cells, and column headers
span several lines ("Three Months Ended" / "June 27," / "2026"). We rebuild columns
from the right edges of numeric cells and attach header fragments to columns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .models import TableCell, TableData, TableRow
from .text_utils import DASHES, is_numeric_cell, normalize_ws, parse_number

if TYPE_CHECKING:  # pragma: no cover
    from .parse import Frag, Row

VALUE_X_MIN = 150.0  # anything left of this is row-label territory
SPACE_W = 2.25       # approx width of a space at ~8pt, for indentation by leading spaces


_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def is_header_row(frags: list["Frag"], body: float) -> bool:
    """Column header line: small or bold cells in the value area, no data values (years allowed).

    A bold cell in the label column (e.g. "Periods") is allowed; a plain row label is not.
    """
    right = [f for f in frags if f.x0 > VALUE_X_MIN]
    left = [f for f in frags if f.x0 <= VALUE_X_MIN]
    if not right:
        return False
    if left and not all(f.bold for f in left):
        return False
    small = all(f.size < body - 0.4 for f in right)
    bold = all(f.bold for f in right)
    if not (small or (bold and len(right) >= 2)):
        return False
    return not any(is_numeric_cell(f.text) and not _YEAR_RE.match(f.text.strip()) for f in frags)


@dataclass
class _Col:
    x0: float
    x1: float

    @property
    def center(self) -> float:
        return (self.x0 + self.x1) / 2


def _cluster_columns(value_frags: list["Frag"], gap: float = 12.0) -> list[_Col]:
    nums = sorted((f for f in value_frags if f.text.strip() not in DASHES), key=lambda f: f.x1)
    cols: list[list["Frag"]] = []
    for f in nums:
        if cols and f.x1 - cols[-1][-1].x1 <= gap:
            cols[-1].append(f)
        else:
            cols.append([f])
    return [_Col(min(f.x0 for f in c), max(f.x1 for f in c)) for c in cols]


def _nearest(cols: list[_Col], f: "Frag") -> int:
    return min(range(len(cols)), key=lambda j: min(abs(cols[j].x1 - f.x1), abs(cols[j].center - (f.x0 + f.x1) / 2)))


def _durations(columns: list[str]) -> list[str]:
    out: list[str] = []
    for c in columns:
        cl = c.lower()
        d = None
        for word, tag in (("three months", "3M"), ("six months", "6M"), ("nine months", "9M"),
                          ("twelve months", "12M"), ("year ended", "12M"), ("years ended", "12M")):
            if word in cl:
                d = tag
                break
        if d is None and re.search(r"(january|february|march|april|may|june|july|august|september|october|"
                                   r"november|december)\s+\d{1,2},?\s+\d{4}", cl):
            d = "instant"
        if d and d not in out:
            out.append(d)
    return out


def make_title(caption: str) -> str:
    c = normalize_ws(caption)
    i = c.lower().find("the following table")
    if i > 0:
        c = c[i:]
    elif c.endswith(":"):
        sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", c)
        c = sents[-1]
    m = re.match(r"^The following tables? (?:shows?|provides?|summarizes?|presents?|sets? forth) (?:the )?(.*)$", c, re.I)
    if m:
        c = m.group(1)
    c = re.sub(r"\s*\([^()]*\bin (?:millions|thousands|billions)[^()]*\)", "", c, flags=re.I)
    c = c.rstrip(":. ")
    if not c:
        return "Table"
    c = c[0].upper() + c[1:]
    return c if len(c) <= 220 else c[:217].rstrip() + "..."


def build_table(rows: list["Row"], *, body_size: float, table_id: str, doc_id: str, caption: str,
                caption_source: str, units_note: Optional[str], section_path: list[str], item_code: str,
                pdf_page: int, page_label: Optional[str]) -> Optional[TableData]:
    header: list["Row"] = []
    body: list["Row"] = []
    for r in rows:
        if not body and is_header_row(r.frags, body_size):
            header.append(r)
        else:
            body.append(r)

    parsed = []  # (row, label_frags, value_frags)
    all_values: list["Frag"] = []
    for r in body:
        label, values = [], []
        for f in r.frags:
            if f.x0 > VALUE_X_MIN and (is_numeric_cell(f.text) or values or label):
                if f.text.strip() != "$":
                    values.append(f)
            else:
                label.append(f)
        parsed.append((r, label, values))
        all_values.extend(values)

    if not any(parse_number(f.text) is not None for f in all_values):
        return None
    cols = _cluster_columns(all_values)
    if not cols:
        return None

    # ---- column headers
    col_heads: list[list[str]] = [[] for _ in cols]
    period_re = re.compile(r"\b(months|weeks|year|years|quarter)\s+ended\b", re.I)
    for hr in header:
        frags = hr.frags
        if len(frags) == 1 and (len(cols) >= 3 or period_re.search(frags[0].text)):
            for j in range(len(cols)):  # table-level title such as "Three Months Ended June 27, 2026"
                col_heads[j].append(frags[0].text)
        elif any(period_re.search(f.text) for f in frags) and len(frags) < len(cols):
            for j, c in enumerate(cols):  # spanning group headers: each column takes the nearest one
                best = min(frags, key=lambda f: abs((f.x0 + f.x1) / 2 - c.center))
                col_heads[j].append(best.text)
        else:
            for f in frags:  # one header fragment per column: nearest column by center
                fc = (f.x0 + f.x1) / 2
                j = min(range(len(cols)), key=lambda k: abs(cols[k].center - fc))
                if abs(cols[j].center - fc) < 80:
                    col_heads[j].append(f.text)
    columns = [normalize_ws(" ".join(h)) or f"col_{j + 1}" for j, h in enumerate(col_heads)]
    if len(columns) == 1 and not header:
        columns = ["Amount"]

    # ---- rows: labels, wrapped labels, indentation, values
    eff_x = sorted({round(l[0].x0 + SPACE_W * l[0].lead) for _, l, _ in parsed if l})
    levels: list[int] = []
    for x in eff_x:
        if not levels or x - levels[-1] > 3:
            levels.append(x)

    def indent_of(label_frags) -> int:
        if not label_frags:
            return 0
        x = label_frags[0].x0 + SPACE_W * label_frags[0].lead
        return min(range(len(levels)), key=lambda k: abs(levels[k] - x)) if levels else 0

    out_rows: list[TableRow] = []
    pending_label: Optional[tuple[str, "Row"]] = None
    group: Optional[tuple[str, int]] = None
    prev_row: Optional["Row"] = None
    for r, label_frags, value_frags in parsed:
        label = normalize_ws(" ".join(f.text for f in label_frags))
        ind = indent_of(label_frags)
        if prev_row is not None and r.y0 - prev_row.y0 > 18:
            group = None
        prev_row = r
        if not value_frags:
            if label.endswith(":") or label.isupper():
                group = (label.rstrip(":").strip(), ind)
                out_rows.append(TableRow(label=label, full_label=label, indent=ind))
                pending_label = None
            else:
                pending_label = (label, r)
            continue
        if pending_label and label[:1].islower() and r.y0 - pending_label[1].y0 < 12:
            label = normalize_ws(pending_label[0] + " " + label)
        elif pending_label:
            out_rows.append(TableRow(label=pending_label[0], full_label=pending_label[0], indent=ind))
        pending_label = None

        cells = [TableCell() for _ in cols]
        for f in value_frags:
            j = _nearest(cols, f)
            raw = f.text.strip()
            if cells[j].raw:
                raw = cells[j].raw + " " + raw
            cells[j] = TableCell(raw=raw, num=parse_number(raw))

        full = label
        if group and ind > group[1]:
            if not label:
                full = group[0]
            elif group[0].lower() not in label.lower():
                full = f"{group[0]}: {label}"
        if label.lower().startswith("total"):
            group = None
        out_rows.append(TableRow(label=label or "(unlabeled)", full_label=full or "(unlabeled)", indent=ind, values=cells))
    if pending_label:
        out_rows.append(TableRow(label=pending_label[0], full_label=pending_label[0]))

    units = None
    src = units_note or caption
    m = re.search(r"\bin (millions|thousands|billions)\b", src or "", re.I)
    if m:
        units = f"USD {m.group(1).lower()}"
    title = make_title(caption)
    if header and header[0].single and header[0].text.lower() not in title.lower():
        title = f"{title} ({header[0].text})"  # e.g. "(Three Months Ended June 27, 2026)"
    return TableData(
        table_id=table_id, doc_id=doc_id, title=title, caption=normalize_ws(caption),
        caption_source=caption_source, units=units, units_note=units_note, durations=_durations(columns),
        columns=columns, rows=out_rows, section_path=section_path, item_code=item_code,
        pdf_page=pdf_page, page_label=page_label,
    )


def render_table(t: TableData) -> str:
    """Plain-text rendering stored in clean.txt; this is the citable raw_text of a table."""
    lines = [f"[Table] {t.title}"]
    if t.units_note:
        lines.append(f"({t.units_note})")
    lines.append("Columns: " + " | ".join(t.columns))
    for r in t.rows:
        if r.values:
            lines.append(r.full_label + " | " + " | ".join(c.raw or "" for c in r.values))
        else:
            lines.append(r.full_label)
    return "\n".join(lines)
