"""Pydantic data models shared by every pipeline stage."""
from __future__ import annotations

import re
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

ContentType = Literal["narrative", "table_summary", "footnote"]


def calendar_quarter_for(period_end: date) -> str:
    """Map a period end to the nearest calendar quarter end.

    Handles 52/53 week fiscal calendars, e.g. a quarter ending on Jan 1 belongs to Q4
    of the prior year, and Apple's Dec 27 quarter end belongs to Q4.
    """
    candidates = []
    for year in (period_end.year - 1, period_end.year, period_end.year + 1):
        for q, (m, d) in enumerate([(3, 31), (6, 30), (9, 30), (12, 31)], start=1):
            candidates.append((abs((date(year, m, d) - period_end).days), year, q))
    _, year, q = min(candidates)
    return f"{year}-Q{q}"


class FilingMeta(BaseModel):
    """One manifest row. The manifest is the source of truth for these fields."""

    ticker: str
    cik: str
    company_name: str
    form_type: str
    accession_no: Optional[str] = None
    filing_date: date
    period_end: date
    fiscal_year: int
    fiscal_quarter: Optional[int] = None
    source_url: Optional[str] = None
    pdf_path: str
    amends: Optional[str] = None

    @field_validator("form_type")
    @classmethod
    def _form(cls, v: str) -> str:
        v = v.strip().upper()
        if not re.fullmatch(r"10-[QK](/A)?", v):
            raise ValueError(f"unsupported form_type {v!r} (expected 10-Q, 10-K, 10-Q/A, 10-K/A)")
        return v

    @field_validator("fiscal_quarter")
    @classmethod
    def _fq(cls, v):
        if v is not None and v not in (1, 2, 3, 4):
            raise ValueError("fiscal_quarter must be 1-4 or empty")
        return v

    @property
    def is_annual(self) -> bool:
        return self.form_type.startswith("10-K")

    @property
    def doc_id(self) -> str:
        """Stable, human-readable id. Accession stays in metadata for citation."""
        form = self.form_type.replace("/", "")
        return f"{self.ticker}_{form}_{self.period_end.isoformat()}"

    @property
    def calendar_quarter(self) -> str:
        return calendar_quarter_for(self.period_end)

    @property
    def short_company(self) -> str:
        return re.sub(r",?\s+(Inc|Corp|Corporation|Co|Ltd|plc|Holdings)\.?$", "", self.company_name).strip()

    @property
    def period_label(self) -> str:
        if self.is_annual:
            return f"Fiscal Year {self.fiscal_year}"
        return f"Fiscal Q{self.fiscal_quarter} {self.fiscal_year}"


class TableCell(BaseModel):
    raw: Optional[str] = None
    num: Optional[float] = None


class TableRow(BaseModel):
    label: str
    full_label: str
    indent: int = 0
    values: list[TableCell] = Field(default_factory=list)


class TableData(BaseModel):
    table_id: str
    doc_id: str
    title: str
    caption: str
    caption_source: str
    units: Optional[str] = None
    units_note: Optional[str] = None
    durations: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    rows: list[TableRow] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    item_code: str
    pdf_page: int
    page_label: Optional[str] = None

    def numbers(self) -> set[str]:
        """Every number that appears in the table, normalized (for hallucination checks)."""
        out: set[str] = set()
        for r in self.rows:
            for c in r.values:
                if c.num is not None:
                    out.add(_norm_num(c.num))
        return out


def _norm_num(x: float) -> str:
    x = abs(x)
    return f"{x:.2f}".rstrip("0").rstrip(".")


class Chunk(BaseModel):
    """Output of chunking, before enrichment."""

    chunk_id: str
    doc_id: str
    content_type: ContentType
    section_path: list[str]
    item_code: str
    raw_text: str
    char_start: int
    char_end: int
    pdf_page_start: int
    pdf_page_end: int
    page_label: Optional[str] = None
    parent_id: str
    table_id: Optional[str] = None
    units: Optional[str] = None
    durations: list[str] = Field(default_factory=list)
    n_tokens: int = 0


class Record(BaseModel):
    """One row in pgvector: a chunk plus everything needed to filter and cite it."""

    id: str  # = chunk_id
    doc_id: str
    ticker: str
    cik: str
    company_name: str
    form_type: str
    accession_no: Optional[str] = None
    filing_date: date
    fiscal_year: int
    fiscal_quarter: Optional[int] = None
    period_end: date
    calendar_quarter: str
    source_url: Optional[str] = None
    section_path: list[str]
    item_code: str
    content_type: ContentType
    parent_id: str
    table_id: Optional[str] = None
    units: Optional[str] = None
    durations: list[str] = Field(default_factory=list)
    char_start: int
    char_end: int
    pdf_page_start: int
    pdf_page_end: int
    page_label: Optional[str] = None
    n_tokens: int
    embed_text: str
    raw_text: str
    content_hash: str
    embedding_model: str
    pipeline_version: str
    embedding: Optional[list[float]] = None
