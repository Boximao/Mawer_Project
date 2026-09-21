"""Build the exact text that gets embedded.

Every chunk starts with the same header format, so the "who and when" part of each
vector is consistent across companies and filings and the content drives the differences.
"""
from __future__ import annotations

import re
from typing import Optional

from .models import Chunk, FilingMeta, TableData

HEADER_REGEX = re.compile(
    r"^.+ \([A-Z.\-]+\) \| Form 10-[QK](/A)? \| (Fiscal Q[1-4] \d{4}|Fiscal Year \d{4}) "
    r"\(period ended \d{4}-\d{2}-\d{2}; calendar \d{4}-Q[1-4]\) \| .+$"
)
MAX_ROW_LABELS = 60


def build_header(f: FilingMeta, section_path: list[str]) -> str:
    return (f"{f.company_name} ({f.ticker}) | Form {f.form_type} | {f.period_label} "
            f"(period ended {f.period_end.isoformat()}; calendar {f.calendar_quarter}) | "
            f"{' > '.join(section_path)}")


def build_embed_text(f: FilingMeta, chunk: Chunk, table: Optional[TableData] = None) -> str:
    header = build_header(f, chunk.section_path)
    if chunk.content_type == "table_summary" and table is not None:
        # Describe the table instead of embedding raw cells: embeddings handle numbers poorly.
        # The full numbers stay in raw_text (for display) and in the table JSON (for a fact store).
        meta = "; ".join(x for x in [table.units or "", ", ".join(table.durations)] if x)
        labels = [r.full_label for r in table.rows if r.values][:MAX_ROW_LABELS]
        return "\n".join([
            header,
            f"Table: {table.title}" + (f" ({meta})" if meta else ""),
            f"Columns: {' | '.join(table.columns)}",
            f"Rows: {', '.join(labels)}",
        ])
    return f"{header}\n{chunk.raw_text}"
