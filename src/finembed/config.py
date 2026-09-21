"""Settings (config.yaml + .env) and manifest loading."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from .models import FilingMeta


@dataclass
class ChunkingConfig:
    min_tokens: int = 100
    target_tokens: int = 500
    max_tokens: int = 800
    overlap_tokens: int = 50
    trivial_item_tokens: int = 15

REQUIRED_COLUMNS = ["ticker", "cik", "company_name", "form_type", "filing_date", "period_end",
                    "fiscal_year", "fiscal_quarter", "pdf_path"]


@dataclass
class Settings:
    root: Path
    pipeline_version: str
    manifest: Path
    processed_dir: Path
    database_url: Optional[str]
    table_name: str
    chunking: ChunkingConfig
    embedding_model: str
    embedding_dimensions: Optional[int]
    embedding_batch_size: int


def load_settings(config_path: str | Path = "config.yaml") -> Settings:
    config_path = Path(config_path).resolve()
    root = config_path.parent
    load_dotenv(root / ".env")
    raw = yaml.safe_load(config_path.read_text()) or {}
    p, db, emb = raw.get("paths", {}), raw.get("database", {}), raw.get("embedding", {})
    return Settings(
        root=root,
        pipeline_version=str(raw.get("pipeline_version", "0.1.0")),
        manifest=root / p.get("manifest", "data/manifest.csv"),
        processed_dir=root / p.get("processed_dir", "data/processed"),
        database_url=os.environ.get(db.get("url_env", "DATABASE_URL")),
        table_name=db.get("table", "filing_chunks"),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        embedding_model=emb.get("model", "text-embedding-3-large"),
        embedding_dimensions=emb.get("dimensions", 1536),
        embedding_batch_size=emb.get("batch_size", 64),
    )


class ManifestError(ValueError):
    pass


def load_manifest(path: str | Path, root: Path) -> list[FilingMeta]:
    """Validate every row and fail with one message listing all bad rows."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(f"{path}: missing columns {missing}")
        rows = list(reader)
    filings, errors, seen = [], [], set()
    for i, row in enumerate(rows, start=2):
        clean = {k: (v.strip() if isinstance(v, str) and v.strip() else None) for k, v in row.items() if k}
        try:
            f = FilingMeta(**clean)
        except ValidationError as e:
            errors.append(f"line {i}: " + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()))
            continue
        if not f.is_annual and f.fiscal_quarter is None:
            errors.append(f"line {i}: fiscal_quarter is required for {f.form_type}")
        pdf = Path(f.pdf_path) if Path(f.pdf_path).is_absolute() else root / f.pdf_path
        if not pdf.exists():
            errors.append(f"line {i}: pdf_path not found: {f.pdf_path}")
        f.pdf_path = str(pdf)
        if f.doc_id in seen:
            errors.append(f"line {i}: duplicate filing {f.doc_id}")
        seen.add(f.doc_id)
        filings.append(f)
    if errors:
        raise ManifestError(f"{path}: {len(errors)} bad row(s)\n  " + "\n  ".join(errors))
    return filings
