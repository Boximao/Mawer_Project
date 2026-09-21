"""Git-controlled allowlist. Query path never fetches URLs."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data" / "corpus_manifest.json"

# Named issuers that are not Apple. Overlapping words like "revenue" must not
# let Tesla questions ride Apple chunks.
_FOREIGN = (
    "tesla", "tsla", "microsoft", "msft", "amazon", "amzn", "google", "alphabet",
    "googl", "nvidia", "nvda", "meta", "netflix", "nflx", "berkshire",
    "ford", "general motors", "gm", "ibm", "oracle", "salesforce", "adobe",
    "intel", "amd", "qualcomm", "samsung", "sony", "spotify", "uber", "airbnb",
    "walmart", "costco", "jpmorgan", "goldman sachs", "coca-cola", "pepsico",
)
_APPLE = ("apple", "aapl")


@dataclass(frozen=True)
class ManifestDoc:
    document_id: str
    source_uri: str
    sha256: str
    mime_type: str
    form_type: str | None
    filing_period: str | None
    page_count: int | None
    ingest_status: str
    embedding_model: str | None = None
    embedding_dim: int | None = None
    chunk_count: int | None = None


def load_manifest(path: Path | None = None) -> list[ManifestDoc]:
    p = path or DEFAULT_MANIFEST
    raw = json.loads(p.read_text(encoding="utf-8"))
    docs = []
    for row in raw.get("documents", []):
        docs.append(ManifestDoc(
            document_id=row["document_id"],
            source_uri=row["source_uri"],
            sha256=row["sha256"],
            mime_type=row.get("mime_type", "application/pdf"),
            form_type=row.get("form_type"),
            filing_period=row.get("filing_period"),
            page_count=row.get("page_count"),
            ingest_status=row.get("ingest_status", "pending"),
            embedding_model=row.get("embedding_model"),
            embedding_dim=row.get("embedding_dim"),
            chunk_count=row.get("chunk_count"),
        ))
    return docs


def ready_document_ids(path: Path | None = None) -> frozenset[str]:
    return frozenset(d.document_id for d in load_manifest(path) if d.ingest_status == "ready")


def validate_processed_corpus(
    processed_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> list[str]:
    """Return integrity errors between the allowlist and Max's processed corpus."""
    import numpy as np

    base = processed_dir or (ROOT / "data" / "processed")
    errors: list[str] = []
    for doc in load_manifest(manifest_path):
        if doc.ingest_status != "ready":
            continue
        folder = base / doc.document_id
        records_path = folder / "records.jsonl"
        vectors_path = folder / (
            "embeddings.npz"
            if not doc.embedding_model or doc.embedding_model.startswith("openai:")
            else "embeddings_bge.npz"
        )
        if not records_path.exists():
            errors.append(f"{doc.document_id}: records.jsonl missing")
            continue
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if doc.chunk_count is not None and len(records) != doc.chunk_count:
            errors.append(
                f"{doc.document_id}: manifest chunks={doc.chunk_count}, processed={len(records)}"
            )
        models = {r.get("embedding_model") for r in records}
        # records.jsonl is Max's immutable parsed source. A later local BGE pass
        # writes a separate NPZ and intentionally does not rewrite those rows.
        if (
            doc.embedding_model
            and doc.embedding_model.startswith("openai:")
            and models != {doc.embedding_model}
        ):
            errors.append(
                f"{doc.document_id}: manifest model={doc.embedding_model}, processed={sorted(models)}"
            )
        if not vectors_path.exists():
            errors.append(f"{doc.document_id}: embeddings.npz missing")
            continue
        with np.load(vectors_path, allow_pickle=False) as vectors:
            shape = vectors["vectors"].shape
        if len(shape) != 2 or shape[0] != len(records):
            errors.append(
                f"{doc.document_id}: vector rows={shape[0] if shape else 0}, records={len(records)}"
            )
        if doc.embedding_dim is not None and (len(shape) != 2 or shape[1] != doc.embedding_dim):
            errors.append(
                f"{doc.document_id}: manifest dim={doc.embedding_dim}, processed={shape[1] if len(shape) == 2 else 'invalid'}"
            )
    return errors


def out_of_corpus_issuer(question: str) -> bool:
    """True when the question names a non-Apple issuer (workshop Tesla path)."""
    q = question.lower()
    if any(re.search(rf"\b{re.escape(t)}\b", q) for t in _FOREIGN):
        if not any(re.search(rf"\b{re.escape(t)}\b", q) for t in _APPLE):
            return True
    return False
