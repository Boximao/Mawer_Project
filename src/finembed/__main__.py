"""CLI.

  python -m finembed embed [--doc-id ID]              embed already-parsed filings (no PyMuPDF needed)
  python -m finembed ingest [--doc-id ID] [--dry-run] parse PDFs, then embed (--dry-run: parse only)
  python -m finembed init-db                          create the pgvector table and indexes
  python -m finembed inspect --doc-id ID [--n 5]      print sample records (embed_text + metadata)

Vectors are always saved to data/processed/<doc_id>/embeddings.npz.
If DATABASE_URL is set they are also written to pgvector.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys

from .config import load_manifest, load_settings


def _embedder(s):
    from .embedder import OpenAIEmbedder

    return OpenAIEmbedder(s.embedding_model, s.embedding_dimensions, s.embedding_batch_size)


def _store(s, required: bool = False):
    if not s.database_url:
        if required:
            sys.exit("DATABASE_URL is not set (put it in .env)")
        print("DATABASE_URL not set: saving vectors to embeddings.npz only (pgvector skipped)", file=sys.stderr)
        return None
    from .store import PgVectorStore

    store = PgVectorStore(s.database_url, s.table_name, s.embedding_dimensions or 3072)
    store.init_schema()  # CREATE ... IF NOT EXISTS, safe to repeat
    return store


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="finembed")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    emb = sub.add_parser("embed")
    emb.add_argument("--doc-id")
    ing = sub.add_parser("ingest")
    ing.add_argument("--doc-id", help="only this filing, e.g. AAPL_10-Q_2026-06-27")
    ing.add_argument("--dry-run", action="store_true", help="parse + chunk only; no OpenAI or database calls")
    ins = sub.add_parser("inspect")
    ins.add_argument("--doc-id", required=True)
    ins.add_argument("--n", type=int, default=5)
    ins.add_argument("--type", choices=["narrative", "table_summary", "footnote"])
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    s = load_settings(args.config)

    if args.cmd == "init-db":
        store = _store(s, required=True)
        print(f"ready: table {s.table_name} (vector({store.dim}))")
        return 0

    if args.cmd == "embed":
        from .pipeline import embed_records, processed_doc_ids

        doc_ids = [args.doc_id] if args.doc_id else processed_doc_ids(s)
        if not doc_ids:
            sys.exit(f"no parsed filings in {s.processed_dir}; run `ingest --dry-run` first")
        embedder, store = _embedder(s), _store(s)
        for d in doc_ids:
            print(json.dumps(embed_records(d, s, embedder, store)))
        return 0

    if args.cmd == "ingest":
        from .pipeline import ingest_filing

        filings = load_manifest(s.manifest, s.root)
        if args.doc_id:
            filings = [f for f in filings if f.doc_id == args.doc_id]
            if not filings:
                sys.exit(f"no manifest row for {args.doc_id}")
        embedder, store = (None, None) if args.dry_run else (_embedder(s), _store(s))
        for f in filings:
            print(json.dumps(ingest_filing(f, s, embedder, store, dry_run=args.dry_run)))
        return 0

    if args.cmd == "inspect":
        path = s.processed_dir / args.doc_id / "records.jsonl"
        if not path.exists():
            sys.exit(f"{path} not found; run ingest (or ingest --dry-run) first")
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        if args.type:
            rows = [r for r in rows if r["content_type"] == args.type]
        random.seed(0)
        for r in random.sample(rows, min(args.n, len(rows))):
            print("=" * 100)
            print(r["embed_text"][:1500])
            meta = {k: r[k] for k in ("id", "content_type", "item_code", "page_label", "pdf_page_start",
                                      "char_start", "char_end", "n_tokens", "table_id", "units", "durations")}
            print("-" * 100)
            print(json.dumps(meta, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
