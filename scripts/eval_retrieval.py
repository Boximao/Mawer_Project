"""Retrieval check: for known questions, does a correct chunk show up in the top k?

A result counts as correct if its section path or text contains the `expected` string
(and, with --no-filter, it also comes from the right filing).

  python scripts/eval_retrieval.py                 # search within the right filing (metadata filter)
  python scripts/eval_retrieval.py --no-filter     # search all filings: tests picking the right quarter
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finembed.config import load_settings  # noqa: E402
from finembed.store import PgVectorStore  # noqa: E402


def search(store: PgVectorStore, vec, k: int, q: dict | None):
    sql = f"SELECT id, section_path, page_label, raw_text, ticker, fiscal_year, fiscal_quarter FROM {store.table}"
    params: list = []
    if q is not None:
        sql += " WHERE ticker = %s AND fiscal_year = %s AND fiscal_quarter = %s"
        params += [q["ticker"], int(q["fiscal_year"]), int(q["fiscal_quarter"])]
    sql += " ORDER BY embedding <=> %s LIMIT %s"
    params += [np.asarray(vec, dtype=np.float32), k]
    with store.connect() as conn:
        return conn.execute(sql, params).fetchall()


def is_hit(row, q: dict) -> bool:
    _, path, _, text, ticker, fy, fq = row
    right_filing = (ticker, fy, fq) == (q["ticker"], int(q["fiscal_year"]), int(q["fiscal_quarter"]))
    return right_filing and q["expected"].lower() in (" > ".join(path) + "\n" + text).lower()


def evaluate(questions: list[dict], embed, store: PgVectorStore, k: int, use_filter: bool) -> dict:
    vectors = embed([q["question"] for q in questions])
    hits, rr = 0, 0.0
    for q, v in zip(questions, vectors):
        rows = search(store, v, k, q if use_filter else None)
        rank = next((i + 1 for i, r in enumerate(rows) if is_hit(r, q)), None)
        if rank:
            hits += 1
            rr += 1 / rank
            print(f"  hit @{rank}  {q['question']}")
        else:
            print(f"  MISS     {q['question']}  (expected: {q['expected']})")
            for r in rows[:3]:
                print(f"             got FQ{r[6]} p.{r[2]}  {' > '.join(r[1][2:])[:90]}")
    n = len(questions)
    return {"questions": n, f"hit@{k}": round(hits / n, 3), "mrr": round(rr / n, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.csv"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-filter", action="store_true", help="search across all filings")
    args = ap.parse_args()

    from finembed.embedder import OpenAIEmbedder

    s = load_settings(ROOT / "config.yaml")
    store = PgVectorStore(s.database_url, s.table_name, s.embedding_dimensions or 3072)
    embedder = OpenAIEmbedder(s.embedding_model, s.embedding_dimensions, s.embedding_batch_size)
    questions = list(csv.DictReader(open(args.questions, encoding="utf-8")))
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        sha = "unknown"
    print("git_commit_sha", sha)
    print(evaluate(questions, embedder.embed, store, args.k, use_filter=not args.no_filter))


if __name__ == "__main__":
    main()
