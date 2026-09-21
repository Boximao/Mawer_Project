"""Answer questions about the indexed filings: pgvector retrieval + an OpenAI answer.

  python scripts/ask.py                              run every test in scripts_to_test.txt
  python scripts/ask.py --test 4                     run one test
  python scripts/ask.py -q "How did iPhone revenue move?"
  python scripts/ask.py --test 2 --show-context      also print the chunks sent to the model

Retrieval takes the top matches overall plus the best few from each filing, so
questions that compare quarters see every quarter. The model is told which
filings exist and must answer only from the chunks it is given.
"""
from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finembed.config import load_settings  # noqa: E402
from finembed.store import PgVectorStore  # noqa: E402

SYSTEM = """You answer questions about SEC filings using only the excerpts provided.

Rules:
- Use only the excerpts. Never use outside knowledge, memory of these companies, or arithmetic on
  figures that are not in the excerpts.
- Every number and quoted phrase must appear verbatim in an excerpt.
- Cite the excerpts you used by their [n] markers, inline, right after the claim they support.
- The excerpts come from the filings listed under COVERAGE. If the question asks about a period or
  topic that COVERAGE does not include, say plainly that the indexed filings do not cover it and
  stop. Do not estimate, extrapolate, or substitute a different period's figure.
- A year-to-date or nine-month figure is not a quarterly figure. Do not present one as the other.
- Completeness matters as much as accuracy. When an excerpt gives a prior-period comparative, a
  percentage, or a percentage change alongside a figure you report, include it.
- When the excerpts state a year-over-year direction, say so explicitly, even if the question is
  about a sequential trend. Do not let a sequential decline imply the figure fell year over year.
- Be direct: lead with the answer, then the supporting detail. A few sentences is usually enough.
"""

CONTEXT_BLOCK = """[{n}] {form} for the quarter ended {period_end} (FY{fy} Q{fq}{acc})
    section: {section} | page {page} | {ctype}
{text}
"""


def parse_tests(path: Path) -> list[dict]:
    """Pull only the `Question:` lines out of scripts_to_test.txt.

    The Answer:/Source: lines in that file are the grader's key. They are never read, so they
    cannot reach the model or this script's output.
    """
    tests = []
    for block in re.split(r"\n(?=Test \d)", path.read_text(encoding="utf-8").strip()):
        label = re.match(r"Test [^\n]*", block)
        m = re.search(r"^Question:\s*(.*?)(?=\n(?:Answer|Source):|\Z)", block, re.S | re.M)
        if m:
            tests.append({"label": label.group(0) if label else "Test", "question": m.group(1).strip()})
    return tests


def coverage(store: PgVectorStore) -> list[tuple]:
    with store.connect() as conn:
        return conn.execute(
            f"SELECT DISTINCT company_name, form_type, period_end, fiscal_year, fiscal_quarter "
            f"FROM {store.table} ORDER BY period_end").fetchall()


def retrieve(store: PgVectorStore, vec, k: int, per_doc: int) -> list[dict]:
    """Top k overall, but at most `per_doc` from any one filing keeps comparisons balanced."""
    v = np.asarray(vec, dtype=np.float32)
    sql = f"""
        WITH ranked AS (
            SELECT id, doc_id, form_type, period_end, fiscal_year, fiscal_quarter, accession_no,
                   section_path, page_label, pdf_page_start, content_type, raw_text,
                   embedding <=> %s AS dist,
                   row_number() OVER (PARTITION BY doc_id ORDER BY embedding <=> %s) AS doc_rank
            FROM {store.table}
        )
        SELECT * FROM ranked WHERE doc_rank <= %s ORDER BY dist LIMIT %s
    """
    with store.connect() as conn:
        return conn.cursor(row_factory=dict_row).execute(sql, (v, v, per_doc, k)).fetchall()


def section_of(h: dict) -> str:
    """Drop the company/form prefix of the path; fall back to the whole path if that empties it."""
    return " > ".join(h["section_path"][2:]) or " > ".join(h["section_path"])


def page_of(h: dict) -> str:
    return h["page_label"] or str(h["pdf_page_start"])


def format_context(hits: list[dict]) -> str:
    return "\n".join(
        CONTEXT_BLOCK.format(
            n=n, form=h["form_type"], period_end=h["period_end"], fy=h["fiscal_year"],
            fq=h["fiscal_quarter"], acc=f", accession {h['accession_no']}" if h["accession_no"] else "",
            section=section_of(h), page=page_of(h), ctype=h["content_type"], text=h["raw_text"].strip())
        for n, h in enumerate(hits, 1))


def build_prompt(question: str, hits: list[dict], filings: list[tuple]) -> str:
    """The entire user message. Its only sources are the question and chunks read from pgvector."""
    cov = "\n".join(f"- {c} {f} for the quarter ended {pe} (FY{fy} Q{fq})" for c, f, pe, fy, fq in filings)
    return (f"COVERAGE — the only filings indexed:\n{cov}\n\n"
            f"EXCERPTS:\n{format_context(hits)}\n\nQUESTION: {question}")


def answer(client, model: str, question: str, hits: list[dict], filings: list[tuple]) -> str:
    user = build_prompt(question, hits, filings)
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    return resp.choices[0].message.content.strip()


def cite(h: dict) -> str:
    return (f"{h['form_type']} q/e {h['period_end']} (FY{h['fiscal_year']} Q{h['fiscal_quarter']}) "
            f"p.{page_of(h)} | {section_of(h)} | sim {1 - h['dist']:.3f}")


def run(question: str, store, embedder, client, args, filings) -> None:
    print("=" * 100)
    print("Q:", question, "\n")
    hits = retrieve(store, embedder.embed([question])[0], args.k, args.per_doc)
    if args.show_context:
        print(format_context(hits), "\n" + "-" * 100)
    text = answer(client, args.model, question, hits, filings)
    print(text if args.raw else "\n".join(textwrap.fill(p, 100) for p in text.split("\n")))
    print("\nRetrieved:")
    for n, h in enumerate(hits, 1):
        print(f"  [{n}] {cite(h)}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question", help="ask one question instead of running the test file")
    ap.add_argument("--file", default=str(ROOT / "scripts_to_test.txt"))
    ap.add_argument("--test", type=int, action="append", help="run only this test number (repeatable)")
    ap.add_argument("--model", default="gpt-4.1", help="OpenAI chat model for the answer")
    ap.add_argument("--k", type=int, default=12, help="chunks sent to the model")
    ap.add_argument("--per-doc", type=int, default=4, help="max chunks from any one filing")
    ap.add_argument("--show-context", action="store_true")
    ap.add_argument("--raw", action="store_true", help="do not wrap the answer to 100 columns")
    args = ap.parse_args()

    from openai import OpenAI

    from finembed.embedder import OpenAIEmbedder

    s = load_settings(ROOT / "config.yaml")
    if not s.database_url:
        sys.exit("DATABASE_URL is not set (put it in .env)")
    store = PgVectorStore(s.database_url, s.table_name, s.embedding_dimensions or 3072)
    filings = coverage(store)
    if not filings:
        sys.exit(f"{s.table_name} is empty; run `python -m finembed embed` first")
    embedder = OpenAIEmbedder(s.embedding_model, s.embedding_dimensions, s.embedding_batch_size)
    client = OpenAI()

    if args.question:
        run(args.question, store, embedder, client, args, filings)
        return
    tests = parse_tests(Path(args.file))
    for i, t in enumerate(tests, 1):
        if args.test and i not in args.test:
            continue
        print(f"\n### {t['label']}")
        run(t["question"], store, embedder, client, args, filings)


if __name__ == "__main__":
    main()
