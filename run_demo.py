"""One-click demo: embed the Apple 10-Qs with OpenAI, store them, and run a live search.

Run it with the VS Code Run button, or:
    python run_demo.py
    python run_demo.py "your own question here"

Needs OPENAI_API_KEY in a .env file next to this script. DATABASE_URL in .env is optional
(if set, the embeddings are also written to pgvector).
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from finembed.config import load_settings  # noqa: E402
from finembed.pipeline import embed_records, load_records, processed_doc_ids  # noqa: E402

DEFAULT_QUESTIONS = [
    "Why did Apple's products gross margin increase in fiscal Q3 2026?",
    "Is Apple exposed to rising memory chip (DRAM) costs?",
    "What was iPhone revenue in the quarter ended June 27, 2026?",
]


def search(question, embedder, s, doc_ids, k=3):
    """Cosine similarity over the saved vectors (no database needed)."""
    q = np.asarray(embedder.embed([question])[0], dtype=np.float32)
    hits = []
    for d in doc_ids:
        z = np.load(s.processed_dir / d / "embeddings.npz")
        records = {r.id: r for r in load_records(s, d)}
        V = z["vectors"]
        sims = V @ q / (np.linalg.norm(V, axis=1) * np.linalg.norm(q) + 1e-9)
        hits += [(float(sim), records[i]) for i, sim in zip(z["ids"].tolist(), sims)]
    return sorted(hits, key=lambda h: -h[0])[:k]


def main(embedder=None):
    s = load_settings(ROOT / "config.yaml")
    doc_ids = processed_doc_ids(s)
    if not doc_ids:
        sys.exit("No parsed filings found in data/processed/.")

    if embedder is None:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit(f"OPENAI_API_KEY not found. Add it to {ROOT / '.env'} and run again.")
        from finembed.embedder import OpenAIEmbedder
        embedder = OpenAIEmbedder(s.embedding_model, s.embedding_dimensions, s.embedding_batch_size)

    store = None
    if s.database_url:
        try:
            from finembed.store import PgVectorStore
            store = PgVectorStore(s.database_url, s.table_name, s.embedding_dimensions or 3072)
            store.init_schema()
        except Exception as e:
            print(f"(pgvector not reachable: {type(e).__name__}. Continuing with local vectors only.)")
            store = None

    print("=" * 90)
    print(f"STEP 1  Embedding {len(doc_ids)} filings with {embedder.name}")
    print("=" * 90)
    total = 0
    for d in doc_ids:
        r = embed_records(d, s, embedder, store)
        total += r["chunks"]
        where = " + pgvector" if store else ""
        print(f"  {d:24}  {r['chunks']:3} chunks  ({r['embedded']} new, {r['reused']} reused)  saved to .npz{where}")
    dim = np.load(s.processed_dir / doc_ids[0] / "embeddings.npz")["vectors"].shape[1]
    print(f"  Done: {total} chunks embedded as {dim}-dimensional vectors.")

    print()
    print("=" * 90)
    print("STEP 2  Semantic search with citations")
    print("=" * 90)
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEFAULT_QUESTIONS
    for question in questions:
        print(f"\nQ: {question}")
        for rank, (score, r) in enumerate(search(question, embedder, s, doc_ids), start=1):
            section = " > ".join(r.section_path[2:]) or r.section_path[-1]
            period = f"FQ{r.fiscal_quarter} {r.fiscal_year}" if r.fiscal_quarter else f"FY{r.fiscal_year}"
            snippet = " ".join(r.raw_text.split())[:220]
            print(f"  {rank}. [{score:.3f}] {r.ticker} {r.form_type} {period}, p.{r.page_label} | {section}")
            print(f"     \"{snippet}...\"")
    print()


if __name__ == "__main__":
    main()
