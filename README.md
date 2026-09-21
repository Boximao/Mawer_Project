# fin-embed (MVP)

Reads SEC filing PDFs (10-Q / 10-K), splits them into section-aware chunks, embeds them with
OpenAI, and stores them in Postgres with pgvector, with the metadata needed to filter and cite.

## Demo

Put `OPENAI_API_KEY=...` in `.env`, then run `run_demo.py` (VS Code Run button or `python run_demo.py`).
It embeds all three filings and runs sample searches with page citations.

## Quickstart (Windows PowerShell)

The three Apple 10-Qs are already parsed in `data/processed/`, so generating embeddings only needs
an OpenAI key. PyMuPDF is not needed for this step.

```powershell
python -m pip install -r requirements.txt
copy .env.example .env                # put your OPENAI_API_KEY in .env (DATABASE_URL optional)
$env:PYTHONPATH = "src"
python -m finembed embed
```

Output per filing: `data/processed/<doc_id>/embeddings.npz` with `ids`, `hashes` and `vectors`
(one 1536-dim vector per chunk, rows aligned with `records.jsonl`). If `DATABASE_URL` is set, the
same rows are also written to pgvector (the table is created automatically). To start Postgres
with pgvector: `docker compose up -d`, then run `python -m finembed embed` again. Unchanged chunks
reuse their saved vectors, so this makes no new OpenAI calls.

New PDFs need parsing first (this step needs a working PyMuPDF):

```powershell
python -m finembed ingest --dry-run   # parse + chunk into data/processed/
python -m finembed inspect --doc-id AAPL_10-Q_2026-06-27 --n 5
python -m finembed embed
```

## Adding filings

Put the PDF under `data/raw/` and add a row to `data/manifest.csv`. The manifest is the source of
truth for filing metadata (the parser never guesses it). EDGAR PDFs carry the accession number in
the PDF title (`pdfinfo filing.pdf`).

`ticker, cik, company_name, form_type, accession_no, filing_date, period_end, fiscal_year, fiscal_quarter, source_url, pdf_path, amends`

## How it works

1. **Read PDF** (`parse.py`, `tables.py`): PyMuPDF gives every line with position and font. Bold,
   italic and bold-italic lines become a section stack (Part > Item > statement or Note > subheading).
   Rows of positioned numbers become tables, rebuilt into columns with proper headers
   ("Three Months Ended June 27, 2026"). Cover page, table of contents, signatures and exhibit
   certifications are skipped.
2. **Chunk** (`chunk.py`): 300 to 800 tokens on paragraph boundaries, never across an Item, tiny
   sibling sections merged (e.g. the five geographic segment paragraphs become one chunk), each table
   is its own chunk. Every chunk is an exact span of `clean.txt`, so `raw_text` is citable.
3. **Embed text** (`embed_text.py`): each chunk starts with the same header format:
   `Apple Inc. (AAPL) | Form 10-Q | Fiscal Q3 2026 (period ended 2026-06-27; calendar 2026-Q2) | Part I > Item 2. ... > Macroeconomic Conditions`
   Tables are embedded as a description (title, units, periods, columns, row labels), not raw
   numbers. The numbers stay in `raw_text` and in `data/processed/<doc_id>/tables/*.json`.
4. **Store** (`store.py`): one row per chunk in `filing_chunks`, HNSW cosine index plus a filter index
   on ticker, form, fiscal year and quarter. Re-ingesting a filing replaces its rows atomically and
   reuses stored vectors for unchanged chunks, so a re-run makes zero OpenAI calls.

`dimensions: 1536` in `config.yaml` is deliberate: pgvector's HNSW index supports at most 2000
dimensions, and `text-embedding-3-large` shortened to 1536 keeps most of its quality.

## Querying

```sql
SELECT id, page_label, section_path, left(raw_text, 200)
FROM filing_chunks
WHERE ticker = 'AAPL' AND fiscal_year = 2026 AND fiscal_quarter = 3
ORDER BY embedding <=> %(query_vector)s
LIMIT 5;
```

Embed the query with the same model and dimensions. Cite with `accession_no`, `page_label`
(printed page) or `pdf_page_start`, and `char_start`/`char_end` into `clean.txt`.

## Tests

```bash
pytest                                                   # parsing and chunking checks
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/finrag pytest   # plus pgvector round trip
```

The pgvector test uses a fake embedder, so it needs no OpenAI key.

## Checking quality

1. Parsing and chunking (free): `ingest --dry-run`, then `inspect`, and compare a few table JSONs in
   `data/processed/<doc_id>/tables/` against the PDF page (`page_label` is the printed page number).
2. Retrieval (after `ingest`): `python scripts/eval_retrieval.py` scores the questions in
   `eval/questions.csv` (hit@5 and MRR) and prints the top results for every miss.
   Add `--no-filter` to test whether the right quarter wins without metadata filters.
