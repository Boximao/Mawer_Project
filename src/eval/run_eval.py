"""Citation-hit / abstain eval over the Git-controlled JSONL (no chat required)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from retrieve.allowlist import out_of_corpus_issuer  # noqa: E402
from retrieve.hybrid import HybridRetriever  # noqa: E402


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    path = ROOT / "eval" / "questions.jsonl"
    retriever = HybridRetriever(dsn=None, embedder=None)
    hits = 0
    n = 0
    print("git_commit_sha", git_sha())
    print("retrieve", "local_hybrid", "top_k_context=6")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        n += 1
        res = retriever.search(q["question"])
        if q.get("out_of_corpus"):
            ok = res.mode == "out_of_corpus" or not res.hits
            ok = ok and out_of_corpus_issuer(q["question"])
            print(("  hit abstain" if ok else "  MISS abstain"), q["question"])
            hits += int(ok)
            continue
        gold = q.get("document_id")
        expected = (q.get("expected") or "").lower().replace(",", "")
        docs = [h.document_id for h in res.hits]
        found = gold in docs
        numeric = False
        if expected:
            numeric = any(
                expected in (h.text.lower() + h.section.lower()).replace(",", "")
                for h in res.hits if h.document_id == gold
            )
        print(("  hit" if found else "  MISS"), q["question"], "numeric" if numeric else "")
        hits += int(found)
    print({"questions": n, "citation_hit": round(hits / n, 3) if n else 0})
    return 0 if n and hits == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
