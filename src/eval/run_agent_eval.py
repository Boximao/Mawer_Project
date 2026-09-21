"""End-to-end bounded-loop evaluation over the Git-controlled question set."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agent.loop import run_loop  # noqa: E402
from agent.tools import ToolDispatcher  # noqa: E402
from finembed.embedder import BGEEmbedder  # noqa: E402
from hitl.store import MemoryStore  # noqa: E402
from retrieve.hybrid import HybridRetriever  # noqa: E402


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    questions = [
        json.loads(line)
        for line in (ROOT / "eval" / "questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    embedder = BGEEmbedder("BAAI/bge-small-en-v1.5")
    retriever = HybridRetriever(dsn=None, embedder=embedder)
    tools = ToolDispatcher(retriever)
    metrics = {
        "questions": len(questions),
        "citation_hit": 0,
        "claim_support": 0,
        "numeric_exact": 0,
        "abstain_quality": 0,
        "draft_fact_hit": 0,
    }
    answerable = sum(not q.get("out_of_corpus") for q in questions)
    unanswerable = len(questions) - answerable

    print("git_commit_sha", git_sha())
    print("embedding", embedder.name, embedder.dimensions)
    print("rerank", "cloud-if-configured-else-rrf")
    for item in questions:
        env = run_loop(
            item["question"],
            tools=tools,
            store=MemoryStore(),
        )
        if item.get("out_of_corpus"):
            ok = (
                env["answer"] is None
                and env["draft_answer"] is None
                and env["human_review"] is True
                and "OUT_OF_CORPUS" in env["reason_codes"]
            )
            metrics["abstain_quality"] += int(ok)
            print(("  hit abstain" if ok else "  MISS abstain"), item["question"])
            continue

        gold = item["document_id"]
        citation_hit = any(
            row.get("document_id") == gold
            for row in env.get("trace", {}).get("kept", [])
        )
        claims = env.get("claims") or []
        support = bool(claims) and all(c.get("supported") for c in claims)
        numeric = "NUMERIC_MISMATCH" not in env.get("reason_codes", [])
        expected = str(item.get("expected") or "").lower().replace(",", "")
        evidence = json.dumps(
            {"claims": claims, "citations": env.get("citations") or []},
            ensure_ascii=False,
        ).lower().replace(",", "")
        fact_hit = bool(expected and expected in evidence)
        metrics["citation_hit"] += int(citation_hit)
        metrics["claim_support"] += int(support)
        metrics["numeric_exact"] += int(numeric)
        metrics["draft_fact_hit"] += int(fact_hit)
        print(
            ("  hit" if citation_hit and support else "  MISS"),
            item["question"],
            f"fact={'yes' if fact_hit else 'no'}",
        )

    result = {
        "questions": metrics["questions"],
        "citation_hit": round(metrics["citation_hit"] / answerable, 3),
        "claim_support": round(metrics["claim_support"] / answerable, 3),
        "numeric_exact": round(metrics["numeric_exact"] / answerable, 3),
        "draft_fact_hit": round(metrics["draft_fact_hit"] / answerable, 3),
        "abstain_quality": round(
            metrics["abstain_quality"] / max(unanswerable, 1), 3
        ),
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if all(value == 1.0 for key, value in result.items() if key != "questions") else 1


if __name__ == "__main__":
    raise SystemExit(main())
