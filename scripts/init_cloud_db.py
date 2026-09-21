"""Apply migrations/*.sql to Neon using DATABASE_URL_UNPOOLED (or DATABASE_URL)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    load_env(ROOT / ".env")
    load_env(ROOT / ".env.local")
    dsn = os.environ.get("DATABASE_URL_UNPOOLED") or os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL_UNPOOLED / DATABASE_URL missing from .env", file=sys.stderr)
        return 1

    import psycopg

    files = sorted((ROOT / "migrations").glob("*.sql"))
    try:
        conn = psycopg.connect(dsn, connect_timeout=30)
    except Exception as exc:
        print(f"connect failed: {type(exc).__name__} (DSN not printed)", file=sys.stderr)
        return 1
    with conn:
        conn.execute("SELECT 1")
        for path in files:
            conn.execute(path.read_text(encoding="utf-8"))
        conn.commit()
        ext = conn.execute(
            "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pgcrypto') ORDER BY 1"
        ).fetchall()
        tables = conn.execute(
            """
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
              AND tablename IN ('documents', 'chunks', 'filing_chunks', 'runs', 'agent_steps', 'approvals')
            ORDER BY 1
            """
        ).fetchall()

    print("wake: SELECT 1 ok")
    print("migrations:", ", ".join(p.name for p in files))
    print("extensions:", ", ".join(r[0] for r in ext) or "(none)")
    print("tables:", ", ".join(r[0] for r in tables) or "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
