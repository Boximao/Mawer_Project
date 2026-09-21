"""Apply migrations/001_initial.sql to Neon using DATABASE_URL_UNPOOLED.

Does not print connection strings or table contents.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = ROOT / "migrations" / "001_initial.sql"


def load_env(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v)


def main() -> int:
    load_env(ROOT / ".env")
    dsn = os.environ.get("DATABASE_URL_UNPOOLED") or os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL_UNPOOLED / DATABASE_URL missing from .env", file=sys.stderr)
        return 1

    import psycopg

    sql = SQL_PATH.read_text(encoding="utf-8")
    with psycopg.connect(dsn, connect_timeout=30) as conn:
        conn.execute("SELECT 1")
        conn.execute(sql)
        conn.commit()
        ext = conn.execute(
            "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pgcrypto') ORDER BY 1"
        ).fetchall()
        tables = conn.execute(
            """
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
              AND tablename IN ('documents', 'chunks', 'runs', 'agent_steps', 'approvals')
            ORDER BY 1
            """
        ).fetchall()
        dim = conn.execute(
            """
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
            """
        ).fetchone()

    print("wake: SELECT 1 ok")
    print("extensions:", ", ".join(r[0] for r in ext) or "(none)")
    print("tables:", ", ".join(r[0] for r in tables) or "(none)")
    print("chunks.embedding typmod:", dim[0] if dim else "missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
