from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from agent.loop import require_not_released_by_model, run_loop
from agent.tools import ToolDispatcher
from finembed.config import load_settings
from hitl.approve import HitlError, approve
from hitl.store import MemoryStore, PgStore
from retrieve.allowlist import load_manifest, validate_processed_corpus
from retrieve.hybrid import HybridRetriever

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger(__name__)


def _prepare_schema(dsn: str, settings) -> bool:
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=15) as conn:
            for path in sorted((ROOT / "migrations").glob("*.sql")):
                conn.execute(path.read_text(encoding="utf-8"))
            conn.commit()
        return True
    except Exception as exc:
        log.warning("schema init skipped (%s); local retrieve still available", type(exc).__name__)
        return False


class QuestionIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class ApproveIn(BaseModel):
    run_id: str
    pin: str = Field(min_length=1, max_length=64)
    to_state: Literal["RELEASED", "ABSTAINED"]
    approver_id: str = Field(default="presenter", min_length=1, max_length=128)


class _LazyBGE:
    def __init__(self, model: str):
        self.model = model
        self._client = None

    def embed(self, texts):
        if self._client is None:
            from finembed.embedder import BGEEmbedder

            self._client = BGEEmbedder(self.model)
        return self._client.embed(texts)


def _embedder(settings):
    models = {
        d.embedding_model
        for d in load_manifest()
        if d.ingest_status == "ready" and d.embedding_model
    }
    if len(models) != 1:
        return None
    model = next(iter(models))
    if not model.startswith("openai:"):
        return _LazyBGE(model)
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    from finembed.embedder import OpenAIEmbedder

    return OpenAIEmbedder(settings.embedding_model, settings.embedding_dimensions, settings.embedding_batch_size)


def _ensure_hitl_key() -> None:
    if os.environ.get("HITL_SIGNING_KEY", "").strip():
        return
    local = ROOT / ".env.local"
    if local.exists():
        for line in local.read_text(encoding="utf-8").splitlines():
            if not line.startswith("HITL_SIGNING_KEY="):
                continue
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                os.environ["HITL_SIGNING_KEY"] = value
                return
    key = secrets.token_urlsafe(32)
    os.environ["HITL_SIGNING_KEY"] = key
    with local.open("a", encoding="utf-8") as fh:
        fh.write(f"\nHITL_SIGNING_KEY={key}\n")
    log.warning("wrote HITL_SIGNING_KEY to .env.local for this workshop machine")


def build_app() -> FastAPI:
    os.environ.setdefault("HITL_PIN", "2026")
    settings = load_settings(ROOT / "config.yaml")
    dsn = settings.database_url
    _ensure_hitl_key()
    db_ready = bool(dsn and _prepare_schema(dsn, settings))
    table = os.environ.get("AGENT_TABLE", "chunks" if db_ready else settings.table_name)
    store = PgStore(dsn) if db_ready else MemoryStore()
    retriever = HybridRetriever(
        dsn=dsn if db_ready else None,
        table=table,
        processed_dir=settings.processed_dir,
        embedder=_embedder(settings),
    )
    tools = ToolDispatcher(retriever)

    app = FastAPI(title="Mawer agentic RAG", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.store = store
    app.state.tools = tools

    @app.get("/health")
    def health():
        corpus_errors = validate_processed_corpus(settings.processed_dir)
        return {
            "ok": not corpus_errors,
            "mode": "live",
            "store": "postgres" if db_ready else "memory",
            "database_fallback": bool(dsn and not db_ready),
            "table": table,
            "embedder": bool(retriever.embedder),
            "corpus_errors": corpus_errors,
        }

    @app.get("/corpus")
    def corpus():
        return {"documents": [d.__dict__ for d in load_manifest() if d.ingest_status == "ready"]}

    @app.post("/runs")
    def create_run(body: QuestionIn):
        env = run_loop(body.question, tools=app.state.tools, store=app.state.store)
        return env

    @app.get("/runs/{run_id}")
    def get_run(run_id: str):
        row = app.state.store.get(run_id)
        if not row:
            raise HTTPException(404, "run not found")
        return require_not_released_by_model(row)

    @app.post("/hitl/approve")
    def hitl_approve(body: ApproveIn):
        row = app.state.store.get(body.run_id)
        if not row:
            raise HTTPException(404, "run not found")
        try:
            row, canon, sig = approve(
                app.state.store,
                row,
                to_state=body.to_state,
                pin=body.pin,
                approver_id=body.approver_id,
            )
        except HitlError as e:
            raise HTTPException(400, str(e)) from e
        env = require_not_released_by_model(row)
        env["canon"] = canon
        env["signature"] = sig
        return env

    demo = ROOT / "demo"
    if demo.is_dir():
        app.mount("/", StaticFiles(directory=str(demo), html=True), name="demo")
    return app


app = build_app()
