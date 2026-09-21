-- Keyword index on the ingest table (filing_chunks) plus HITL tables if 001 was skipped.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS documents (
    document_id text PRIMARY KEY,
    source_uri text NOT NULL,
    sha256 text NOT NULL UNIQUE,
    mime_type text NOT NULL,
    form_type text,
    filing_period date,
    page_count integer CHECK (page_count > 0),
    ingest_status text NOT NULL DEFAULT 'pending'
        CHECK (ingest_status IN ('pending', 'ready', 'failed')),
    ingest_git_sha text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
    run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    question text NOT NULL,
    state text NOT NULL DEFAULT 'RECEIVED',
    draft jsonb,
    draft_sha256 text,
    answer text,
    reason_codes jsonb NOT NULL DEFAULT '[]'::jsonb,
    stop_reason text,
    usage jsonb NOT NULL DEFAULT '{"steps":0,"tokens":0,"latency_ms":0}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_steps (
    step_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step_number integer NOT NULL,
    state text NOT NULL,
    event_type text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id, step_number, event_type)
);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    from_state text NOT NULL,
    to_state text NOT NULL CHECK (to_state IN ('RELEASED', 'ABSTAINED')),
    step_id text NOT NULL,
    payload_sha256 text NOT NULL,
    approver_id text NOT NULL,
    unix_ts bigint NOT NULL,
    signature text NOT NULL,
    prev_audit_hash text,
    audit_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id, signature)
);

CREATE OR REPLACE FUNCTION guard_terminal_transition()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.state IN ('RELEASED', 'ABSTAINED') AND OLD.state <> NEW.state THEN
        IF OLD.state <> 'AWAITING_HUMAN' THEN
            RAISE EXCEPTION 'terminal transition must originate at AWAITING_HUMAN';
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM approvals a
            WHERE a.run_id = NEW.run_id
              AND a.from_state = OLD.state
              AND a.to_state = NEW.state
              AND a.payload_sha256 = NEW.draft_sha256
        ) THEN
            RAISE EXCEPTION 'verified approval row required';
        END IF;
    END IF;
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS runs_terminal_guard ON runs;
CREATE TRIGGER runs_terminal_guard
BEFORE UPDATE OF state ON runs
FOR EACH ROW EXECUTE FUNCTION guard_terminal_transition();

DO $$
BEGIN
    IF to_regclass('public.filing_chunks') IS NOT NULL THEN
        BEGIN
            ALTER TABLE filing_chunks ADD COLUMN search_vector tsvector
                GENERATED ALWAYS AS (to_tsvector('simple', coalesce(raw_text, ''))) STORED;
        EXCEPTION WHEN duplicate_column THEN
            NULL;
        END;
        CREATE INDEX IF NOT EXISTS filing_chunks_search_gin ON filing_chunks USING gin (search_vector);
    END IF;
END $$;
