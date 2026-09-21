ALTER TABLE runs ADD COLUMN IF NOT EXISTS trace jsonb NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS visited jsonb NOT NULL DEFAULT '["RECEIVED"]'::jsonb;

CREATE OR REPLACE FUNCTION guard_run_transition()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    transition_ok boolean := false;
BEGIN
    IF OLD.state = NEW.state THEN
        transition_ok := true;
    ELSIF OLD.state = 'RECEIVED' AND NEW.state IN ('RETRIEVING', 'FAILED') THEN
        transition_ok := true;
    ELSIF OLD.state = 'RETRIEVING' AND NEW.state IN ('RERANKING', 'AWAITING_HUMAN', 'FAILED') THEN
        transition_ok := true;
    ELSIF OLD.state = 'RERANKING' AND NEW.state IN ('DRAFTING', 'AWAITING_HUMAN', 'FAILED') THEN
        transition_ok := true;
    ELSIF OLD.state = 'DRAFTING' AND NEW.state IN ('VALIDATING', 'AWAITING_HUMAN', 'FAILED') THEN
        transition_ok := true;
    ELSIF OLD.state = 'VALIDATING' AND NEW.state IN ('RETRIEVING', 'AWAITING_HUMAN', 'FAILED') THEN
        transition_ok := true;
    ELSIF OLD.state = 'AWAITING_HUMAN' AND NEW.state IN ('RELEASED', 'ABSTAINED', 'RETURN_TO_DRAFT') THEN
        transition_ok := true;
    ELSIF OLD.state = 'RETURN_TO_DRAFT' AND NEW.state IN ('RETRIEVING', 'FAILED') THEN
        transition_ok := true;
    ELSIF OLD.state = 'FAILED' AND NEW.state = 'ABSTAINED' THEN
        transition_ok := true;
    END IF;

    IF NOT transition_ok THEN
        RAISE EXCEPTION 'illegal run transition: % -> %', OLD.state, NEW.state;
    END IF;

    IF NEW.state IN ('RELEASED', 'ABSTAINED') AND OLD.state <> NEW.state THEN
        IF NOT (
            OLD.state = 'AWAITING_HUMAN'
            OR (OLD.state = 'FAILED' AND NEW.state = 'ABSTAINED')
        ) THEN
            RAISE EXCEPTION 'terminal transition requires human gate';
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM approvals a
            WHERE a.run_id = NEW.run_id
              AND a.from_state = OLD.state
              AND a.to_state = NEW.state
              AND a.payload_sha256 = NEW.draft_sha256
              AND abs(extract(epoch FROM now())::bigint - a.unix_ts) <= 300
        ) THEN
            RAISE EXCEPTION 'fresh verified approval row required';
        END IF;
    END IF;

    IF NEW.state <> 'RELEASED' THEN
        NEW.answer := NULL;
    ELSIF NEW.answer IS DISTINCT FROM NEW.draft->>'draft_answer' THEN
        RAISE EXCEPTION 'released answer must equal signed draft';
    END IF;

    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS runs_terminal_guard ON runs;
DROP TRIGGER IF EXISTS runs_integrity_guard ON runs;
CREATE TRIGGER runs_integrity_guard
BEFORE UPDATE ON runs
FOR EACH ROW EXECUTE FUNCTION guard_run_transition();
