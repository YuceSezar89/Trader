ALTER TABLE signals
    ADD COLUMN IF NOT EXISTS z_score_entry  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS is_confluence  BOOLEAN DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_signals_confluence
    ON signals (is_confluence, opened_at DESC)
    WHERE is_confluence = TRUE;
