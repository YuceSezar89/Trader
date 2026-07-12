ALTER TABLE signals
    ADD COLUMN IF NOT EXISTS ha_ultra_confirm SMALLINT;

CREATE INDEX IF NOT EXISTS idx_signals_ha_ultra_confirm
    ON signals (ha_ultra_confirm, opened_at DESC)
    WHERE ha_ultra_confirm = 3;
