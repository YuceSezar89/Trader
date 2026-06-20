-- Migration 010: Trailing stop kolonu ekle
ALTER TABLE signals
    ADD COLUMN IF NOT EXISTS trailing_stop_price DOUBLE PRECISION;

-- Constraint'e trailing_stop ekle
ALTER TABLE signals DROP CONSTRAINT IF EXISTS signals_close_reason_check;
ALTER TABLE signals ADD CONSTRAINT signals_close_reason_check
    CHECK (close_reason IN (
        'reversal', 'timeout', 'manual', 'reconciliation',
        'stop_loss', 'take_profit', 'trailing_stop'
    ));
