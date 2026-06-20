-- Risk yönetim kolonları: SL/TP fiyat seviyeleri ve çarpanlar
ALTER TABLE signals
    ADD COLUMN IF NOT EXISTS stop_loss_price  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS take_profit_price DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS sl_multiplier     DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS tp_multiplier     DOUBLE PRECISION;

-- close_reason kısıtına stop_loss ve take_profit ekle
ALTER TABLE signals DROP CONSTRAINT IF EXISTS signals_close_reason_check;
ALTER TABLE signals ADD CONSTRAINT signals_close_reason_check
    CHECK (close_reason IS NULL OR close_reason = ANY (ARRAY[
        'reversal', 'timeout', 'manual', 'stop_loss', 'take_profit'
    ]));
