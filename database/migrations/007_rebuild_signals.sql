-- signals tablosunu temiz state-machine tasarımıyla yeniden oluşturur.
-- Eski veriler silinir; signal_performance da sıfırlanır.

DROP TABLE IF EXISTS signal_performance CASCADE;
DROP TABLE IF EXISTS signals CASCADE;

CREATE TABLE signals (
    id           BIGSERIAL PRIMARY KEY,
    symbol       TEXT    NOT NULL,
    interval     TEXT    NOT NULL,
    indicators   TEXT    NOT NULL,
    signal_type  TEXT    NOT NULL CHECK (signal_type IN ('Long', 'Short')),

    opened_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    open_price   DOUBLE PRECISION NOT NULL,

    vpms_score   DOUBLE PRECISION,
    mtf_score    DOUBLE PRECISION,
    st_confirmed BOOLEAN,
    rsi          DOUBLE PRECISION,
    strength     INTEGER,
    atr          DOUBLE PRECISION,
    alpha        DOUBLE PRECISION,
    beta         DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,

    status       TEXT NOT NULL DEFAULT 'active'
                 CHECK (status IN ('active', 'closed')),
    closed_at    TIMESTAMP,
    close_price  DOUBLE PRECISION,
    close_reason TEXT
                 CHECK (close_reason IS NULL OR close_reason IN ('reversal', 'timeout', 'manual')),
    closed_by    BIGINT REFERENCES signals(id),

    realized_pnl     DOUBLE PRECISION,
    duration_minutes INTEGER
);

-- En fazla 1 aktif sinyal per (symbol, interval, indicator)
CREATE UNIQUE INDEX one_active_per_key
    ON signals (symbol, interval, indicators)
    WHERE status = 'active';

CREATE INDEX idx_signals_status_time ON signals (status, opened_at DESC);
CREATE INDEX idx_signals_symbol_key  ON signals (symbol, interval, indicators, opened_at DESC);
CREATE INDEX idx_signals_closed      ON signals (closed_at DESC) WHERE status = 'closed';

-- signal_performance
CREATE TABLE signal_performance (
    id                     SERIAL PRIMARY KEY,
    signal_id              INTEGER NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    entry_price            NUMERIC(20,8) NOT NULL,
    entry_timestamp        TIMESTAMP NOT NULL,
    atr_at_entry           NUMERIC(20,8) NOT NULL DEFAULT 0,
    interval               VARCHAR(10) NOT NULL,
    return_t3_atr          NUMERIC(10,4),
    return_t5_atr          NUMERIC(10,4),
    return_t10_atr         NUMERIC(10,4),
    return_t3_pct          NUMERIC(10,4),
    return_t5_pct          NUMERIC(10,4),
    return_t10_pct         NUMERIC(10,4),
    mfe_atr                NUMERIC(10,4),
    mae_atr                NUMERIC(10,4),
    risk_reward            NUMERIC(10,4),
    mfe_bar_index          INTEGER,
    mae_bar_index          INTEGER,
    is_calculated          BOOLEAN DEFAULT FALSE,
    calculation_attempts   INTEGER DEFAULT 0,
    last_calculation_error TEXT,
    created_at             TIMESTAMP DEFAULT NOW(),
    updated_at             TIMESTAMP DEFAULT NOW(),
    calculated_at          TIMESTAMP,
    UNIQUE (signal_id)
);

CREATE INDEX idx_signal_perf_signal_id ON signal_performance (signal_id);
CREATE INDEX idx_signal_perf_pending   ON signal_performance (is_calculated, entry_timestamp)
    WHERE is_calculated = FALSE;

-- Trigger: yeni sinyal açılınca signal_performance kaydı oluştur
CREATE OR REPLACE FUNCTION auto_create_signal_performance()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO signal_performance (
        signal_id, entry_price, entry_timestamp, atr_at_entry, interval,
        is_calculated, calculation_attempts, created_at, updated_at
    ) VALUES (
        NEW.id, NEW.open_price, NEW.opened_at,
        COALESCE(NEW.atr, 0), NEW.interval,
        FALSE, 0, NOW(), NOW()
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_auto_signal_performance
    AFTER INSERT ON signals
    FOR EACH ROW EXECUTE FUNCTION auto_create_signal_performance();

-- Trigger: sinyal kapanınca duration_minutes hesapla
CREATE OR REPLACE FUNCTION compute_signal_duration()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.closed_at IS NOT NULL AND OLD.closed_at IS NULL THEN
        NEW.duration_minutes = EXTRACT(EPOCH FROM (NEW.closed_at - NEW.opened_at)) / 60;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_compute_duration
    BEFORE UPDATE ON signals
    FOR EACH ROW EXECUTE FUNCTION compute_signal_duration();

-- signal_performance updated_at trigger
CREATE OR REPLACE FUNCTION update_signal_performance_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_signal_performance_updated_at
    BEFORE UPDATE ON signal_performance
    FOR EACH ROW EXECUTE FUNCTION update_signal_performance_timestamp();
