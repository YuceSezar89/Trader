-- Paper Trading Modülü
-- paper_trades: Her konfluans sinyali için simüle edilmiş pozisyon
-- paper_portfolio: Strateji bazında bakiye ve istatistik

CREATE TABLE IF NOT EXISTS paper_trades (
    id              SERIAL PRIMARY KEY,
    signal_id       INTEGER REFERENCES signals(id) ON DELETE SET NULL,
    strategy        VARCHAR(50)  NOT NULL DEFAULT 'conf_100',

    -- Pozisyon
    symbol          VARCHAR(30)  NOT NULL,
    signal_type     VARCHAR(10)  NOT NULL,
    interval        VARCHAR(10)  NOT NULL,
    position_usd    DOUBLE PRECISION NOT NULL DEFAULT 100.0,
    entry_price     DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION,
    stop_loss_price DOUBLE PRECISION,
    take_profit_price DOUBLE PRECISION,
    trailing_stop_price DOUBLE PRECISION,

    -- Finansal sonuç
    fee_usd         DOUBLE PRECISION,
    pnl_usd         DOUBLE PRECISION,
    pnl_pct         DOUBLE PRECISION,
    balance_after   DOUBLE PRECISION,

    -- Durum
    status          VARCHAR(20)  NOT NULL DEFAULT 'open',
    close_reason    VARCHAR(50),
    opened_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    closed_at       TIMESTAMPTZ,

    -- ML snapshot (açılış anında snaplenir)
    btc_z_score     DOUBLE PRECISION,
    btc_trend       VARCHAR(20),
    hour_utc        SMALLINT,
    day_of_week     SMALLINT,
    funding_rate    DOUBLE PRECISION,
    recent_win_rate DOUBLE PRECISION,

    -- Signal features (denormalize, ML join kolaylığı)
    vpms_score      DOUBLE PRECISION,
    z_score_entry   DOUBLE PRECISION,
    mtf_score       DOUBLE PRECISION,
    atr             DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_paper_trades_strategy_status
    ON paper_trades (strategy, status);

CREATE INDEX IF NOT EXISTS idx_paper_trades_symbol
    ON paper_trades (symbol, opened_at DESC);

CREATE INDEX IF NOT EXISTS idx_paper_trades_signal_id
    ON paper_trades (signal_id);

-- Portföy özeti (strateji başına tek satır)
CREATE TABLE IF NOT EXISTS paper_portfolio (
    id               SERIAL PRIMARY KEY,
    strategy         VARCHAR(50)  NOT NULL UNIQUE DEFAULT 'conf_100',
    balance          DOUBLE PRECISION NOT NULL DEFAULT 10000.0,
    initial_balance  DOUBLE PRECISION NOT NULL DEFAULT 10000.0,
    peak_balance     DOUBLE PRECISION NOT NULL DEFAULT 10000.0,
    max_drawdown_pct DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    total_trades     INTEGER NOT NULL DEFAULT 0,
    winning_trades   INTEGER NOT NULL DEFAULT 0,
    total_pnl_usd    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO paper_portfolio (strategy, balance, initial_balance, peak_balance)
VALUES ('conf_100', 10000.0, 10000.0, 10000.0)
ON CONFLICT (strategy) DO NOTHING;
