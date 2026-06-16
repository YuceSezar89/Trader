-- ============================================================
-- Migration 006: TimescaleDB Continuous Aggregates
-- 1m price_data'dan 5m / 15m / 1h / 4h otomatik türetilir.
-- Binance'ten doğrudan 5m/15m/1h/4h çekme tamamen kalkar.
-- ============================================================

-- ─── 5 Dakika ───────────────────────────────────────────────
CREATE MATERIALIZED VIEW cagg_5m
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket('5 minutes', timestamp) AS bucket,
    symbol,
    first(open,   timestamp)            AS open,
    max(high)                           AS high,
    min(low)                            AS low,
    last(close,   timestamp)            AS close,
    sum(volume)                         AS volume
FROM price_data
WHERE interval = '1m'
GROUP BY bucket, symbol
WITH NO DATA;

CREATE INDEX ON cagg_5m  (symbol, bucket DESC);

SELECT add_continuous_aggregate_policy('cagg_5m',
    start_offset      => INTERVAL '15 minutes',
    end_offset        => INTERVAL '0 seconds',
    schedule_interval => INTERVAL '1 minute');

-- ─── 15 Dakika ──────────────────────────────────────────────
CREATE MATERIALIZED VIEW cagg_15m
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket('15 minutes', timestamp) AS bucket,
    symbol,
    first(open,   timestamp)             AS open,
    max(high)                            AS high,
    min(low)                             AS low,
    last(close,   timestamp)             AS close,
    sum(volume)                          AS volume
FROM price_data
WHERE interval = '1m'
GROUP BY bucket, symbol
WITH NO DATA;

CREATE INDEX ON cagg_15m (symbol, bucket DESC);

SELECT add_continuous_aggregate_policy('cagg_15m',
    start_offset      => INTERVAL '45 minutes',
    end_offset        => INTERVAL '0 seconds',
    schedule_interval => INTERVAL '2 minutes');

-- ─── 1 Saat ─────────────────────────────────────────────────
CREATE MATERIALIZED VIEW cagg_1h
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket('1 hour', timestamp) AS bucket,
    symbol,
    first(open,   timestamp)         AS open,
    max(high)                        AS high,
    min(low)                         AS low,
    last(close,   timestamp)         AS close,
    sum(volume)                      AS volume
FROM price_data
WHERE interval = '1m'
GROUP BY bucket, symbol
WITH NO DATA;

CREATE INDEX ON cagg_1h  (symbol, bucket DESC);

SELECT add_continuous_aggregate_policy('cagg_1h',
    start_offset      => INTERVAL '3 hours',
    end_offset        => INTERVAL '0 seconds',
    schedule_interval => INTERVAL '5 minutes');

-- ─── 4 Saat ─────────────────────────────────────────────────
-- 1m verisi üzerinden türetilir; Binance 4h cache'i artık gerekmez.
CREATE MATERIALIZED VIEW cagg_4h
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket('4 hours', timestamp) AS bucket,
    symbol,
    first(open,   timestamp)          AS open,
    max(high)                         AS high,
    min(low)                          AS low,
    last(close,   timestamp)          AS close,
    sum(volume)                       AS volume
FROM price_data
WHERE interval = '1m'
GROUP BY bucket, symbol
WITH NO DATA;

CREATE INDEX ON cagg_4h  (symbol, bucket DESC);

SELECT add_continuous_aggregate_policy('cagg_4h',
    start_offset      => INTERVAL '12 hours',
    end_offset        => INTERVAL '0 seconds',
    schedule_interval => INTERVAL '10 minutes');

-- ─── İlk Backfill (mevcut 1m verisinden doldur) ─────────────
CALL refresh_continuous_aggregate('cagg_5m',  NULL, NULL);
CALL refresh_continuous_aggregate('cagg_15m', NULL, NULL);
CALL refresh_continuous_aggregate('cagg_1h',  NULL, NULL);
CALL refresh_continuous_aggregate('cagg_4h',  NULL, NULL);
