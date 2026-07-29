-- ============================================================
-- Migration 023: CA view'lara yönlü hacim (buy_volume/sell_volume) eklendi
--
-- Sorun: cagg_5m/15m/1h/4h zincirinde (021) sadece open/high/low/close/volume
-- aggregate ediliyordu, buy_volume/sell_volume hiç taşınmıyordu. Bu yüzden
-- signals/market_context.py::compute_cvd_slope() bu view'lardan beslenen
-- her çağrıda (trade_snapshot.py, sürdürme fazı izleme) "buy_volume" kolonu
-- df'de HİÇ olmadığı için %100 geometrik proxy'ye düşüyordu — sinyal
-- anındaki (canlı buffer, gerçek buy_volume) CVD ile karşılaştırılamaz
-- hale geliyordu. Ampirik test (19 Tem): gerçek vs proxy CVD slope
-- korelasyonu 0.40, işaret uyumu %64.7 — birbirinin yerine geçmiyor.
--
-- Aynı hiyerarşik zincir (5m←price_data, 15m←5m, 1h←15m, 4h←1h) korunuyor,
-- sadece sum(buy_volume)/sum(sell_volume) eklendi (additive, sum-of-sum
-- matematiksel olarak doğru).
-- ============================================================

SELECT remove_continuous_aggregate_policy('cagg_4h', if_exists => true);
SELECT remove_continuous_aggregate_policy('cagg_1h', if_exists => true);
SELECT remove_continuous_aggregate_policy('cagg_15m', if_exists => true);
SELECT remove_continuous_aggregate_policy('cagg_5m', if_exists => true);

DROP MATERIALIZED VIEW IF EXISTS cagg_4h;
DROP MATERIALIZED VIEW IF EXISTS cagg_1h;
DROP MATERIALIZED VIEW IF EXISTS cagg_15m;
DROP MATERIALIZED VIEW IF EXISTS cagg_5m;

-- ─── 5 Dakika — ham 1m'den ───────────────────────────────────
CREATE MATERIALIZED VIEW cagg_5m
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket('5 minutes', timestamp, origin => '2000-01-01 03:00:00'::timestamp) AS bucket,
    symbol,
    first(open,   timestamp) AS open,
    max(high)                AS high,
    min(low)                 AS low,
    last(close,   timestamp) AS close,
    sum(volume)              AS volume,
    sum(buy_volume)          AS buy_volume,
    sum(sell_volume)         AS sell_volume
FROM price_data
WHERE interval = '1m'
GROUP BY time_bucket('5 minutes', timestamp, origin => '2000-01-01 03:00:00'::timestamp), symbol
WITH NO DATA;

CREATE INDEX ON cagg_5m (symbol, bucket DESC);

SELECT add_continuous_aggregate_policy('cagg_5m',
    start_offset      => INTERVAL '15 minutes',
    end_offset        => INTERVAL '0 seconds',
    schedule_interval => INTERVAL '1 minute');

CALL refresh_continuous_aggregate('cagg_5m', NULL, NULL);

-- ─── 15 Dakika — cagg_5m'den ────────────────────────────────
CREATE MATERIALIZED VIEW cagg_15m
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket('15 minutes', bucket, origin => '2000-01-01 03:00:00'::timestamp) AS bucket,
    symbol,
    first(open,   bucket) AS open,
    max(high)             AS high,
    min(low)              AS low,
    last(close,   bucket) AS close,
    sum(volume)           AS volume,
    sum(buy_volume)       AS buy_volume,
    sum(sell_volume)      AS sell_volume
FROM cagg_5m
GROUP BY time_bucket('15 minutes', bucket, origin => '2000-01-01 03:00:00'::timestamp), symbol
WITH NO DATA;

CREATE INDEX ON cagg_15m (symbol, bucket DESC);

SELECT add_continuous_aggregate_policy('cagg_15m',
    start_offset      => INTERVAL '45 minutes',
    end_offset        => INTERVAL '0 seconds',
    schedule_interval => INTERVAL '2 minutes');

CALL refresh_continuous_aggregate('cagg_15m', NULL, NULL);

-- ─── 1 Saat — cagg_15m'den ──────────────────────────────────
CREATE MATERIALIZED VIEW cagg_1h
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket('1 hour', bucket, origin => '2000-01-01 03:00:00'::timestamp) AS bucket,
    symbol,
    first(open,   bucket) AS open,
    max(high)             AS high,
    min(low)              AS low,
    last(close,   bucket) AS close,
    sum(volume)           AS volume,
    sum(buy_volume)       AS buy_volume,
    sum(sell_volume)      AS sell_volume
FROM cagg_15m
GROUP BY time_bucket('1 hour', bucket, origin => '2000-01-01 03:00:00'::timestamp), symbol
WITH NO DATA;

CREATE INDEX ON cagg_1h (symbol, bucket DESC);

SELECT add_continuous_aggregate_policy('cagg_1h',
    start_offset      => INTERVAL '7 days',
    end_offset        => INTERVAL '0 seconds',
    schedule_interval => INTERVAL '5 minutes');

CALL refresh_continuous_aggregate('cagg_1h', NULL, NULL);

-- ─── 4 Saat — cagg_1h'den (aynı UTC anchor korunuyor) ───────
CREATE MATERIALIZED VIEW cagg_4h
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket('4 hours', bucket, origin => '2000-01-01 03:00:00'::timestamp) AS bucket,
    symbol,
    first(open,   bucket) AS open,
    max(high)             AS high,
    min(low)              AS low,
    last(close,   bucket) AS close,
    sum(volume)           AS volume,
    sum(buy_volume)       AS buy_volume,
    sum(sell_volume)      AS sell_volume
FROM cagg_1h
GROUP BY time_bucket('4 hours', bucket, origin => '2000-01-01 03:00:00'::timestamp), symbol
WITH NO DATA;

CREATE INDEX ON cagg_4h (symbol, bucket DESC);

SELECT add_continuous_aggregate_policy('cagg_4h',
    start_offset      => INTERVAL '30 days',
    end_offset        => INTERVAL '0 seconds',
    schedule_interval => INTERVAL '10 minutes');

CALL refresh_continuous_aggregate('cagg_4h', NULL, NULL);
