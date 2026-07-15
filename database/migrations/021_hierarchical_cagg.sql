-- ============================================================
-- Migration 021: Hiyerarşik Continuous Aggregate (aggregate-on-aggregate)
-- cagg_15m artık cagg_5m'den, cagg_1h artık cagg_15m'den, cagg_4h artık
-- cagg_1h'den türetilir — her seviye bir alttakinin ÇOK daha küçük
-- (zaten agregeli) veri kümesini tarar.
--
-- ÖNEMLİ: TimescaleDB zincirleme CA'larda üst-alt bucket origin'lerinin
-- BİREBİR eşleşmesini zorunlu kılıyor. Bu yüzden cagg_5m dahil TÜM
-- seviyelere aynı origin ('2000-01-01 03:00:00') veriliyor — bu değer
-- 5dk/15dk/1sa sınırlarına zaten tam oturduğu için (03:00:00, 5'in de
-- 15'in de katı) gerçek bucket sınırlarında HİÇBİR kayma olmaz, sadece
-- cagg_4h'nin ihtiyaç duyduğu özel UTC anchor ile tutarlılık sağlanır.
--
-- Şema/kolonlar (bucket, symbol, open, high, low, close, volume) ve
-- refresh_policy parametreleri AYNEN korunuyor — uygulama kodu
-- (database/crud.py::get_cagg_klines vb.) DEĞİŞMEDEN çalışmaya devam eder.
-- ============================================================

SELECT remove_continuous_aggregate_policy('cagg_4h', if_exists => true);
SELECT remove_continuous_aggregate_policy('cagg_1h', if_exists => true);
SELECT remove_continuous_aggregate_policy('cagg_15m', if_exists => true);
SELECT remove_continuous_aggregate_policy('cagg_5m', if_exists => true);

DROP MATERIALIZED VIEW IF EXISTS cagg_4h;
DROP MATERIALIZED VIEW IF EXISTS cagg_1h;
DROP MATERIALIZED VIEW IF EXISTS cagg_15m;
DROP MATERIALIZED VIEW IF EXISTS cagg_5m;

-- ─── 5 Dakika — ham 1m'den (değişmedi, sadece ortak origin eklendi) ─
CREATE MATERIALIZED VIEW cagg_5m
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket('5 minutes', timestamp, origin => '2000-01-01 03:00:00'::timestamp) AS bucket,
    symbol,
    first(open,   timestamp) AS open,
    max(high)                AS high,
    min(low)                 AS low,
    last(close,   timestamp) AS close,
    sum(volume)              AS volume
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

-- ─── 15 Dakika — artık cagg_5m'den ──────────────────────────
CREATE MATERIALIZED VIEW cagg_15m
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket('15 minutes', bucket, origin => '2000-01-01 03:00:00'::timestamp) AS bucket,
    symbol,
    first(open,   bucket) AS open,
    max(high)             AS high,
    min(low)              AS low,
    last(close,   bucket) AS close,
    sum(volume)           AS volume
FROM cagg_5m
GROUP BY time_bucket('15 minutes', bucket, origin => '2000-01-01 03:00:00'::timestamp), symbol
WITH NO DATA;

CREATE INDEX ON cagg_15m (symbol, bucket DESC);

SELECT add_continuous_aggregate_policy('cagg_15m',
    start_offset      => INTERVAL '45 minutes',
    end_offset        => INTERVAL '0 seconds',
    schedule_interval => INTERVAL '2 minutes');

CALL refresh_continuous_aggregate('cagg_15m', NULL, NULL);

-- ─── 1 Saat — artık cagg_15m'den ────────────────────────────
CREATE MATERIALIZED VIEW cagg_1h
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket('1 hour', bucket, origin => '2000-01-01 03:00:00'::timestamp) AS bucket,
    symbol,
    first(open,   bucket) AS open,
    max(high)             AS high,
    min(low)              AS low,
    last(close,   bucket) AS close,
    sum(volume)           AS volume
FROM cagg_15m
GROUP BY time_bucket('1 hour', bucket, origin => '2000-01-01 03:00:00'::timestamp), symbol
WITH NO DATA;

CREATE INDEX ON cagg_1h (symbol, bucket DESC);

SELECT add_continuous_aggregate_policy('cagg_1h',
    start_offset      => INTERVAL '7 days',
    end_offset        => INTERVAL '0 seconds',
    schedule_interval => INTERVAL '5 minutes');

CALL refresh_continuous_aggregate('cagg_1h', NULL, NULL);

-- ─── 4 Saat — artık cagg_1h'den (aynı UTC anchor korunuyor) ─
CREATE MATERIALIZED VIEW cagg_4h
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket('4 hours', bucket, origin => '2000-01-01 03:00:00'::timestamp) AS bucket,
    symbol,
    first(open,   bucket) AS open,
    max(high)             AS high,
    min(low)              AS low,
    last(close,   bucket) AS close,
    sum(volume)           AS volume
FROM cagg_1h
GROUP BY time_bucket('4 hours', bucket, origin => '2000-01-01 03:00:00'::timestamp), symbol
WITH NO DATA;

CREATE INDEX ON cagg_4h (symbol, bucket DESC);

SELECT add_continuous_aggregate_policy('cagg_4h',
    start_offset      => INTERVAL '30 days',
    end_offset        => INTERVAL '0 seconds',
    schedule_interval => INTERVAL '10 minutes');

CALL refresh_continuous_aggregate('cagg_4h', NULL, NULL);
