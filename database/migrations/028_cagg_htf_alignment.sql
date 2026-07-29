-- ============================================================
-- Migration 028: HTF hizalanma takibi için cagg_6h / cagg_8h / cagg_12h
--
-- Amaç: 4h/6h/8h/12h Heikin-Ashi hizalanmasını TÜM izlenen sembol evreni
-- için arka planda hesaplayıp takip edebilmek (27 Tem 2026, kullanıcı
-- isteği — HTF hizalanan coinlerde oluşan sinyallere işlem açma planının
-- ilk altyapı adımı).
--
-- SADECE EKLEME — mevcut cagg_5m/15m/1h/4h zincirine (021, 023) hiçbir
-- şekilde dokunulmuyor, drop/recreate YOK. Aynı hiyerarşik desen ve aynı
-- ortak origin ('2000-01-01 03:00:00', 021'de açıklanan nedenle) korunuyor
-- — TimescaleDB nested CA kısıtı: çocuk bucket genişliği ebeveynin TAM
-- katı olmalı VE origin birebir eşleşmeli.
--
--   cagg_6h  (6h = 6×1h)  ← cagg_1h
--   cagg_8h  (8h = 2×4h)  ← cagg_4h
--   cagg_12h (12h = 3×4h) ← cagg_4h
--
-- Şema cagg_4h ile birebir aynı (buy_volume/sell_volume dahil, 023 ile
-- tutarlı). database/crud.py::get_cagg_klines KULLANILMADAN ÖNCE
-- _CAGG_MAP/_INTERVAL_MINUTES'e "6h"/"8h"/"12h" eklenmesi GEREKİYOR
-- (ayrı adım, bu migration sadece DB tarafı).
-- ============================================================

CREATE MATERIALIZED VIEW cagg_6h
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket('6 hours', bucket, origin => '2000-01-01 03:00:00'::timestamp) AS bucket,
    symbol,
    first(open,   bucket) AS open,
    max(high)             AS high,
    min(low)              AS low,
    last(close,   bucket) AS close,
    sum(volume)           AS volume,
    sum(buy_volume)       AS buy_volume,
    sum(sell_volume)      AS sell_volume
FROM cagg_1h
GROUP BY time_bucket('6 hours', bucket, origin => '2000-01-01 03:00:00'::timestamp), symbol
WITH NO DATA;

CREATE INDEX ON cagg_6h (symbol, bucket DESC);

SELECT add_continuous_aggregate_policy('cagg_6h',
    start_offset      => INTERVAL '60 days',
    end_offset        => INTERVAL '0 seconds',
    schedule_interval => INTERVAL '15 minutes');

CALL refresh_continuous_aggregate('cagg_6h', NULL, NULL);

-- ─── 8 Saat — cagg_4h'den (2×4h) ─────────────────────────────
CREATE MATERIALIZED VIEW cagg_8h
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket('8 hours', bucket, origin => '2000-01-01 03:00:00'::timestamp) AS bucket,
    symbol,
    first(open,   bucket) AS open,
    max(high)             AS high,
    min(low)              AS low,
    last(close,   bucket) AS close,
    sum(volume)           AS volume,
    sum(buy_volume)       AS buy_volume,
    sum(sell_volume)      AS sell_volume
FROM cagg_4h
GROUP BY time_bucket('8 hours', bucket, origin => '2000-01-01 03:00:00'::timestamp), symbol
WITH NO DATA;

CREATE INDEX ON cagg_8h (symbol, bucket DESC);

SELECT add_continuous_aggregate_policy('cagg_8h',
    start_offset      => INTERVAL '90 days',
    end_offset        => INTERVAL '0 seconds',
    schedule_interval => INTERVAL '20 minutes');

CALL refresh_continuous_aggregate('cagg_8h', NULL, NULL);

-- ─── 12 Saat — cagg_4h'den (3×4h) ────────────────────────────
CREATE MATERIALIZED VIEW cagg_12h
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket('12 hours', bucket, origin => '2000-01-01 03:00:00'::timestamp) AS bucket,
    symbol,
    first(open,   bucket) AS open,
    max(high)             AS high,
    min(low)              AS low,
    last(close,   bucket) AS close,
    sum(volume)           AS volume,
    sum(buy_volume)       AS buy_volume,
    sum(sell_volume)      AS sell_volume
FROM cagg_4h
GROUP BY time_bucket('12 hours', bucket, origin => '2000-01-01 03:00:00'::timestamp), symbol
WITH NO DATA;

CREATE INDEX ON cagg_12h (symbol, bucket DESC);

SELECT add_continuous_aggregate_policy('cagg_12h',
    start_offset      => INTERVAL '120 days',
    end_offset        => INTERVAL '0 seconds',
    schedule_interval => INTERVAL '30 minutes');

CALL refresh_continuous_aggregate('cagg_12h', NULL, NULL);
