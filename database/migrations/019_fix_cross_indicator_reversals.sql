-- Çapraz-indicator supersede kirliliği düzeltmesi (12 Tem 2026)
--
-- signal_lifecycle_manager._get_active() önceden sadece (symbol, interval) ile
-- aktif sinyal arıyordu, indicators'a bakmıyordu. Farklı indicator'lar
-- (HA_Cross/RSI_Cross/MA200_Cross/Supertrend) aynı sembol+interval'de sinyal
-- üretince birbirinin sinyalini "reversal" ile kapatıyordu. Kod artık
-- (symbol, interval, indicators) üçlüsüyle scope ediyor (bkz. signals/
-- signal_lifecycle_manager.py). Bu migration geçmiş veriyi düzeltir.
--
-- Yöntem: ham kline replay GEREKMİYOR — sinyallerin açılış verisi (open_price,
-- opened_at, indicators, signal_type) hiç bozulmadı, sadece kapanış bağlantısı
-- yanlıştı. Her (symbol, interval, indicators) grubunu kendi kronolojik
-- sırasında yürüyüp GERÇEK bir sonraki ters-yönlü aynı-indicator sinyali
-- bulup ona göre yeniden bağlıyoruz.

BEGIN;

ALTER TABLE signals
    ADD COLUMN IF NOT EXISTS cross_indicator_close BOOLEAN;

-- 1) Düzeltilebilir satırlar (%82,3 — kendi indicator'ünde gerçek bir sonraki
--    ters sinyal var): close_price/closed_at/closed_by/realized_pnl/
--    duration_minutes gerçek ardıla göre yeniden hesaplanır.
WITH true_seq AS (
    SELECT id, symbol, interval, indicators, opened_at,
           LEAD(id) OVER w         AS true_closer_id,
           LEAD(open_price) OVER w AS true_closer_price,
           LEAD(opened_at) OVER w  AS true_closer_time
    FROM signals
    WINDOW w AS (PARTITION BY symbol, interval, indicators ORDER BY opened_at)
),
kirli AS (
    SELECT ts.id, ts.true_closer_id, ts.true_closer_price, ts.true_closer_time
    FROM true_seq ts
    JOIN signals old ON old.id = ts.id
    JOIN signals n   ON old.closed_by = n.id
    WHERE old.close_reason = 'reversal'
      AND old.indicators != n.indicators
      AND ts.true_closer_id IS NOT NULL
)
UPDATE signals s
SET close_price = k.true_closer_price,
    closed_at   = k.true_closer_time,
    closed_by   = k.true_closer_id,
    realized_pnl = CASE
        WHEN s.open_price = 0 THEN 0
        WHEN s.signal_type = 'Long' THEN (k.true_closer_price - s.open_price) / s.open_price * 100
        ELSE (s.open_price - k.true_closer_price) / s.open_price * 100
    END,
    duration_minutes = EXTRACT(EPOCH FROM (k.true_closer_time - s.opened_at)) / 60
FROM kirli k
WHERE s.id = k.id;

-- 2) Kurtarılamaz satırlar (%17,7 — o indicator o sembol+interval'de bir daha
--    hiç sinyal üretmemiş): close verisi güvenilmez olarak işaretlenir,
--    dokunulmaz, PnL analizlerinde dışlanmalı.
WITH true_seq AS (
    SELECT id, symbol, interval, indicators, opened_at,
           LEAD(id) OVER w AS true_closer_id
    FROM signals
    WINDOW w AS (PARTITION BY symbol, interval, indicators ORDER BY opened_at)
),
kurtarilamaz AS (
    SELECT ts.id
    FROM true_seq ts
    JOIN signals old ON old.id = ts.id
    JOIN signals n   ON old.closed_by = n.id
    WHERE old.close_reason = 'reversal'
      AND old.indicators != n.indicators
      AND ts.true_closer_id IS NULL
)
UPDATE signals s
SET cross_indicator_close = TRUE
FROM kurtarilamaz k
WHERE s.id = k.id;

COMMIT;
