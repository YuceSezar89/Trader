-- ============================================================
-- Migration 034: candle_kategori'nin sayısal kırılımı (body_pct/wick_pct)
--
-- 13 Ağu 2026 — Sinyal Mumu Ratioları entegrasyonunun son parçası.
-- _classify_candle() zaten gövde/üst fitil/alt fitil yüzdelerini içeride
-- hesaplıyordu, sadece hangisinin baskın olduğu (candle_kategori) tutulup
-- sayılar atılıyordu. body_pct = gövde %, wick_pct = üst+alt fitil toplamı
-- (=100-body_pct).
-- ============================================================

ALTER TABLE signals
    ADD COLUMN IF NOT EXISTS body_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS wick_pct DOUBLE PRECISION;

ALTER TABLE paper_trades
    ADD COLUMN IF NOT EXISTS body_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS wick_pct DOUBLE PRECISION;
