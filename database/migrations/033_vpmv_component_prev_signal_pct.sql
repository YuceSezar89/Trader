-- ============================================================
-- Migration 033: baseline'ı "5-bar pencere ortalaması"ndan "bir önceki
-- sinyal"e çevirir (032'yi düzeltir)
--
-- 13 Ağu 2026 — Kullanıcı kararı: Hoca'nın Telegram külliyatındaki Sinyal
-- Mumu Ratioları sisteminin gerçek formülüne (`volume/lastSignalVolume-1`,
-- `rsiValue/lastSignalMomentum-1`) sadık kalınsın — baseline bir pencere
-- ortalaması değil, AYNI (symbol, interval, indicators, signal_type)
-- için BİR ÖNCEKİ sinyalin ham değeri olsun.
--
-- vol/mom/volat/price_pre_avg + *_pre_change_pct (032) → vol/mom/volat/
-- price_raw (BU sinyalin ham değeri, bir sonraki sinyalin baseline'ı
-- olarak kullanılacak) + *_change_pct (önceki sinyale göre % değişim).
-- ============================================================

ALTER TABLE signals
    DROP COLUMN IF EXISTS vol_pre_avg,
    DROP COLUMN IF EXISTS mom_pre_avg,
    DROP COLUMN IF EXISTS volat_pre_avg,
    DROP COLUMN IF EXISTS price_pre_avg,
    DROP COLUMN IF EXISTS vol_pre_change_pct,
    DROP COLUMN IF EXISTS mom_pre_change_pct,
    DROP COLUMN IF EXISTS volat_pre_change_pct,
    DROP COLUMN IF EXISTS price_pre_change_pct,
    ADD COLUMN IF NOT EXISTS vol_raw DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mom_raw DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS volat_raw DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS price_raw DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS vol_change_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mom_change_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS volat_change_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS price_change_pct DOUBLE PRECISION;

ALTER TABLE paper_trades
    DROP COLUMN IF EXISTS vol_pre_avg,
    DROP COLUMN IF EXISTS mom_pre_avg,
    DROP COLUMN IF EXISTS volat_pre_avg,
    DROP COLUMN IF EXISTS price_pre_avg,
    DROP COLUMN IF EXISTS vol_pre_change_pct,
    DROP COLUMN IF EXISTS mom_pre_change_pct,
    DROP COLUMN IF EXISTS volat_pre_change_pct,
    DROP COLUMN IF EXISTS price_pre_change_pct,
    ADD COLUMN IF NOT EXISTS vol_raw DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mom_raw DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS volat_raw DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS price_raw DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS vol_change_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mom_change_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS volat_change_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS price_change_pct DOUBLE PRECISION;
