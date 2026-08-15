-- ============================================================
-- Migration 032: 031'deki skor-bazlı pre_avg/delta kolonlarını YÜZDE-bazlı
-- versiyona çevirir
--
-- 13 Ağu 2026 — 031'de eklenen vol/mom/volat/price_score_pre_avg/delta,
-- 0-100'e normalize edilmiş SKORLARIN farkını tutuyordu (puan cinsinden,
-- % değil). Kullanıcı kararı: bunun yerine 4 bileşenin HAM (normalize
-- edilmemiş — hacim, RSI seviyesi, ATR, kapanış fiyatı) değerlerinin
-- sinyal-öncesi ortalamaya göre YÜZDESEL değişimi tutulsun (Hoca
-- Telegram külliyatındaki Sinyal Mumu Ratioları'nın `volume/lastSignalVolume-1`
-- mantığıyla aynı aile, ama pre-window ortalamasına göre).
--
-- 031 canlıda sadece birkaç dakika kaldığı için (migration hemen ardından
-- değiştirildi) veri kaybı önemsiz — eski kolonlar drop edilip yerine
-- yenisi ekleniyor, RENAME değil (semantik tamamen farklı, isim de
-- öyle olmalı: *_score_* yerine ham birim + *_change_pct*).
-- ============================================================

ALTER TABLE signals
    DROP COLUMN IF EXISTS vol_score_pre_avg,
    DROP COLUMN IF EXISTS mom_score_pre_avg,
    DROP COLUMN IF EXISTS volat_score_pre_avg,
    DROP COLUMN IF EXISTS price_score_pre_avg,
    DROP COLUMN IF EXISTS vol_score_pre_delta,
    DROP COLUMN IF EXISTS mom_score_pre_delta,
    DROP COLUMN IF EXISTS volat_score_pre_delta,
    DROP COLUMN IF EXISTS price_score_pre_delta,
    ADD COLUMN IF NOT EXISTS vol_pre_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mom_pre_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS volat_pre_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS price_pre_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS vol_pre_change_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mom_pre_change_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS volat_pre_change_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS price_pre_change_pct DOUBLE PRECISION;

ALTER TABLE paper_trades
    DROP COLUMN IF EXISTS vol_score_pre_avg,
    DROP COLUMN IF EXISTS mom_score_pre_avg,
    DROP COLUMN IF EXISTS volat_score_pre_avg,
    DROP COLUMN IF EXISTS price_score_pre_avg,
    DROP COLUMN IF EXISTS vol_score_pre_delta,
    DROP COLUMN IF EXISTS mom_score_pre_delta,
    DROP COLUMN IF EXISTS volat_score_pre_delta,
    DROP COLUMN IF EXISTS price_score_pre_delta,
    ADD COLUMN IF NOT EXISTS vol_pre_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mom_pre_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS volat_pre_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS price_pre_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS vol_pre_change_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mom_pre_change_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS volat_pre_change_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS price_pre_change_pct DOUBLE PRECISION;
