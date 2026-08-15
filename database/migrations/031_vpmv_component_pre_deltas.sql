-- ============================================================
-- Migration 031: VPMV bileşenlerinin (vol/mom/volat/price) sinyal-öncesi
-- pre_avg + delta kırılımı
--
-- 13 Ağu 2026 — Devisso Döngüsü çalışması. vpmv_pre_avg zaten KOMBİNE
-- (ağırlıklı toplam) skorun 5-barlık pre-ortalamasını tutuyordu; bu
-- migration aynı ölçümü 4 ham bileşen için ayrı ayrı ekliyor, hangi
-- bileşenin sinyal öncesi ortalamadan ne kadar saptığını görebilmek için
-- (bkz. utils/vpmv.py::compute_pre_components).
--
-- delta = sinyal anındaki bileşen değeri - kendi pre_avg'ı.
-- ============================================================

ALTER TABLE signals
    ADD COLUMN IF NOT EXISTS vol_score_pre_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mom_score_pre_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS volat_score_pre_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS price_score_pre_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS vol_score_pre_delta DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mom_score_pre_delta DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS volat_score_pre_delta DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS price_score_pre_delta DOUBLE PRECISION;

ALTER TABLE paper_trades
    ADD COLUMN IF NOT EXISTS vol_score_pre_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mom_score_pre_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS volat_score_pre_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS price_score_pre_avg DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS vol_score_pre_delta DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mom_score_pre_delta DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS volat_score_pre_delta DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS price_score_pre_delta DOUBLE PRECISION;
