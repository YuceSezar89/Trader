-- ============================================================
-- Migration 030: signal_performance.last_calculation_error kolonunu sil
--
-- 13 Ağu 2026 — DB_KOLONLARI.md envanteri sırasında bulundu: kod tabanında
-- (signal_performance_analyzer.py dahil, bu tabloyu dolduran tek yer)
-- hiçbir yerde okunmuyor/yazılmıyor, tabloda 0 dolu satır var. Gerçek
-- ölü kolon (cross_indicator_close ile karıştırılmasın — o, bilinçli
-- bırakılmış bir veri güvenilirlik bayrağı, migration 019, silinmedi).
-- ============================================================

ALTER TABLE signal_performance
    DROP COLUMN IF EXISTS last_calculation_error;
