-- ============================================================
-- Migration 037: trade_snapshots'a VPMV bileşen bazlı (hacim/momentum/
-- volatilite/fiyat, ayrı ayrı) girişe göre %değişim eklendi
--
-- 20 Ağu 2026 — totalamount_rank1 için: paper_trades.vol_raw/mom_raw/
-- volat_raw/price_raw (giriş anı ham değerleri, aynı migration'la
-- open_direct yoluna da dolduruluyor) referans alınarak, her dakika
-- güncel ham bileşenlerin girişe göre yüzdesel değişimi hesaplanıp
-- buraya yazılıyor. Formül signal_processor.py::_compute_component_
-- change_pct ile AYNI ((şimdi-giriş)/|giriş|*100) — sadece kıyas
-- noktası "bir önceki sinyal" değil "kendi girişimiz".
-- ============================================================

ALTER TABLE trade_snapshots
    ADD COLUMN IF NOT EXISTS vol_change_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mom_change_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS volat_change_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS price_change_pct DOUBLE PRECISION;
