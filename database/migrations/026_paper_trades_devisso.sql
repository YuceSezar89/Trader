-- ============================================================
-- Migration 026: paper_trades'e devisso (ERSI/"kolaylık") skoru eklendi
--
-- Kullanıcı isteği (20 Tem 2026): açık işlemlerde "kolaylık/verimlilik"
-- verisi izlensin. Bu, Devisso Sıralama tablosunun formülüyle AYNI:
-- ERSI = ΔFiyat% / ΔRSI(14) → EMA(7) → son-100-bar percentile rank (0-100).
-- devisso_delta/devisso_ratio, sinyalin BİR ÖNCEKİ aynı sembol/TF/yön
-- sinyaline göre karşılaştırması (signals tablosundaki AYNI mantık).
--
-- NOT: Bu metrik 6 Tem 2026'da gerçek $ ile test edilmiş, realized_pnl ile
-- anlamlı ilişkisi bulunmamıştı (bkz. memory/project_devisso_ersi.md) —
-- burada SADECE bilgi/izleme amaçlı ekleniyor, filtre/karar verisi DEĞİL.
-- ============================================================

ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS devisso_score DOUBLE PRECISION;
ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS devisso_delta DOUBLE PRECISION;
ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS devisso_ratio DOUBLE PRECISION;
