-- ============================================================
-- Migration 027: signals'a Güç Sıralaması anlık görüntüsü eklendi (Faz 1)
--
-- Kullanıcı isteği (20 Tem 2026): Ranking panelinin ("Güç Sıralaması")
-- verisi hiç kaydedilmiyordu, sadece Redis'te canlı anlık görüntü olarak
-- duruyordu (ranking:snapshot, 90sn'de bir üstüne yazılıyor). rank_score/
-- vs_btc zaten sinyal açılış anında kopyalanıyordu — bu migration Ranking
-- panelinin GERİ KALAN alanlarını da aynı yolla (sinyal açılış anı,
-- look-ahead yok) yakalar. Bkz. docs/plan_radar_data_persistence.md.
--
-- Bilgi/izleme amaçlı — 20 Tem'deki ön-testte rank_score'un PnL ile zayıf/
-- tutarsız ilişkisi bulundu, bu yeni alanlar FİLTRE değil, henüz test
-- edilmemiş ham veri.
-- ============================================================

ALTER TABLE signals ADD COLUMN IF NOT EXISTS rank_combined DOUBLE PRECISION;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS rank_rsi_cross DOUBLE PRECISION;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS rank_z_confluence DOUBLE PRECISION;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS rank_r_score DOUBLE PRECISION;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS rank_aligned BOOLEAN;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS rank_alignment_count INTEGER;
