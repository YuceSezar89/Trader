-- =============================================================================
-- SIGNAL RANK SCORE + VS BTC
-- =============================================================================
-- Güç Sıralaması panelinin (desktop/workers/ranking_worker.py) her zaten
-- hesapladığı ama hiçbir yere kaydetmediği kesitsel yüzdelik sıralamayı
-- (rank_score, tüm takip edilen sembollere göre percentile) ve BTC'ye göre
-- farkı (vs_btc) artık sinyal açılışında da kaydediyoruz — devisso_score/
-- z_score_entry ile aynı desen, geçmişe dönük analiz/backtest için.
--
-- ÖNEMLİ SINIRLAMA: ranking:snapshot Redis key'ini SADECE masaüstü uygulaması
-- (ranking_worker.py, bir QThread) yazıyor — backend (signal_service.py/
-- live_data_manager.py) bunu üretmiyor. Masaüstü uygulaması kapalıyken açılan
-- sinyallerde bu iki kolon NULL kalır (TTL=600s, worker 180sn'de bir yazıyor).

ALTER TABLE signals ADD COLUMN IF NOT EXISTS rank_score FLOAT;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS vs_btc FLOAT;
