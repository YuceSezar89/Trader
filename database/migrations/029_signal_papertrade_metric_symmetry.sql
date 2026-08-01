-- ============================================================
-- Migration 029: signals ↔ paper_trades metrik simetrisi
--
-- Amaç: aynı anda hesaplanan ama sadece TEK tabloya yazılan metrikleri
-- her iki tabloya da kolon olarak ekleyip sorgulanabilir hale getirmek
-- (1-2 Ağu 2026, kullanıcı isteği — "metrikler dağınık" toparlama,
-- Senaryo A). SADECE EKLEME + 1 RENAME — mevcut kolonlara/veriye
-- dokunulmuyor, DROP yok.
--
-- signals'a eklenenler: şu an sadece paper_trades'te olan piyasa
-- bağlamı/ML alanları (signal_processor.py'de zaten hesaplanıyor,
-- sadece enriched_signal dict'ine yazılmıyordu).
--
-- paper_trades'e eklenenler: şu an sadece signals'ta olan sinyal
-- kalite/sıralama metrikleri (paper_trade_manager.py zaten signal_id
-- üzerinden Signal satırını tekrar okuyor — devisso_score gibi, aynı
-- okumadan bu alanlar da kopyalanacak).
--
-- vlt_score → volat_score RENAME: trade_snapshots'ta signals/
-- paper_trades'teki "volat_score" ile aynı kavram farklı isimle
-- duruyordu.
-- ============================================================

ALTER TABLE signals
    ADD COLUMN IF NOT EXISTS regime_trend VARCHAR(20),
    ADD COLUMN IF NOT EXISTS volatility_regime VARCHAR(20),
    ADD COLUMN IF NOT EXISTS btc_z_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS btc_trend VARCHAR(20),
    ADD COLUMN IF NOT EXISTS funding_rate DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS hour_utc SMALLINT,
    ADD COLUMN IF NOT EXISTS day_of_week SMALLINT;

ALTER TABLE paper_trades
    ADD COLUMN IF NOT EXISTS alpha DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS beta DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS sharpe_ratio DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS sortino_ratio DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS calmar_ratio DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS information_ratio DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS vpmv_pre_proxy DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS vpmv_pre_total DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS vp_score_real DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS market_structure VARCHAR(10),
    ADD COLUMN IF NOT EXISTS fvg_tfs VARCHAR(40),
    ADD COLUMN IF NOT EXISTS candle_pattern VARCHAR(100),
    ADD COLUMN IF NOT EXISTS rank_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS vs_btc DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS rank_combined DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS rank_rsi_cross DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS rank_z_confluence DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS rank_r_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS rank_aligned BOOLEAN,
    ADD COLUMN IF NOT EXISTS rank_alignment_count INTEGER,
    ADD COLUMN IF NOT EXISTS ha_ultra_confirm SMALLINT,
    ADD COLUMN IF NOT EXISTS vol_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mom_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS volat_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS price_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS candle_kategori VARCHAR(20),
    ADD COLUMN IF NOT EXISTS all_up BOOLEAN,
    ADD COLUMN IF NOT EXISTS sl_multiplier DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS tp_multiplier DOUBLE PRECISION;

ALTER TABLE trade_snapshots RENAME COLUMN vlt_score TO volat_score;
