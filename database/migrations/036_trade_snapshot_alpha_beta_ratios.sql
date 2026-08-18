-- ============================================================
-- Migration 036: trade_snapshots'a alpha/beta + finansal oranlar +
-- Sinyal Mumu Ratioları eklendi
--
-- 17 Ağu 2026 — totalamount_rank1 (şu an TEK aktif strateji) hiç Signal
-- satırı oluşturmadığı için (open_direct yolu) alpha/beta/finansal oran/
-- mum-oranı ailesi bu strateji için HİÇ kaydedilmiyordu. trade_snapshots
-- zaten stratejiden bağımsız (status='open' bazlı, signal_id gerektirmiyor)
-- çalıştığı için buraya eklenince TÜM açık pozisyonlar için otomatik devreye
-- giriyor.
--
-- KASITLI OLARAK normalized_composite/scaled_avg_normalized DAHIL DEĞİL —
-- bu skorlar normalize_series'in global min-max "50'ye yapışma" bug'ını
-- taşıyor (bkz. memory/project_financial_metrics_pine_mismatch_10agu.md),
-- bug çözülmeden buggy veri biriktirmemek için ham (normalize edilmemiş)
-- oranlar tutuluyor.
-- ============================================================

ALTER TABLE trade_snapshots
    ADD COLUMN IF NOT EXISTS alpha DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS beta DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS sharpe_ratio DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS sortino_ratio DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS calmar_ratio DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS treynor_ratio DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS information_ratio DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS signal_rsi_change DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS signal_mfi_change DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS signal_macd_change DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS signal_obv DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS body_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS wick_pct DOUBLE PRECISION;
