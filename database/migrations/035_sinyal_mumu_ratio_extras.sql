-- ============================================================
-- Migration 035: Sinyal Mumu Ratioları'nın kalan metrikleri
-- (RSI/MFI/MACD değişimi + OBV)
--
-- 13 Ağu 2026 — Hoca Telegram külliyatındaki "Sinyal Mumu Ratioları" (S29)
-- Pine kodunun geri kalan 4 metriği: RSI(14) ve MFI(14)'ün sinyal
-- barındaki bar-to-bar değişimi (ta.change), MACD çizgisinin RSI'sinin
-- (ta.rsi(macdLine,14)) değişimi, ve OBV seviyesi. calculate_mfi/
-- calculate_obv daha önce tanımlı ama hiç çağrılmayan ("ölü") fonksiyonlardı
-- (bkz. DB_KOLONLARI.md envanteri) — burada ilk kez kullanıma alındı.
-- ============================================================

ALTER TABLE signals
    ADD COLUMN IF NOT EXISTS signal_rsi_change DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS signal_mfi_change DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS signal_macd_change DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS signal_obv DOUBLE PRECISION;

ALTER TABLE paper_trades
    ADD COLUMN IF NOT EXISTS signal_rsi_change DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS signal_mfi_change DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS signal_macd_change DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS signal_obv DOUBLE PRECISION;
