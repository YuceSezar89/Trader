-- ============================================================
-- Migration 022: İşlem Analizi — entry_features (JSONB) + trade_snapshots
-- (19 Tem 2026)
--
-- paper_trades zaten 40+ kolona çıkmıştı, her yeni metrik için migration
-- yapmak sürdürülemez hale gelmişti — signal_processor.py'nin sinyal
-- açılışında hesapladığı 30+ metriğin çoğu (VPMV bileşenleri, SMC, finansal
-- oranlar, all_up, vb.) hiç kaydedilmiyordu. Endüstri standardı deseni:
-- giriş-anı özellikleri için TEK bir JSONB "feature snapshot" kolonu
-- (gelecekte yeni metrik eklemek migration gerektirmez).
--
-- Sürekli/periyodik izleme (CVD/VP/VPMV/SMC — 5dk'da bir, pozisyon açıkken)
-- için ayrı bir zaman-serisi hypertable'ı — signals tablosuyla aynı desen.
-- ============================================================

ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS entry_features JSONB;

CREATE TABLE IF NOT EXISTS trade_snapshots (
    id BIGSERIAL,
    trade_id INTEGER NOT NULL REFERENCES paper_trades(id) ON DELETE CASCADE,
    symbol VARCHAR(30) NOT NULL,
    taken_at TIMESTAMP NOT NULL DEFAULT now(),
    price FLOAT,
    cvd_slope FLOAT,
    vp_buy FLOAT,
    vp_sell FLOAT,
    vp_score FLOAT,
    vol_score FLOAT,
    mom_score FLOAT,
    vlt_score FLOAT,
    price_score FLOAT,
    vpmv_combined FLOAT,
    smc_market_structure VARCHAR(20),
    PRIMARY KEY (id, taken_at)
);

CREATE INDEX IF NOT EXISTS idx_trade_snapshots_trade_id ON trade_snapshots(trade_id, taken_at DESC);

SELECT create_hypertable(
    'trade_snapshots', 'taken_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);
