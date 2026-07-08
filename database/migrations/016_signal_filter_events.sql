-- =============================================================================
-- SIGNAL FILTER EVENTS
-- =============================================================================
-- SignalFilter.check() her denemeyi (kabul/red fark etmeksizin) buraya loglar.
-- Amaç: filtrenin referans noktalarını (last_short_high/last_long_low) Redis
-- cache yerine doğrudan bu tablodan, restart/çoklu-process senkron sorunu
-- olmadan okumak. signals tablosundan farkı: sadece KABUL EDİLEN sinyalleri
-- değil, REDDEDİLEN denemeleri de tutar — PineScript'in "referans her denemede
-- güncellenir" davranışını korumak için gerekli (bkz. signals/signal_filter.py).

CREATE TABLE IF NOT EXISTS signal_filter_events (
    id          BIGSERIAL PRIMARY KEY,
    symbol      VARCHAR(30)  NOT NULL,
    interval    VARCHAR(10)  NOT NULL,
    indicator   VARCHAR(50)  NOT NULL,
    signal_type VARCHAR(10)  NOT NULL,  -- 'Long' / 'Short'
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    passed      BOOLEAN NOT NULL,
    bar_time    TIMESTAMP NOT NULL,     -- denemenin ait olduğu mumun zamanı
    created_at  TIMESTAMP NOT NULL      -- kayıt anı, uygulama datetime.now() ile dolduruyor (KESİN kural, bkz. CLAUDE.md)
);

-- check()'in "bu key için en son karşıt yönlü olay" sorgusu bu index'i kullanır.
CREATE INDEX IF NOT EXISTS idx_sfe_lookup
    ON signal_filter_events (symbol, interval, indicator, signal_type, bar_time DESC);
