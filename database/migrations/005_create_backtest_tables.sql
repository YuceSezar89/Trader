-- Migration 005: Backtest ve Paper Trading Tabloları
-- Created: 2025-10-14

-- Backtest sonuçları tablosu
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    
    -- Test parametreleri
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    initial_balance DECIMAL(18, 2) DEFAULT 10000,
    
    -- Performans metrikleri
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 2),
    
    -- Finansal metrikler
    total_pnl DECIMAL(18, 2),
    total_pnl_percentage DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    profit_factor DECIMAL(10, 4),
    
    -- Risk metrikleri
    avg_win DECIMAL(18, 2),
    avg_loss DECIMAL(18, 2),
    max_consecutive_losses INTEGER DEFAULT 0,
    max_consecutive_wins INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- Backtest işlemleri tablosu
CREATE TABLE IF NOT EXISTS backtest_trades (
    id SERIAL PRIMARY KEY,
    backtest_id INTEGER REFERENCES backtest_results(id) ON DELETE CASCADE,
    
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- 'BUY' veya 'SELL'
    
    -- Fiyat bilgileri
    entry_price DECIMAL(18, 8) NOT NULL,
    exit_price DECIMAL(18, 8),
    quantity DECIMAL(18, 8) NOT NULL,
    
    -- PnL
    pnl DECIMAL(18, 8),
    pnl_percentage DECIMAL(10, 4),
    
    -- Risk yönetimi
    stop_loss DECIMAL(18, 8),
    take_profit DECIMAL(18, 8),
    
    -- Sinyal bilgileri
    signal_id INTEGER,
    vpm_score DECIMAL(10, 4),
    signal_strength INTEGER,
    
    -- Zaman bilgileri
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    timeframe VARCHAR(10) NOT NULL,
    
    -- Durum
    status VARCHAR(20) DEFAULT 'CLOSED',  -- 'OPEN', 'CLOSED', 'CANCELLED'
    exit_reason VARCHAR(50),  -- 'TAKE_PROFIT', 'STOP_LOSS', 'SIGNAL', 'TIMEOUT'
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Paper trading işlemleri tablosu
CREATE TABLE IF NOT EXISTS paper_trades (
    id SERIAL PRIMARY KEY,
    
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- 'BUY' veya 'SELL'
    
    -- Fiyat bilgileri
    entry_price DECIMAL(18, 8) NOT NULL,
    exit_price DECIMAL(18, 8),
    quantity DECIMAL(18, 8) NOT NULL,
    
    -- PnL
    pnl DECIMAL(18, 8),
    pnl_percentage DECIMAL(10, 4),
    
    -- Risk yönetimi
    stop_loss DECIMAL(18, 8),
    take_profit DECIMAL(18, 8),
    
    -- Sinyal bilgileri (foreign key yok - sadece referans)
    signal_id INTEGER,
    vpm_score DECIMAL(10, 4),
    signal_strength INTEGER,
    
    -- Zaman bilgileri
    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    exit_time TIMESTAMP,
    timeframe VARCHAR(10) NOT NULL,
    
    -- Durum
    status VARCHAR(20) DEFAULT 'OPEN',  -- 'OPEN', 'CLOSED', 'CANCELLED'
    exit_reason VARCHAR(50),
    notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- İndeksler
-- İndeksler
CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy ON backtest_results(strategy_name);
CREATE INDEX IF NOT EXISTS idx_backtest_results_dates ON backtest_results(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_backtest_id ON backtest_trades(backtest_id);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_symbol ON backtest_trades(symbol);

-- Paper trades indeksleri (tablo varsa)
DO $$
BEGIN
    IF EXISTS (SELECT FROM pg_tables WHERE tablename = 'paper_trades') THEN
        CREATE INDEX IF NOT EXISTS idx_paper_trades_symbol ON paper_trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status);
        CREATE INDEX IF NOT EXISTS idx_paper_trades_entry_time ON paper_trades(entry_time);
    END IF;
END $$;

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_backtest_results_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER backtest_results_updated_at
    BEFORE UPDATE ON backtest_results
    FOR EACH ROW
    EXECUTE FUNCTION update_backtest_results_updated_at();

CREATE OR REPLACE FUNCTION update_paper_trades_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER paper_trades_updated_at
    BEFORE UPDATE ON paper_trades
    FOR EACH ROW
    EXECUTE FUNCTION update_paper_trades_updated_at();

-- Başarı mesajı
DO $$
BEGIN
    RAISE NOTICE 'Migration 005: Backtest tabloları başarıyla oluşturuldu!';
END $$;
