-- =====================================================
-- Signal Performance Tracking Table
-- =====================================================
-- Bu tablo her sinyalin T+N bar sonrası performansını
-- ve risk metriklerini (MFE/MAE) saklar.
-- =====================================================

-- Mevcut tabloyu kaldır (varsa)
DROP TABLE IF EXISTS signal_performance CASCADE;

-- Performance tablosunu oluştur
CREATE TABLE signal_performance (
    id SERIAL PRIMARY KEY,
    signal_id INTEGER NOT NULL,  -- Foreign key yok (signals tablosunda PK yok)
    
    -- ==================== SNAPSHOT (T0) ====================
    entry_price DECIMAL(20, 8) NOT NULL,
    entry_timestamp TIMESTAMP NOT NULL,
    atr_at_entry DECIMAL(20, 8) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    
    -- ==================== T+N GETİRİLERİ (ATR Normalize) ====================
    -- NULL = henüz hesaplanmadı
    return_t3_atr DECIMAL(10, 4),      -- 3 bar sonra getiri
    return_t5_atr DECIMAL(10, 4),      -- 5 bar sonra getiri
    return_t10_atr DECIMAL(10, 4),     -- 10 bar sonra getiri
    
    -- Ham getiriler (yüzde)
    return_t3_pct DECIMAL(10, 4),
    return_t5_pct DECIMAL(10, 4),
    return_t10_pct DECIMAL(10, 4),
    
    -- ==================== RİSK METRİKLERİ ====================
    mfe_atr DECIMAL(10, 4),            -- Max Favorable Excursion (ATR)
    mae_atr DECIMAL(10, 4),            -- Max Adverse Excursion (ATR)
    risk_reward DECIMAL(10, 4),        -- MFE/MAE oranı
    
    -- MFE/MAE zamanlaması (kaçıncı barda gerçekleşti)
    mfe_bar_index INTEGER,
    mae_bar_index INTEGER,
    
    -- ==================== HESAPLAMA DURUMU ====================
    is_calculated BOOLEAN DEFAULT FALSE,
    calculation_attempts INTEGER DEFAULT 0,
    last_calculation_error TEXT,
    
    -- ==================== ZAMAN DAMGALARI ====================
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    calculated_at TIMESTAMP,
    
    -- ==================== CONSTRAINTS ====================
    UNIQUE(signal_id),
    CHECK (atr_at_entry > 0),
    CHECK (calculation_attempts >= 0)
);

-- ==================== İNDEXLER ====================
CREATE INDEX idx_signal_perf_signal_id ON signal_performance(signal_id);
CREATE INDEX idx_signal_perf_calculated ON signal_performance(is_calculated);
CREATE INDEX idx_signal_perf_entry_timestamp ON signal_performance(entry_timestamp);
CREATE INDEX idx_signal_perf_interval ON signal_performance(interval);
CREATE INDEX idx_signal_perf_updated_at ON signal_performance(updated_at);

-- Composite index: Henüz hesaplanmamış ve son 24 saatteki sinyaller
CREATE INDEX idx_signal_perf_pending ON signal_performance(is_calculated, entry_timestamp) 
WHERE is_calculated = FALSE;

-- ==================== TRIGGER: updated_at otomatik güncelleme ====================
CREATE OR REPLACE FUNCTION update_signal_performance_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_signal_performance_updated_at
    BEFORE UPDATE ON signal_performance
    FOR EACH ROW
    EXECUTE FUNCTION update_signal_performance_timestamp();

-- ==================== YARDIMCI VIEW: Sinyal Kalite Özeti ====================
CREATE OR REPLACE VIEW signal_quality_summary AS
SELECT 
    s.signal_type,
    s.interval,
    COUNT(*) as total_signals,
    COUNT(CASE WHEN sp.is_calculated THEN 1 END) as calculated_signals,
    
    -- T+5 metrikleri
    AVG(sp.return_t5_atr) as avg_return_t5_atr,
    STDDEV(sp.return_t5_atr) as std_return_t5_atr,
    
    -- Hit rate (T+5 > 0)
    ROUND(
        COUNT(CASE WHEN sp.return_t5_atr > 0 THEN 1 END)::NUMERIC / 
        NULLIF(COUNT(CASE WHEN sp.is_calculated THEN 1 END), 0) * 100, 
        2
    ) as hit_rate_pct,
    
    -- Risk metrikleri
    AVG(sp.mfe_atr) as avg_mfe_atr,
    AVG(sp.mae_atr) as avg_mae_atr,
    AVG(sp.risk_reward) as avg_risk_reward,
    
    -- Sharpe-like
    CASE 
        WHEN STDDEV(sp.return_t5_atr) > 0 THEN
            AVG(sp.return_t5_atr) / STDDEV(sp.return_t5_atr)
        ELSE NULL
    END as sharpe_like,
    
    -- Zaman aralığı
    MIN(sp.entry_timestamp) as first_signal,
    MAX(sp.entry_timestamp) as last_signal
FROM 
    signals s
    LEFT JOIN signal_performance sp ON s.id = sp.signal_id
WHERE 
    sp.is_calculated = TRUE
GROUP BY 
    s.signal_type, s.interval
ORDER BY 
    hit_rate_pct DESC NULLS LAST;

-- ==================== YORUM ====================
COMMENT ON TABLE signal_performance IS 'Sinyal performans metrikleri: T+N getirileri, MFE/MAE, risk/reward';
COMMENT ON COLUMN signal_performance.return_t5_atr IS 'T+5 bar getiri (ATR normalize): pozitif=kazanç, negatif=kayıp';
COMMENT ON COLUMN signal_performance.mfe_atr IS 'Max Favorable Excursion: En yüksek kazanç noktası (ATR)';
COMMENT ON COLUMN signal_performance.mae_atr IS 'Max Adverse Excursion: En kötü düşüş noktası (ATR)';
COMMENT ON COLUMN signal_performance.risk_reward IS 'MFE/MAE oranı: yüksek=iyi sinyal';

-- ==================== BAŞARILI MESAJI ====================
DO $$
BEGIN
    RAISE NOTICE '✅ signal_performance tablosu başarıyla oluşturuldu';
    RAISE NOTICE '✅ İndexler eklendi';
    RAISE NOTICE '✅ Trigger kuruldu';
    RAISE NOTICE '✅ signal_quality_summary view oluşturuldu';
END $$;
