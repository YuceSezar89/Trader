-- Sinyal Lifecycle Yönetimi için Signals Tablosu Genişletme
-- Supersede yaklaşımı implementasyonu

-- Yeni kolonlar ekle
ALTER TABLE signals ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'active';
ALTER TABLE signals ADD COLUMN IF NOT EXISTS superseded_by INTEGER;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMP;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS performance_period INTEGER;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS lifecycle_end_reason VARCHAR(50);

-- Status için check constraint
ALTER TABLE signals ADD CONSTRAINT check_signal_status 
    CHECK (status IN ('active', 'superseded', 'completed', 'timeout'));

-- Lifecycle end reason için check constraint  
ALTER TABLE signals ADD CONSTRAINT check_lifecycle_reason
    CHECK (lifecycle_end_reason IN ('supersede', 'reversal', 'timeout', 'target', 'stop', 'manual'));

-- Superseded_by foreign key (self-reference)
ALTER TABLE signals ADD CONSTRAINT fk_superseded_by 
    FOREIGN KEY (superseded_by) REFERENCES signals(id);

-- Performance period hesaplama için trigger function
CREATE OR REPLACE FUNCTION calculate_performance_period()
RETURNS TRIGGER AS $$
BEGIN
    -- Eğer superseded_at set edilirse, performance_period'u hesapla (dakika cinsinden)
    IF NEW.superseded_at IS NOT NULL AND OLD.superseded_at IS NULL THEN
        NEW.performance_period = EXTRACT(EPOCH FROM (NEW.superseded_at - NEW.timestamp)) / 60;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger oluştur
DROP TRIGGER IF EXISTS trigger_calculate_performance_period ON signals;
CREATE TRIGGER trigger_calculate_performance_period
    BEFORE UPDATE ON signals
    FOR EACH ROW
    EXECUTE FUNCTION calculate_performance_period();

-- Aktif sinyaller için index (performans optimizasyonu)
CREATE INDEX IF NOT EXISTS idx_signals_active_status ON signals(symbol, interval, status) 
    WHERE status = 'active';

-- Superseded sinyaller için index
CREATE INDEX IF NOT EXISTS idx_signals_superseded ON signals(superseded_by, superseded_at) 
    WHERE status = 'superseded';

-- Mevcut tüm sinyalleri 'active' olarak işaretle (migration için)
UPDATE signals SET status = 'active' WHERE status IS NULL;

-- Yorum: Bu migration'ı çalıştırdıktan sonra:
-- 1. Yeni sinyal eklenirken supersede kontrolü yapılmalı
-- 2. Panel'de aktif sinyaller filtrelenebilir
-- 3. Performans analizi için lifecycle verileri kullanılabilir
