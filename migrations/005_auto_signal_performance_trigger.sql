-- =============================================================================
-- AUTO SIGNAL PERFORMANCE TRIGGER
-- =============================================================================
-- Yeni sinyal eklendiğinde otomatik olarak signal_performance kaydı oluşturur
-- is_calculated = FALSE olarak başlar, batch job daha sonra günceller

-- Trigger fonksiyonu
CREATE OR REPLACE FUNCTION auto_create_signal_performance()
RETURNS TRIGGER AS $$
BEGIN
    -- Yeni sinyal için signal_performance kaydı oluştur
    INSERT INTO signal_performance (
        signal_id,
        entry_price,
        entry_timestamp,
        atr_at_entry,
        interval,
        is_calculated,
        calculation_attempts,
        created_at,
        updated_at
    ) VALUES (
        NEW.id,
        NEW.price,
        NEW.timestamp,
        COALESCE(NEW.atr, 0),  -- ATR yoksa 0
        NEW.interval,
        FALSE,  -- Henüz hesaplanmadı
        0,      -- Henüz deneme yapılmadı
        NOW(),
        NOW()
    );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger oluştur
DROP TRIGGER IF EXISTS trigger_auto_signal_performance ON signals;

CREATE TRIGGER trigger_auto_signal_performance
    AFTER INSERT ON signals
    FOR EACH ROW
    EXECUTE FUNCTION auto_create_signal_performance();

-- =============================================================================
-- VERIFICATION
-- =============================================================================
-- Trigger'ı test et:
-- INSERT INTO signals (symbol, timestamp, signal_type, interval, price) 
-- VALUES ('TEST', NOW(), 'TEST_LONG', '5m', 100.0);
-- 
-- SELECT * FROM signal_performance WHERE signal_id = (SELECT id FROM signals WHERE symbol = 'TEST' ORDER BY id DESC LIMIT 1);
-- 
-- Cleanup:
-- DELETE FROM signals WHERE symbol = 'TEST';
