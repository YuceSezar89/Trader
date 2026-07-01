-- Migration: signal_performance tablosuna ON DELETE CASCADE FK ekle
-- Orphan kayıtları temizle, ardından referential integrity kur

-- 1. Orphan kayıtları sil
DELETE FROM signal_performance
WHERE NOT EXISTS (
    SELECT 1 FROM signals s WHERE s.id = signal_performance.signal_id
);

-- 2. FK constraint ekle
ALTER TABLE signal_performance
ADD CONSTRAINT fk_signal_performance_signal_id
FOREIGN KEY (signal_id)
REFERENCES signals(id)
ON DELETE CASCADE;
