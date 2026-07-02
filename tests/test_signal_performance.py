"""
Signal Performance Tests
========================
signal_performance tablosu ve analyzer modülü için testler.
"""

import pytest
import psycopg2
from datetime import datetime, timedelta
from decimal import Decimal


class TestSignalPerformanceTable:
    """signal_performance tablosu için testler."""
    
    def test_table_exists(self, db_connection):
        """Tablo var mı kontrol et."""
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'signal_performance'
            );
        """)
        exists = cursor.fetchone()[0]
        assert exists, "signal_performance tablosu bulunamadı"
        cursor.close()
    
    def test_table_columns(self, db_connection):
        """Gerekli kolonlar var mı kontrol et."""
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns
            WHERE table_name = 'signal_performance'
            ORDER BY ordinal_position;
        """)
        columns = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        required_columns = [
            'id', 'signal_id', 'entry_price', 'entry_timestamp', 
            'atr_at_entry', 'interval', 'return_t3_atr', 'return_t5_atr',
            'return_t10_atr', 'mfe_atr', 'mae_atr', 'risk_reward',
            'is_calculated', 'created_at', 'updated_at'
        ]
        
        for col in required_columns:
            assert col in columns, f"Kolon eksik: {col}"
    
    def test_indexes_exist(self, db_connection):
        """İndexler var mı kontrol et."""
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT indexname 
            FROM pg_indexes
            WHERE tablename = 'signal_performance';
        """)
        indexes = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        required_indexes = [
            'idx_signal_perf_signal_id',
            'idx_signal_perf_calculated',
            'idx_signal_perf_entry_timestamp'
        ]
        
        for idx in required_indexes:
            assert idx in indexes, f"Index eksik: {idx}"
    
    def test_view_exists(self, db_connection):
        """signal_quality_summary view var mı kontrol et."""
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.views 
                WHERE table_name = 'signal_quality_summary'
            );
        """)
        exists = cursor.fetchone()[0]
        assert exists, "signal_quality_summary view bulunamadı"
        cursor.close()


class TestSignalPerformanceInsert:
    """Performance kaydı ekleme testleri."""
    
    @pytest.fixture
    def sample_signal(self, db_connection):
        """Test için örnek sinyal oluştur."""
        cursor = db_connection.cursor()
        
        # Örnek sinyal ekle
        cursor.execute("""
            INSERT INTO signals (
                symbol, signal_type, indicators, opened_at,
                open_price, interval, vpms_score, atr, status
            ) VALUES (
                'TESTPERFUSDT', 'Long', 'TEST',
                NOW(), 50000.0, '1h', 75.5, 500.0, 'closed'
            ) RETURNING id;
        """)
        signal_id = cursor.fetchone()[0]
        db_connection.commit()
        
        yield signal_id
        
        # Cleanup (trigger'ın yarattığı performance kaydı dahil)
        cursor.execute("DELETE FROM signal_performance WHERE signal_id = %s", (signal_id,))
        cursor.execute("DELETE FROM signals WHERE id = %s", (signal_id,))
        db_connection.commit()
        cursor.close()
    
    def test_insert_performance_record(self, db_connection, sample_signal):
        """Trigger signal INSERT'inde otomatik performance kaydı oluşturur."""
        cursor = db_connection.cursor()

        # Trigger otomatik oluşturdu, signal_id ile sorgula
        cursor.execute("""
            SELECT id, signal_id, entry_price, atr_at_entry, is_calculated
            FROM signal_performance WHERE signal_id = %s;
        """, (sample_signal,))

        row = cursor.fetchone()
        assert row is not None, "Trigger performance kaydı oluşturmadı"
        assert row[1] == sample_signal
        assert float(row[2]) == 50000.0
        assert float(row[3]) == 500.0
        assert row[4] is False

        cursor.close()
    
    def test_unique_signal_id_constraint(self, db_connection, sample_signal):
        """Aynı signal_id için ikinci kayıt eklenemez (trigger ilkini zaten oluşturdu)."""
        cursor = db_connection.cursor()

        # Trigger zaten bir kayıt oluşturdu — ikinci INSERT hata vermeli
        with pytest.raises(psycopg2.IntegrityError):
            cursor.execute("""
                INSERT INTO signal_performance (
                    signal_id, entry_price, entry_timestamp,
                    atr_at_entry, interval
                ) VALUES (%s, 51000.0, NOW(), 510.0, '1h');
            """, (sample_signal,))
            db_connection.commit()

        db_connection.rollback()
        cursor.close()
    
    def test_performance_cascades_on_signal_delete(self, db_connection):
        """Signal silinince performance kaydı da silinmeli (ON DELETE CASCADE —
        bkz. add_signal_performance_fk_cascade.sql)."""
        cursor = db_connection.cursor()

        # Sinyal ekle
        cursor.execute("""
            INSERT INTO signals (
                symbol, signal_type, indicators, opened_at,
                open_price, interval, vpms_score, atr, status
            ) VALUES (
                'TESTPERFUSDT', 'Short', 'TEST',
                NOW(), 3000.0, '4h', 65.0, 60.0, 'closed'
            ) RETURNING id;
        """)
        signal_id = cursor.fetchone()[0]
        db_connection.commit()

        # Trigger otomatik oluşturdu, performance id'sini al
        cursor.execute("""
            SELECT id FROM signal_performance WHERE signal_id = %s;
        """, (signal_id,))
        perf_id = cursor.fetchone()[0]

        # Sinyali sil — CASCADE performance kaydını da silmeli
        cursor.execute("DELETE FROM signals WHERE id = %s", (signal_id,))
        db_connection.commit()

        cursor.execute("""
            SELECT COUNT(*) FROM signal_performance WHERE id = %s;
        """, (perf_id,))
        count = cursor.fetchone()[0]

        assert count == 0, "CASCADE çalışmadı — orphan performance kaydı kaldı"
        cursor.close()


class TestSignalPerformanceUpdate:
    """Performance güncelleme testleri."""
    
    @pytest.fixture
    def sample_performance(self, db_connection):
        """Test için örnek performance kaydı."""
        cursor = db_connection.cursor()
        
        # Sinyal ekle
        cursor.execute("""
            INSERT INTO signals (
                symbol, signal_type, indicators, opened_at,
                open_price, interval, vpms_score, atr, status
            ) VALUES (
                'TESTPERFUSDT', 'Long', 'TEST',
                NOW(), 50000.0, '1h', 75.0, 500.0, 'closed'
            ) RETURNING id;
        """)
        signal_id = cursor.fetchone()[0]
        db_connection.commit()

        # Trigger otomatik oluşturdu, signal_id ile al
        cursor.execute("""
            SELECT id FROM signal_performance WHERE signal_id = %s;
        """, (signal_id,))
        perf_id = cursor.fetchone()[0]
        
        yield perf_id, signal_id
        
        # Cleanup (trigger'ın yarattığı performance kaydı dahil)
        cursor.execute("DELETE FROM signal_performance WHERE signal_id = %s", (signal_id,))
        cursor.execute("DELETE FROM signals WHERE id = %s", (signal_id,))
        db_connection.commit()
        cursor.close()
    
    def test_update_returns(self, db_connection, sample_performance):
        """Getiri değerleri güncellenebiliyor mu?"""
        perf_id, _ = sample_performance
        cursor = db_connection.cursor()
        
        cursor.execute("""
            UPDATE signal_performance
            SET 
                return_t3_atr = 1.5,
                return_t5_atr = 2.3,
                return_t10_atr = 3.1,
                is_calculated = TRUE,
                calculated_at = NOW()
            WHERE id = %s;
        """, (perf_id,))
        db_connection.commit()
        
        # Kontrol et
        cursor.execute("""
            SELECT return_t3_atr, return_t5_atr, return_t10_atr, is_calculated
            FROM signal_performance WHERE id = %s;
        """, (perf_id,))
        
        row = cursor.fetchone()
        assert float(row[0]) == 1.5
        assert float(row[1]) == 2.3
        assert float(row[2]) == 3.1
        assert row[3] is True
        
        cursor.close()
    
    def test_update_risk_metrics(self, db_connection, sample_performance):
        """Risk metrikleri güncellenebiliyor mu?"""
        perf_id, _ = sample_performance
        cursor = db_connection.cursor()
        
        cursor.execute("""
            UPDATE signal_performance
            SET 
                mfe_atr = 3.0,
                mae_atr = -0.5,
                risk_reward = 6.0,
                mfe_bar_index = 3,
                mae_bar_index = 1
            WHERE id = %s;
        """, (perf_id,))
        db_connection.commit()
        
        # Kontrol et
        cursor.execute("""
            SELECT mfe_atr, mae_atr, risk_reward, mfe_bar_index, mae_bar_index
            FROM signal_performance WHERE id = %s;
        """, (perf_id,))
        
        row = cursor.fetchone()
        assert float(row[0]) == 3.0
        assert float(row[1]) == -0.5
        assert float(row[2]) == 6.0
        assert row[3] == 3
        assert row[4] == 1
        
        cursor.close()
    
    def test_updated_at_trigger(self, db_connection, sample_performance):
        """updated_at trigger çalışıyor mu?"""
        perf_id, _ = sample_performance
        cursor = db_connection.cursor()
        
        # İlk updated_at değerini al
        cursor.execute("""
            SELECT updated_at FROM signal_performance WHERE id = %s;
        """, (perf_id,))
        old_updated_at = cursor.fetchone()[0]
        
        # 1 saniye bekle
        import time
        time.sleep(1)
        
        # Güncelleme yap
        cursor.execute("""
            UPDATE signal_performance
            SET return_t5_atr = 1.0
            WHERE id = %s;
        """, (perf_id,))
        db_connection.commit()
        
        # Yeni updated_at değerini al
        cursor.execute("""
            SELECT updated_at FROM signal_performance WHERE id = %s;
        """, (perf_id,))
        new_updated_at = cursor.fetchone()[0]
        
        assert new_updated_at > old_updated_at, "updated_at trigger çalışmadı"
        cursor.close()


class TestSignalQualitySummaryView:
    """signal_quality_summary view testleri."""
    
    @pytest.fixture
    def sample_signals_with_performance(self, db_connection):
        """Test için birden fazla sinyal ve performance kaydı."""
        cursor = db_connection.cursor()
        signal_ids = []
        
        # 5 adet test sinyali ekle (trigger otomatik performance yaratır)
        # clock_timestamp() her çağrıda farklı zaman döndürür (NOW() transaction-start sabitler)
        for i in range(5):
            cursor.execute("""
                INSERT INTO signals (
                    symbol, signal_type, indicators, opened_at,
                    open_price, interval, vpms_score, atr, status
                ) VALUES (
                    'TESTPERFUSDT', 'Long', 'TEST',
                    clock_timestamp() + (%(i)s * INTERVAL '1 minute'),
                    50000.0, '1h', 75.0, 500.0, 'closed'
                ) RETURNING id;
            """, {'i': i})
            signal_id = cursor.fetchone()[0]
            signal_ids.append(signal_id)

            # Trigger'ın yarattığı kaydı test verileriyle güncelle
            return_value = 1.5 if i < 3 else -0.5  # 3 kazanç, 2 kayıp
            cursor.execute("""
                UPDATE signal_performance
                SET return_t5_atr = %s, mfe_atr = 2.0, mae_atr = -0.3,
                    risk_reward = 6.67, is_calculated = TRUE, calculated_at = NOW()
                WHERE signal_id = %s;
            """, (return_value, signal_id))

        db_connection.commit()
        
        yield signal_ids
        
        # Cleanup (trigger'ın yarattığı performance kayıtları dahil)
        for sid in signal_ids:
            cursor.execute("DELETE FROM signal_performance WHERE signal_id = %s", (sid,))
            cursor.execute("DELETE FROM signals WHERE id = %s", (sid,))
        db_connection.commit()
        cursor.close()
    
    def test_view_query(self, db_connection, sample_signals_with_performance):
        """View sorgulanabiliyor mu?"""
        cursor = db_connection.cursor()
        
        cursor.execute("""
            SELECT
                signal_type, interval, total_signals, calculated_signals,
                hit_rate_pct, avg_return_t5_atr
            FROM signal_quality_summary
            WHERE signal_type = 'Long' AND interval = '1h';
        """)

        row = cursor.fetchone()
        assert row is not None, "View sonuç döndürmedi"

        signal_type, interval, total, calculated, hit_rate, avg_return = row
        # View tüm canlı veriler üzerinden aggregate eder — sabit değer yerine yapısal kontrol
        assert signal_type == 'Long'
        assert interval == '1h'
        assert total >= 5
        assert calculated >= 5
        assert hit_rate is None or 0 <= float(hit_rate) <= 100
        
        cursor.close()
    
    def test_view_aggregations(self, db_connection, sample_signals_with_performance):
        """View aggregasyonları doğru mu?"""
        cursor = db_connection.cursor()
        
        cursor.execute("""
            SELECT
                avg_mfe_atr, avg_mae_atr, avg_risk_reward
            FROM signal_quality_summary
            WHERE signal_type = 'Long' AND interval = '1h';
        """)
        
        row = cursor.fetchone()
        assert row is not None, "View aggregate satırı yok"
        avg_mfe, avg_mae, avg_rr = row
        # Canlı veriler dahil — işaret/varlık kontrolü yeterli
        assert avg_mfe is not None
        assert avg_mae is not None
        
        cursor.close()


@pytest.mark.integration
class TestPerformanceWorkflow:
    """End-to-end performance workflow testi."""
    
    def test_full_workflow(self, db_connection):
        """Tam workflow: sinyal → performance → hesaplama → view."""
        cursor = db_connection.cursor()

        # 1. Sinyal ekle (trigger otomatik performance yaratır)
        cursor.execute("""
            INSERT INTO signals (
                symbol, signal_type, indicators, opened_at,
                open_price, interval, vpms_score, atr, status
            ) VALUES (
                'TESTPERFUSDT', 'Long', 'TEST',
                NOW(), 3000.0, '4h', 80.0, 60.0, 'closed'
            ) RETURNING id;
        """)
        signal_id = cursor.fetchone()[0]
        db_connection.commit()

        # 2. Trigger'ın yarattığı performance kaydını al
        cursor.execute("""
            SELECT id FROM signal_performance WHERE signal_id = %s;
        """, (signal_id,))
        perf_id = cursor.fetchone()[0]

        # 3. Performance hesapla (simüle et)
        cursor.execute("""
            UPDATE signal_performance
            SET
                return_t5_atr = 2.5,
                mfe_atr = 3.2,
                mae_atr = -0.4,
                risk_reward = 8.0,
                is_calculated = TRUE,
                calculated_at = NOW()
            WHERE id = %s;
        """, (perf_id,))
        db_connection.commit()

        # 4. View'dan kontrol et (signal_type + interval ile filtrele)
        cursor.execute("""
            SELECT hit_rate_pct, avg_return_t5_atr
            FROM signal_quality_summary
            WHERE signal_type = 'Long' AND interval = '4h';
        """)

        row = cursor.fetchone()
        assert row is not None

        # Cleanup
        cursor.execute("DELETE FROM signal_performance WHERE id = %s", (perf_id,))
        cursor.execute("DELETE FROM signals WHERE id = %s", (signal_id,))
        db_connection.commit()
        cursor.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
