"""
Signal Performance Analyzer
============================
Sinyal performans metriklerini hesaplar:
- T+N bar getirileri (ATR normalize)
- MFE/MAE (Max Favorable/Adverse Excursion)
- Risk/Reward oranı
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras

from config import Config

logger = logging.getLogger(__name__)


class SignalPerformanceAnalyzer:
    """Sinyal performans analizi için ana sınıf."""

    def __init__(self, db_connection=None):
        """
        Args:
            db_connection: Mevcut database bağlantısı (opsiyonel)
        """
        self.db = db_connection
        self.own_connection = db_connection is None

        if self.own_connection:
            self.db = self._create_connection()

    def _create_connection(self):
        """Database bağlantısı oluştur."""
        return psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
        )

    def close(self):
        """Bağlantıyı kapat (sadece kendi oluşturduğumuz bağlantı için)."""
        if self.own_connection and self.db:
            self.db.close()

    def __enter__(self):
        """Context manager desteği."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()

    # =========================================================================
    # YARDIMCI FONKSİYONLAR
    # =========================================================================

    def _interval_to_minutes(self, interval: str) -> int:
        """
        Interval string'ini dakikaya çevir.

        Args:
            interval: '1m', '5m', '15m', '1h', '4h', '1d' vb.

        Returns:
            Dakika cinsinden süre
        """
        interval = interval.lower()

        if interval.endswith("m"):
            return int(interval[:-1])
        elif interval.endswith("h"):
            return int(interval[:-1]) * 60
        elif interval.endswith("d"):
            return int(interval[:-1]) * 1440
        elif interval.endswith("w"):
            return int(interval[:-1]) * 10080
        else:
            raise ValueError(f"Geçersiz interval: {interval}")

    def _get_price_at_time(
        self, symbol: str, target_time: datetime, interval: str
    ) -> Optional[float]:
        """
        Belirli bir zamandaki fiyatı database'den çek.
        MTF interval'ler için 1m verisinden hesaplar.

        Args:
            symbol: Sembol (örn: BTCUSDT)
            target_time: Hedef zaman
            interval: Zaman dilimi

        Returns:
            Close fiyatı veya None
        """
        # MTF interval'ler için 1m verisinden hesapla
        if interval in ["5m", "15m", "1h", "4h"]:
            return self._get_price_from_1m_aggregate(symbol, target_time, interval)

        # 1m ve 15m için direkt price_data'dan çek
        cursor = self.db.cursor()

        try:
            # price_data tablosundan en yakın fiyatı çek
            cursor.execute(
                """
                SELECT close
                FROM price_data
                WHERE symbol = %s
                  AND interval = %s
                  AND timestamp <= %s
                ORDER BY timestamp DESC
                LIMIT 1;
            """,
                (symbol, interval, target_time),
            )

            result = cursor.fetchone()
            return float(result[0]) if result else None

        finally:
            cursor.close()

    def _get_price_from_1m_aggregate(
        self, symbol: str, target_time: datetime, interval: str
    ) -> Optional[float]:
        """
        1m verilerinden MTF fiyatı hesapla.

        Args:
            symbol: Sembol
            target_time: Hedef zaman
            interval: MTF interval (5m, 15m, 1h, 4h)

        Returns:
            Aggregate edilmiş close fiyatı veya None
        """
        cursor = self.db.cursor()

        try:
            # Interval'e göre kaç 1m bar gerekli
            interval_minutes = self._interval_to_minutes(interval)

            # Target time'dan geriye doğru interval_minutes kadar 1m bar al
            # Örnek: 5m için son 5 adet 1m bar
            cursor.execute(
                """
                SELECT close
                FROM price_data
                WHERE symbol = %s
                  AND interval = '1m'
                  AND timestamp <= %s
                  AND timestamp > %s - INTERVAL '%s minutes'
                ORDER BY timestamp DESC
                LIMIT 1;
            """,
                (symbol, target_time, target_time, interval_minutes),
            )

            result = cursor.fetchone()

            if result:
                return float(result[0])

            # Fallback: En yakın 1m bar'ı al
            cursor.execute(
                """
                SELECT close
                FROM price_data
                WHERE symbol = %s
                  AND interval = '1m'
                  AND timestamp <= %s
                ORDER BY timestamp DESC
                LIMIT 1;
            """,
                (symbol, target_time),
            )

            result = cursor.fetchone()
            return float(result[0]) if result else None

        finally:
            cursor.close()

    def _get_price_range(
        self, symbol: str, start_time: datetime, end_time: datetime, interval: str
    ) -> List[Tuple[float, float]]:
        """
        Belirli zaman aralığındaki high/low fiyatları çek.

        Args:
            symbol: Sembol
            start_time: Başlangıç zamanı
            end_time: Bitiş zamanı
            interval: Zaman dilimi

        Returns:
            [(high, low), ...] listesi
        """
        cursor = self.db.cursor()

        try:
            cursor.execute(
                """
                SELECT high, low
                FROM price_data
                WHERE symbol = %s
                  AND interval = %s
                  AND timestamp >= %s
                  AND timestamp <= %s
                ORDER BY timestamp ASC;
            """,
                (symbol, interval, start_time, end_time),
            )

            return cursor.fetchall()

        finally:
            cursor.close()

    # =========================================================================
    # PERFORMANS HESAPLAMA
    # =========================================================================

    def calculate_t_plus_n_return(
        self, signal_id: int, n_bars: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        T+N bar sonrası getiriyi hesapla.

        Args:
            signal_id: Sinyal ID
            n_bars: Kaç bar sonra (default: 5)

        Returns:
            {
                'return_atr': ATR normalize getiri,
                'return_pct': Yüzde getiri,
                'target_price': N bar sonraki fiyat
            }
        """
        cursor = self.db.cursor(cursor_factory=psycopg2.extras.DictCursor)

        try:
            # Signal ve performance bilgilerini çek
            cursor.execute(
                """
                SELECT 
                    s.symbol,
                    s.signal_type,
                    s.timestamp,
                    s.interval,
                    s.price as entry_price,
                    sp.atr_at_entry
                FROM signals s
                JOIN signal_performance sp ON s.id = sp.signal_id
                WHERE s.id = %s;
            """,
                (signal_id,),
            )

            signal = cursor.fetchone()
            if not signal:
                logger.warning(f"Signal {signal_id} bulunamadı")
                return None

            # N bar sonraki zamanı hesapla
            interval_minutes = self._interval_to_minutes(signal["interval"])
            target_time = signal["timestamp"] + timedelta(
                minutes=interval_minutes * n_bars
            )

            # Hedef zamandaki fiyatı çek
            target_price = self._get_price_at_time(
                signal["symbol"], target_time, signal["interval"]
            )

            if target_price is None:
                logger.warning(f"Signal {signal_id} için T+{n_bars} fiyatı bulunamadı")
                return None

            # Getiri hesapla
            entry_price = float(signal["entry_price"])
            atr = float(signal["atr_at_entry"])

            # Yön (LONG=+1, SHORT=-1)
            side = (
                1
                if signal["signal_type"] and "LONG" in signal["signal_type"].upper()
                else -1
            )

            # Yüzde getiri
            return_pct = ((target_price - entry_price) / entry_price) * 100 * side

            # ATR normalize getiri
            return_atr = ((target_price - entry_price) / atr) * side if atr > 0 else 0

            return {
                "return_atr": round(return_atr, 4),
                "return_pct": round(return_pct, 4),
                "target_price": target_price,
                "entry_price": entry_price,
                "side": side,
            }

        except Exception as e:
            logger.error(f"T+{n_bars} hesaplama hatası (signal_id={signal_id}): {e}")
            try:
                self.db.rollback()
            except Exception as rollback_err:
                logger.warning(f"Rollback başarısız: {rollback_err}")
            return None
        finally:
            cursor.close()

    def calculate_mfe_mae(
        self, signal_id: int, lookback_bars: int = 20
    ) -> Optional[Dict[str, Any]]:
        """
        MFE (Max Favorable Excursion) ve MAE (Max Adverse Excursion) hesapla.

        Args:
            signal_id: Sinyal ID
            lookback_bars: Kaç bar geriye bak (default: 20)

        Returns:
            {
                'mfe_atr': En yüksek kazanç (ATR),
                'mae_atr': En kötü düşüş (ATR),
                'risk_reward': MFE/MAE oranı,
                'mfe_bar_index': MFE'nin gerçekleştiği bar,
                'mae_bar_index': MAE'nin gerçekleştiği bar
            }
        """
        cursor = self.db.cursor(cursor_factory=psycopg2.extras.DictCursor)

        try:
            # Signal bilgilerini çek
            cursor.execute(
                """
                SELECT 
                    s.symbol,
                    s.signal_type,
                    s.timestamp,
                    s.interval,
                    s.price as entry_price,
                    sp.atr_at_entry
                FROM signals s
                JOIN signal_performance sp ON s.id = sp.signal_id
                WHERE s.id = %s;
            """,
                (signal_id,),
            )

            signal = cursor.fetchone()
            if not signal:
                return None

            # Lookback zaman aralığı
            interval_minutes = self._interval_to_minutes(signal["interval"])
            end_time = signal["timestamp"] + timedelta(
                minutes=interval_minutes * lookback_bars
            )

            # Fiyat aralığını çek
            price_range = self._get_price_range(
                signal["symbol"], signal["timestamp"], end_time, signal["interval"]
            )

            if not price_range:
                return None

            entry_price = float(signal["entry_price"])
            atr = float(signal["atr_at_entry"])

            # Yön
            side = (
                1
                if signal["signal_type"] and "LONG" in signal["signal_type"].upper()
                else -1
            )

            # MFE/MAE hesapla
            mfe = 0.0
            mae = 0.0
            mfe_bar = 0
            mae_bar = 0

            for i, (high, low) in enumerate(price_range):
                high = float(high)
                low = float(low)

                if side == 1:  # LONG
                    # En yüksek kazanç
                    favorable = (high - entry_price) / atr if atr > 0 else 0
                    if favorable > mfe:
                        mfe = favorable
                        mfe_bar = i

                    # En kötü düşüş
                    adverse = (low - entry_price) / atr if atr > 0 else 0
                    if adverse < mae:
                        mae = adverse
                        mae_bar = i
                else:  # SHORT
                    # En yüksek kazanç
                    favorable = (entry_price - low) / atr if atr > 0 else 0
                    if favorable > mfe:
                        mfe = favorable
                        mfe_bar = i

                    # En kötü düşüş
                    adverse = (entry_price - high) / atr if atr > 0 else 0
                    if adverse < mae:
                        mae = adverse
                        mae_bar = i

            # Risk/Reward
            risk_reward = abs(mfe / mae) if mae != 0 else 0

            return {
                "mfe_atr": round(mfe, 4),
                "mae_atr": round(mae, 4),
                "risk_reward": round(risk_reward, 4),
                "mfe_bar_index": mfe_bar,
                "mae_bar_index": mae_bar,
            }

        except Exception as e:
            logger.error(f"MFE/MAE hesaplama hatası (signal_id={signal_id}): {e}")
            try:
                self.db.rollback()
            except Exception as rollback_err:
                logger.warning(f"Rollback başarısız: {rollback_err}")
            return None
        finally:
            cursor.close()

    def update_signal_performance(self, signal_id: int, dry_run: bool = False) -> bool:
        """
        Tek bir sinyal için performans metriklerini hesapla ve güncelle.

        Args:
            signal_id: Sinyal ID

        Returns:
            Başarılı ise True
        """
        try:
            # T+3, T+5, T+10 getirilerini hesapla
            ret_t3 = self.calculate_t_plus_n_return(signal_id, n_bars=3)
            ret_t5 = self.calculate_t_plus_n_return(signal_id, n_bars=5)
            ret_t10 = self.calculate_t_plus_n_return(signal_id, n_bars=10)

            # MFE/MAE hesapla
            mfe_mae = self.calculate_mfe_mae(signal_id, lookback_bars=20)

            # En az biri hesaplanmışsa güncelle
            if ret_t3 or ret_t5 or ret_t10 or mfe_mae:
                # Prepare values for update
                params = (
                    ret_t3["return_atr"] if ret_t3 else None,
                    ret_t3["return_pct"] if ret_t3 else None,
                    ret_t5["return_atr"] if ret_t5 else None,
                    ret_t5["return_pct"] if ret_t5 else None,
                    ret_t10["return_atr"] if ret_t10 else None,
                    ret_t10["return_pct"] if ret_t10 else None,
                    mfe_mae["mfe_atr"] if mfe_mae else None,
                    mfe_mae["mae_atr"] if mfe_mae else None,
                    mfe_mae["risk_reward"] if mfe_mae else None,
                    mfe_mae["mfe_bar_index"] if mfe_mae else None,
                    mfe_mae["mae_bar_index"] if mfe_mae else None,
                    signal_id,
                )

                if dry_run:
                    # Log what we would write and skip DB changes
                    logger.info(
                        f"[DRY-RUN] Signal {signal_id} would be updated with: return_t3_pct={params[1]}, return_t5_pct={params[3]}, return_t10_pct={params[5]}, mfe_atr={params[6]}, mae_atr={params[7]}, risk_reward={params[8]}"
                    )
                    return True

                cursor = self.db.cursor()
                try:
                    cursor.execute(
                        """
                        UPDATE signal_performance
                        SET 
                            return_t3_atr = %s,
                            return_t3_pct = %s,
                            return_t5_atr = %s,
                            return_t5_pct = %s,
                            return_t10_atr = %s,
                            return_t10_pct = %s,
                            mfe_atr = %s,
                            mae_atr = %s,
                            risk_reward = %s,
                            mfe_bar_index = %s,
                            mae_bar_index = %s,
                            is_calculated = TRUE,
                            calculated_at = NOW(),
                            calculation_attempts = calculation_attempts + 1
                        WHERE signal_id = %s;
                    """,
                        params,
                    )

                    self.db.commit()
                    logger.info(f"Signal {signal_id} performansı güncellendi")
                    return True
                finally:
                    cursor.close()
            else:
                logger.warning(f"Signal {signal_id} için hiçbir metrik hesaplanamadı")
                return False

        except Exception as e:
            logger.error(f"Performance güncelleme hatası (signal_id={signal_id}): {e}")
            self.db.rollback()
            return False

    # =========================================================================
    # BATCH İŞLEMLER
    # =========================================================================

    def batch_update_performance(
        self,
        hours_back: int = 24,
        max_signals: Optional[int] = None,
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """
        Son N saatteki sinyallerin performansını toplu güncelle.

        Args:
            hours_back: Kaç saat geriye bak (default: 24)
            max_signals: Maksimum sinyal sayısı (None=sınırsız)

        Returns:
            {
                'total': Toplam sinyal,
                'success': Başarılı güncelleme,
                'failed': Başarısız güncelleme,
                'skipped': Atlanan
            }
        """
        cursor = self.db.cursor()

        try:
            # Henüz hesaplanmamış sinyalleri bul
            query = """
                SELECT sp.signal_id
                FROM signal_performance sp
                JOIN signals s ON s.id = sp.signal_id
                WHERE sp.is_calculated = FALSE
                  AND s.timestamp >= NOW() - INTERVAL '%s hours'
                ORDER BY s.timestamp ASC
            """

            if max_signals:
                query += f" LIMIT {max_signals}"

            cursor.execute(query, (hours_back,))
            signal_ids = [row[0] for row in cursor.fetchall()]

            logger.info(f"{len(signal_ids)} sinyal performansı hesaplanacak")

            # Her sinyali güncelle
            batch_stats = {"total": len(signal_ids), "success": 0, "failed": 0, "skipped": 0}

            for signal_id in signal_ids:
                if self.update_signal_performance(signal_id, dry_run=dry_run):
                    batch_stats["success"] += 1
                else:
                    batch_stats["failed"] += 1

            logger.info(f"Batch güncelleme tamamlandı: {batch_stats}")
            return batch_stats

        finally:
            cursor.close()


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)

    with SignalPerformanceAnalyzer() as analyzer:
        stats = analyzer.batch_update_performance(hours_back=24, max_signals=10)
        print(f"Sonuç: {stats}")
