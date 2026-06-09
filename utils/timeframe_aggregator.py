"""
Timeframe Aggregation Modülü

Bu modül 1m OHLCV verilerini daha yüksek timeframe'lere (5m, 15m, 1h, 4h) 
dönüştürmek için kullanılır.

Aggregation Kuralları:
- Open: İlk bar'ın open değeri
- High: Tüm barların en yüksek high değeri  
- Low: Tüm barların en düşük low değeri
- Close: Son bar'ın close değeri
- Volume: Tüm barların volume toplamı
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import logging
from config import Config

from utils.logger import get_logger

logger = get_logger(__name__)


class TimeframeAggregator:
    """
    OHLCV verilerini farklı timeframe'lere aggregate eden sınıf.
    """
    
    # Desteklenen timeframe dönüşümleri
    SUPPORTED_AGGREGATIONS = {
        '1m': ['5m', '15m', '1h', '4h', '1d'],
        '5m': ['15m', '1h', '4h', '1d'],
        '15m': ['1h', '4h', '1d'],
        '1h': ['4h', '1d'],
        '4h': ['1d']
    }
    
    # Timeframe'leri dakika cinsinden
    TIMEFRAME_MINUTES = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }
    
    @classmethod
    def can_aggregate(cls, from_tf: str, to_tf: str) -> bool:
        """
        Bir timeframe'den diğerine aggregate edilip edilemeyeceğini kontrol eder.
        
        Args:
            from_tf: Kaynak timeframe (örn: '1m')
            to_tf: Hedef timeframe (örn: '5m')
            
        Returns:
            bool: Aggregate edilebilirse True
        """
        return from_tf in cls.SUPPORTED_AGGREGATIONS and to_tf in cls.SUPPORTED_AGGREGATIONS[from_tf]
    
    @classmethod
    def get_aggregation_ratio(cls, from_tf: str, to_tf: str) -> Optional[int]:
        """
        İki timeframe arasındaki oran hesaplar.
        
        Args:
            from_tf: Kaynak timeframe
            to_tf: Hedef timeframe
            
        Returns:
            int: Kaç adet kaynak bar'ın 1 hedef bar oluşturduğu
        """
        if not cls.can_aggregate(from_tf, to_tf):
            return None
            
        from_minutes = cls.TIMEFRAME_MINUTES.get(from_tf)
        to_minutes = cls.TIMEFRAME_MINUTES.get(to_tf)
        
        if not from_minutes or not to_minutes:
            return None
            
        ratio = to_minutes // from_minutes
        return ratio if to_minutes % from_minutes == 0 else None
    
    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> bool:
        """
        DataFrame'in aggregation için uygun olup olmadığını kontrol eder.
        
        Args:
            df: Kontrol edilecek DataFrame
            
        Returns:
            bool: Uygunsa True
        """
        required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        
        if df.empty:
            logger.warning("DataFrame boş")
            return False
            
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Eksik kolonlar: {missing_columns}")
            return False
            
        # Numeric kolonları kontrol et
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(f"Kolon {col} numeric değil")
                return False
                
        return True
    
    @classmethod
    def aggregate_ohlcv(cls, df: pd.DataFrame, from_tf: str, to_tf: str) -> pd.DataFrame:
        """
        OHLCV verilerini belirtilen timeframe'e aggregate eder.
        
        Args:
            df: Kaynak OHLCV DataFrame
            from_tf: Kaynak timeframe (örn: '1m')
            to_tf: Hedef timeframe (örn: '5m')
            
        Returns:
            pd.DataFrame: Aggregate edilmiş OHLCV verisi
        """
        if not cls.validate_dataframe(df):
            logger.error("DataFrame validation başarısız")
            return pd.DataFrame()
            
        if not cls.can_aggregate(from_tf, to_tf):
            logger.error(f"Desteklenmeyen aggregation: {from_tf} -> {to_tf}")
            return pd.DataFrame()
            
        ratio = cls.get_aggregation_ratio(from_tf, to_tf)
        if not ratio:
            logger.error(f"Aggregation ratio hesaplanamadı: {from_tf} -> {to_tf}")
            return pd.DataFrame()
            
        logger.info(f"Aggregation başlatılıyor: {from_tf} -> {to_tf} (ratio: {ratio})")
        
        # DataFrame'i kopyala ve sırala
        df_sorted = df.copy().sort_values('open_time').reset_index(drop=True)

        # --- Boundary Alignment (Timeframe hizalaması) ---
        # Amaç: Hedef timeframe'in doğal sınırlarından başlamak (ör. 5m için dakika%5==0)
        try:
            dt_utc = pd.to_datetime(df_sorted['open_time'], unit='ms', utc=True)

            def _is_aligned(ts: pd.Timestamp, tf: str) -> bool:
                if tf == '5m':
                    return ts.minute % 5 == 0 and ts.second == 0
                if tf == '15m':
                    return ts.minute % 15 == 0 and ts.second == 0
                if tf == '1h':
                    return ts.minute == 0 and ts.second == 0
                if tf == '4h':
                    # Yerel saate göre hizalama
                    try:
                        tz_name = getattr(Config, 'TIMEZONE', 'Europe/Istanbul')
                        ts_local = ts.tz_convert(tz_name)
                    except Exception:
                        # Beklenmedik timezone hatalarında UTC ile devam (dakika==0 koşulu korunur)
                        ts_local = ts
                    return ts_local.minute == 0 and ts_local.second == 0 and (ts_local.hour % 4 == 0)
                if tf == '1d':
                    return ts.hour == 0 and ts.minute == 0 and ts.second == 0
                # Varsayılan: 1m ve diğerleri için ek şart yok
                return True

            first_aligned_idx = None
            for i, ts in enumerate(dt_utc):
                if _is_aligned(ts, to_tf):
                    first_aligned_idx = i
                    break

            if first_aligned_idx is None:
                logger.warning(f"Boundary alignment bulunamadı: {to_tf} için uygun başlangıç yok. Boş döndürülüyor.")
                return pd.DataFrame()

            if first_aligned_idx > 0:
                # Uygun hizayı bulana kadar baştaki satırları kırp
                df_sorted = df_sorted.iloc[first_aligned_idx:].reset_index(drop=True)
                # Log için ilk hizalı timestamp'i ayrı al
                first_ts = dt_utc.iloc[first_aligned_idx]
                logger.info(f"Boundary alignment uygulandı: ilk hizalı indeks={first_aligned_idx} | ilk ts={first_ts}")
        except Exception as e:
            logger.warning(f"Boundary alignment sırasında hata: {e}. Devam ediliyor (mevcut sırayla).")
        
        # Grup sayısını hesapla
        total_bars = len(df_sorted)
        complete_groups = total_bars // ratio
        
        if complete_groups == 0:
            logger.warning(f"Yeterli veri yok. Gerekli: {ratio}, Mevcut: {total_bars}")
            return pd.DataFrame()
            
        # Sadece tam grupları al (kalan barları göz ardı et)
        df_complete = df_sorted.iloc[:complete_groups * ratio].copy()
        
        # Grup indekslerini oluştur
        df_complete['group'] = df_complete.index // ratio
        
        # Aggregation işlemi
        aggregated_data = []
        
        for group_id in range(complete_groups):
            group_data = df_complete[df_complete['group'] == group_id]
            
            if len(group_data) != ratio:
                logger.warning(f"Grup {group_id} eksik veri içeriyor: {len(group_data)}/{ratio}")
                continue
                
            # OHLCV aggregation
            agg_bar = {
                'open_time': int(group_data['open_time'].iloc[0]),  # İlk bar'ın zamanı
                'open': float(group_data['open'].iloc[0]),          # İlk bar'ın open'ı
                'high': float(group_data['high'].max()),           # En yüksek high
                'low': float(group_data['low'].min()),             # En düşük low
                'close': float(group_data['close'].iloc[-1]),      # Son bar'ın close'u
                'volume': float(group_data['volume'].sum()),       # Volume toplamı
            }
            
            # Opsiyonel kolonları koru
            optional_columns = ['close_time', 'quote_asset_volume', 'number_of_trades', 
                              'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            volume_columns = ['quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            
            for col in optional_columns:
                if col in group_data.columns:
                    if col == 'close_time':
                        val = group_data[col].iloc[-1]
                        agg_bar[col] = int(val) if pd.notna(val) else 0
                    elif col in volume_columns:
                        # Önce numeric'e çevir, sonra topla
                        numeric_values = pd.to_numeric(group_data[col], errors='coerce').fillna(0)
                        agg_bar[col] = float(numeric_values.sum())   # Toplamları al
                    elif col == 'number_of_trades':
                        agg_bar[col] = int(group_data[col].sum())     # Trade sayısı toplamı
            
            aggregated_data.append(agg_bar)
        
        result_df = pd.DataFrame(aggregated_data)
        
        logger.info(f"Aggregation tamamlandı: {len(df_sorted)} -> {len(result_df)} bars ({from_tf} -> {to_tf})")
        
        return result_df
    
    @classmethod
    def aggregate_to_multiple_timeframes(cls, df: pd.DataFrame, from_tf: str, 
                                       target_timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Tek bir DataFrame'i birden fazla timeframe'e aggregate eder.
        
        Args:
            df: Kaynak OHLCV DataFrame
            from_tf: Kaynak timeframe
            target_timeframes: Hedef timeframe listesi
            
        Returns:
            Dict[str, pd.DataFrame]: Timeframe -> DataFrame mapping
        """
        results = {}
        
        for target_tf in target_timeframes:
            try:
                aggregated_df = cls.aggregate_ohlcv(df, from_tf, target_tf)
                if not aggregated_df.empty:
                    results[target_tf] = aggregated_df
                    logger.info(f"✅ {from_tf} -> {target_tf}: {len(aggregated_df)} bars")
                else:
                    logger.warning(f"❌ {from_tf} -> {target_tf}: Aggregation başarısız")
            except Exception as e:
                logger.error(f"❌ {from_tf} -> {target_tf}: Hata - {e}")
        
        return results
    
    @classmethod
    def create_test_data(cls, bars: int = 50, start_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Test amaçlı sample OHLCV verisi oluşturur.
        
        Args:
            bars: Oluşturulacak bar sayısı
            start_time: Başlangıç zamanı (None ise şu anki zaman)
            
        Returns:
            pd.DataFrame: Test OHLCV verisi
        """
        if start_time is None:
            # Saat başından (dakika=0) başla; boundary alignment hiçbir bar kesmez
            start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        data = []
        base_price = 50000.0  # BTCUSDT benzeri fiyat
        
        for i in range(bars):
            # Basit random walk
            price_change = (np.random.random() - 0.5) * 100  # -50 ile +50 arası
            current_price = base_price + price_change
            
            # OHLC oluştur (mantıklı OHLC değerleri)
            open_price = current_price
            close_change = (np.random.random() - 0.5) * 20
            close_price = current_price + close_change
            
            # High ve Low değerlerini Open ve Close'a göre ayarla
            min_price = min(open_price, close_price)
            max_price = max(open_price, close_price)
            
            high_offset = np.random.random() * 50
            low_offset = np.random.random() * 50
            
            high_price = max_price + high_offset
            low_price = min_price - low_offset
            
            bar_time = start_time + timedelta(minutes=i)
            bar_time_utc = bar_time.replace(tzinfo=timezone.utc)

            bar_data = {
                'open_time': int(bar_time_utc.timestamp() * 1000),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': round(np.random.random() * 1000000, 2),
                'close_time': int((bar_time_utc + timedelta(minutes=1)).timestamp() * 1000),
                'quote_asset_volume': round(np.random.random() * 50000000, 2),
                'number_of_trades': int(np.random.random() * 1000),
                'taker_buy_base_asset_volume': round(np.random.random() * 500000, 2),
                'taker_buy_quote_asset_volume': round(np.random.random() * 25000000, 2),
            }
            
            data.append(bar_data)
            base_price = bar_data['close']  # Sonraki bar için base price güncelle
        
        return pd.DataFrame(data)


# Convenience functions
def aggregate_1m_to_5m(df: pd.DataFrame) -> pd.DataFrame:
    """1m verilerini 5m'e çevirir."""
    return TimeframeAggregator.aggregate_ohlcv(df, '1m', '5m')


def aggregate_1m_to_15m(df: pd.DataFrame) -> pd.DataFrame:
    """1m verilerini 15m'e çevirir."""
    return TimeframeAggregator.aggregate_ohlcv(df, '1m', '15m')


def aggregate_5m_to_15m(df: pd.DataFrame) -> pd.DataFrame:
    """5m verilerini 15m'e çevirir."""
    return TimeframeAggregator.aggregate_ohlcv(df, '5m', '15m')


def create_multi_timeframe_data(df_1m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    1m verilerinden 5m ve 15m verilerini oluşturur.
    
    Args:
        df_1m: 1m OHLCV DataFrame
        
    Returns:
        Dict: {'1m': df_1m, '5m': df_5m, '15m': df_15m}
    """
    results = {'1m': df_1m}
    
    # 1m -> 5m
    df_5m = aggregate_1m_to_5m(df_1m)
    if not df_5m.empty:
        results['5m'] = df_5m
        
        # 5m -> 15m (daha verimli)
        df_15m = aggregate_5m_to_15m(df_5m)
        if not df_15m.empty:
            results['15m'] = df_15m
    
    return results
