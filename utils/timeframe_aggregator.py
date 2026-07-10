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
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from config import Config

from utils.logger import get_logger

logger = get_logger(__name__)


class TimeframeAggregator:
    """
    OHLCV verilerini farklı timeframe'lere aggregate eden sınıf.
    """
    
    # Desteklenen timeframe dönüşümleri
    # 10 Tem 2026: 30m/6h/8h/12h eklendi (1m-türetme projesi, MTF_TIMEFRAMES'in
    # tamamını kapsaması için) — sadece 1m satırına eklendi, mevcut 5m/15m/1h/4h
    # satırları DOKUNULMADI (geriye dönük uyumluluk, var olan çağıranlar etkilenmez).
    SUPPORTED_AGGREGATIONS = {
        '1m': ['5m', '15m', '30m', '1h', '4h', '6h', '8h', '12h', '1d'],
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
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '6h': 360,
        '8h': 480,
        '12h': 720,
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
    def get_period_start(cls, ts_ms: int, tf: str) -> int:
        """Verilen zaman damgasının (epoch ms, UTC) ait olduğu tf periyodunun
        BAŞLANGICINI (epoch ms) döner. 10 Tem 2026, 1m-türetme Adım 4: oluşum
        halindeki (forming/henüz kapanmamış) üst TF barının hangi 1m barlarından
        oluştuğunu bulmak için — is_boundary_aligned ile AYNI hizalama tanımını
        (yerel saat bazlı 4h/6h/8h/12h dahil) kullanır, tutarsızlık riski yok."""
        minutes = cls.TIMEFRAME_MINUTES.get(tf)
        if not minutes:
            return ts_ms
        if tf in ('4h', '6h', '8h', '12h'):
            hour_mod = {'4h': 4, '6h': 6, '8h': 8, '12h': 12}[tf]
            tz_name = getattr(Config, 'TIMEZONE', 'Europe/Istanbul')
            ts_local = pd.Timestamp(ts_ms, unit='ms', tz='UTC').tz_convert(tz_name)
            period_start_local = ts_local.replace(
                hour=(ts_local.hour // hour_mod) * hour_mod,
                minute=0, second=0, microsecond=0, nanosecond=0,
            )
            return int(period_start_local.tz_convert('UTC').value // 10**6)
        if tf == '1d':
            # is_boundary_aligned ile tutarlı: 1d burada UTC gün başı (Binance'in
            # kendi 1d kline tanımıyla aynı) — do_kirilimi._daily_open'daki YEREL
            # gün açılışı farklı bir kavram (trading-day referansı), karıştırılmamalı.
            ts = pd.Timestamp(ts_ms, unit='ms', tz='UTC')
            period_start = ts.replace(hour=0, minute=0, second=0, microsecond=0, nanosecond=0)
            return int(period_start.value // 10**6)
        period_ms = minutes * 60_000
        return (ts_ms // period_ms) * period_ms

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
    def is_boundary_aligned(cls, ts: pd.Timestamp, tf: str) -> bool:
        """Verilen zaman damgası (tz-aware, UTC) tf'nin doğal periyot sınırında mı?
        10 Tem 2026: aggregate_ohlcv içindeki özel _is_aligned closure'ından
        çıkarıldı — hem aggregation'ın boundary-alignment adımı hem 1m-türetme
        projesinin "bu 1m barı hangi üst TF'leri de kapattı" tespiti (bkz.
        get_closing_timeframes) AYNI hizalama tanımını paylaşmalı; iki ayrı
        kopya, aralarında sessizce tutarsızlık oluşma riski taşırdı."""
        if tf == '5m':
            return ts.minute % 5 == 0 and ts.second == 0
        if tf == '15m':
            return ts.minute % 15 == 0 and ts.second == 0
        if tf == '30m':
            return ts.minute % 30 == 0 and ts.second == 0
        if tf == '1h':
            return ts.minute == 0 and ts.second == 0
        if tf in ('4h', '6h', '8h', '12h'):
            # Yerel saate göre hizalama.
            try:
                tz_name = getattr(Config, 'TIMEZONE', 'Europe/Istanbul')
                ts_local = ts.tz_convert(tz_name)
            except Exception:
                # Beklenmedik timezone hatalarında UTC ile devam (dakika==0 koşulu korunur)
                ts_local = ts
            hour_mod = {'4h': 4, '6h': 6, '8h': 8, '12h': 12}[tf]
            return ts_local.minute == 0 and ts_local.second == 0 and (ts_local.hour % hour_mod == 0)
        if tf == '1d':
            return ts.hour == 0 and ts.minute == 0 and ts.second == 0
        # Varsayılan: 1m ve diğerleri için ek şart yok
        return True

    @classmethod
    def get_closing_timeframes(cls, next_open_time_ms: int, tf_list: Optional[List[str]] = None) -> List[str]:
        """1m-türetme projesi için: bir 1m barı kapandığında (close_time+1ms =
        next_open_time_ms, yani bir sonraki 1m barının başlangıcı), bu ANDA
        hangi üst TF'lerin de kapandığını döner.

        Mantık: next_open_time_ms, sonraki 1m barının başlangıcıdır. Eğer bu an
        aynı zamanda TF X için de bir periyot sınırıysa (ör. dakika%5==0), demek
        ki az önce kapanan 1m barı TF X'in de SON barıydı — yani TF X de kapandı.
        Bu, live_data_manager.py::_update_mtf_data'daki mevcut
        `_is_mtf_bar_complete` mantığıyla aynı ilkeyi (next_bar_open_time
        kontrolü) paylaşır, sadece TimeframeAggregator'ın merkezi hizalama
        tanımını (is_boundary_aligned) kullanır.

        Args:
            next_open_time_ms: kapanan 1m barının close_time'ından hemen sonraki
                (bir sonraki 1m barının) open_time'ı, epoch ms (UTC).
            tf_list: kontrol edilecek TF listesi (varsayılan: tüm desteklenenler).

        Returns:
            Kapanan üst TF'lerin listesi (ör. ['5m', '15m'] — saat başıysa ayrıca '1h' de eklenir).
        """
        if tf_list is None:
            tf_list = ['5m', '15m', '30m', '1h', '4h', '6h', '8h', '12h', '1d']
        ts = pd.Timestamp(next_open_time_ms, unit='ms', tz='UTC')
        return [tf for tf in tf_list if cls.is_boundary_aligned(ts, tf)]

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
        # 10 Tem 2026: hizalama tanımı artık is_boundary_aligned()'a taşındı (paylaşılan,
        # tek kaynak — bkz. metodun docstring'i).
        try:
            dt_utc = pd.to_datetime(df_sorted['open_time'], unit='ms', utc=True)

            first_aligned_idx = None
            for i, ts in enumerate(dt_utc):
                if cls.is_boundary_aligned(ts, to_tf):
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
        
        # Zaman-damgası tabanlı gruplama (index-tabanlı DEĞİL) — 10 Tem 2026 düzeltmesi:
        # eski `index // ratio` gruplaması, 1m buffer'ında bir boşluk (WS kesintisi,
        # restart, senkronizasyon gecikmesi) varsa boşluktan SONRAKİ TÜM barları
        # kaydırıp yanlış hizalıyordu — gerçek veriyle (BTCUSDT 1m buffer'ında restart
        # kaynaklı 2 adet 2dk'lık boşluk) test edilirken bulundu: 30m+ türetmede art
        # arda barlar arası 31dk gibi tutarsız aralıklar oluşuyordu. Artık her bar
        # KENDİ zaman damgasından bağımsız olarak doğru periyoda atanıyor; boşluk
        # varsa o periyot eksik üyeyle oluşur ve aşağıdaki len(group_data)!=ratio
        # kontrolüyle atlanır — takip eden periyotlar ARTIK KAYMIYOR.
        period_ms = ratio * 60_000
        if to_tf in ('4h', '6h', '8h', '12h'):
            hour_mod = {'4h': 4, '6h': 6, '8h': 8, '12h': 12}[to_tf]
            tz_name = getattr(Config, 'TIMEZONE', 'Europe/Istanbul')
            dt_local = pd.to_datetime(df_sorted['open_time'], unit='ms', utc=True).dt.tz_convert(tz_name)
            period_start_local = dt_local.apply(
                lambda ts: ts.replace(hour=(ts.hour // hour_mod) * hour_mod,
                                       minute=0, second=0, microsecond=0, nanosecond=0)
            )
            group_key = period_start_local.dt.tz_convert('UTC').astype('int64') // 10**6
        else:
            group_key = (df_sorted['open_time'] // period_ms) * period_ms

        df_sorted['group'] = group_key
        group_ids = sorted(df_sorted['group'].unique())

        if not group_ids:
            logger.warning(f"Yeterli veri yok. Gerekli: {ratio}, Mevcut: {len(df_sorted)}")
            return pd.DataFrame()

        # Aggregation işlemi
        aggregated_data = []

        for group_id in group_ids:
            group_data = df_sorted[df_sorted['group'] == group_id]

            if len(group_data) != ratio:
                logger.debug(
                    f"Grup {group_id} eksik veri içeriyor: {len(group_data)}/{ratio} "
                    "(boşluk ya da tamamlanmamış son periyot olabilir, atlanıyor)"
                )
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
            # 10 Tem 2026: buy_volume/sell_volume eklendi — canonical kline şemasının
            # (bkz. live_data_manager.py new_row, 546d4da) parçası ve utils/vpmv.py::
            # directional_volume bunları okuyor, 1m-türetme ile üretilen üst TF barları
            # için VPMV hesaplanabilsin diye toplanması şart.
            optional_columns = ['close_time', 'quote_asset_volume', 'number_of_trades',
                              'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                              'buy_volume', 'sell_volume']
            volume_columns = ['quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                              'buy_volume', 'sell_volume']
            
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

            bar_data = {
                'open_time': int(bar_time.timestamp() * 1000),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': round(np.random.random() * 1000000, 2),
                'close_time': int((bar_time + timedelta(minutes=1)).timestamp() * 1000),
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
