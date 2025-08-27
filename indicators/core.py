import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import Optional, Tuple, List, Union

# Config import
from config import Config

# Error handling
from utils.exceptions import (
    CalculationError,
    ValidationError,
    raise_missing_column,
    raise_insufficient_data,
    raise_calculation_error,
    ErrorCodes
)
from utils.logger import get_logger

# Logger
logger = get_logger(__name__)

def calculate_ema(df: pd.DataFrame, period: int = Config.EMA_DEFAULT_PERIOD, price_col: str = 'close') -> pd.Series:
    """
    DataFrame'den EMA serisi hesaplar ve döndürür.
    
    Args:
        df: OHLCV verisi içeren DataFrame
        period: EMA periyodu (varsayılan 21)
        price_col: Hangi fiyat sütunu üzerinden EMA hesaplanacak
        
    Returns:
        pd.Series: EMA değerleri
        
    Raises:
        ValueError: Geçersiz sütun adı
    """
    if price_col not in df.columns:
        raise ValueError(f"{price_col} sütunu bulunamadı!")
    ema = df[price_col].ewm(span=period, adjust=False).mean()
    return ema

def calculate_sma(df: pd.DataFrame, period: int = Config.SMA_DEFAULT_PERIOD, price_col: str = 'close') -> pd.Series:
    """
    DataFrame'den SMA serisi hesaplar ve döndürür.
    
    Args:
        df: OHLCV verisi içeren DataFrame
        period: SMA periyodu (varsayılan 21)
        price_col: Hangi fiyat sütunu üzerinden SMA hesaplanacak
        
    Returns:
        pd.Series: SMA değerleri
        
    Raises:
        ValueError: Geçersiz sütun adı
    """
    if price_col not in df.columns:
        raise ValueError(f"{price_col} sütunu bulunamadı!")
    sma = df[price_col].rolling(window=period).mean()
    return sma

def calculate_rsi(df, period=Config.RSI_PERIOD_DEFAULT, price_col='close'):
    """
    DataFrame'den RSI serisi hesaplar ve döndürür (TradingView/PineScript uyumlu EMA tabanlı).
    
    Args:
        df: OHLCV verisi içeren DataFrame
        period: RSI periyodu (varsayılan 14)
        price_col: Hangi fiyat sütunu üzerinden RSI hesaplanacak
        
    Returns:
        pd.Series: RSI değerleri (0-100 arası)
        
    Raises:
        ValidationError: Geçersiz parametreler
        CalculationError: Hesaplama hatası
    """
    try:
        # Input validation
        if df is None or df.empty:
            raise ValidationError("DataFrame boş veya None", field="df")
            
        if period <= 0:
            raise ValidationError(f"Period pozitif olmalı, alınan: {period}", field="period", value=period)
            
        if price_col not in df.columns:
            raise_missing_column(price_col, list(df.columns))
            
        if len(df) < period:
            raise_insufficient_data("RSI calculation", period, len(df))
            
        # Check for valid price data
        price_series = df[price_col]
        if price_series.isna().all():
            raise CalculationError("Tüm fiyat verileri NaN", indicator="RSI")
            
        # RSI calculation
        delta = price_series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # EMA-based smoothing (TradingView compatible)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        # Sıfıra bölme hatasını önle ve RSI'ı doğru hesapla
        # avg_loss'un 0 olduğu yerlerde RSI 100 olmalıdır (güçlü yükseliş).
        rs = avg_gain / avg_loss
        rsi_values = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
        rsi = pd.Series(rsi_values, index=df.index)
        
        # Validate output
        if rsi.isna().all():
            raise CalculationError("RSI hesaplaması tüm NaN değerler üretti", indicator="RSI")
            
        logger.debug(f"RSI hesaplandı: period={period}, price_col={price_col}, valid_values={rsi.notna().sum()}")
        return rsi
        
    except (ValidationError, CalculationError):
        raise
    except Exception as e:
        raise CalculationError(f"RSI hesaplama hatası: {str(e)}", indicator="RSI") from e


def calculate_rsi_sma(df, period=Config.RSI_PERIOD_DEFAULT, price_col='close'):
    """
    DataFrame'den SMA (basit ortalama) ile RSI serisi hesaplar ve döndürür.
    period: RSI periyodu (varsayılan 14)
    price_col: Hangi fiyat sütunu üzerinden RSI hesaplanacak
    """
    if price_col not in df.columns:
        raise ValueError(f"{price_col} sütunu bulunamadı!")
    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi



def calculate_bollinger_bands(df, period=20, price_col='close', num_std=2):
    """
    DataFrame'den Bollinger Bands üst, alt ve orta bantlarını döndürür.
    period: Bant periyodu (varsayılan 20)
    price_col: Hangi fiyat sütunu üzerinden hesaplanacak
    num_std: Standart sapma katsayısı
    """
    if price_col not in df.columns:
        raise ValueError(f"{price_col} sütunu bulunamadı!")
    sma = df[price_col].rolling(window=period).mean()
    std = df[price_col].rolling(window=period).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

def calculate_macd(df, fast=12, slow=26, signal=9, price_col='close'):
    """
    DataFrame'den MACD, signal ve histogram serilerini döndürür.
    fast: Hızlı EMA periyodu (varsayılan 12)
    slow: Yavaş EMA periyodu (varsayılan 26)
    signal: Signal EMA periyodu (varsayılan 9)
    price_col: Hangi fiyat sütunu üzerinden hesaplanacak
    """
    if price_col not in df.columns:
        raise ValueError(f"{price_col} sütunu bulunamadı!")
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def wilder_rma(series, period):
    result = [series.iloc[0]]
    for val in series.iloc[1:]:
        result.append((result[-1] * (period - 1) + val) / period)
    return pd.Series(result, index=series.index)

def calculate_adx(df, adxlen=14, dilen=14):
    """
    DataFrame'den ADX, +DI ve -DI serilerini döndürür (TradingView/Pine Script uyumlu).
    adxlen: ADX periyodu (varsayılan 14)
    dilen: DI periyodu (varsayılan 14)
    Gereken sütunlar: 'high', 'low', 'close'
    """
    high = df['high']
    low = df['low']
    close = df['close']
    up = high.diff()
    down = -low.diff()
    plusDM = up.where((up > down) & (up > 0), 0.0)
    minusDM = down.where((down > up) & (down > 0), 0.0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    truerange = wilder_rma(tr, dilen)
    plus = 100 * wilder_rma(plusDM, dilen) / truerange
    minus = 100 * wilder_rma(minusDM, dilen) / truerange
    sum_ = plus + minus
    dx = abs(plus - minus) / sum_.replace(0, 1)
    adx = 100 * wilder_rma(dx, adxlen)
    return adx, plus, minus


def calculate_mfi(df, period=Config.MFI_PERIOD):
    """
    DataFrame'den Money Flow Index (MFI) serisini döndürür (TradingView/Pine Script uyumlu).
    period: MFI periyodu (varsayılan 14)
    Gereken sütunlar: 'high', 'low', 'close', 'volume'
    """
    import numpy as np
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
    positive_mf = pd.Series(positive_flow).rolling(window=period, min_periods=period).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=period, min_periods=period).sum()
    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    mfi = pd.Series(mfi, index=df.index)
    return mfi


def calculate_obv(df, price_col='close', volume_col='volume'):
    """
    DataFrame'den On Balance Volume (OBV) serisini döndürür (TradingView/Pine Script uyumlu).
    price_col: Fiyat sütunu (varsayılan 'close')
    volume_col: Hacim sütunu (varsayılan 'volume')
    """
    obv = []
    for i in range(len(df)):
        if i == 0:
            obv.append(df[volume_col].iloc[0])  # İlk barın hacmiyle başla
        else:
            if df[price_col].iloc[i] > df[price_col].iloc[i - 1]:
                obv.append(obv[-1] + df[volume_col].iloc[i])
            elif df[price_col].iloc[i] < df[price_col].iloc[i - 1]:
                obv.append(obv[-1] - df[volume_col].iloc[i])
            else:
                obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def calculate_atr(df, period=Config.ATR_PERIOD):
    """
    DataFrame'den Average True Range (ATR) serisini döndürür (TradingView/Pine Script uyumlu).
    period: ATR periyodu (varsayılan 14)
    Gereken sütunlar: 'high', 'low', 'close'
    """
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr


def calculate_cci(df, period=20):
    """
    Commodity Channel Index (CCI) hesaplar.
    Gereken sütunlar: 'high', 'low', 'close'
    """
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=period).mean()
    # .mad() yerine manuel hesaplama
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    return cci





def calculate_stochastic(df, k_period=Config.STOCH_K_PERIOD, d_period=Config.STOCH_D_PERIOD):
    """
    Stochastic Oscillator hesaplar. %K ve %D döndürür.
    Gereken sütunlar: 'high', 'low', 'close'
    """
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    percent_k = 100 * (df['close'] - low_min) / (high_max - low_min)
    percent_d = percent_k.rolling(window=d_period).mean()
    return percent_k, percent_d


def calculate_williams_r(df, period=Config.WILLIAMS_R_PERIOD):
    """
    Williams %R hesaplar.
    Gereken sütunlar: 'high', 'low', 'close'
    """
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    williams_r = -100 * (high_max - df['close']) / (high_max - low_min)
    return williams_r


def calculate_donchian_channel(df, period=20):
    """
    Donchian Channel üst ve alt bantlarını döndürür.
    Gereken sütunlar: 'high', 'low'
    """
    upper_band = df['high'].rolling(window=period).max()
    lower_band = df['low'].rolling(window=period).min()
    return upper_band, lower_band


def calculate_keltner_channel(df, period=20, atr_mult=2):
    """
    Keltner Channel orta, üst ve alt bantlarını döndürür.
    Gereken sütunlar: 'high', 'low', 'close'
    """
    typical_price = df['close']
    ema = typical_price.ewm(span=period, adjust=False).mean()
    atr = calculate_atr(df, period)
    upper_band = ema + atr_mult * atr
    lower_band = ema - atr_mult * atr
    return ema, upper_band, lower_band

def calculate_vwap(df, price_col='close', volume_col='volume'):
    """
    Volume Weighted Average Price (VWAP) hesaplar.
    Fiyat ve hacimden zaman boyunca kümülatif hacim ağırlıklı ortalama döndürür.
    """
    price = df[price_col]
    volume = df[volume_col]
    vwap = (price * volume).cumsum() / volume.cumsum()
    return vwap

def calculate_parabolic_sar(df, step=0.02, max_step=0.2):
    """
    TradingView/Pine Script ile uyumlu Parabolic SAR hesaplama fonksiyonu.
    Trend yönü (bullish/bearish) bilgisini de döndürür.
    Hata düzeltilmiş ve standartlara uygun hale getirilmiş versiyon.
    """
    import numpy as np
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    length = len(df)
    sar = np.zeros(length)
    trend = np.zeros(length, dtype=bool)  # True: Bull (Yükseliş), False: Bear (Düşüş)

    if length < 2:
        return pd.Series(sar, index=df.index), trend

    # Başlangıç değerlerini ve trendi ata
    if close[1] > close[0]:
        bull = True
        ep = high[1]
        sar[1] = low[0]
    else:
        bull = False
        ep = low[1]
        sar[1] = high[0]

    sar[0] = sar[1]
    trend[0] = trend[1] = bull
    af = step

    for i in range(2, length):
        prev_sar = sar[i-1]

        if bull:  # Yükseliş trendi
            sar_candidate = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar_candidate, low[i-1], low[i-2])

            if low[i] < sar[i]:  # Trend düşüşe dönüyor
                bull = False
                sar[i] = ep  # Son tepe noktası yeni SAR olur
                ep = low[i]
                af = step
            else:  # Trend devam ediyor
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
        else:  # Düşüş trendi
            sar_candidate = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar_candidate, high[i-1], high[i-2])

            if high[i] > sar[i]:  # Trend yükselişe dönüyor
                bull = True
                sar[i] = ep  # Son dip noktası yeni SAR olur
                ep = high[i]
                af = step
            else:  # Trend devam ediyor
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)
        
        trend[i] = bull

    return pd.Series(sar, index=df.index), trend


def calculate_awesome_oscillator(df, fast_period=5, slow_period=34):
    """
    Awesome Oscillator (AO) hesaplar.
    Gereken sütunlar: 'high', 'low'
    """
    median_price = (df['high'] + df['low']) / 2
    ao = median_price.rolling(window=fast_period).mean() - median_price.rolling(window=slow_period).mean()
    return ao


def calculate_roc(df, period=Config.ROC_PERIOD, price_col='close'):
    """
    Rate of Change (ROC) hesaplar.
    TradingView ile uyumlu olması için varsayılan periyot 9'dur.

    Args:
        df (pd.DataFrame): Fiyat verilerini içeren DataFrame.
        period (int): ROC hesaplaması için periyot.
        price_col (str): Fiyatların alınacağı sütun adı.

    Returns:
        pd.Series: ROC değerlerini içeren pandas Serisi.
    """
    price = df[price_col]
    roc = ((price - price.shift(period)) / price.shift(period)) * 100
    return roc


def calculate_stoch_rsi(df, rsi_period=Config.RSI_PERIOD_DEFAULT, stoch_period=Config.RSI_PERIOD_DEFAULT, k_period=Config.STOCH_D_PERIOD, d_period=Config.STOCH_D_PERIOD, price_col='close'):
    """
    Stochastic RSI (StochRSI) hesaplar. StochK ve StochD döndürür.
    Gereken sütunlar: 'close'
    """
    # StochRSI için RSI hesaplaması gerekir, bu yüzden mevcut RSI fonksiyonunu çağırıyoruz.
    rsi = calculate_rsi(df, period=rsi_period, price_col=price_col)
    
    min_rsi = rsi.rolling(window=stoch_period).min()
    max_rsi = rsi.rolling(window=stoch_period).max()
    
    # NaN veya sonsuz bölünmeleri önlemek için (max_rsi - min_rsi) farkının 0 olduğu durumları yönetiyoruz.
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)
    
    stoch_k = stoch_rsi.rolling(window=k_period).mean() * 100
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return stoch_k, stoch_d


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm gerekli teknik göstergeleri hesaplar ve DataFrame'e ekler.
    SignalEngine'in ihtiyaç duyduğu tüm sütunları oluşturur.
    """
    try:
        # MA200
        if len(df) >= 200:
            df['ma200'] = calculate_sma(df, period=200)
        else:
            df['ma200'] = np.nan

        # RSI (Dinamik periyotlar)
        fast_rsi_period = Config.RSI_FAST_WINDOW
        slow_rsi_period = Config.RSI_SLOW_WINDOW
        fast_rsi_col = f'rsi_{fast_rsi_period}'
        slow_rsi_col = f'rsi_{slow_rsi_period}'

        if len(df) >= slow_rsi_period:
            df[fast_rsi_col] = calculate_rsi(df, period=fast_rsi_period)
            df[slow_rsi_col] = calculate_rsi(df, period=slow_rsi_period)
            df['rsi_change'] = df[fast_rsi_col] - df[slow_rsi_col]
        else:
            df[fast_rsi_col] = np.nan
            df[slow_rsi_col] = np.nan
            df['rsi_change'] = np.nan

        # MACD
        if len(df) >= 26:
            df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df)
            df['macd_change'] = df['macd'].diff()
        else:
            df['macd'] = np.nan
            df['macd_signal'] = np.nan
            df['macd_hist'] = np.nan
            df['macd_change'] = np.nan

        # ADX
        if len(df) >= 28: # ADX için genellikle 2*dilen kadar veri gerekir
            df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df)
        else:
            df['adx'] = np.nan
            df['plus_di'] = np.nan
            df['minus_di'] = np.nan

        # ATR
        if len(df) >= Config.ATR_PERIOD:
            df['atr'] = calculate_atr(df, period=Config.ATR_PERIOD)
        else:
            df['atr'] = np.nan

        # Momentum (ROC)
        if len(df) >= Config.ROC_PERIOD:
            df['momentum'] = calculate_roc(df, period=Config.ROC_PERIOD)
        else:
            df['momentum'] = np.nan

        logger.debug("Tüm göstergeler başarıyla DataFrame'e eklendi.")

    except Exception as e:
        logger.error(f"Gösterge hesaplama hatası: {e}", exc_info=True)
        # Hata durumunda bile DataFrame'i döndür, eksik sütunlar loglanacaktır.
    
    return df

def find_support_resistance(df, order=10, tolerance=0.002):
    """
    Verilen DataFrame'de destek ve direnç seviyelerini bulur.
    order: Kaç barda bir lokal min/max aranacak (örn: 10)
    tolerance: Seviye kümelenme toleransı (örn: 0.002 = %0.2)
    """
    if df.empty: 
        return [], []
    
    local_min = argrelextrema(df['low'].values, np.less_equal, order=order)[0]
    local_max = argrelextrema(df['high'].values, np.greater_equal, order=order)[0]
    support_levels = []
    resistance_levels = []

    for i in local_min:
        level = df['low'].iloc[i]
        if not any(abs(level - l) < level * tolerance for l in support_levels):
            support_levels.append(level)

    for i in local_max:
        level = df['high'].iloc[i]
        if not any(abs(level - l) < level * tolerance for l in resistance_levels):
            resistance_levels.append(level)

    return sorted(support_levels), sorted(resistance_levels)

