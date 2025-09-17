import pandas as pd
import numpy as np

def normalize_series(series, index=None):
    """
    Seriyi 0-1 aralığında normalize eder
    """
    if series.empty or series.isna().all():
        return pd.Series(0.0, index=index if index is not None else series.index)
    
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series(0.5, index=index if index is not None else series.index)
    
    normalized = (series - min_val) / (max_val - min_val)
    
    if index is not None:
        normalized.index = index
    
    return normalized.fillna(0.0)

def smooth_series(series, period=3, index=None):
    """
    Seriyi hareketli ortalama ile yumuşatır
    """
    if series.empty:
        return pd.Series(0.0, index=index if index is not None else series.index)
    
    smoothed = series.rolling(window=period, min_periods=1).mean()
    
    if index is not None:
        smoothed.index = index
    
    return smoothed.fillna(0.0)

def normalize_price_0_100(series, window=50, index=None):
    """
    Fiyat verilerini robust normalizasyon ile 0-100 aralığında normalize eder
    Outlier'lara dayanıklı
    """
    if series.empty or series.isna().all():
        return pd.Series(50.0, index=index if index is not None else series.index)
    
    median = series.rolling(window, min_periods=1).median()
    q25 = series.rolling(window, min_periods=1).quantile(0.25)
    q75 = series.rolling(window, min_periods=1).quantile(0.75)
    iqr = q75 - q25
    
    normalized = ((series - median) / (iqr + 1e-8)) * 20 + 50
    normalized = np.clip(normalized, 0, 100)
    
    if index is not None:
        normalized.index = index
    
    return normalized.fillna(50.0)

def normalize_momentum_0_100(series, k=1.0, index=None):
    """
    Momentum verilerini sigmoid normalizasyon ile 0-100 aralığında normalize eder
    Smooth geçişler sağlar
    """
    if series.empty or series.isna().all():
        return pd.Series(50.0, index=index if index is not None else series.index)
    
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return pd.Series(50.0, index=index if index is not None else series.index)
    
    z_score = (series - mean) / std
    sigmoid = 100 / (1 + np.exp(-k * z_score))
    
    if index is not None:
        sigmoid.index = index
    
    return sigmoid.fillna(50.0)

def normalize_volume_0_100(series, window=50, index=None):
    """
    Hacim verilerini log transformation + min-max ile 0-100 aralığında normalize eder
    Hacim patlamalarını yumuşatır
    """
    if series.empty or series.isna().all():
        return pd.Series(50.0, index=index if index is not None else series.index)
    
    log_series = np.log1p(np.maximum(series, 0))  # log(1 + x), negatif değerleri 0 yap
    
    rolling_min = log_series.rolling(window, min_periods=1).min()
    rolling_max = log_series.rolling(window, min_periods=1).max()
    
    range_val = rolling_max - rolling_min
    normalized = np.where(range_val > 1e-8, 
                         ((log_series - rolling_min) / range_val) * 100,
                         50.0)
    
    normalized = pd.Series(normalized, index=series.index)
    
    if index is not None:
        normalized.index = index
    
    return normalized.fillna(50.0)

def normalize_volatility_0_100(series, window=200, index=None):
    """
    Volatilite verilerini quantile normalizasyon ile 0-100 aralığında normalize eder
    Percentile tabanlı, outlier'lara çok dayanıklı
    """
    if series.empty or series.isna().all():
        return pd.Series(50.0, index=index if index is not None else series.index)
    
    def percentile_rank(x):
        if len(x) < 10:
            return 50.0
        current_val = x.iloc[-1]
        rank = (current_val > x).sum() / len(x)
        return rank * 100
    
    normalized = series.rolling(window, min_periods=10).apply(percentile_rank, raw=False)
    
    if index is not None:
        normalized.index = index
    
    return normalized.fillna(50.0)
