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
