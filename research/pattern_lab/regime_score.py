"""
Piyasa rejimi kompozit skoru — bugünkü 3 ayrı denemenin (BTC z-score 15m,
BTC ADX 1h, piyasa genişliği ham/trailing-24h) hepsi KISMEN açıklayıcı ama
tek başına yetersiz çıkmasından sonra tasarlandı. İki YAVAŞ/smoothed
bileşen birleştiriliyor — rejim değişimlerinin gün mertebesinde (10-11 gün)
olduğu gözlemine dayanarak, saat-mertebesi ölçütlerden kaçınılıyor:

1. Piyasa genişliği — trailing-24h (96 bar @ 15m) pozitif getirili sembol
   oranı, 3 GÜNLÜK EMA (span=288 bar) ile yumuşatılmış — whipsaw azaltılıyor.
2. BTC'nin çok-günlük trendi — 4h barlarından türetilen GÜNLÜK kapanış,
   fiyatın SMA(20 gün)'e göre % konumu — BİR GÜN GECİKMELİ (look-ahead yok,
   sadece tam biten günün verisi kullanılıyor).

İkisi ayrı ayrı 0-100 percentile rank'e çevrilip ortalanıyor — 50=nötr,
düşük=ayı rejimi, yüksek=boğa rejimi.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.rsi_cross_vpmv_jump_bt import _fetch_symbol_history  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_market_breadth_bt import _build_breadth_series  # pylint: disable=wrong-import-position

BREADTH_EMA_SPAN = 288  # 3 gün @ 15m
BTC_DAILY_SMA_DAYS = 20


def _rolling_rank_0_100(series: pd.Series, window: int = 200) -> pd.Series:
    def _rank(x):
        return (x <= x[-1]).mean() * 100.0
    return series.rolling(window, min_periods=30).apply(_rank, raw=True)


def _build_btc_daily_trend() -> pd.DataFrame:
    """BTC'nin 4h barlarından günlük kapanış türetip SMA(20 gün)'e göre %
    konumunu hesaplar — BİR GÜN GECİKMELİ (sadece tam biten günün verisi,
    look-ahead yok). Döner: date -> trend_pct (lagged)."""
    hist = _fetch_symbol_history("BTCUSDT", "4h")
    hist = hist.sort_values("ts").reset_index(drop=True)
    hist["date"] = pd.to_datetime(hist["ts"]).dt.date
    daily = hist.groupby("date")["close"].last().reset_index()
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["sma"] = daily["close"].rolling(BTC_DAILY_SMA_DAYS).mean()
    daily["trend_pct"] = (daily["close"] - daily["sma"]) / daily["sma"] * 100.0
    daily["trend_pct_lagged"] = daily["trend_pct"].shift(1)  # bugünün verisini KULLANMA
    return daily[["date", "trend_pct_lagged"]]


def build_regime_score(df_all_symbols: pd.DataFrame) -> pd.Series:
    """df_all_symbols: symbol/ts/close (ve varsa başka) kolonlarını içeren,
    TÜM sembollerin 15m verisi (ör. rsi_cross_combined_sl_bt._fetch_with_volume()).
    Döner: ts -> regime_score (0-100, 50=nötr) Series'i."""
    breadth = _build_breadth_series(df_all_symbols)
    breadth_smooth = breadth.ewm(span=BREADTH_EMA_SPAN, adjust=False).mean()
    breadth_rank = _rolling_rank_0_100(breadth_smooth)

    btc_daily = _build_btc_daily_trend()
    date_to_trend = dict(zip(btc_daily["date"], btc_daily["trend_pct_lagged"]))

    ts_index = breadth_rank.index
    btc_trend_series = pd.Series(
        [date_to_trend.get(pd.Timestamp(t).date(), np.nan) for t in ts_index],
        index=ts_index,
    )
    btc_trend_rank = _rolling_rank_0_100(btc_trend_series)

    composite = (breadth_rank + btc_trend_rank) / 2.0
    return composite
