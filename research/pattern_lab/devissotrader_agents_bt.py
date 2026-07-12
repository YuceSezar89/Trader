"""
Hocanın (Selçuk Deveci) paylaştığı DevisSoTrader kodundaki
(/Users/yusuf/Documents/Sezar/control_decision.py::MultiAgentSystem) 3 kural-tabanlı
ajanın (trend_follower, mean_reversion, volatility_breakout) TRader'ın gerçek
verisiyle bağımsız testi. Sentiment ajanı gerçek veri kullanmadığı (sadece
momentum'un tanh'i) için dahil edilmedi — [[project-turtle-traders]].

Eşikler ve mantık control_decision.py'den BİREBİR alındı, tek fark:
DevisSoTrader'ın kendi SQLite/ccxt altyapısı yerine TRader'ın cagg_{interval}
tabloları kullanılıyor (hiç çalıştırılmamıştı, trades.log boştu).

Look-ahead yok: göstergeler rolling/ewm ile SADECE geçmiş barlarla hesaplanıyor
(bar i'nin sma/rsi/macd değeri close[i] dahil geçmişten), ileri getiri i'den
i+h'ye ayrı ölçülüyor. Baseline = ilgili TF'deki TÜM barlar, sinyal = ajanın
ürettiği buy/sell anları. Kronolojik ilk/ikinci yarı ile sağlamlık kontrolü
(vol_exhaustion_bt / do_open_streak_split_check ile aynı desen).
"""

import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from indicators.core import (  # pylint: disable=wrong-import-position
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi_sma,
)
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position

DAYS = 270
MIN_BARS = 250
WARMUP = 200

# trend_follower — control_decision.py::_trend_follower_decision
TF_TREND = "4h"
MIN_MA_SEPARATION = 0.02
CONFIRMATION_PERIODS = 3
TREND_HORIZONS = {"1g": 6, "3g": 18, "7g": 42}

# mean_reversion — control_decision.py::_mean_reversion_decision
TF_MR = "15m"
OVERSOLD, OVERBOUGHT = 30, 70
BB_WIDTH_MIN = 0.03
MR_HORIZONS = {"4s": 16, "12s": 48, "24s": 96}

# volatility_breakout — control_decision.py::_volatility_breakout_decision
TF_VB = "15m"
VOLUME_SURGE_THRESHOLD = 2.0
BREAKOUT_PERIODS = 20
MIN_CONSOLIDATION_PERIODS = 5
VB_HORIZONS = MR_HORIZONS


def _fetch(interval: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, open, high, low, close, volume
        FROM cagg_{interval}
        WHERE bucket > NOW() - INTERVAL '{DAYS} days'
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def _trend_state_series(g: pd.DataFrame) -> pd.Series:
    """control_decision.py::_trend_follower_decision'daki uptrend/downtrend
    durumu — MACD onayından ÖNCEKİ ham durum, rejim testinde tetikleyiciden
    bağımsız kullanılabilir (rsi_cross_trend_meanrev_regime_bt.py).
    +1=uptrend, -1=downtrend, 0=nötr."""
    close = g["close"]
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    ma_separation = sma_20 / sma_50 - 1
    uptrend = (close > sma_20) & (sma_20 > sma_50) & (ma_separation > MIN_MA_SEPARATION)
    downtrend = (close < sma_20) & (sma_20 < sma_50) & (ma_separation < -MIN_MA_SEPARATION)
    return (uptrend.astype(int) - downtrend.astype(int)).fillna(0)


def _trend_follower_signals(g: pd.DataFrame) -> tuple:
    _, _, macd_hist = calculate_macd(g)
    state = _trend_state_series(g)

    hist_pos = macd_hist > 0
    hist_neg = macd_hist < 0
    macd_up = hist_pos & hist_pos.shift(1).fillna(False) & hist_pos.shift(2).fillna(False)
    macd_down = hist_neg & hist_neg.shift(1).fillna(False) & hist_neg.shift(2).fillna(False)

    buy = ((state == 1) & macd_up).fillna(False).to_numpy()
    sell = ((state == -1) & macd_down).fillna(False).to_numpy()
    return buy, sell


def _mean_reversion_state_series(g: pd.DataFrame) -> pd.Series:
    """control_decision.py::_mean_reversion_decision'daki aşırı-satım/alım
    durumu — rejim testinde tetikleyiciden bağımsız kullanılabilir.
    +1=aşırı-satım(oversold), -1=aşırı-alım(overbought), 0=nötr."""
    close = g["close"]
    rsi = calculate_rsi_sma(g, period=14)
    sma_20, bb_upper, bb_lower = calculate_bollinger_bands(g, period=20, num_std=2)
    bb_width = (bb_upper - bb_lower) / sma_20
    bb_position = (close - bb_lower) / (bb_upper - bb_lower)

    mr_up = (rsi < OVERSOLD) & (bb_position < 0.1) & (bb_width > BB_WIDTH_MIN)
    mr_down = (rsi > OVERBOUGHT) & (bb_position > 0.9) & (bb_width > BB_WIDTH_MIN)
    return (mr_up.astype(int) - mr_down.astype(int)).fillna(0)


def _mean_reversion_signals(g: pd.DataFrame) -> tuple:
    state = _mean_reversion_state_series(g)
    buy = (state == 1).fillna(False).to_numpy()
    sell = (state == -1).fillna(False).to_numpy()
    return buy, sell


def _is_consolidating_series(g: pd.DataFrame) -> pd.Series:
    """control_decision.py::_volatility_breakout_decision'daki konsolidasyon
    durumu — sinyal tetikleyicisinden BAĞIMSIZ, rejim testinde (regime_score_bt
    ailesiyle aynı desen) tek başına da kullanılabilir."""
    high, low, close = g["high"], g["low"], g["close"]
    n_high = high.rolling(BREAKOUT_PERIODS).max()
    n_low = low.rolling(BREAKOUT_PERIODS).min()
    price_range = (n_high - n_low) / close
    return price_range.rolling(MIN_CONSOLIDATION_PERIODS).mean() < 0.05


def _volatility_breakout_signals(g: pd.DataFrame) -> tuple:
    high, low, close, open_, volume = g["high"], g["low"], g["close"], g["open"], g["volume"]

    volume_ratio = volume / volume.rolling(20).mean()

    n_high = high.rolling(BREAKOUT_PERIODS).max()
    n_low = low.rolling(BREAKOUT_PERIODS).min()
    is_consolidating = _is_consolidating_series(g)

    volume_surge = volume_ratio > VOLUME_SURGE_THRESHOLD
    upside = (close > n_high * 0.99) & (close > open_) & volume_surge
    downside = (close < n_low * 1.01) & (close < open_) & volume_surge

    buy = (upside & is_consolidating).fillna(False).to_numpy()
    sell = (downside & is_consolidating).fillna(False).to_numpy()
    return buy, sell


def _run_agent(name: str, interval: str, horizons: dict, signal_fn) -> None:
    df = _fetch(interval)
    t_min, t_max = df["ts"].min(), df["ts"].max()
    mid = t_min + (t_max - t_min) / 2
    print(
        f"\n{'='*74}\n{name} ({interval})  —  {df['symbol'].nunique()} sembol, {len(df):,} bar\n"
        f"dönem: {t_min} .. {t_max} | orta nokta: {mid}\n{'='*74}"
    )

    labels = ["baseline", "buy", "sell"]
    halves = ["tum", "ilk_yari", "ikinci_yari"]
    res = {h: {lbl: {half: [] for half in halves} for lbl in labels} for h in horizons}
    n_syms = 0

    for _sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1
        ts_np = g["ts"].to_numpy()
        close = g["close"].to_numpy(float)
        max_h = max(horizons.values())
        buy_sig, sell_sig = signal_fn(g)

        all_idx = np.arange(WARMUP, len(g) - max_h)
        idx_by_label = {
            "baseline": all_idx,
            "buy": all_idx[buy_sig[all_idx]],
            "sell": all_idx[sell_sig[all_idx]],
        }

        for lbl, idxs in idx_by_label.items():
            if len(idxs) == 0:
                continue
            first_mask = ts_np[idxs] < np.datetime64(mid)
            for h_name, h_bars in horizons.items():
                rets = close[idxs + h_bars] / close[idxs] - 1
                if lbl == "sell":
                    rets = -rets
                res[h_name][lbl]["tum"].append(rets)
                res[h_name][lbl]["ilk_yari"].append(rets[first_mask])
                res[h_name][lbl]["ikinci_yari"].append(rets[~first_mask])

    print(f"analiz edilen sembol: {n_syms}\n")
    for h_name in horizons:
        print(f"── ufuk: {h_name} ──")
        print(f"{'grup':10} {'dönem':12} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
        for lbl in labels:
            for half in halves:
                arrs = res[h_name][lbl][half]
                rets = np.concatenate(arrs) if arrs else np.array([])
                s = _stats(rets)
                print(
                    f"{lbl:10} {half:12} {s.get('n',0):>7} {s.get('wr',0):>6} "
                    f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}"
                )
        print()


def run() -> None:
    _run_agent("trend_follower", TF_TREND, TREND_HORIZONS, _trend_follower_signals)
    _run_agent("mean_reversion", TF_MR, MR_HORIZONS, _mean_reversion_signals)
    _run_agent("volatility_breakout", TF_VB, VB_HORIZONS, _volatility_breakout_signals)


if __name__ == "__main__":
    run()
