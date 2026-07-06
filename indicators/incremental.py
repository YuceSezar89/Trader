"""
İndikatörlerin incremental (tek adımlı) hesaplanması.

Amaç: canlı akışta her bar kapanışında add_all_indicators'ı TÜM buffer üzerinde
yeniden hesaplamak yerine (ölçüldü: ~48ms/çağrı, 1000 barlık 1m buffer için —
200 sembolün 1m mumu aynı dakikada kapanınca bu ~9.7s CPU işi demek), sadece
yeni barın indikatör değerlerini önceki durumdan O(1) hesaplamak.

indicators/core.py'deki mevcut fonksiyonlara DOKUNULMADI — oradaki formüller
burada bootstrap için bilerek tekrar kullanıldı (kopyalandı), mevcut çağıranları
(backtest, signal_processor, panel) etkilememek için. İlk yükleme (soğuk
başlangıç, _load_symbol_all_timeframes) hâlâ add_all_indicators kullanır; bu
modül SADECE canlı akıştaki (bar kapanışı) güncellemeler için kullanılacak.

MA200 ve momentum (ROC) burada YOK — bunlar N-bar-önceki ham fiyata ihtiyaç
duyar, state gerektirmez; çağıran taraf buffer'dan doğrudan okuyup ekler.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import Config

_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
_ST_ATR_LENGTH = 10
_ST_FACTOR = 3.0
_ADX_DILEN = 14
_ADX_ADXLEN = 14

# State'in kendi içinde biriken floating-point farkını sınırlamak için: her bu kadar
# incremental adımdan sonra state ground-truth'tan (tam yeniden hesaplama) yeniden
# bootstrap edilir. Canlıda ölçüldü: birkaç adımda ~1e-6 mertebesinde fark oluşabiliyor
# (pandas'ın vektörel .ewm()'i ile elle yazılan iteratif formül arasındaki işlem sırası
# farkından) — resync bunu periyodik olarak sıfırlar.
RESYNC_INTERVAL = 200


@dataclass
class IndicatorState:
    """Bir sembol+TF çifti için incremental hesaplama durumu."""
    rsi_fast_avg_gain: float
    rsi_fast_avg_loss: float
    rsi_slow_avg_gain: float
    rsi_slow_avg_loss: float
    ema_fast: float
    ema_slow: float
    macd_signal: float
    prev_macd: float
    atr: float
    adx_truerange: float
    adx_plus_dm_rma: float
    adx_minus_dm_rma: float
    adx_rma: float
    st_atr: float
    st_upper: float
    st_lower: float
    st_direction: float
    ha_open: float
    ha_close: float
    prev_close: float
    prev_high: float
    prev_low: float
    steps_since_bootstrap: int = 0


def _wilder_rma_series(series: pd.Series, period: int) -> pd.Series:
    """indicators/core.py:wilder_rma ile birebir aynı — bootstrap için bilerek kopyalandı."""
    result = [series.iloc[0]]
    for val in series.iloc[1:]:
        result.append((result[-1] * (period - 1) + val) / period)
    return pd.Series(result, index=series.index)


def bootstrap_state(df: pd.DataFrame) -> IndicatorState:
    """Geçmiş buffer'dan (zaten add_all_indicators ile dolu) incremental hesaplama
    için gereken gizli iç durumu çıkarır. Sembol+TF ilk yüklendiğinde BİR KEZ
    çalışır — maliyeti önemli değil, doğruluğu önemli."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)

    # --- RSI (dual period), Wilder/EMA smoothing (core.py:calculate_rsi ile aynı) ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    rsi_fast_avg_gain = gain.ewm(alpha=1 / Config.RSI_FAST_WINDOW, min_periods=Config.RSI_FAST_WINDOW, adjust=False).mean()
    rsi_fast_avg_loss = loss.ewm(alpha=1 / Config.RSI_FAST_WINDOW, min_periods=Config.RSI_FAST_WINDOW, adjust=False).mean()
    rsi_slow_avg_gain = gain.ewm(alpha=1 / Config.RSI_SLOW_WINDOW, min_periods=Config.RSI_SLOW_WINDOW, adjust=False).mean()
    rsi_slow_avg_loss = loss.ewm(alpha=1 / Config.RSI_SLOW_WINDOW, min_periods=Config.RSI_SLOW_WINDOW, adjust=False).mean()

    # --- MACD (core.py:calculate_macd ile aynı) ---
    ema_fast = close.ewm(span=_MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=_MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=_MACD_SIGNAL, adjust=False).mean()

    # --- ATR, genel (core.py:calculate_atr ile aynı) ---
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / Config.ATR_PERIOD, min_periods=Config.ATR_PERIOD, adjust=False).mean()

    # --- ADX (core.py:calculate_adx ile aynı) ---
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    adx_truerange_s = _wilder_rma_series(tr, _ADX_DILEN)
    adx_plus_dm_rma_s = _wilder_rma_series(plus_dm, _ADX_DILEN)
    adx_minus_dm_rma_s = _wilder_rma_series(minus_dm, _ADX_DILEN)
    plus = 100 * adx_plus_dm_rma_s / adx_truerange_s
    minus = 100 * adx_minus_dm_rma_s / adx_truerange_s
    sum_ = plus + minus
    dx = (plus - minus).abs() / sum_.replace(0, 1)
    adx_rma_s = _wilder_rma_series(dx, _ADX_ADXLEN)

    # --- SuperTrend (core.py:calculate_supertrend ile aynı, kendi özel ATR'si) ---
    st_alpha = 1.0 / _ST_ATR_LENGTH
    st_atr_s = tr.copy().astype(float)
    st_atr_s.iloc[:_ST_ATR_LENGTH] = np.nan
    st_atr_s.iloc[_ST_ATR_LENGTH - 1] = tr.iloc[:_ST_ATR_LENGTH].mean()
    for i in range(_ST_ATR_LENGTH, len(tr)):
        st_atr_s.iloc[i] = st_alpha * tr.iloc[i] + (1 - st_alpha) * st_atr_s.iloc[i - 1]

    hl2 = (high + low) / 2
    upper_b = hl2 + _ST_FACTOR * st_atr_s
    lower_b = hl2 - _ST_FACTOR * st_atr_s

    st_upper_s = pd.Series(np.nan, index=df.index, dtype=float)
    st_lower_s = pd.Series(np.nan, index=df.index, dtype=float)
    st_direction_s = pd.Series(np.nan, index=df.index, dtype=float)

    s = _ST_ATR_LENGTH - 1
    st_upper_s.iloc[s] = upper_b.iloc[s]
    st_lower_s.iloc[s] = lower_b.iloc[s]
    st_direction_s.iloc[s] = 1.0

    for i in range(s + 1, len(df)):
        ub, lb = upper_b.iloc[i], lower_b.iloc[i]
        pu, pl = st_upper_s.iloc[i - 1], st_lower_s.iloc[i - 1]
        prev_close_i = close.iloc[i - 1]
        prev_dir = st_direction_s.iloc[i - 1]

        st_upper_s.iloc[i] = ub if ub < pu or prev_close_i > pu else pu
        st_lower_s.iloc[i] = lb if lb > pl or prev_close_i < pl else pl

        if prev_dir == -1:
            st_direction_s.iloc[i] = 1.0 if close.iloc[i] < st_lower_s.iloc[i] else -1.0
        else:
            st_direction_s.iloc[i] = -1.0 if close.iloc[i] > st_upper_s.iloc[i] else 1.0

    # --- Heikin-Ashi (core.py:calculate_ha ile aynı) ---
    ha_close_s = (open_ + high + low + close) / 4
    ha_open_vals = np.zeros(len(df))
    ha_open_vals[0] = (open_.iloc[0] + close.iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open_vals[i] = (ha_open_vals[i - 1] + ha_close_s.iloc[i - 1]) / 2

    return IndicatorState(
        rsi_fast_avg_gain=float(rsi_fast_avg_gain.iloc[-1]),
        rsi_fast_avg_loss=float(rsi_fast_avg_loss.iloc[-1]),
        rsi_slow_avg_gain=float(rsi_slow_avg_gain.iloc[-1]),
        rsi_slow_avg_loss=float(rsi_slow_avg_loss.iloc[-1]),
        ema_fast=float(ema_fast.iloc[-1]),
        ema_slow=float(ema_slow.iloc[-1]),
        macd_signal=float(macd_signal.iloc[-1]),
        prev_macd=float(macd.iloc[-1]),
        atr=float(atr.iloc[-1]),
        adx_truerange=float(adx_truerange_s.iloc[-1]),
        adx_plus_dm_rma=float(adx_plus_dm_rma_s.iloc[-1]),
        adx_minus_dm_rma=float(adx_minus_dm_rma_s.iloc[-1]),
        adx_rma=float(adx_rma_s.iloc[-1]),
        st_atr=float(st_atr_s.iloc[-1]),
        st_upper=float(st_upper_s.iloc[-1]),
        st_lower=float(st_lower_s.iloc[-1]),
        st_direction=float(st_direction_s.iloc[-1]),
        ha_open=float(ha_open_vals[-1]),
        ha_close=float(ha_close_s.iloc[-1]),
        prev_close=float(close.iloc[-1]),
        prev_high=float(high.iloc[-1]),
        prev_low=float(low.iloc[-1]),
    )


def update_state(state: IndicatorState, new_bar: dict) -> dict:
    """Yeni bar için O(1) incremental güncelleme — state'i in-place günceller,
    add_all_indicators ile aynı isimli kolonları içeren dict döndürür (ma200 ve
    momentum hariç — çağıran taraf buffer'dan N-bar-önce bakıp ekler)."""
    close = float(new_bar["close"])
    high = float(new_bar["high"])
    low = float(new_bar["low"])
    open_ = float(new_bar["open"])

    prev_close = state.prev_close
    prev_high = state.prev_high
    prev_low = state.prev_low

    result: dict = {}

    # --- RSI (dual period) ---
    delta = close - prev_close
    gain = max(delta, 0.0)
    loss = max(-delta, 0.0)

    alpha_fast = 1.0 / Config.RSI_FAST_WINDOW
    state.rsi_fast_avg_gain += alpha_fast * (gain - state.rsi_fast_avg_gain)
    state.rsi_fast_avg_loss += alpha_fast * (loss - state.rsi_fast_avg_loss)
    rsi_fast = 100.0 if state.rsi_fast_avg_loss == 0 else 100 - (100 / (1 + state.rsi_fast_avg_gain / state.rsi_fast_avg_loss))

    alpha_slow = 1.0 / Config.RSI_SLOW_WINDOW
    state.rsi_slow_avg_gain += alpha_slow * (gain - state.rsi_slow_avg_gain)
    state.rsi_slow_avg_loss += alpha_slow * (loss - state.rsi_slow_avg_loss)
    rsi_slow = 100.0 if state.rsi_slow_avg_loss == 0 else 100 - (100 / (1 + state.rsi_slow_avg_gain / state.rsi_slow_avg_loss))

    result[f"rsi_{Config.RSI_FAST_WINDOW}"] = rsi_fast
    result[f"rsi_{Config.RSI_SLOW_WINDOW}"] = rsi_slow
    result["rsi_change"] = rsi_fast - rsi_slow

    # --- MACD ---
    alpha_ema_fast = 2.0 / (_MACD_FAST + 1)
    alpha_ema_slow = 2.0 / (_MACD_SLOW + 1)
    alpha_ema_signal = 2.0 / (_MACD_SIGNAL + 1)
    state.ema_fast += alpha_ema_fast * (close - state.ema_fast)
    state.ema_slow += alpha_ema_slow * (close - state.ema_slow)
    macd = state.ema_fast - state.ema_slow
    state.macd_signal += alpha_ema_signal * (macd - state.macd_signal)
    result["macd"] = macd
    result["macd_signal"] = state.macd_signal
    result["macd_hist"] = macd - state.macd_signal
    result["macd_change"] = macd - state.prev_macd
    state.prev_macd = macd

    # --- ATR (genel) ---
    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
    alpha_atr = 1.0 / Config.ATR_PERIOD
    state.atr += alpha_atr * (tr - state.atr)
    result["atr"] = state.atr

    # --- ADX ---
    up = high - prev_high
    down = prev_low - low
    plus_dm = up if (up > down and up > 0) else 0.0
    minus_dm = down if (down > up and down > 0) else 0.0

    alpha_di = 1.0 / _ADX_DILEN
    state.adx_truerange += alpha_di * (tr - state.adx_truerange)
    state.adx_plus_dm_rma += alpha_di * (plus_dm - state.adx_plus_dm_rma)
    state.adx_minus_dm_rma += alpha_di * (minus_dm - state.adx_minus_dm_rma)

    plus_di = 100 * state.adx_plus_dm_rma / state.adx_truerange if state.adx_truerange != 0 else 0.0
    minus_di = 100 * state.adx_minus_dm_rma / state.adx_truerange if state.adx_truerange != 0 else 0.0
    sum_di = plus_di + minus_di
    dx = abs(plus_di - minus_di) / (sum_di if sum_di != 0 else 1.0)
    alpha_adx = 1.0 / _ADX_ADXLEN
    state.adx_rma += alpha_adx * (dx - state.adx_rma)

    result["adx"] = 100 * state.adx_rma
    result["plus_di"] = plus_di
    result["minus_di"] = minus_di

    # --- SuperTrend ---
    st_alpha = 1.0 / _ST_ATR_LENGTH
    state.st_atr += st_alpha * (tr - state.st_atr)
    hl2 = (high + low) / 2
    upper_b = hl2 + _ST_FACTOR * state.st_atr
    lower_b = hl2 - _ST_FACTOR * state.st_atr

    new_upper = upper_b if (upper_b < state.st_upper or prev_close > state.st_upper) else state.st_upper
    new_lower = lower_b if (lower_b > state.st_lower or prev_close < state.st_lower) else state.st_lower

    if state.st_direction == -1:
        new_direction = 1.0 if close < new_lower else -1.0
    else:
        new_direction = -1.0 if close > new_upper else 1.0

    state.st_upper = new_upper
    state.st_lower = new_lower
    state.st_direction = new_direction
    result["st_line"] = new_lower if new_direction == -1 else new_upper
    result["st_direction"] = new_direction

    # --- Heikin-Ashi ---
    ha_close = (open_ + high + low + close) / 4
    ha_open = (state.ha_open + state.ha_close) / 2
    result["ha_open"] = ha_open
    result["ha_close"] = ha_close
    result["ha_high"] = max(high, ha_open, ha_close)
    result["ha_low"] = min(low, ha_open, ha_close)
    state.ha_open = ha_open
    state.ha_close = ha_close

    # --- Önceki bar durumunu güncelle ---
    state.prev_close = close
    state.prev_high = high
    state.prev_low = low
    state.steps_since_bootstrap += 1

    return result
