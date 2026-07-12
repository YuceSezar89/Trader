"""
rsi_cross_btc_regime_bt.py'nin devamı — o testte BTC z-score (15m, EMA200/std200)
rejim filtresi işe yaramamıştı (kısa-pencereli "aşırı sapma" ölçüsü, ~11 günlük
yavaş bir çöküşü yakalayamadı). Bu script, /Users/yusuf/Downloads/hüseyin sistemi/
panel_topsis...html'deki Matrix/Koopman katmanının esinlendirdiği fikri (piyasa
YAPISI: trend/kırılma/yatay) kendi altyapımızdaki bir araçla, endüstri standardı
ADX+DI (`indicators/core.py::calculate_adx`) ile basitleştirir — Koopman/TOPSIS/
TBRS/BBO gibi hiç doğrulanmamış özel göstergeleri PORTE ETMEDEN.

ADX, kalıcı YÖN+GÜÇ ölçmek için tasarlanmıştır (kısa-pencereli z-score'un aksine)
— BTC'nin 1h serisinde hesaplanıyor (15m'den daha yavaş, gürültüyü azaltmak için).
Kural: ADX>25 VE -DI>+DI ise "güçlü düşüş trendi" — bu rejimde Long RSI_Cross
sinyali elenir (simetrik olarak ADX>25 VE +DI>-DI ise Short elenir).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from indicators.core import calculate_adx, calculate_rsi  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_bt import (  # pylint: disable=wrong-import-position
    DAYS,
    HORIZON_BARS,
    MIN_BARS,
)
from research.pattern_lab.features import _atr  # pylint: disable=wrong-import-position
from research.pattern_lab.mtf_helpers import TF_DURATION  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_combined_sl_bt import (  # pylint: disable=wrong-import-position
    _fetch_with_volume,
    _report,
)
from research.pattern_lab.rsi_cross_sl_sweep_bt import (  # pylint: disable=wrong-import-position
    _apply_signal_filter,
    _rsi_cross_events,
)
from research.pattern_lab.rsi_cross_vpmv_jump_bt import (
    _fetch_symbol_history,  # pylint: disable=wrong-import-position
)
from utils.vpmv import compute_series  # pylint: disable=wrong-import-position

ADX_TF = "1h"
ADX_LEN = 14
ADX_TREND_THRESHOLD = 25.0
EVENT_TF_DURATION = TF_DURATION["15m"]  # RSI_Cross olayının kendi barının (cagg_15m) süresi


def _build_btc_adx_regime() -> tuple:
    """BTCUSDT'nin ADX_TF serisinde ADX+DI hesaplar. Döner: (ts_arr, regime_arr)
    regime_arr elemanları: 'bearish_trend' | 'bullish_trend' | 'neutral'."""
    hist = _fetch_symbol_history("BTCUSDT", ADX_TF)
    hist = hist.sort_values("ts").reset_index(drop=True)
    adx, plus_di, minus_di = calculate_adx(hist, adxlen=ADX_LEN, dilen=ADX_LEN)
    adx_np, plus_np, minus_np = adx.to_numpy(), plus_di.to_numpy(), minus_di.to_numpy()

    regime = np.full(len(hist), "neutral", dtype=object)
    strong = adx_np > ADX_TREND_THRESHOLD
    regime[strong & (minus_np > plus_np)] = "bearish_trend"
    regime[strong & (plus_np > minus_np)] = "bullish_trend"
    return hist["ts"].to_numpy(), regime


def _regime_before(ts_arr: np.ndarray, regime_arr: np.ndarray, event_ts) -> str:
    """event_ts, RSI_Cross olayının kendi (15m) barının AÇILIŞ zamanı — olayın
    GERÇEK/canlıda bilindiği an bu barın KAPANIŞI (event_ts+15m). ADX_TF (1h)
    barının o ana kadar GERÇEKTEN kapanmış sayılması için bucket+1h <= o an
    şartı aranır — DÜZELTME (11 Tem 2026, bkz. v2-24): önceki hali bucket'ı
    doğrudan event_ts ile kıyaslıyordu, henüz kapanmamış bir 1h barını
    kullanma (look-ahead) riskini taşıyordu."""
    event_close = np.datetime64(event_ts) + EVENT_TF_DURATION
    cutoff = event_close - TF_DURATION[ADX_TF]
    idx = np.searchsorted(ts_arr, cutoff, side="right") - 1
    if idx < 0:
        return None
    return regime_arr[idx]


def _regime_ok(direction: str, regime: str) -> bool:
    if regime is None:
        return False
    if direction == "Long":
        return regime != "bearish_trend"
    return regime != "bullish_trend"


def run():
    btc_ts_arr, btc_regime_arr = _build_btc_adx_regime()
    print(f"BTC {ADX_TF} ADX serisi: {len(btc_ts_arr):,} bar\n")

    df = _fetch_with_volume()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    clean_events = []  # (direction, h, l, c, atr_val, i, entry_price, ts, jump, regime)

    for _sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue

        ts = g["ts"]
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)
        fast = calculate_rsi(g, period=Config.RSI_FAST_WINDOW).to_numpy()
        slow = calculate_rsi(g, period=Config.RSI_SLOW_WINDOW).to_numpy()
        atr = _atr(g[["high", "low", "close"]]).to_numpy()

        raw_events = _rsi_cross_events(fast, slow)
        filtered_events = _apply_signal_filter(raw_events, h, l)
        if not filtered_events:
            continue

        series_long = compute_series(g, "Long")
        series_short = compute_series(g, "Short")

        for i, direction in filtered_events:
            if (
                i + HORIZON_BARS >= len(c)
                or i - 1 < 0
                or i + 1 >= len(g)
                or not (np.isfinite(atr[i]) and atr[i] > 0)
            ):
                continue
            series = series_long if direction == "Long" else series_short
            pre_v, post_v = series.iloc[i - 1], series.iloc[i + 1]
            if not (np.isfinite(pre_v) and np.isfinite(post_v)):
                continue

            regime = _regime_before(btc_ts_arr, btc_regime_arr, ts.iloc[i])
            clean_events.append(
                (
                    direction,
                    h,
                    l,
                    c,
                    atr[i],
                    i,
                    c[i],
                    ts.iloc[i],
                    post_v - pre_v,
                    regime,
                )
            )

    t_min = min(e[7] for e in clean_events)
    t_max = max(e[7] for e in clean_events)
    mid = t_min + (t_max - t_min) / 2
    oos_days = (t_max - mid).total_seconds() / 86400
    print(f"dönem: {t_min} .. {t_max}")
    print(f"kalibrasyon (in-sample): {t_min} .. {mid}")
    print(f"test (out-of-sample):    {mid} .. {t_max}  ({oos_days:.1f} gün)\n")

    is_events = [e for e in clean_events if e[7] < mid]
    oos_events = [e for e in clean_events if e[7] >= mid]
    jump_threshold = float(np.percentile([e[8] for e in is_events], 66.7))
    print(f"in-sample olay: {len(is_events)} | SABİT VPMV eşiği: {jump_threshold:.2f}\n")

    vpmv_only = [e for e in oos_events if e[8] >= jump_threshold]
    vpmv_regime = [e for e in vpmv_only if _regime_ok(e[0], e[9])]

    regime_counts = {}
    for e in oos_events:
        regime_counts[e[9]] = regime_counts.get(e[9], 0) + 1
    print(f"OOS rejim dağılımı (tüm olaylar): {regime_counts}\n")

    def _by_dir(events: list, direction: str) -> list:
        return [e for e in events if e[0] == direction]

    oos_mid = mid + (t_max - mid) / 2
    half_days = oos_days / 2

    for dir_label, direction in (
        ("LONG+SHORT (birlikte)", None),
        ("SADECE LONG", "Long"),
        ("SADECE SHORT", "Short"),
    ):
        vo = vpmv_only if direction is None else _by_dir(vpmv_only, direction)
        vr = vpmv_regime if direction is None else _by_dir(vpmv_regime, direction)

        print(f"\n\n╔══════════ {dir_label} ══════════╗")
        print("\n════ TÜM OOS ════")
        _report("VPMV sıçraması (rejimsiz, v2-18)", [e[:7] for e in vo], oos_days)
        _report("VPMV + ADX rejim filtresi (1h)", [e[:7] for e in vr], oos_days)

        print(f"\n════ OOS — İLK YARI (biten: {oos_mid}) ════")
        _report("VPMV sıçraması (rejimsiz)", [e[:7] for e in vo if e[7] < oos_mid], half_days)
        _report("VPMV + ADX rejim filtresi", [e[:7] for e in vr if e[7] < oos_mid], half_days)

        print(f"\n════ OOS — İKİNCİ YARI (başlayan: {oos_mid}) ════")
        _report("VPMV sıçraması (rejimsiz)", [e[:7] for e in vo if e[7] >= oos_mid], half_days)
        _report("VPMV + ADX rejim filtresi", [e[:7] for e in vr if e[7] >= oos_mid], half_days)


if __name__ == "__main__":
    run()
