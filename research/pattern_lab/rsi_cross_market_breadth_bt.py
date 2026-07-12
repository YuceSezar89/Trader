"""
rsi_cross_adx_regime_bt.py'nin devamı — BTC'nin KENDİ göstergeleri (z-score,
ADX/DI) Long'un ilk-yarı çöküşünü tam açıklayamadı/düzeltemedi. Elle yapılan
teşhiste (bu sohbette) en net ayrım BTC'den değil PİYASA GENİŞLİĞİNDEN
gelmişti: ilk yarıda 189 sembolün %81.5'i düşüyordu (medyan %-10.3), ikinci
yarıda %61.4'ü yükseliyordu. Bu script bu gözlemi resmi bir filtreye çevirip
test eder — kaç sembolün (trailing 24h, 96 bar @ 15m) pozitif getirili
olduğu (breadth_pct), evrendeki TÜM sembollerden hesaplanır (look-ahead yok
— her an SADECE o ana kadarki 96 bar kullanılıyor).

Eşik (33/67 percentile) SADECE ilk yarıdan (in-sample) türetilip ikinci
yarıya (out-of-sample) sabit uygulanıyor — VPMV eşiğiyle aynı disiplin.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from indicators.core import calculate_rsi  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_bt import (  # pylint: disable=wrong-import-position
    DAYS,
    HORIZON_BARS,
    MIN_BARS,
)
from research.pattern_lab.features import _atr  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_combined_sl_bt import (  # pylint: disable=wrong-import-position
    _fetch_with_volume,
    _report,
)
from research.pattern_lab.rsi_cross_sl_sweep_bt import (  # pylint: disable=wrong-import-position
    _apply_signal_filter,
    _rsi_cross_events,
)
from utils.vpmv import compute_series  # pylint: disable=wrong-import-position

BREADTH_LOOKBACK_BARS = HORIZON_BARS  # 96 bar @ 15m = 24h, mevcut ufukla tutarlı


def _build_breadth_series(df: pd.DataFrame) -> pd.Series:
    """Her 15m barında, evrendeki sembollerin kaçının trailing 24h getirisi
    pozitif olduğunu (0-100) döner. Look-ahead yok — sadece o ana kadarki
    BREADTH_LOOKBACK_BARS bar kullanılıyor."""
    wide = df.pivot_table(index="ts", columns="symbol", values="close")
    wide = wide.sort_index()
    trailing_ret = wide.pct_change(BREADTH_LOOKBACK_BARS)
    breadth = (trailing_ret > 0).sum(axis=1) / trailing_ret.notna().sum(axis=1) * 100.0
    return breadth


def _regime_ok(direction: str, breadth: float, low_th: float, high_th: float) -> bool:
    if breadth is None or not np.isfinite(breadth):
        return False
    if direction == "Long":
        return breadth >= low_th
    return breadth <= high_th


def run():
    df = _fetch_with_volume()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    breadth_series = _build_breadth_series(df)
    breadth_lookup = breadth_series.to_dict()
    print(
        f"Piyasa genişliği serisi: {len(breadth_series):,} zaman noktası "
        f"(ort=%{breadth_series.mean():.1f}, min=%{breadth_series.min():.1f}, max=%{breadth_series.max():.1f})\n"
    )

    clean_events = []  # (direction, h, l, c, atr_val, i, entry_price, ts, jump, breadth)

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

            breadth = breadth_lookup.get(ts.iloc[i])
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
                    breadth,
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

    is_breadth_vals = [e[9] for e in is_events if e[9] is not None and np.isfinite(e[9])]
    breadth_low = float(np.percentile(is_breadth_vals, 33.3))
    breadth_high = float(np.percentile(is_breadth_vals, 66.7))
    print(f"in-sample olay: {len(is_events)} | SABİT VPMV eşiği: {jump_threshold:.2f}")
    print(
        f"SABİT (OOS'a uygulanan) genişlik eşikleri: düşük=%{breadth_low:.1f} yüksek=%{breadth_high:.1f}\n"
    )

    vpmv_only = [e for e in oos_events if e[8] >= jump_threshold]
    vpmv_breadth = [e for e in vpmv_only if _regime_ok(e[0], e[9], breadth_low, breadth_high)]

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
        vb = vpmv_breadth if direction is None else _by_dir(vpmv_breadth, direction)

        print(f"\n\n╔══════════ {dir_label} ══════════╗")
        print("\n════ TÜM OOS ════")
        _report("VPMV sıçraması (rejimsiz, v2-18)", [e[:7] for e in vo], oos_days)
        _report("VPMV + piyasa genişliği filtresi", [e[:7] for e in vb], oos_days)

        print(f"\n════ OOS — İLK YARI (biten: {oos_mid}) ════")
        _report("VPMV sıçraması (rejimsiz)", [e[:7] for e in vo if e[7] < oos_mid], half_days)
        _report(
            "VPMV + piyasa genişliği filtresi", [e[:7] for e in vb if e[7] < oos_mid], half_days
        )

        print(f"\n════ OOS — İKİNCİ YARI (başlayan: {oos_mid}) ════")
        _report("VPMV sıçraması (rejimsiz)", [e[:7] for e in vo if e[7] >= oos_mid], half_days)
        _report(
            "VPMV + piyasa genişliği filtresi", [e[:7] for e in vb if e[7] >= oos_mid], half_days
        )


if __name__ == "__main__":
    run()
