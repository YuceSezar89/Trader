"""
do_break_sl_sweep_bt.py'nin RSI_Cross karşılığı — "MTF-onaylı do_kirilimi'nde
SL işe yaramadı, RSI_Cross'ta işe yarar mı?" sorusunu test eder.

RSI_Cross'un GERÇEK tetikleme mantığı (`signals/signal_engine.py::
rsi_crossover_signal` — Config.RSI_FAST_WINDOW(9)/RSI_SLOW_WINDOW(24)
kesişimi) doğrudan cagg_15m üzerinde yeniden üretiliyor (SignalFilter kapısı
HARİÇ — sadece çekirdek crossover koşulu, `signals` tablosundaki gerçek
sinyallerin bar-içi fiyat yolu saklanmadığı için SL simülasyonu ancak ham
veriden yeniden üretimle mümkün). Hem Long hem Short yönü dahil.

Aynı disiplin: cagg_15m, 45 gün, 24h/96-bar ufuk, ${POSITION_USD} pozisyon +
gerçek fee, split-period sağlamlık.
"""
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from indicators.core import calculate_rsi  # pylint: disable=wrong-import-position
from research.pattern_lab.features import _atr  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_bt import DAYS, HORIZON_BARS, MIN_BARS, _fetch  # pylint: disable=wrong-import-position
from research.pattern_lab.do_break_gauss_economic_bt import POSITION_USD, ROUND_TRIP_FEE  # pylint: disable=wrong-import-position

SL_MULTIPLES = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
WARMUP_BARS = 30  # RSI(24)/ATR(14) ısınma payı


def _rsi_cross_events(fast: np.ndarray, slow: np.ndarray) -> list[tuple[int, str]]:
    events = []
    n = len(fast)
    for i in range(WARMUP_BARS, n):
        if not (np.isfinite(fast[i - 1]) and np.isfinite(slow[i - 1])
                and np.isfinite(fast[i]) and np.isfinite(slow[i])):
            continue
        if fast[i - 1] < slow[i - 1] and fast[i] > slow[i]:
            events.append((i, "Long"))
        elif fast[i - 1] > slow[i - 1] and fast[i] < slow[i]:
            events.append((i, "Short"))
    return events


def _apply_signal_filter(events: list[tuple[int, str]], high: np.ndarray, low: np.ndarray) -> list[tuple[int, str]]:
    """signals/signal_filter.py::SignalFilter.check ile BİREBİR mantık — DB'siz,
    deterministik replay (do_kirilimi.py'nin kendi gate'lerini ham veride
    yeniden ürettiği desenle aynı). Kural: Long geçerli → bu bar'ın high'ı, EN
    SON Short olayının high'ından büyük; Short geçerli → bu bar'ın low'u, EN
    SON Long olayının low'undan küçük. Referans (last_long/last_short) GEÇTİ/
    KALDI farketmeksizin HER yeni olayın kendi high/low'uyla güncellenir —
    ilk olayda karşıt referans yoksa her zaman geçersiz (PineScript na koruması).
    `events` bar indeksine göre artan sırada olmalı (zaten öyle üretiliyor)."""
    last_long: Optional[tuple[float, float]] = None
    last_short: Optional[tuple[float, float]] = None
    passed: list[tuple[int, str]] = []
    for i, direction in events:
        if direction == "Long":
            if last_short is not None and high[i] > last_short[0]:
                passed.append((i, direction))
            last_long = (high[i], low[i])
        else:
            if last_long is not None and low[i] < last_long[1]:
                passed.append((i, direction))
            last_short = (high[i], low[i])
    return passed


def _simulate_sl_exit(direction: str, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                       entry_idx: int, entry_price: float, atr_val: float,
                       sl_mult: float, horizon: int) -> tuple[float, str]:
    sl = entry_price - sl_mult * atr_val if direction == "Long" else entry_price + sl_mult * atr_val
    last_i = min(entry_idx + horizon, len(close) - 1)
    for j in range(entry_idx + 1, last_i + 1):
        hit = low[j] <= sl if direction == "Long" else high[j] >= sl
        if hit:
            return sl, "stop_loss"
    return close[last_i], "timeout"


def _signed_ret(direction: str, exit_price: float, entry_price: float) -> float:
    ret = exit_price / entry_price - 1
    return ret if direction == "Long" else -ret


def _dollar_stats(pnls: np.ndarray, days_span: float) -> dict:
    if len(pnls) == 0:
        return {"n": 0}
    total = float(pnls.sum())
    return {
        "n": len(pnls),
        "wr": round(float((pnls > 0).mean() * 100), 1),
        "avg_usd": round(float(pnls.mean()), 3),
        "total_usd": round(total, 1),
        "usd_per_month": round(total / days_span * 30, 1) if days_span > 0 else 0.0,
    }


def _report(label: str, events: list, days_span: float) -> None:
    print(f"\n=== {label} (n={len(events)}) ===")
    if not events:
        return

    blind_pnls = []
    sl_pnls: dict[float, list] = {sl: [] for sl in SL_MULTIPLES}
    sl_reasons: dict[float, dict] = {sl: {} for sl in SL_MULTIPLES}

    for direction, h, l, c, atr_val, i, entry_price in events:
        last_i = min(i + HORIZON_BARS, len(c) - 1)
        blind_pnls.append(_signed_ret(direction, c[last_i], entry_price) * POSITION_USD - ROUND_TRIP_FEE)

        for sl in SL_MULTIPLES:
            exit_price, reason = _simulate_sl_exit(direction, h, l, c, i, entry_price, atr_val, sl, HORIZON_BARS)
            sl_reasons[sl][reason] = sl_reasons[sl].get(reason, 0) + 1
            sl_pnls[sl].append(_signed_ret(direction, exit_price, entry_price) * POSITION_USD - ROUND_TRIP_FEE)

    print(f"{'yöntem':24} {'n':>6} {'WR%':>6} {'ort $/işlem':>12} {'toplam $':>10} {'$/ay':>10}")
    s = _dollar_stats(np.array(blind_pnls), days_span)
    print(f"{'kör 24h bekleme':24} {s['n']:>6} {s['wr']:>6} {s['avg_usd']:>12} "
          f"{s['total_usd']:>10} {s['usd_per_month']:>10}")
    for sl in SL_MULTIPLES:
        s = _dollar_stats(np.array(sl_pnls[sl]), days_span)
        print(f"{'SL='+str(sl)+'×ATR (TP yok)':24} {s['n']:>6} {s['wr']:>6} {s['avg_usd']:>12} "
              f"{s['total_usd']:>10} {s['usd_per_month']:>10}")

    print("\n-- çıkış nedeni dağılımı --")
    for sl in SL_MULTIPLES:
        total = sum(sl_reasons[sl].values())
        breakdown = ", ".join(f"{r}=%{c/total*100:.0f}" for r, c in
                               sorted(sl_reasons[sl].items(), key=lambda x: -x[1]))
        print(f"  SL={sl}×ATR: {breakdown}")


def run():
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)")
    print(f"RSI_Cross tanımı: fast={Config.RSI_FAST_WINDOW}, slow={Config.RSI_SLOW_WINDOW}\n")

    all_events = []    # (direction, h, l, c, atr_val, i, entry_price, ts)
    clean_events = []  # SignalFilter'dan geçenler (canlıya en yakın küme)

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
        filtered_events = set(_apply_signal_filter(raw_events, h, l))

        for i, direction in raw_events:
            if i + HORIZON_BARS >= len(c) or not (np.isfinite(atr[i]) and atr[i] > 0):
                continue
            rec = (direction, h, l, c, atr[i], i, c[i], ts.iloc[i])
            all_events.append(rec)
            if (i, direction) in filtered_events:
                clean_events.append(rec)

    t_min = min(t for *_, t in all_events)
    t_max = max(t for *_, t in all_events)
    days_span = (t_max - t_min).total_seconds() / 86400
    print(f"dönem: {t_min} .. {t_max} ({days_span:.1f} gün)")

    _report("BASELINE (ham, SignalFilter'sız, Long+Short)", [e[:-1] for e in all_events], days_span)
    _report("TEMİZ (SignalFilter geçen, canlıya en yakın küme)", [e[:-1] for e in clean_events], days_span)

    mid = t_min + (t_max - t_min) / 2
    half_days = days_span / 2
    print("\n\n──────── SPLIT-PERIOD SAĞLAMLIK (TEMİZ küme) ────────")
    first = [e[:-1] for e in clean_events if e[-1] < mid]
    second = [e[:-1] for e in clean_events if e[-1] >= mid]
    _report("TEMİZ — ilk yarı", first, half_days)
    _report("TEMİZ — ikinci yarı", second, half_days)


if __name__ == "__main__":
    run()
