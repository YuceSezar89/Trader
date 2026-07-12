"""
v2-18/v2-19'daki VPMV-sıçraması filtresinde bulunan look-ahead düzeltmesi:
jump = post[i+1] - pre[i-1] hesaplamak için sinyal barından (i) BİR SONRAKİ
barı (i+1) bilmek gerekiyor — ama backtest entry fiyatı olarak c[i] (sinyal
barının kendi kapanışı) kullanmıştı. Canlıda bu MÜMKÜN DEĞİL: filtre ancak
i+1 kapandıktan SONRA değerlendirilebilir, yani gerçek giriş c[i+1] olur
(veya o anki güncel fiyat), c[i] değil.

Bu script SADECE SHORT için üç versiyonu yan yana karşılaştırır:
1. ESKİ (look-ahead'li, jump=post[i+1]-pre[i-1], entry=c[i])
2. DÜZELTİLMİŞ-GECİKMELİ (aynı jump, ama entry=c[i+1] — gerçekçi ama 1 bar geç)
3. MUM İÇİ (kullanıcı fikri, 11 Tem 2026): jump_live=VPMV[i]-VPMV[i-1] — sinyal
   barının KENDİ kapanışını bir önceki barla kıyaslar, gelecek bar gerekmez,
   entry=c[i] (gecikmesiz, canlıda tam uygulanabilir).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from indicators.core import calculate_rsi  # pylint: disable=wrong-import-position
from utils.vpmv import compute_series  # pylint: disable=wrong-import-position
from research.pattern_lab.features import _atr  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_bt import HORIZON_BARS, MIN_BARS  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_sl_sweep_bt import _rsi_cross_events, _apply_signal_filter  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_combined_sl_bt import _fetch_with_volume, _report  # pylint: disable=wrong-import-position

DIRECTION = "Short"


def run():
    df = _fetch_with_volume()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar\n")

    old_events = []      # entry = c[i] (look-ahead'li, eski yöntem)
    fixed_events = []    # entry = c[i+1] (düzeltilmiş, canlıda mümkün)
    intrabar_events = []  # entry = c[i], jump = VPMV[i]-VPMV[i-1] (mum içi, canlıda mümkün)
    all_ts = []

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

        series_short = compute_series(g, "Short")

        for i, direction in filtered_events:
            if direction != DIRECTION:
                continue
            if i - 1 < 0 or i + 1 >= len(g):
                continue
            pre_v, post_v, sig_v = series_short.iloc[i - 1], series_short.iloc[i + 1], series_short.iloc[i]
            if not (np.isfinite(pre_v) and np.isfinite(post_v) and np.isfinite(sig_v)):
                continue
            jump = post_v - pre_v
            jump_live = sig_v - pre_v

            atr_ok = np.isfinite(atr[i]) and atr[i] > 0
            if i + HORIZON_BARS < len(c) and atr_ok:
                old_events.append((direction, h, l, c, atr[i], i, c[i], ts.iloc[i], jump))
                intrabar_events.append((direction, h, l, c, atr[i], i, c[i], ts.iloc[i], jump_live))

            i2 = i + 1
            if i2 + HORIZON_BARS < len(c) and np.isfinite(atr[i2]) and atr[i2] > 0:
                fixed_events.append((direction, h, l, c, atr[i2], i2, c[i2], ts.iloc[i2], jump))

            all_ts.append(ts.iloc[i])

    t_min, t_max = min(all_ts), max(all_ts)
    mid = t_min + (t_max - t_min) / 2
    oos_days = (t_max - mid).total_seconds() / 86400
    print(f"dönem: {t_min} .. {t_max}")
    print(f"kalibrasyon (in-sample): {t_min} .. {mid}")
    print(f"test (out-of-sample):    {mid} .. {t_max}  ({oos_days:.1f} gün)\n")

    for label, events in (("ESKİ (look-ahead'li, entry=c[i])", old_events),
                          ("DÜZELTİLMİŞ-GECİKMELİ (entry=c[i+1])", fixed_events),
                          ("MUM İÇİ (jump=VPMV[i]-VPMV[i-1], entry=c[i])", intrabar_events)):
        is_ev = [e for e in events if e[7] < mid]
        oos_ev = [e for e in events if e[7] >= mid]
        jump_threshold = float(np.percentile([e[8] for e in is_ev], 66.7))
        oos_filtered = [e[:7] for e in oos_ev if e[8] >= jump_threshold]

        print(f"\n\n╔══════════ {label} ══════════╗")
        print(f"in-sample olay: {len(is_ev)} | SABİT VPMV eşiği: {jump_threshold:.2f}")
        _report("VPMV sıçraması (SADECE SHORT)", oos_filtered, oos_days)


if __name__ == "__main__":
    run()
