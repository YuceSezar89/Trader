"""
rsi_cross_combined_sl_bt.py'nin devamı — VPMV-sıçraması+SL=1.5×ATR
kombinasyonunun (v2-18) OOS penceresinin kendi içinde tutarsız çıkması
(ilk yarı ~$0/ay, ikinci yarı +$4145/ay — rejime bağlı görünüyor) üzerine,
kullanıcının önerdiği "üst TF'lerin yönü sinyalle aynı mı" (HA_Cross/
do_kirilimi'de doğrulanmış ULTRA/MTF onayı fikri, v2-14/v2-15) RSI_Cross'a
da uygulanıyor: rejim-bağımlılığı MTF onayı düzeltiyor mu?

RSI_Cross 15m tabanlı olduğu için onay TF'leri (do_kirilimi ile aynı
gerekçeyle) 1h + 4h — DB'de native 30m/45m yok. Kural: sinyal Long ise
1h VE 4h'nin son kapanmış HA barı bullish olmalı (2/2 TAM onay).
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
from research.pattern_lab.mtf_helpers import _fetch_dir_data, _confirm_count  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_sl_sweep_bt import _rsi_cross_events, _apply_signal_filter  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_combined_sl_bt import _fetch_with_volume, _report  # pylint: disable=wrong-import-position

CONFIRM_TFS = ["1h", "4h"]


def run():
    df = _fetch_with_volume()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar\n")

    clean_events = []  # (direction, h, l, c, atr_val, i, entry_price, ts, jump, confirm_count)

    for sym, g in df.groupby("symbol"):
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
        dir_data = _fetch_dir_data(sym, CONFIRM_TFS)

        for i, direction in filtered_events:
            if i + HORIZON_BARS >= len(c) or i - 1 < 0 or i + 1 >= len(g) \
                    or not (np.isfinite(atr[i]) and atr[i] > 0):
                continue
            series = series_long if direction == "Long" else series_short
            pre_v, post_v = series.iloc[i - 1], series.iloc[i + 1]
            if not (np.isfinite(pre_v) and np.isfinite(post_v)):
                continue

            confirm_count = None
            if dir_data is not None:
                confirm_count = _confirm_count(
                    dir_data, CONFIRM_TFS, ts.iloc[i], want_bullish=(direction == "Long")
                )

            clean_events.append((
                direction, h, l, c, atr[i], i, c[i], ts.iloc[i], post_v - pre_v, confirm_count,
            ))

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
    vpmv_mtf = [e for e in oos_events if e[8] >= jump_threshold and e[9] == len(CONFIRM_TFS)]

    def _by_dir(events: list, direction: str) -> list:
        return [e for e in events if e[0] == direction]

    oos_mid = mid + (t_max - mid) / 2
    half_days = oos_days / 2

    for dir_label, direction in (("LONG+SHORT (birlikte)", None), ("SADECE LONG", "Long"), ("SADECE SHORT", "Short")):
        vo = vpmv_only if direction is None else _by_dir(vpmv_only, direction)
        vm = vpmv_mtf if direction is None else _by_dir(vpmv_mtf, direction)

        print(f"\n\n╔══════════ {dir_label} ══════════╗")
        print("\n════ TÜM OOS ════")
        _report("VPMV sıçraması (MTF'siz, v2-18)", [e[:7] for e in vo], oos_days)
        _report(f"VPMV + MTF onayı ({len(CONFIRM_TFS)}/{len(CONFIRM_TFS)}, 1h+4h)", [e[:7] for e in vm], oos_days)

        print(f"\n════ OOS — İLK YARI (biten: {oos_mid}) ════")
        _report("VPMV sıçraması (MTF'siz)", [e[:7] for e in vo if e[7] < oos_mid], half_days)
        _report("VPMV + MTF onayı", [e[:7] for e in vm if e[7] < oos_mid], half_days)

        print(f"\n════ OOS — İKİNCİ YARI (başlayan: {oos_mid}) ════")
        _report("VPMV sıçraması (MTF'siz)", [e[:7] for e in vo if e[7] >= oos_mid], half_days)
        _report("VPMV + MTF onayı", [e[:7] for e in vm if e[7] >= oos_mid], half_days)


if __name__ == "__main__":
    run()
