"""
MAE/MFE (Maximum Adverse/Favorable Excursion) analizi — do_break+streak==3
olaylarının (do_kirilimi/do_open_streak'in temel setup'ı) 24h'lik gözlem
penceresinde fiyatın entry'ye göre ne kadar ALEYHE (MAE) ve ne kadar LEHTE
(MFE) gittiğini ölçer. Amaç: SL/TP çarpanlarını sezgiyle seçmek yerine
veriden kalibre etmek (bkz. docs/sl_tp_mimarisi.md, "MAE/MFE analizi" —
López de Prado'nun triple-barrier yönteminin ilham verdiği bir teşhis aracı).

Sorular:
1. Kazanan işlemlerin ne kadarı, çeşitli SL çarpanlarına TAKILIP durdurulurdu
   (gerçekte kazanacakken erken kapanma riski = "false stop-out")?
2. Kaybeden işlemlerin ne kadarı, çeşitli TP seviyelerine ULAŞTI ama sonra
   geri döndü (TP olsaydı kurtarılabilirdi)?

Aynı disiplin: cagg_15m, 45 gün, do_break_gate + streak==3 (do_open_streak_bt.py
ile birebir event tanımı), 24h/96-bar gözlem penceresi, look-ahead yok (MAE/MFE
sadece entry SONRASI barlardan hesaplanıyor). MTF-onaylı (ULTRA, v2-15) alt
küme ayrıca gösteriliyor — daha temiz sinyallerin MAE/MFE profili farklı mı diye.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from indicators.core import calculate_atr  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_bt import (  # pylint: disable=wrong-import-position
    DAYS,
    HORIZON_BARS,
    MIN_BARS,
    MTF_CONFIRM_TFS,
    MTF_STREAK_TARGET,
    _do_break_gate,
    _fetch,
    _streak_events,
)
from research.pattern_lab.mtf_helpers import (  # pylint: disable=wrong-import-position
    _confirm_count,
    _fetch_dir_data,
)
from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position

TP_MULTIPLES = [2.0, 3.0, 4.0, 6.0]
SL_MULTIPLES = [1.5, 2.0, 3.0, 4.0]
PERCENTILES = [10, 25, 50, 75, 90]


def _mae_mfe(h: np.ndarray, l: np.ndarray, c: np.ndarray, atr: np.ndarray, i: int, horizon: int):
    """entry'den (bar i'nin kapanışı) SONRAKİ `horizon` bar içinde en aleyhe
    ve en lehte gidişi ATR biriminde döner. Sadece entry SONRASI barlar
    kullanılıyor — look-ahead yok."""
    entry = c[i]
    atr_val = atr[i]
    if not np.isfinite(atr_val) or atr_val <= 0:
        return None
    window_low = l[i + 1 : i + 1 + horizon]
    window_high = h[i + 1 : i + 1 + horizon]
    if len(window_low) < horizon:
        return None
    mae_atr = (entry - window_low.min()) / atr_val  # pozitif = aleyhe mesafe
    mfe_atr = (window_high.max() - entry) / atr_val  # pozitif = lehte mesafe
    final_ret = (c[i + horizon] - entry) / entry
    return mae_atr, mfe_atr, final_ret


def _percentiles(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {}
    return {q: round(float(np.percentile(arr, q)), 2) for q in PERCENTILES}


def _report(label: str, rows: list) -> None:
    df = pd.DataFrame(rows, columns=["mae_atr", "mfe_atr", "final_ret"])
    n = len(df)
    print(f"\n=== {label} (n={n}) ===")
    if n == 0:
        return

    winners = df[df["final_ret"] > 0]
    losers = df[df["final_ret"] <= 0]
    print(
        f"kazanan={len(winners)} (%{100*len(winners)/n:.1f}) "
        f"kaybeden={len(losers)} (%{100*len(losers)/n:.1f})"
    )

    print(f"MAE (ATR) percentile — TÜMÜ: {_percentiles(df['mae_atr'].to_numpy())}")
    print(f"MAE (ATR) percentile — KAZANANLAR: {_percentiles(winners['mae_atr'].to_numpy())}")
    print(f"MFE (ATR) percentile — TÜMÜ: {_percentiles(df['mfe_atr'].to_numpy())}")
    print(f"MFE (ATR) percentile — KAYBEDENLER: {_percentiles(losers['mfe_atr'].to_numpy())}")

    print("\n-- SL çarpanına göre 'false stop-out' oranı (kazananların o SL'e takılma ihtimali) --")
    for sl in SL_MULTIPLES:
        if len(winners) == 0:
            continue
        false_stop = (winners["mae_atr"] >= sl).mean() * 100
        print(f"  SL={sl}×ATR: kazananların %{false_stop:.1f}'i bu SL'e TAKILIRDI")

    print(
        "\n-- TP çarpanına göre 'kurtarılabilirdi' oranı (kaybedenlerin o TP'ye ulaşma ihtimali) --"
    )
    for tp in TP_MULTIPLES:
        if len(losers) == 0:
            continue
        would_hit_tp = (losers["mfe_atr"] >= tp).mean() * 100
        print(
            f"  TP={tp}×ATR: kaybedenlerin %{would_hit_tp:.1f}'i bu TP'ye ULAŞMIŞTI (sonra geri döndü)"
        )


def run():
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    baseline_rows = []
    ultra_rows = []

    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue

        ts = g["ts"]
        o = g["open"].to_numpy(float)
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)

        daily_open, _ = _daily_open(ts, o)
        gate = _do_break_gate(o, c, daily_open)
        events = _streak_events(o, c, gate=gate)
        target_events = [i for i in events[MTF_STREAK_TARGET] if i + HORIZON_BARS < len(c)]
        if not target_events:
            continue

        atr = calculate_atr(g, period=14).to_numpy()
        dir_data = _fetch_dir_data(sym, MTF_CONFIRM_TFS)

        for i in target_events:
            res = _mae_mfe(h, l, c, atr, i, HORIZON_BARS)
            if res is None:
                continue
            baseline_rows.append(res)

            if dir_data is not None:
                confirm_count = _confirm_count(
                    dir_data, MTF_CONFIRM_TFS, ts.iloc[i], want_bullish=True
                )
                if confirm_count == len(MTF_CONFIRM_TFS):
                    ultra_rows.append(res)

    _report("BASELINE (tüm do_break+streak3)", baseline_rows)
    _report("ULTRA (2/2 MTF onaylı alt küme, v2-15)", ultra_rows)


if __name__ == "__main__":
    run()
