"""
do_open_streak — farklı Gauss eşiklerinde kâr-yoğunlaşması teşhisi (22 Tem 2026).

concentration_diagnostics.py'deki üç teşhisi (top-N katkı, trim edilmiş toplam,
bootstrap aylık dağılım) mevcut eşik (4.5) ve önerilen adaylarda (6.0/8.0/10.0)
karşılaştırmalı olarak uygular — "iyi görünen $/ay rakamı birkaç uç işlemden mi
geliyor" sorusuna sistemli bir cevap.

Kullanım: python -m research.pattern_lab.do_open_streak_concentration_bt
"""

from research.pattern_lab.concentration_diagnostics import summarize
from research.pattern_lab.do_open_streak_percentile_bt import _collect_events

_THRESHOLDS = [2.0, 4.5, 6.0, 8.0, 10.0, 12.0]


def main() -> None:
    df = _collect_events()
    if df.empty:
        print("Olay yok.")
        return
    df = df.sort_values("ts").reset_index(drop=True)
    days_span = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400

    for th in _THRESHOLDS:
        sub = df[df["gauss_val"] >= th]
        if sub.empty:
            continue
        summarize(f"GAUSS_THRESHOLD >= {th}", sub["pnl_usd"].to_numpy(), sub["ts"], days_span)


if __name__ == "__main__":
    main()
