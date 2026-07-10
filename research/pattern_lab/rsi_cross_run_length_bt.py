"""
Kullanıcının hipotezi (10 Tem 2026): RSI_Cross doğası gereği momentum
DÖNÜŞÜNÜ yakalar (dip/tepe civarında tetiklenir). Art arda aynı yönde
RSI_Cross sinyali gelmesi, güçlü tek-yönlü bir trendde MANTIKLI değil —
RSI zaten bir bant içinde sallanıyor demektir. Yani ardışık sinyal SAYISI
arttıkça, bu muhtemelen yatay/kararsız piyasanın belirtisi ve performans
KÖTÜLEŞMELİ (cumDelta testinin aksine, "daha çok = daha iyi" değil
"daha çok = daha kötü/kararsız piyasa" hipotezi).

run_position: bu sinyal, son ters-yönlü sinyalden beri aynı yönde kaçıncı
sinyal (1 = ters dönüşten sonraki İLK sinyal — "temiz" dönüş; 2, 3, 4... =
aynı yönde tekrarlayan sinyaller — muhtemelen sallanan piyasa).

Metodoloji: SADECE 3 Tem 19:22:16 sonrası (temiz rejim), split-period.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_vpmv_jump_bt import INTERVALS, _fetch_signals  # pylint: disable=wrong-import-position


def _print_table(df: pd.DataFrame, groups: list) -> None:
    print(f"{'run_position':14} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for name in groups:
        rets = df.loc[df["run_group"] == name, "realized_pnl"].to_numpy() / 100
        s = _stats(rets)
        print(f"{name:14} {s.get('n',0):>7} {s.get('wr',0):>6} "
              f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")


def run():
    rows = []
    for interval in INTERVALS:
        sigs = _fetch_signals(interval)
        print(f"{interval}: {len(sigs):,} kapanmış RSI_Cross sinyali (3 Tem 19:22 sonrası)")

        for _symbol, sub in sigs.groupby("symbol"):
            sub_sorted = sub.sort_values("opened_at")
            prev_type = None
            run_pos = 0
            for _, row in sub_sorted.iterrows():
                if prev_type is None or prev_type != row["signal_type"]:
                    run_pos = 1
                else:
                    run_pos += 1
                prev_type = row["signal_type"]
                rows.append({
                    "run_position": run_pos,
                    "realized_pnl": row["realized_pnl"],
                    "opened_at": row["opened_at"],
                })

    df = pd.DataFrame(rows)
    print(f"\ntoplam sinyal: {len(df):,}\n")

    df["run_group"] = np.where(df["run_position"] == 1, "1 (temiz dönüş)",
                        np.where(df["run_position"] == 2, "2",
                        np.where(df["run_position"] == 3, "3", "4+")))
    groups = ["1 (temiz dönüş)", "2", "3", "4+"]

    print("dağılım:")
    print(df["run_group"].value_counts().reindex(groups))

    s = _stats(df["realized_pnl"].to_numpy() / 100)
    print(f"\n{'grup':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    print(f"{'baseline (tümü)':20} {s.get('n',0):>7} {s.get('wr',0):>6} "
          f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")

    print("\n── run_position'a göre (1=ters dönüşten sonraki ilk sinyal) ──")
    _print_table(df, groups)

    t_min, t_max = df["opened_at"].min(), df["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    print(f"\ndönem: {t_min} .. {t_max}\norta nokta: {mid}\n")

    first = df[df["opened_at"] < mid]
    second = df[df["opened_at"] >= mid]

    print(f"══ 1_ilk_yari (n={len(first)}) ══")
    if len(first) >= 30:
        _print_table(first, groups)
    else:
        print("örneklem çok küçük")

    print(f"\n══ 2_ikinci_yari (n={len(second)}) ══")
    if len(second) >= 30:
        _print_table(second, groups)
    else:
        print("örneklem çok küçük")


if __name__ == "__main__":
    run()
