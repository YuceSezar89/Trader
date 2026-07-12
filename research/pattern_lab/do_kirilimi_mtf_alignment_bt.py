"""
"Ultra hizalama" (v2-14, HA_Cross'ta bulunan MTF onay fikri) do_kirilimi'nin
kendi en güçlü bulgusuna (do_break + 3 ardışık yeşil mum, v2-3/v2-4)
uygulanıyor. do_kirilimi 15m tabanlı olduğu için üst TF onayı HA_Cross'taki
gibi 3 değil SADECE 2 (1h/4h) — Long-only bir aile olduğundan onay şartı:
sinyal anında 1h/4h'nin en son kapanmış HA barı bullish mi.

Aynı veri/disiplin: cagg_15m, 45 gün, do_break_gate + streak==3 event'leri
(do_open_streak_bt.py/do_open_touch_gauss_bt.py'den birebir import), 24h
ileri getiri (_fwd_returns, look-ahead'siz — event SONRASI 96 bar), MTF
yönü ise event ANINDAN ÖNCE kapanmış barlardan alınıyor (_last_direction_before,
ha_cross_mtf_alignment_bt.py'den import). Split-period sağlamlık kontrolü var.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.do_open_streak_bt import (  # pylint: disable=wrong-import-position
    DAYS,
    HORIZON_BARS,
    MIN_BARS,
    _do_break_gate,
    _fetch,
)
from research.pattern_lab.do_open_touch_gauss_bt import (  # pylint: disable=wrong-import-position
    GAUSS_STREAK_THRESHOLD,
    _streak_state,
    _threshold_events,
)
from research.pattern_lab.mtf_helpers import (  # pylint: disable=wrong-import-position
    _confirm_count,
    _fetch_dir_data,
)
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position

CONFIRM_TFS = ["1h", "4h"]
POSITION_USD = 100.0
FEE_RATE = 0.0005
ROUND_TRIP_FEE = POSITION_USD * FEE_RATE * 2


def _dollar_stats(rets: np.ndarray, days_span: float) -> dict:
    if len(rets) == 0:
        return {"n": 0}
    pnl = rets * POSITION_USD - ROUND_TRIP_FEE
    total = float(pnl.sum())
    per_month = total / days_span * 30 if days_span > 0 else 0.0
    return {
        "n": len(rets),
        "wr": round(float((pnl > 0).mean() * 100), 1),
        "avg_usd": round(float(pnl.mean()), 3),
        "total_usd": round(total, 1),
        "usd_per_month": round(per_month, 1),
    }


def _print_by_count(d: pd.DataFrame) -> None:
    print(f"{'üst TF onayı':14} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for count in (0, 1, 2):
        rets = d.loc[d["confirm_count"] == count, "ret"].to_numpy()
        s = _stats(rets)
        label = f"{count}/2" + (" (ULTRA)" if count == 2 else "")
        print(
            f"{label:14} {s.get('n',0):>7} {s.get('wr',0):>6} "
            f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}"
        )


def run():
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    rows = []
    n_syms = 0

    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue

        dir_data = _fetch_dir_data(sym, CONFIRM_TFS)
        if dir_data is None:
            continue
        n_syms += 1

        ts = g["ts"]
        o = g["open"].to_numpy(float)
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)

        daily_open, _ = _daily_open(ts, o)
        gate = _do_break_gate(o, c, daily_open)
        count_long, _long_perc = _streak_state(o, h, l, c)
        ev3 = _threshold_events(count_long, gate, GAUSS_STREAK_THRESHOLD)
        ev3 = [i for i in ev3 if i < len(c) - HORIZON_BARS]

        for i in ev3:
            # Long-only sinyal, onay = üst TF bullish
            confirm_count = _confirm_count(dir_data, CONFIRM_TFS, ts.iloc[i], want_bullish=True)
            if confirm_count is None:
                continue
            ret = c[i + HORIZON_BARS] / c[i] - 1
            rows.append({"confirm_count": confirm_count, "ret": ret, "ts": ts.iloc[i]})

    print(f"analize giren sembol: {n_syms}\n")
    res = pd.DataFrame(rows)
    print(f"toplam do_break+streak3 olayı (MTF verisi eşleşen): {len(res):,}\n")
    if len(res) < 50:
        print("Örneklem çok küçük, güvenilir analiz yapılamaz.")
        return

    s = _stats(res["ret"].to_numpy())
    print(f"{'grup':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    print(
        f"{'baseline (tümü)':20} {s.get('n',0):>7} {s.get('wr',0):>6} "
        f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}\n"
    )

    print("── Üst TF (1h/4h) onay sayısına göre ──")
    _print_by_count(res)

    t_min, t_max = res["ts"].min(), res["ts"].max()
    mid = t_min + (t_max - t_min) / 2
    print(f"\ndönem: {t_min} .. {t_max}\norta nokta: {mid}\n")

    first = res[res["ts"] < mid]
    second = res[res["ts"] >= mid]

    print(f"══ 1_ilk_yari (n={len(first)}) ══")
    if len(first) >= 20:
        _print_by_count(first)
    else:
        print("örneklem çok küçük")

    print(f"\n══ 2_ikinci_yari (n={len(second)}) ══")
    if len(second) >= 20:
        _print_by_count(second)
    else:
        print("örneklem çok küçük")

    days_span = (t_max - t_min).total_seconds() / 86400
    print(
        f"\n── Ekonomik etki (TÜM dönem, ${POSITION_USD:.0f} pozisyon, in-sample/OOS ayrımı YOK — "
        f"confirm_count sabit bir eşik, kalibre edilen serbest parametre yok) ──"
    )
    print(f"{'grup':20} {'n':>6} {'WR%':>6} {'ort $/işlem':>12} {'toplam $':>10} {'$/ay':>10}")
    for name, sub in (
        ("baseline (tümü)", res),
        ("ULTRA (2/2)", res[res["confirm_count"] == 2]),
        ("0-1/2", res[res["confirm_count"] < 2]),
    ):
        stat = _dollar_stats(sub["ret"].to_numpy(), days_span)
        if stat.get("n", 0) == 0:
            print(f"{name:20} {'0':>6}")
            continue
        print(
            f"{name:20} {stat['n']:>6} {stat['wr']:>6} {stat['avg_usd']:>12} "
            f"{stat['total_usd']:>10} {stat['usd_per_month']:>10}"
        )


if __name__ == "__main__":
    run()
