"""
do_open_streak'in Gauss/long_perc filtresi SABİT bir % eşiği kullanıyor
(long_perc=(h-start_low)/start_low*100, gauss_sum>=4.5) — bu, tüm sembollere
aynı "%2.5 hareket" barını uyguluyor, ama coinlerin tipik volatilitesi çok
farklı (düşük-vol bir coin için %2.5 nadir/anlamlı, yüksek-vol bir coin için
gürültü). Hipotez: hareketi sembolün kendi ATR'sine göre normalize etmek
(ATR_MULTIPLE = (h-start_low)/ATR, "kaç ATR'lik hareket") daha iyi ayrım
sağlar mı?

Adil kıyas için SEÇİCİLİK EŞİTLENİYOR: iki yöntem de gate_active+count==3 aday
barları arasından AYNI ORANDA (IS'te kalibre edilen persentil) olay seçiyor —
"daha az işlem seçmek zaten PF'yi şişirir" karıştırmasını önlemek için
([[project_pattern_lab]] — bugünkü gauss-eşik-optimizasyonu denemesinde
görülen overfitting tuzağı).

IS'te persentil kalibre edilip OOS'a SABİT uygulanıyor + split-period + placebo.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from indicators.core import calculate_atr  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_gauss_threshold_bt import (  # pylint: disable=wrong-import-position
    HORIZON_BARS,
    MIN_BARS,
    N_PLACEBO,
    WARMUP,
    _fetch,
    _simulate,
)
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position
from signals.do_open_streak import (  # pylint: disable=wrong-import-position
    SL_ATR_MULT,
    STREAK_THRESHOLD,
)

SELECTIVITY_PERCENTILE = 66.0  # mevcut eşiğin (4.5) yaklaşık seçiciliğine denk


def _candidates(df: pd.DataFrame) -> pd.DataFrame:
    """gate_active+count==3 olan TÜM aday barları toplar (gauss filtresi
    UYGULANMADAN), her biri için hem long_perc hem atr_multiple hesaplanır."""
    rows = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        ts = g["ts"]
        o = g["open"].to_numpy(float)
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)
        ts_np = ts.to_numpy()
        n = len(c)

        do, _ = _daily_open(ts, o)
        prev_c = np.roll(c, 1)
        prev_c[0] = np.nan
        do_break = (c > do) & (prev_c <= do) & np.isfinite(do)
        is_long = c > o
        atr_series = calculate_atr(g, period=14).to_numpy()

        gate_active = False
        count_long = 0
        start_low = np.nan
        for i in range(n):
            if do_break[i]:
                gate_active = True
            elif not is_long[i]:
                gate_active = False
            if is_long[i]:
                count_long += 1
                if count_long == 1 or np.isnan(start_low):
                    start_low = l[i]
            else:
                count_long = 0
                start_low = np.nan

            if gate_active and count_long == STREAK_THRESHOLD:
                if i >= WARMUP and i < n - HORIZON_BARS:
                    atr = atr_series[i]
                    if np.isfinite(atr) and atr > 0:
                        long_perc = (h[i] - start_low) / start_low * 100.0
                        atr_mult = (h[i] - start_low) / atr
                        entry = c[i]
                        sl = entry - SL_ATR_MULT * atr
                        ret = _simulate(l, c, i, entry, sl, HORIZON_BARS)
                        rows.append(
                            {
                                "symbol": sym,
                                "opened_at": pd.Timestamp(ts_np[i]),
                                "long_perc": long_perc,
                                "atr_mult": atr_mult,
                                "ret": ret,
                            }
                        )
    return pd.DataFrame(rows)


def _report(name: str, ev: pd.DataFrame) -> dict:
    s = _stats(ev["ret"].to_numpy())
    print(f"{name:34} n={s.get('n',0):>5}  WR%={s.get('wr',0):>6}  PF={s.get('pf',0):>7}")
    return s


def run() -> None:
    df = _fetch()
    cands = _candidates(df)
    print(f"{df['symbol'].nunique()} sembol | aday bar (gate+count==3): {len(cands)}\n")

    t_min, t_max = cands["opened_at"].min(), cands["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    is_c = cands[cands["opened_at"] < mid]
    oos_c = cands[cands["opened_at"] >= mid]
    print(f"IS aday: {len(is_c)} | OOS aday: {len(oos_c)}\n")

    # IS'te iki yöntemin de eşiğini AYNI persentile göre kalibre et
    pct_thr_long = np.percentile(is_c["long_perc"], SELECTIVITY_PERCENTILE)
    pct_thr_atr = np.percentile(is_c["atr_mult"], SELECTIVITY_PERCENTILE)
    print(
        f"IS'te kalibre edilen eşikler (persentil={SELECTIVITY_PERCENTILE}): "
        f"long_perc>={pct_thr_long:.3f}  |  atr_mult>={pct_thr_atr:.3f}\n"
    )

    print(f"{'varyant':34} sonuç")
    is_pct = _report("% bazlı (IS)", is_c[is_c["long_perc"] >= pct_thr_long])
    is_atr = _report("ATR bazlı (IS)", is_c[is_c["atr_mult"] >= pct_thr_atr])
    print()
    oos_pct_ev = oos_c[oos_c["long_perc"] >= pct_thr_long]
    oos_atr_ev = oos_c[oos_c["atr_mult"] >= pct_thr_atr]
    _report("% bazlı (OOS) <<< ASIL SINAV", oos_pct_ev)
    _report("ATR bazlı (OOS) <<< ASIL SINAV", oos_atr_ev)

    for label, ev in (("% bazlı OOS", oos_pct_ev), ("ATR bazlı OOS", oos_atr_ev)):
        if len(ev) < 30:
            continue
        ts_ = ev["opened_at"]
        m = ts_.min() + (ts_.max() - ts_.min()) / 2
        fh = _stats(ev[ev["opened_at"] < m]["ret"].to_numpy())
        sh = _stats(ev[ev["opened_at"] >= m]["ret"].to_numpy())
        print(
            f"\n{label} split-period: ilk_yari n={fh.get('n',0)} PF={fh.get('pf',0)} | "
            f"ikinci_yari n={sh.get('n',0)} PF={sh.get('pf',0)}"
        )

    # Placebo: OOS aday havuzundan (gate+count==3, filtre UYGULANMAMIŞ) rastgele
    # aynı sayıda seçip PF dağılımını ölç.
    rng = np.random.default_rng(42)
    for label, ev in (("% bazlı OOS", oos_pct_ev), ("ATR bazlı OOS", oos_atr_ev)):
        if len(ev) < 30:
            continue
        real_pf = _stats(ev["ret"].to_numpy()).get("pf", 0.0)
        n_pick = len(ev)
        pool = oos_c["ret"].to_numpy()
        placebo_pfs = []
        for _ in range(N_PLACEBO):
            picks = rng.choice(pool, size=n_pick, replace=False)
            s = _stats(picks)
            if s.get("n", 0) >= 20:
                placebo_pfs.append(s.get("pf", 0.0))
        if placebo_pfs:
            arr = np.array(placebo_pfs)
            rank = float((arr < real_pf).mean() * 100)
            print(
                f"\n{label} placebo (n={len(arr)}): ort={arr.mean():.3f} p90={np.percentile(arr,90):.3f} "
                f"| gerçek PF={real_pf:.3f} (placebo'nun %{rank:.0f}'ini geçiyor)"
            )


if __name__ == "__main__":
    run()
