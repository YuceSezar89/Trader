"""
do_open_streak_gauss_threshold_bt.py'nin devamı — GAUSS_THRESHOLD için tek tek
sabit değer denemek yerine, PROJENİN standart IS/OOS disiplinini uyguluyor:

1. Dönemi ortadan ikiye böl (IS = ilk yarı, OOS = ikinci yarı).
2. SADECE IS'teki olaylarla bir eşik ARALIĞI tara, PF'yi maksimize eden eşiği
   seç (min örneklem şartıyla — aşırı küçük n'de gürültüye kilitlenmeyi önler).
3. Bu SABİT eşiği OOS'a uygula, OOS'ta gerçekten tutuyor mu bak (asıl sınav).
4. Seçilen eşiğin OOS performansını placebo ile karşılaştır (rastgele giriş
   zamanlaması karıştırması, do_open_streak_gauss_threshold_bt.py ile aynı yöntem).

Mevcut üretim değeri (4.5) referans olarak OOS'ta da gösteriliyor.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position
from indicators.core import calculate_atr  # pylint: disable=wrong-import-position
from signals.do_open_streak import SL_ATR_MULT  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_gauss_threshold_bt import (  # pylint: disable=wrong-import-position
    _fetch, _signal_series, _simulate, MIN_BARS, WARMUP, HORIZON_BARS, N_PLACEBO,
)

CANDIDATE_THRESHOLDS = [round(x, 2) for x in np.arange(0.5, 10.01, 0.25)]
MIN_IS_N = 50
PROD_THRESHOLD = 4.5


def _events_for_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Her sembol için sinyal olaylarını (opened_at, ret) toplar."""
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

        do, _ = _daily_open(ts, o)
        signal = _signal_series(o, h, l, c, do, threshold)
        atr_series = calculate_atr(g, period=14).to_numpy()

        idxs = np.where(signal)[0]
        idxs = idxs[(idxs >= WARMUP) & (idxs < len(g) - HORIZON_BARS)]

        for i in idxs:
            atr = atr_series[i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            entry = c[i]
            sl = entry - SL_ATR_MULT * atr
            ret = _simulate(l, c, i, entry, sl, HORIZON_BARS)
            rows.append({"symbol": sym, "opened_at": pd.Timestamp(ts_np[i]), "ret": ret})
    return pd.DataFrame(rows)


def run() -> None:
    df = _fetch()
    t_min, t_max = df["ts"].min(), df["ts"].max()
    mid = t_min + (t_max - t_min) / 2
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar")
    print(f"dönem: {t_min} .. {t_max} | IS: {t_min}..{mid} | OOS: {mid}..{t_max}\n")

    # 1) SADECE IS'te eşik taraması
    print(f"{'esik':>6} {'IS_n':>7} {'IS_WR%':>7} {'IS_PF':>7}")
    is_results = {}
    for th in CANDIDATE_THRESHOLDS:
        events = _events_for_threshold(df, th)
        if events.empty:
            continue
        is_events = events[events["opened_at"] < mid]
        oos_events = events[events["opened_at"] >= mid]
        is_results[th] = (is_events, oos_events)
        s = _stats(is_events["ret"].to_numpy())
        if s.get("n", 0) >= MIN_IS_N:
            print(f"{th:>6} {s.get('n',0):>7} {s.get('wr',0):>7} {s.get('pf',0):>7}")

    # En iyi IS PF'ye sahip (min örneklem şartını sağlayan) eşiği seç
    best_th, best_pf = None, -1.0
    for th, (is_ev, _oos_ev) in is_results.items():
        s = _stats(is_ev["ret"].to_numpy())
        if s.get("n", 0) >= MIN_IS_N and s.get("pf", 0) > best_pf:
            best_pf = s.get("pf", 0)
            best_th = th

    if best_th is None:
        print("\nYeterli örneklemli eşik bulunamadı.")
        return

    print(f"\n>>> IS'te seçilen eşik: {best_th} (IS PF={best_pf:.3f})\n")

    # 2) Seçilen eşiği OOS'a SABİT uygula
    sel_is_ev, sel_oos_ev = is_results[best_th]
    prod_is_ev, prod_oos_ev = is_results.get(PROD_THRESHOLD, (pd.DataFrame(), pd.DataFrame()))

    print(f"{'varyant':28} {'dönem':6} {'n':>6} {'WR%':>6} {'PF':>7}")

    s_is = _stats(sel_is_ev["ret"].to_numpy())
    s_oos = _stats(sel_oos_ev["ret"].to_numpy())
    p_is = _stats(prod_is_ev["ret"].to_numpy()) if not prod_is_ev.empty else {}
    p_oos = _stats(prod_oos_ev["ret"].to_numpy()) if not prod_oos_ev.empty else {}

    print(f"{'seçilen eşik=' + str(best_th):28} {'IS':6} {s_is.get('n',0):>6} {s_is.get('wr',0):>6} {s_is.get('pf',0):>7}")
    print(f"{'seçilen eşik=' + str(best_th):28} {'OOS':6} {s_oos.get('n',0):>6} {s_oos.get('wr',0):>6} {s_oos.get('pf',0):>7}  <<< ASIL SINAV")
    print(f"{'üretim eşik=' + str(PROD_THRESHOLD):28} {'IS':6} {p_is.get('n',0):>6} {p_is.get('wr',0):>6} {p_is.get('pf',0):>7}")
    print(f"{'üretim eşik=' + str(PROD_THRESHOLD):28} {'OOS':6} {p_oos.get('n',0):>6} {p_oos.get('wr',0):>6} {p_oos.get('pf',0):>7}")

    if s_oos.get("n", 0) < 30:
        print("\nOOS örneklemi çok küçük, güvenilir yorum yapılamaz.")
        return

    # 3) Split-period (OOS'un kendi içinde ilk/ikinci yarı)
    oos_ts = sel_oos_ev["opened_at"]
    oos_mid = oos_ts.min() + (oos_ts.max() - oos_ts.min()) / 2
    first_half = sel_oos_ev[sel_oos_ev["opened_at"] < oos_mid]
    second_half = sel_oos_ev[sel_oos_ev["opened_at"] >= oos_mid]
    sf = _stats(first_half["ret"].to_numpy())
    ss = _stats(second_half["ret"].to_numpy())
    print(f"\nOOS split-period: ilk_yari n={sf.get('n',0)} PF={sf.get('pf',0)} | "
          f"ikinci_yari n={ss.get('n',0)} PF={ss.get('pf',0)}")

    # 4) Placebo (OOS'ta, seçilen eşiğin olay sayısını koruyarak rastgele zamanlama)
    rng = np.random.default_rng(42)
    per_symbol_meta = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        ts_np = g["ts"].to_numpy()
        oos_mask_bar = ts_np >= np.datetime64(mid)
        valid_idx = np.where(oos_mask_bar)[0]
        valid_idx = valid_idx[(valid_idx >= WARMUP) & (valid_idx < len(g) - HORIZON_BARS)]
        if len(valid_idx) == 0:
            continue
        n_ev = int((sel_oos_ev["symbol"] == sym).sum())
        if n_ev == 0:
            continue
        c = g["close"].to_numpy(float)
        l = g["low"].to_numpy(float)
        atr_series = calculate_atr(g, period=14).to_numpy()
        per_symbol_meta.append((l, c, atr_series, valid_idx, n_ev))

    placebo_pfs = []
    for _ in range(N_PLACEBO):
        rets = []
        for l, c, atr_series, valid_idx, n_ev in per_symbol_meta:
            if n_ev > len(valid_idx):
                continue
            picks = rng.choice(valid_idx, size=n_ev, replace=False)
            for i in picks:
                atr = atr_series[i]
                if not np.isfinite(atr) or atr <= 0:
                    continue
                entry = c[i]
                sl = entry - SL_ATR_MULT * atr
                rets.append(_simulate(l, c, i, entry, sl, HORIZON_BARS))
        if len(rets) >= 20:
            placebo_pfs.append(_stats(np.array(rets)).get("pf", 0.0))

    if placebo_pfs:
        arr = np.array(placebo_pfs)
        real_pf = s_oos.get("pf", 0.0)
        rank = float((arr < real_pf).mean() * 100)
        print(f"\nOOS placebo (n={len(arr)}): ort={arr.mean():.3f} p90={np.percentile(arr,90):.3f} "
              f"max={arr.max():.3f} | gerçek OOS PF={real_pf:.3f} (placebo'nun %{rank:.0f}'ini geçiyor)")


if __name__ == "__main__":
    run()
