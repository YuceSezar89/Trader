"""
"Ardışık Volume" Pine göstergesi (engineerofmoney) do_open_streak'e filtre
olarak test ediliyor (21 Tem 2026, kullanıcının paylaştığı Pine kodu).

Pine formülü BİREBİR:
  buyerVolume  = high==low ? 0 : volume*(close-low)/(high-low)
  sellerVolume = high==low ? 0 : volume*(high-close)/(high-low)
  consecutiveUp   := close>close[1] ? consecutiveUp[1]+buyerVolume   : 0
  consecutiveDown := close<close[1] ? consecutiveDown[1]+sellerVolume : 0

Bar-bar "hacim arttı mı" (do_open_streak_volume_rising_bt.py, placebo %60,
ÇÜRÜDÜ) DEĞİL — yükseliş serisi boyunca biriken alıcı-ağırlıklı hacim toplamı.

do_open_streak'in kendi D-open kırılım + 3 ardışık yeşil mum tetikleyici
barında bu consecutiveUp değeri hesaplanıp, sabit 96-bar (24h) ileri
getiriyle SÜREKLİ değişken olarak korelasyona sokuluyor — placebo + çeyrek
kırılımı + split-period ile.

Kullanım: python -m research.pattern_lab.do_open_streak_consecutive_volume_bt
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import psycopg2  # pylint: disable=wrong-import-position

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _fwd_returns, _stats  # pylint: disable=wrong-import-position
from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position

DAYS = 45
HORIZON_BARS = 96
MIN_BARS = 700
STREAK_TH = 3
_PLACEBO_ITER = 300


def _fetch() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, open, high, low, close, volume, buy_volume
        FROM cagg_15m
        WHERE bucket > NOW() - INTERVAL '{DAYS} days'
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def _do_break_gate(o: np.ndarray, c: np.ndarray, daily_open: np.ndarray) -> np.ndarray:
    n = len(c)
    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan
    do_break = (c > daily_open) & (prev_c <= daily_open) & np.isfinite(daily_open)
    is_long = c > o
    gate = np.zeros(n, dtype=bool)
    active = False
    for i in range(n):
        if do_break[i]:
            active = True
        elif not is_long[i]:
            active = False
        gate[i] = active
    return gate


def _consecutive_up(c: np.ndarray, buy_vol: np.ndarray) -> np.ndarray:
    """Pine'ın mantığı (consecutiveUp := close>close[1] ? consecutiveUp[1]+buyerVolume : 0)
    AMA buyerVolume proxy (volume*(close-low)/(high-low)) YERİNE Binance'in
    gerçek taker_buy_base_asset_volume alanı (cagg_15m.buy_volume) kullanılıyor
    (21 Tem 2026 — kullanıcı proxy yerine gerçek veri kullanılmasını istedi)."""
    n = len(c)
    consec_up = np.zeros(n)
    for i in range(1, n):
        if c[i] > c[i - 1]:
            consec_up[i] = consec_up[i - 1] + buy_vol[i]
        else:
            consec_up[i] = 0.0
    return consec_up


def _streak_events(o: np.ndarray, c: np.ndarray, gate: np.ndarray) -> list[int]:
    n = len(c)
    is_long = c > o
    count_long = 0
    fired = False
    events: list[int] = []
    for i in range(n):
        if is_long[i]:
            count_long += 1
        else:
            count_long = 0
            fired = False
            continue
        if not gate[i]:
            continue
        if count_long == STREAK_TH and not fired:
            fired = True
            events.append(i)
    return events


def run() -> None:
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    records = []
    n_syms = 0
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1

        ts = g["ts"]
        o = g["open"].to_numpy(float)
        c = g["close"].to_numpy(float)
        buy_vol = g["buy_volume"].to_numpy(float)

        daily_open, _ = _daily_open(ts, o)
        gate = _do_break_gate(o, c, daily_open)
        events = _streak_events(o, c, gate)
        events = [i for i in events if i < len(c) - HORIZON_BARS]
        if not events:
            continue

        consec_up = _consecutive_up(c, buy_vol)
        for i in events:
            fwd_ret = c[i + HORIZON_BARS] / c[i] - 1.0
            records.append({"symbol": sym, "ts": ts.iloc[i], "consec_up": consec_up[i], "fwd_ret": fwd_ret * 100.0})

    print(f"analize giren sembol: {n_syms}, toplam olay: {len(records)}\n")
    rdf = pd.DataFrame(records)
    if rdf.empty:
        print("Olay yok.")
        return

    print("=== [1] Korelasyon (consecutiveUp, ham değer) ===")
    rho, p = spearmanr(rdf["consec_up"], rdf["fwd_ret"])
    print(f"  rho={rho:+.4f} (p={p:.4f})")

    print("\n=== [1b] Sembol-içi normalize (consecutiveUp / o sembolün medyan hacmi) ===")
    vol_med = df.groupby("symbol")["volume"].median().rename("vol_med")
    rdf = rdf.merge(vol_med, on="symbol", how="left")
    rdf["consec_up_norm"] = rdf["consec_up"] / rdf["vol_med"].replace(0, np.nan)
    rdf = rdf.dropna(subset=["consec_up_norm"])
    rho2, p2 = spearmanr(rdf["consec_up_norm"], rdf["fwd_ret"])
    print(f"  rho={rho2:+.4f} (p={p2:.4f})  (semboller arası hacim ölçek farkını gidermek için)")

    print("\n  Normalize değer çeyrekleri:")
    tercile = pd.qcut(rdf["consec_up_norm"], 4, labels=["1.düşük", "2", "3", "4.yüksek"], duplicates="drop")
    grp = rdf.groupby(tercile, observed=True)["fwd_ret"].agg(ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100)
    print(grp.to_string())

    def _pf_stats(sub):
        return _stats(sub.to_numpy() / 100)

    top_q = rdf[rdf["consec_up_norm"] >= rdf["consec_up_norm"].quantile(0.75)]
    bot_q = rdf[rdf["consec_up_norm"] <= rdf["consec_up_norm"].quantile(0.25)]
    print(f"\n  en yüksek %25: {_pf_stats(top_q['fwd_ret'])}")
    print(f"  en düşük  %25: {_pf_stats(bot_q['fwd_ret'])}")

    rng = np.random.default_rng(42)
    vals = rdf["consec_up_norm"].to_numpy()
    target = rdf["fwd_ret"].to_numpy()
    count_ge = sum(
        1 for _ in range(_PLACEBO_ITER)
        if abs(spearmanr(rng.permutation(vals), target)[0]) >= abs(rho2)
    )
    print(f"\n  placebo (korelasyon karıştırma): %{count_ge/_PLACEBO_ITER*100:.1f}")

    if len(top_q) >= 40:
        tq_sorted = top_q.sort_values("ts")
        mid = tq_sorted["ts"].iloc[len(tq_sorted)//2]
        fh = _pf_stats(tq_sorted[tq_sorted["ts"] < mid]["fwd_ret"])
        sh = _pf_stats(tq_sorted[tq_sorted["ts"] >= mid]["fwd_ret"])
        print(f"\n  en yüksek %25 içinde split-period: ilk yarı {fh} | ikinci yarı {sh}")


if __name__ == "__main__":
    run()
