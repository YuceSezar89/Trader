"""
v2-1b — "Satıcı kalmayınca ne olur" (hacim sönmesi) bağımsız replay backtest.

Vaka-kontrol turunda (t0 = büyük pump başlangıcı) hacim sönmesi hiçbir şey
göstermedi — muhtemelen kavramsal uyumsuzluk: trader'ların işareti YEREL
destek sıçraması, bizim t0 ise çok-günlük mega-pump başlangıcı. Bu script
soruyu doğru çerçevede sorar: pump olsun olmasın, TÜM evrende, "hacim
kırmızıdan (≥75) maviye (≤20) tam söndüğü her an" sonrasında fiyat ne yapar?

Ayna testi: "hacim kırmızıdayken" (≥75, henüz soğumamış) giriş kötü mü
(tepe/dağıtım uyarısı doğrulanıyor mu)?

Look-ahead: sinyal SADECE geçmiş barlarla (rolling percentile + state
machine) üretilir; ileri getiri ayrı ve sonradan ölçülür.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import psycopg2

from config import Config

DAYS = 45
VOL_LEN = 100
VOL_SMOOTH = 3
HOT_LEVEL = 75
COLD_LEVEL = 20
HORIZONS_BARS = {"1h": 4, "4h": 16, "12h": 48, "24h": 96}   # 15m barlarda
MIN_BARS = 700


def _fetch() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, close, volume
        FROM cagg_15m
        WHERE bucket > NOW() - INTERVAL '{DAYS} days'
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def _vol_rank(volume: np.ndarray) -> np.ndarray:
    smooth = pd.Series(volume).rolling(VOL_SMOOTH).mean().to_numpy()
    n = len(smooth)
    rank = np.full(n, np.nan)
    for i in range(VOL_LEN, n):
        w = smooth[i - VOL_LEN:i + 1]
        if np.isnan(w).any():
            continue
        rank[i] = (w < w[-1]).mean() * 100
    return rank


def _events(rank: np.ndarray) -> tuple[list[int], list[int]]:
    """COLD (tam sönme) ve HOT (sıcak, henüz sönmemiş) bar indeksleri."""
    cold_idx, hot_idx = [], []
    was_hot = False
    for i in range(len(rank)):
        r = rank[i]
        if np.isnan(r):
            continue
        if r >= HOT_LEVEL:
            was_hot = True
            hot_idx.append(i)
        elif was_hot and r <= COLD_LEVEL:
            cold_idx.append(i)
            was_hot = False
    return cold_idx, hot_idx


def _fwd_returns(close: np.ndarray, idx: list[int], bars: int) -> np.ndarray:
    out = []
    n = len(close)
    for i in idx:
        if i + bars < n:
            out.append(close[i + bars] / close[i] - 1)
    return np.array(out)


def _stats(rets: np.ndarray) -> dict:
    if len(rets) == 0:
        return {"n": 0}
    g, b = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "n": len(rets),
        "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean() * 100), 3),
        "pf": round(float(g / b), 3) if b > 0 else float("inf"),
    }


def run():
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    baseline_fwd = {h: [] for h in HORIZONS_BARS}
    cold_fwd = {h: [] for h in HORIZONS_BARS}
    hot_fwd = {h: [] for h in HORIZONS_BARS}
    n_syms, n_cold_events, n_hot_events = 0, 0, 0

    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        close = g["close"].to_numpy(float)
        rank = _vol_rank(g["volume"].to_numpy(float))
        cold_idx, hot_idx = _events(rank)
        n_syms += 1
        n_cold_events += len(cold_idx)
        n_hot_events += len(hot_idx)

        all_idx = list(range(VOL_LEN, len(close) - max(HORIZONS_BARS.values()), 4))
        for h, bars in HORIZONS_BARS.items():
            baseline_fwd[h].append(_fwd_returns(close, all_idx, bars))
            cold_fwd[h].append(_fwd_returns(close, cold_idx, bars))
            hot_fwd[h].append(_fwd_returns(close, hot_idx, bars))

    print(f"analize giren sembol: {n_syms} | COLD olay: {n_cold_events} | HOT bar: {n_hot_events}\n")
    print(f"{'ufuk':6} {'grup':22} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for h in HORIZONS_BARS:
        for name, store in (("baseline (tüm barlar)", baseline_fwd),
                            ("COLD (tam sönme)", cold_fwd),
                            ("HOT (henüz kırmızı)", hot_fwd)):
            rets = np.concatenate(store[h]) if store[h] else np.array([])
            s = _stats(rets)
            print(f"{h:6} {name:22} {s.get('n',0):>7} {s.get('wr',0):>6} "
                  f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")
        print()


if __name__ == "__main__":
    run()
