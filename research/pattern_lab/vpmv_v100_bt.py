"""
VPMV'nin Hacim (V) bileşeni tam 100 olduğunda (o barın hacmi, son 50 barın
hacim rekorunu kırdığında — utils/preprocessing.py::normalize_volume_0_100,
BİREBİR aynı fonksiyon, canlı üretimle aynı) fiyat sonrasında ne yapıyor?

V=100 kendi başına yönsüz (rekor hacim yeşil de kırmızı da mumda olabilir) —
bu yüzden olay ayrıca o barın rengine (yeşil/kırmızı) göre ayrıştırılıyor:
yeşil = alım patlaması/kırılım adayı, kırmızı = satım patlaması/kapitülasyon
adayı olabilir, çok farklı davranabilirler.

Look-ahead yok: V, sadece geçmiş 50 barla (rolling min/max) hesaplanıyor,
ileri getiri ayrı ölçülüyor. Baseline = ilgili TF'deki TÜM barlar.
"""

import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from utils.preprocessing import normalize_volume_0_100  # pylint: disable=wrong-import-position

DAYS = 60
MIN_BARS = 250
WARMUP = 200
V_WINDOW = 50
HORIZONS = {
    "15dk": 1,
    "30dk": 2,
    "1s": 4,
    "2s": 8,
    "4s": 16,
}  # 15m bar cinsinden — 5m/15m sinyallerin gerçek tutma süresine yakın


def _fetch() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, open, high, low, close, volume
        FROM cagg_15m
        WHERE bucket > NOW() - INTERVAL '{DAYS} days'
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def run() -> None:
    df = _fetch()
    t_min, t_max = df["ts"].min(), df["ts"].max()
    mid = t_min + (t_max - t_min) / 2
    print(
        f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n"
        f"dönem: {t_min} .. {t_max} | orta nokta: {mid}\n"
    )

    labels = ["baseline", "V100_yesil", "V100_kirmizi"]
    halves = ["tum", "ilk_yari", "ikinci_yari"]
    res = {h: {lbl: {half: [] for half in halves} for lbl in labels} for h in HORIZONS}
    n_syms, n_v100_yesil, n_v100_kirmizi = 0, 0, 0

    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1

        v_score = normalize_volume_0_100(g["volume"], window=V_WINDOW).to_numpy()
        close = g["close"].to_numpy(float)
        open_ = g["open"].to_numpy(float)
        ts_np = g["ts"].to_numpy()

        is_v100 = v_score >= 99.99
        is_green = close > open_
        v100_yesil = is_v100 & is_green
        v100_kirmizi = is_v100 & ~is_green
        n_v100_yesil += int(v100_yesil.sum())
        n_v100_kirmizi += int(v100_kirmizi.sum())

        max_h = max(HORIZONS.values())
        all_idx = np.arange(WARMUP, len(g) - max_h)
        idx_by_label = {
            "baseline": all_idx,
            "V100_yesil": all_idx[v100_yesil[all_idx]],
            "V100_kirmizi": all_idx[v100_kirmizi[all_idx]],
        }

        for lbl, idxs in idx_by_label.items():
            if len(idxs) == 0:
                continue
            first_mask = ts_np[idxs] < np.datetime64(mid)
            for h_name, h_bars in HORIZONS.items():
                rets = close[idxs + h_bars] / close[idxs] - 1
                res[h_name][lbl]["tum"].append(rets)
                res[h_name][lbl]["ilk_yari"].append(rets[first_mask])
                res[h_name][lbl]["ikinci_yari"].append(rets[~first_mask])

    print(
        f"analiz edilen sembol: {n_syms} | V100-yeşil olay: {n_v100_yesil} | V100-kırmızı olay: {n_v100_kirmizi}\n"
    )
    for h_name in HORIZONS:
        print(f"── ufuk: {h_name} ──")
        print(f"{'grup':14} {'dönem':12} {'n':>8} {'WR%':>6} {'ort%':>8} {'PF':>7}")
        for lbl in labels:
            for half in halves:
                arrs = res[h_name][lbl][half]
                rets = np.concatenate(arrs) if arrs else np.array([])
                s = _stats(rets)
                print(
                    f"{lbl:14} {half:12} {s.get('n',0):>8} {s.get('wr',0):>6} "
                    f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}"
                )
        print()


if __name__ == "__main__":
    run()
