"""
devisso_ema_cross_bt.py — Hoca'nın Devis'So göstergesinden çıkarılan, kendi
kodunun KULLANMADIĞI bir karşılaştırma: ema1 (yukarı-momentum) vs ema2
(aşağı-momentum) kesişimi.

Neden: devisso_original_bt.py'de bulundu — ema1/ema2'nin paydası aynı TİP
şey (ardışık mum SAYISI), bu yüzden ikisi de aynı ölçekte (|ort|≈2.4,
sembol/TF'den bağımsız SABİT). ema3'ün (ana momentum, paydası küçük bir
%değişim) ölçeği çok daha büyük ve değişken olduğu için ema1/ema3 veya
ema2/ema3 kesişimi (Hoca'nın orijinal AL/SAT mantığı) pratikte neredeyse
hiç tetiklenmiyordu (40 günde BTCUSDT'de 0 kez). ema1 vs ema2 ise 40 günde
~360 kez kesişiyor — kullanılabilir bir frekans.

Hoca'nın kodu bu iki çizgiyi ASLA doğrudan karşılaştırmıyor (sadece ayrı
ayrı ema3 ile) — bu script o boşluğu, aynı disiplinle (canlı formül,
placebo, split-period) test ediyor.

BUY: ema1, ema2'yi YUKARI keser (yukarı-momentum aşağı-momentumu geçti)
SELL: ema1, ema2'yi AŞAĞI keser

Kullanım: python -m research.pattern_lab.devisso_ema_cross_bt
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import research.pattern_lab.devisso_original_bt as base  # noqa: E402  pylint: disable=wrong-import-position

FORWARD_BARS = {"1h": 4, "4h": 16, "12h": 48, "24h": 96}
_PLACEBO_SEED = 42
BURN_IN = 200


def _find_ema12_crossovers(ind: dict) -> tuple[list[int], list[int]]:
    ema1, ema2 = ind["ema1"], ind["ema2"]
    n = len(ema1)
    ups, downs = [], []
    for i in range(BURN_IN, n):
        if not (np.isfinite(ema1[i]) and np.isfinite(ema2[i]) and np.isfinite(ema1[i - 1]) and np.isfinite(ema2[i - 1])):
            continue
        if ema1[i - 1] <= ema2[i - 1] and ema1[i] > ema2[i]:
            ups.append(i)
        elif ema1[i - 1] >= ema2[i - 1] and ema1[i] < ema2[i]:
            downs.append(i)
    return ups, downs


def _forward_returns(c: np.ndarray, idx: int) -> dict[str, float]:
    entry = c[idx]
    out = {}
    for name, bars in FORWARD_BARS.items():
        j = min(idx + bars, len(c) - 1)
        out[name] = (c[j] - entry) / entry * 100.0
    return out


def _stats(vals: np.ndarray) -> dict:
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return {"n": 0}
    return {
        "n": len(vals), "ort_%": round(float(vals.mean()), 3),
        "medyan_%": round(float(np.median(vals)), 3), "wr": round(float((vals > 0).mean() * 100), 1),
    }


def run() -> None:
    conn = base._conn()  # pylint: disable=protected-access
    bad = base._bad_symbols(conn.cursor())  # pylint: disable=protected-access
    conn.close()
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    df_all = base._fetch(bad)  # pylint: disable=protected-access
    print(f"{df_all['symbol'].nunique()} sembol, {len(df_all):,} 15m bar ({base.DAYS} gün)\n")

    up_records: list[dict] = []
    down_records: list[dict] = []
    placebo_records: list[dict] = []
    rng = np.random.default_rng(_PLACEBO_SEED)

    n_syms = 0
    for sym, g in df_all.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < base.MIN_BARS:
            continue
        n_syms += 1
        c = g["close"].to_numpy(float)
        ts = g["ts"]

        ind = base._compute_devisso_original(g)  # pylint: disable=protected-access
        ups, downs = _find_ema12_crossovers(ind)

        for idx in ups:
            if idx + max(FORWARD_BARS.values()) >= len(c):
                continue
            rec = {"symbol": sym, "ts": ts.iloc[idx]}
            rec.update(_forward_returns(c, idx))
            up_records.append(rec)

        for idx in downs:
            if idx + max(FORWARD_BARS.values()) >= len(c):
                continue
            rec = {"symbol": sym, "ts": ts.iloc[idx]}
            rec.update(_forward_returns(c, idx))
            down_records.append(rec)

        valid_range = np.arange(BURN_IN, len(c) - max(FORWARD_BARS.values()))
        if len(valid_range) > 0 and ups:
            sample = rng.choice(valid_range, size=min(len(ups), len(valid_range)), replace=False)
            for idx in sample:
                rec = {"symbol": sym, "ts": ts.iloc[idx]}
                rec.update(_forward_returns(c, idx))
                placebo_records.append(rec)

    print(f"[tarama] {n_syms} sembol")
    print(f"Toplam UP-cross (ema1 ema2'yi yukarı kesti) sinyali: {len(up_records)}")
    print(f"Toplam DOWN-cross (ema1 ema2'yi aşağı kesti) sinyali: {len(down_records)}")
    print(f"Toplam PLACEBO noktası: {len(placebo_records)}\n")

    udf = pd.DataFrame(up_records)
    ddf = pd.DataFrame(down_records)
    plc = pd.DataFrame(placebo_records)

    print("=== UP-cross (yukarı-momentum aşağıyı geçti) sonrası ileri getiri ===")
    for name in FORWARD_BARS:
        print(f"  {name:4}: {_stats(udf[name].to_numpy())}")

    print("\n=== DOWN-cross (aşağı-momentum yukarıyı geçti) sonrası ileri getiri ===")
    for name in FORWARD_BARS:
        print(f"  {name:4}: {_stats(ddf[name].to_numpy())}")

    print("\n=== PLACEBO (rastgele noktalar) sonrası ileri getiri ===")
    for name in FORWARD_BARS:
        print(f"  {name:4}: {_stats(plc[name].to_numpy())}")

    print("\n=== SPLIT-PERIOD (UP-cross, 24h) ===")
    mid = udf["ts"].median()
    for label, sub in [("IS", udf[udf["ts"] < mid]), ("OOS", udf[udf["ts"] >= mid])]:
        print(f"  {label}: {_stats(sub['24h'].to_numpy())}")

    print("\n=== SPLIT-PERIOD (DOWN-cross, 24h) ===")
    mid_d = ddf["ts"].median()
    for label, sub in [("IS", ddf[ddf["ts"] < mid_d]), ("OOS", ddf[ddf["ts"] >= mid_d])]:
        print(f"  {label}: {_stats(sub['24h'].to_numpy())}")


if __name__ == "__main__":
    run()
