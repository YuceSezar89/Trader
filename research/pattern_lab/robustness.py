"""
Adım 5 — Sağlamlık kapıları (3 aday özellik için).

Kapı 1 PLACEBO : sahte t0 = gerçek t0 − 72h (taban: korpus başı + 62h).
                 Gerçek parmak izi patlamaya özgüyse sahte anda KAYBOLMALI
                 (|delta_placebo| < 0.20). Kaybolmuyorsa ölçtüğümüz şey
                 "kazanan coin karakteri"dir, "patlama arifesi" değil.
Kapı 2 SEED    : kontrol grubu SEED+1000 ile yeniden seçilir (aynı eşleştirme
                 kuralı, yeni semboller REST'ten çekilir). İşaret korunmalı
                 ve |delta| >= 0.20 kalmalı.
Kapı 3 PENCERE : LONG bağlam 48h → 24h. İşaret korunmalı, |delta| >= 0.20.
"""
import asyncio
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab import config as C
from research.pattern_lab import features as F
from research.pattern_lab.rank import RankProvider
from research.pattern_lab.report import cliffs_delta

CANDIDATES = ["top10pct_saat", "akis_gercek_6h", "rsi_delta_6h"]


def _load():
    with open(f"{C.CORPUS_DIR}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    la = pd.read_parquet(f"{C.CORPUS_DIR}/layer_a_detail.parquet")
    lb = pd.read_parquet(f"{C.CORPUS_DIR}/layer_b_universe.parquet")
    t0s = pd.read_parquet(f"{C.CORPUS_DIR}/t0_table.parquet")
    btc = lb[lb["symbol"] == "BTCUSDT"].set_index("ts")["close"].sort_index()
    return meta, la, lb, t0s, btc


def _observe(sym, t0, la, btc, rp, hours_back=48):
    feats = F.extract_features(la[la["symbol"] == sym], btc, t0)
    if not feats:
        return None
    traj = rp.rank_series(sym, pd.Timestamp(t0), hours_back=hours_back, window_bars=288)
    valid = traj.dropna()
    feats["top10pct_saat"] = float((valid > 90).mean() * hours_back) if len(valid) >= 2 else np.nan
    return feats


def _delta(pairs_vals):
    a = np.array([v for v, _ in pairs_vals], dtype=float)
    b = np.array([v for _, v in pairs_vals], dtype=float)
    return cliffs_delta(a, b)


def _compare(pairs, t0_map, la, btc, rp, hours_back=48):
    """Her aday için Cliff's delta (vaka vs kontrol, verilen t0 haritasıyla)."""
    obs = {}
    for case, ctrl in pairs:
        t0 = t0_map[case]
        fo = _observe(case, t0, la, btc, rp, hours_back)
        fc = _observe(ctrl, t0, la, btc, rp, hours_back)
        if fo and fc:
            obs[case] = (fo, fc)
    out = {}
    for feat in CANDIDATES:
        vals = [(fo.get(feat, np.nan), fc.get(feat, np.nan)) for fo, fc in obs.values()]
        out[feat] = round(_delta(vals), 3)
    return out


async def _fetch_new_controls(symbols, anchor):
    from research.pattern_lab.corpus import fetch_layer_a
    return await fetch_layer_a(symbols, anchor)


def run():
    meta, la, lb, t0s, btc = _load()
    anchor = pd.Timestamp(meta["anchor"])
    rp = RankProvider(lb)
    pairs = [tuple(p) for p in meta["pairs"]]
    real_t0 = dict(zip(t0s["vaka"], t0s["t0"]))

    print("── Referans (gerçek t0, 48h) ──")
    base = _compare(pairs, real_t0, la, btc, rp)
    print(base)

    print("\n── KAPI 1: Placebo (t0 − 72h) ──")
    floor = la["ts"].min() + pd.Timedelta(hours=62)
    fake_t0 = {c: max(pd.Timestamp(t) - pd.Timedelta(hours=72), floor) for c, t in real_t0.items()}
    placebo = _compare(pairs, fake_t0, la, btc, rp)
    print(placebo)

    print("\n── KAPI 2: Yeni seed'li kontrol grubu ──")
    stats = pd.read_parquet(f"{C.CORPUS_DIR}/universe_stats.parquet")
    lo, hi = stats["ret_pct"].quantile(C.MID_QUANTILE[0]), stats["ret_pct"].quantile(C.MID_QUANTILE[1])
    pool = stats[(stats["ret_pct"] >= lo) & (stats["ret_pct"] <= hi)]
    pool = pool[~pool["symbol"].isin([s for p in pairs for s in p])]
    used, new_pairs = set(), []
    dec_map = dict(zip(stats["symbol"], stats["vol_decile"]))
    for i, (case, _) in enumerate(pairs):
        dec = dec_map[case]
        for widen in range(C.VOL_DECILES):
            cand = pool[(pool["vol_decile"].sub(dec).abs() <= widen) & (~pool["symbol"].isin(used))]
            if not cand.empty:
                pick = cand.sample(1, random_state=C.SEED + 1000 + i).iloc[0]
                used.add(pick["symbol"])
                new_pairs.append((case, pick["symbol"]))
                break
    new_syms = [c for _, c in new_pairs]
    print(f"yeni kontroller: {new_syms}")
    la_new = asyncio.run(_fetch_new_controls(new_syms, anchor.to_pydatetime()))
    la2 = pd.concat([la, la_new], ignore_index=True)
    seed_res = _compare(new_pairs, real_t0, la2, btc, rp)
    print(seed_res)

    print("\n── KAPI 3: Dar pencere (LONG 48h → 24h) ──")
    orig_long = F.LONG_BARS
    F.LONG_BARS = (24 - C.FEAT_SHORT_H) * 12
    try:
        win_res = _compare(pairs, real_t0, la, btc, rp, hours_back=24)
    finally:
        F.LONG_BARS = orig_long
    print(win_res)

    print("\n══ HÜKÜM TABLOSU ══")
    print(f"{'özellik':22} {'ref':>7} {'placebo':>8} {'seed':>7} {'pencere':>8}  karar")
    for feat in CANDIDATES:
        b, p, s, w = base[feat], placebo[feat], seed_res[feat], win_res[feat]
        placebo_ok = abs(p) < 0.20
        seed_ok = (np.sign(s) == np.sign(b)) and abs(s) >= 0.20
        win_ok = (np.sign(w) == np.sign(b)) and abs(w) >= 0.20
        verdict = "✅ ADAY" if (placebo_ok and seed_ok and win_ok) else "❌ eleniyor"
        detail = f"(placebo{'✓' if placebo_ok else '✗'} seed{'✓' if seed_ok else '✗'} pencere{'✓' if win_ok else '✗'})"
        print(f"{feat:22} {b:>7} {p:>8} {s:>7} {w:>8}  {verdict} {detail}")


if __name__ == "__main__":
    run()
