"""
Adım 3 koşucusu — 40 gözlemlik özellik matrisi.

Her çift için: vaka t0'ında hem vaka hem kontrol ölçülür (aynı damga —
piyasa fazı karıştırıcısı düşer). Sıralama özellikleri RankProvider'dan;
evren < 150 ise None (UB kuralı). Sinyal yoğunluğu signals tablosundan
tek sorguyla önceden çekilir (uptime kirliliği bayrağıyla raporlanacak).
"""
import json
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import psycopg2

from config import Config
from research.pattern_lab import config as C
from research.pattern_lab.features import extract_features
from research.pattern_lab.rank import RankProvider


def _signal_counts(symbols: list[str], t_start: datetime) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = "SELECT symbol, opened_at FROM signals WHERE symbol = ANY(%s) AND opened_at >= %s"
    df = pd.read_sql(q, conn, params=(symbols, t_start))
    conn.close()
    return df


def run() -> pd.DataFrame:
    with open(f"{C.CORPUS_DIR}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    layer_a = pd.read_parquet(f"{C.CORPUS_DIR}/layer_a_detail.parquet")
    layer_b = pd.read_parquet(f"{C.CORPUS_DIR}/layer_b_universe.parquet")
    t0_tbl = pd.read_parquet(f"{C.CORPUS_DIR}/t0_table.parquet")

    btc = layer_b[layer_b["symbol"] == "BTCUSDT"].set_index("ts")["close"].sort_index()
    assert len(btc) > 1000, "BTC referans serisi eksik!"

    rp = RankProvider(layer_b)
    all_syms = [s for pair in meta["pairs"] for s in pair]
    sig = _signal_counts(all_syms, datetime.fromisoformat(meta["anchor"]) - pd.Timedelta(days=C.CORPUS_DAYS))

    rows = []
    for _, r in t0_tbl.iterrows():
        t0 = pd.Timestamp(r["t0"])
        for role, sym in (("vaka", r["vaka"]), ("kontrol", r["kontrol"])):
            sdf = layer_a[layer_a["symbol"] == sym]
            feats = extract_features(sdf, btc, t0)
            if not feats:
                print(f"  UYARI: {sym}@{t0} özellik üretilemedi")
                continue
            for name, bars in C.RANK_WINDOWS_BARS.items():
                feats[f"rank_{name}"] = rp.rank_pct(sym, t0, bars)
            # sıra yörüngesi: 24h penceresi sırası, t0-48h..t0 (tırmanış ölçüsü)
            traj = rp.rank_series(sym, t0, hours_back=48, window_bars=288)
            valid = traj.dropna()
            feats["rank24_tirmanis"] = float(valid.iloc[-1] - valid.iloc[0]) if len(valid) >= 2 else None
            feats["top10pct_saat"] = float((valid > 90).mean() * 48) if len(valid) >= 2 else None
            n_sig = ((sig["symbol"] == sym) & (sig["opened_at"] >= t0 - pd.Timedelta(hours=48))
                     & (sig["opened_at"] < t0)).sum()
            feats["sinyal_yogunlugu_48h"] = int(n_sig)
            rows.append({"sembol": sym, "rol": role, "t0": t0, **feats})
        print(f"  çift işlendi: {r['vaka']} / {r['kontrol']} @ {t0}")

    out = pd.DataFrame(rows)
    out.to_parquet(f"{C.CORPUS_DIR}/feature_matrix.parquet", index=False)
    print(f"\nÖzellik matrisi: {out.shape[0]} gözlem × {out.shape[1]} kolon → parquet")
    return out


if __name__ == "__main__":
    run()
