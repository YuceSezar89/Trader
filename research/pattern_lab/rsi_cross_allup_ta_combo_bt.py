"""
all_up + TA-uyumu (5m+1H hemfikir) KOMBİNASYONU — RSI_Cross(9,24) Long / 5m
(22 Tem 2026, kullanıcı isteği: bugünün iki en güçlü bulgusunu birleştir).

all_up: vol/mom/volat/price skorlarının HEPSİ bir önceki aynı sembol
sinyaline göre artmış mı (rsi_cross_allup_candleshape_clean_bt.py).
TA-uyumu: 5m VE 1H'in kümülatif 12-saatlik momentumu (Total Amount) sinyal
yönüyle (Long: net>0) İKİSİ DE hemfikir mi (rsi_cross_ta_alignment_bt.py).

Kullanım: python -m research.pattern_lab.rsi_cross_allup_ta_combo_bt
"""

import numpy as np
import pandas as pd

from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import (
    _add_all_up,
    _bad_symbols,
    _collect,
    _conn,
)
from research.pattern_lab.rsi_cross_ta_alignment_bt import _deep_validate, _fetch_series

_PLACEBO_ITER = 300


def _add_ta_alignment(df: pd.DataFrame, conn) -> pd.DataFrame:
    cur = conn.cursor()
    aligns = []
    symbols = df["symbol"].unique()
    print(f"{len(symbols)} sembol için 5m+1H TA serisi hesaplanacak")
    cache: dict[str, pd.DataFrame] = {}
    for si, sym in enumerate(symbols):
        df5 = _fetch_series(cur, sym, "cagg_5m")
        df1h = _fetch_series(cur, sym, "cagg_1h")
        cache[sym] = (df5, df1h)
        if (si + 1) % 150 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")

    for _, row in df.iterrows():
        df5, df1h = cache.get(row["symbol"], (pd.DataFrame(), pd.DataFrame()))
        if df5.empty or df1h.empty:
            aligns.append(None)
            continue
        b5 = df5["bucket"].to_numpy()
        net5arr = df5["net_ta"].to_numpy()
        b1h = df1h["bucket"].to_numpy()
        net1harr = df1h["net_ta"].to_numpy()
        idx5 = np.searchsorted(b5, np.datetime64(row["opened_at"]), side="right") - 1
        idx1h = np.searchsorted(b1h, np.datetime64(row["opened_at"]), side="right") - 1
        if idx5 < 0 or idx1h < 0:
            aligns.append(None)
            continue
        match5 = net5arr[idx5] > 0
        match1h = net1harr[idx1h] > 0
        if match5 and match1h:
            aligns.append("UYUMLU")
        elif match5 or match1h:
            aligns.append("KISMİ")
        else:
            aligns.append("UYUMSUZ")
    df = df.copy()
    df["ta_align"] = aligns
    return df


def main() -> None:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    print("\n[all_up verisi] sinyaller + canlı formül bileşenleri toplanıyor...")
    raw = _collect(conn, "RSI_Cross(9,24)", "Long", bad)
    raw = raw[raw["interval"] == "5m"].reset_index(drop=True)
    print(f"  5m'e indirgendi: n={len(raw)}")

    df = _add_all_up(raw)
    print(f"[all_up hesaplandı] n={len(df)} (bir önceki aynı sembol sinyali olanlar)")

    df = _add_ta_alignment(df, conn)
    conn.close()
    df = df.dropna(subset=["ta_align"]).reset_index(drop=True)
    print(f"\n[collect] TA uyumu da hesaplandı: n={len(df)}\n")

    print("=== [1] all_up × ta_align çapraz tablo ===")
    g = df.groupby(["all_up", "ta_align"])["fwd_ret"].agg(ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100)
    print(g.to_string())

    df["_g"] = df["all_up"] & (df["ta_align"] == "UYUMLU")
    print(f"\n=== [2] KOMBO: all_up=True VE ta_align=UYUMLU ===")
    _deep_validate("all_up + TA UYUMLU", df[df["_g"]], df[~df["_g"]], df)

    print(f"\n=== [3] Referans: sadece all_up=True (TA'sız) ===")
    df["_g"] = df["all_up"]
    _deep_validate("sadece all_up=True", df[df["all_up"]], df[~df["all_up"]], df)

    print(f"\n=== [4] Referans: sadece ta_align=UYUMLU (all_up'sız) ===")
    df["_g"] = df["ta_align"] == "UYUMLU"
    _deep_validate("sadece TA UYUMLU", df[df["ta_align"] == "UYUMLU"], df[df["ta_align"] != "UYUMLU"], df)


if __name__ == "__main__":
    main()
