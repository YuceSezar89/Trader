"""
R-Score, zaten dogrulanmis RSI Cross combined skorunun (bkz.
[[rsi_cross_combined_score_bt]], rho +0.196/+0.215, 4 kapili) UZERINE bonus
filtre olarak eklenince gercek bir katki sagliyor mu? 18 Tem 2026 — devisso_score
bonus testiyle ([[rsi_cross_devisso_bonus_bt]]) AYNI kademeli/stacking
mantigi: standalone zayif-ama-gercek bir filtre, guclu bir filtrenin ustune
eklendiginde COGU ZAMAN ek deger katmiyor (ornek zaten kucultulmus, bilgi
ortusuyor) — bunu R-Score icin test ediyoruz.

Kullanım: python -m research.pattern_lab.r_score_bonus_stack_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config
from research.pattern_lab.r_score_signal_quality_bt import _add_r_score
from research.pattern_lab.rsi_cross_combined_score_bt import _add_combined_score, _fetch_signals as _fetch_rsi_signals


def _fetch_real_trades() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = "SELECT signal_id, pnl_usd FROM paper_trades WHERE strategy = 'rsi_cross_live' AND status = 'closed'"
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def main() -> None:
    print("RSI_Cross(9,24) kapanmış sinyalleri çekiliyor...")
    df = _fetch_rsi_signals()
    print(f"Toplam sinyal: {len(df)}")

    print("\nRSI Cross combined skoru hesaplanıyor...")
    df = _add_combined_score(df)
    df = df.dropna(subset=["combined_adj"])

    print("R-Score hesaplanıyor...")
    df = _add_r_score(df)
    df = df.dropna(subset=["r_score"])
    print(f"Her ikisi de geçerli: {len(df)}")

    # Yon-ayarli R-Score: Long'da yuksek iyi, Short'ta dusuk iyi (bugunku bulgu)
    df["r_score_adj"] = np.where(df["signal_type"] == "Long", df["r_score"], -df["r_score"])

    real = _fetch_real_trades()
    merged = df.merge(real, left_on="id", right_on="signal_id", how="inner")
    print(f"Gerçek $ kesişimi: {len(merged)}")

    for sig_type in ["Long", "Short"]:
        sub_all = merged[merged["signal_type"] == sig_type]
        if len(sub_all) < 30:
            continue
        print(f"\n  {sig_type}:")

        # (1) Sadece RSI Cross ust tercile
        s1 = sub_all[sub_all["combined_adj"] >= sub_all["combined_adj"].quantile(0.667)]
        print(
            f"    (1) Sadece RSI Cross üst tercile:      n={len(s1)}, "
            f"ort_$={s1['pnl_usd'].mean():+.3f}, toplam_$={s1['pnl_usd'].sum():+.2f}, "
            f"wr={(s1['pnl_usd']>0).mean()*100:.1f}%"
        )

        # (2) + R-Score bonus (S1 icinde de ust yariya gecen)
        if len(s1) >= 10:
            s2 = s1[s1["r_score_adj"] >= s1["r_score_adj"].median()]
            print(
                f"    (2) + R-Score bonus (üst yarı):        n={len(s2)}, "
                f"ort_$={s2['pnl_usd'].mean():+.3f}, toplam_$={s2['pnl_usd'].sum():+.2f}, "
                f"wr={(s2['pnl_usd']>0).mean()*100:.1f}%"
            )
            s2_alt = s1[s1["r_score_adj"] < s1["r_score_adj"].median()]
            print(
                f"    (2b) R-Score alt yarı (karşılaştırma): n={len(s2_alt)}, "
                f"ort_$={s2_alt['pnl_usd'].mean():+.3f}, toplam_$={s2_alt['pnl_usd'].sum():+.2f}, "
                f"wr={(s2_alt['pnl_usd']>0).mean()*100:.1f}%"
            )


if __name__ == "__main__":
    main()
