"""
devisso_score'u BONUS filtre olarak dogru sekilde test etme — 18 Tem 2026,
kullanicinin itirazi haklı çıktı: kohort-içi çifte-sıralama (rank-of-rank)
bilgiyi bozuyordu. devisso_score zaten kendi 100 barlık geçmişine göre bir
percentile (0-100) — bu HAM haliyle (kohort içinde TEKRAR sıralamadan) tüm
evrende zayıf ama tutarlı bir yön gösterdi (4/4 testte üst tercile > alt
tercile, bkz. [[project_tf_alignment_early_divergence]]).

Bu script, HAM devisso_score>medyan şartını, zaten doğrulanmış TF-hizalanma+
erken-ayrışma yığınının ÜSTÜNE ekliyor (kohort içinde tekrar sıralamadan).

Kullanım: python -m research.pattern_lab.rsi_cross_devisso_bonus_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config
from research.pattern_lab.rsi_cross_tf_alignment_bt import (
    _EARLY_BARS,
    _add_early_pct,
    _add_htf_alignment,
    _build_cohorts,
    _fetch,
)


def _add_devisso(df: pd.DataFrame) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    sig = pd.read_sql(
        "SELECT id, symbol, signal_type, interval, opened_at, devisso_score FROM signals "
        "WHERE indicators='RSI_Cross(9,24)' AND devisso_score IS NOT NULL",
        conn,
    )
    conn.close()
    return df.merge(sig, on=["symbol", "signal_type", "interval", "opened_at"], how="inner")


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
    print("RSI_Cross(9,24) verisi + devisso_score hazırlanıyor...")
    df = _fetch(_EARLY_BARS)
    df = _add_early_pct(df)
    df = _add_devisso(df)
    cohorts = _build_cohorts(df)
    print(f"Kohort içi sinyal: {len(cohorts)}")

    print("1h/4h Heikin Ashi hizalanması hesaplanıyor...")
    aligned = _add_htf_alignment(cohorts)
    devisso_median = aligned["devisso_score"].median()
    print(f"devisso_score medyanı (bonus eşik): {devisso_median:.1f}\n")

    print("=== BACKTEST realized_pnl: kademeli filtre ===")
    for sig_type in ["Long", "Short"]:
        sub = aligned[aligned["signal_type"] == sig_type]
        print(f"\n  {sig_type}:")
        s1 = sub[sub["aligned_count"] == 2]
        print(f"    (1) +hizalı:                 n={len(s1)}, ort_pnl={s1['realized_pnl'].mean():+.4f}, wr={(s1['realized_pnl']>0).mean()*100:.1f}%")

        s2 = s1[s1["early_rank"] >= s1["early_rank"].quantile(0.667)] if len(s1) > 0 else s1
        print(f"    (2) +çok-ayrışan:            n={len(s2)}, ort_pnl={s2['realized_pnl'].mean():+.4f}, wr={(s2['realized_pnl']>0).mean()*100:.1f}%" if len(s2) > 0 else "    (2) örnek yok")

        s3 = s2[s2["devisso_score"] > devisso_median] if len(s2) > 0 else s2
        print(f"    (3) +devisso_score>medyan:   n={len(s3)}, ort_pnl={s3['realized_pnl'].mean():+.4f}, wr={(s3['realized_pnl']>0).mean()*100:.1f}%" if len(s3) > 0 else "    (3) örnek yok")

    print("\n=== GERÇEK $ DOĞRULAMASI (paper_trades.pnl_usd) — kademeli filtre ===")
    real = _fetch_real_trades()
    merged = aligned.merge(real, left_on="id", right_on="signal_id", how="inner")
    print(f"Kohort + hizalanma + GERÇEK işlem kesişimi: {len(merged)}")
    for sig_type in ["Long", "Short"]:
        sub_all = merged[merged["signal_type"] == sig_type]
        if len(sub_all) < 5:
            print(f"  {sig_type}: yetersiz örnek")
            continue
        print(f"\n  {sig_type}:")
        r1 = sub_all[sub_all["aligned_count"] == 2]
        print(f"    (1) +hizalı:                 n={len(r1)}, ort_$={r1['pnl_usd'].mean():+.3f}, toplam_$={r1['pnl_usd'].sum():+.2f}, wr={(r1['pnl_usd']>0).mean()*100:.1f}%" if len(r1) > 0 else "    (1) örnek yok")

        r2 = r1[r1["early_rank"] >= r1["early_rank"].quantile(0.667)] if len(r1) > 0 else r1
        print(f"    (2) +çok-ayrışan:            n={len(r2)}, ort_$={r2['pnl_usd'].mean():+.3f}, toplam_$={r2['pnl_usd'].sum():+.2f}, wr={(r2['pnl_usd']>0).mean()*100:.1f}%" if len(r2) > 0 else "    (2) örnek yok")

        r3 = r2[r2["devisso_score"] > devisso_median] if len(r2) > 0 else r2
        print(f"    (3) +devisso_score>medyan:   n={len(r3)}, ort_$={r3['pnl_usd'].mean():+.3f}, toplam_$={r3['pnl_usd'].sum():+.2f}, wr={(r3['pnl_usd']>0).mean()*100:.1f}%" if len(r3) > 0 else "    (3) örnek yok")


if __name__ == "__main__":
    main()
