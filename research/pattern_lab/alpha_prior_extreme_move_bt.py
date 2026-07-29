"""
Mevcut alpha/beta'nin (indicators/financial_metrics.py) penceresi sinyal
ufkuyla uyumsuz (beta: 1 gun-1 ay, alpha: TF'ye gore sabit-bar, coin'e ozgu
degil) — 18 Tem 2026, kullanici gozlemi: "short geldi ama sembol 2-3 gun once
asiri hareket yapmisti, sonuc olumluydu". Bu script, sinyalden ONCE (24h/48h/
72h) coin'in BTC'ye gore YAPTIGI HAM excess return'u (basit fark, beta
regresyonu yok — "asiri hareket" kavramini dogrudan olcuyor) hesaplayip
realized_pnl + gercek $ ile iliskisini tek SQL sorgusuyla (JOIN LATERAL,
[[rsi_cross_tf_alignment_bt]] ile ayni verimli desen) test eder.

Kullanım: python -m research.pattern_lab.alpha_prior_extreme_move_bt
"""

import warnings

import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_WINDOWS_HOURS = [24, 48, 72]
_REF_SYMBOL = Config.MARKET_REFERENCE_SYMBOL


def _fetch_window(hours: int) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT s.id, s.symbol, s.signal_type, s.opened_at, s.realized_pnl,
               coin_now.close AS coin_now, coin_prior.close AS coin_prior,
               btc_now.close AS btc_now, btc_prior.close AS btc_prior
        FROM signals s
        JOIN LATERAL (
            SELECT p.close FROM price_data p
            WHERE p.symbol = s.symbol AND p.interval = '1m' AND p.timestamp <= s.opened_at
            ORDER BY p.timestamp DESC LIMIT 1
        ) coin_now ON true
        JOIN LATERAL (
            SELECT p.close FROM price_data p
            WHERE p.symbol = s.symbol AND p.interval = '1m'
              AND p.timestamp <= s.opened_at - INTERVAL '{hours} hours'
            ORDER BY p.timestamp DESC LIMIT 1
        ) coin_prior ON true
        JOIN LATERAL (
            SELECT p.close FROM price_data p
            WHERE p.symbol = %s AND p.interval = '1m' AND p.timestamp <= s.opened_at
            ORDER BY p.timestamp DESC LIMIT 1
        ) btc_now ON true
        JOIN LATERAL (
            SELECT p.close FROM price_data p
            WHERE p.symbol = %s AND p.interval = '1m'
              AND p.timestamp <= s.opened_at - INTERVAL '{hours} hours'
            ORDER BY p.timestamp DESC LIMIT 1
        ) btc_prior ON true
        WHERE s.indicators = 'RSI_Cross(9,24)' AND s.status = 'closed'
          AND s.realized_pnl IS NOT NULL AND s.open_price IS NOT NULL AND s.open_price > 0
    """
    df = pd.read_sql(q, conn, params=(_REF_SYMBOL, _REF_SYMBOL))
    conn.close()
    df["coin_ret"] = (df["coin_now"] - df["coin_prior"]) / df["coin_prior"] * 100.0
    df["btc_ret"] = (df["btc_now"] - df["btc_prior"]) / df["btc_prior"] * 100.0
    df["excess"] = df["coin_ret"] - df["btc_ret"]
    return df


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
    real = _fetch_real_trades()
    for hours in _WINDOWS_HOURS:
        print(f"\n=== {hours} saatlik pencere: coin'in {_REF_SYMBOL}'e göre excess return'ü ===")
        df = _fetch_window(hours)
        print(f"Toplam sinyal: {len(df)}")

        for sig_type in ["Long", "Short"]:
            sub = df[df["signal_type"] == sig_type]
            if len(sub) < 30:
                continue
            rho, p = spearmanr(sub["excess"], sub["realized_pnl"])
            print(f"  {sig_type}: n={len(sub)}, rho={rho:+.3f} (p={p:.4f})")
            tercile = pd.qcut(sub["excess"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
            g = sub.groupby(tercile, observed=True)["realized_pnl"].agg(
                ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean() * 100
            )
            print("   ", g.to_string().replace("\n", "\n    "))

        merged = df.merge(real, left_on="id", right_on="signal_id", how="inner")
        if len(merged) >= 30:
            print(f"  --- GERÇEK $ (n={len(merged)}) ---")
            for sig_type in ["Long", "Short"]:
                sub = merged[merged["signal_type"] == sig_type]
                if len(sub) < 15:
                    print(f"    {sig_type}: yetersiz örnek (n={len(sub)})")
                    continue
                tercile = pd.qcut(sub["excess"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
                g = sub.groupby(tercile, observed=True).agg(
                    n=("pnl_usd", "count"),
                    ort_usd=("pnl_usd", "mean"),
                    toplam_usd=("pnl_usd", "sum"),
                    wr=("pnl_usd", lambda s: (s > 0).mean() * 100),
                )
                print(f"    {sig_type}:")
                print("     ", g.to_string().replace("\n", "\n      "))


if __name__ == "__main__":
    main()
