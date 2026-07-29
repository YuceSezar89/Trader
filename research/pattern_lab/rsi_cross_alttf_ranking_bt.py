"""
Alt-TF kohort sıralaması testi — 18 Tem 2026, kullanıcının fikri: 15m/5m
sinyalinde kohortu sıralamak için sinyalin KENDİ TF'sinde 5 bar (15m'de 75dk,
5m'de 25dk) beklemek yerine, 1m'nin ilk 5 barına (5dk) bakarak çok daha erken
sıralama/seçim yapılabilir mi? [[rsi_cross_5m_early_entry_bt]]'deki "5m'nin
kendi erken hareketi 15m onayından daha güçlü" bulgusunun kohort-sıralama
versiyonu.

İki early_pct hesaplanıyor, AYNI kohort üzerinde:
  - early_pct_own: sinyalin KENDİ TF'sinde 5 bar sonrası (mevcut yöntem)
  - early_pct_1m : HER interval için 1m'nin 5 barı sonrası (5 dakika, sabit)

Kullanım: python -m research.pattern_lab.rsi_cross_alttf_ranking_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_MIN_COHORT_SIZE = 2
_COHORT_KEY = ["signal_type", "interval", "opened_at"]
_EARLY_BARS = 5
_INTERVAL_MIN = {"5m": 5, "15m": 15}  # 1m/1h/4h disinda birakildi (odak: bekleme maliyeti yuksek olanlar)


def _fetch_interval(interval: str, early_bars: int) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    own_offset = early_bars * _INTERVAL_MIN[interval] - 1
    alt_offset = early_bars * 1 - 1  # 1m'de 5 bar = 5dk, interval'den bagimsiz sabit
    q = """
        SELECT s.symbol, s.signal_type, s.interval, s.opened_at,
               s.open_price, s.realized_pnl,
               pd_own.close AS price_early_own,
               pd_alt.close AS price_early_1m
        FROM signals s
        JOIN LATERAL (
            SELECT p.close FROM price_data p
            WHERE p.symbol = s.symbol AND p.interval = '1m' AND p.timestamp > s.opened_at
            ORDER BY p.timestamp ASC LIMIT 1 OFFSET %s
        ) pd_own ON true
        JOIN LATERAL (
            SELECT p.close FROM price_data p
            WHERE p.symbol = s.symbol AND p.interval = '1m' AND p.timestamp > s.opened_at
            ORDER BY p.timestamp ASC LIMIT 1 OFFSET %s
        ) pd_alt ON true
        WHERE s.indicators = 'RSI_Cross(9,24)' AND s.interval = %s
          AND s.status = 'closed' AND s.realized_pnl IS NOT NULL
          AND s.open_price IS NOT NULL AND s.open_price > 0
    """
    df = pd.read_sql(q, conn, params=(own_offset, alt_offset, interval))
    conn.close()
    return df


def _fetch(early_bars: int) -> pd.DataFrame:
    pieces = [_fetch_interval(iv, early_bars) for iv in _INTERVAL_MIN]
    return pd.concat(pieces, ignore_index=True)


def _add_pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    side = np.where(df["signal_type"] == "Long", 1.0, -1.0)
    df["early_pct_own"] = (df["price_early_own"] - df["open_price"]) / df["open_price"] * 100.0 * side
    df["early_pct_1m"] = (df["price_early_1m"] - df["open_price"]) / df["open_price"] * 100.0 * side
    return df


def _build_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cohort_size"] = df.groupby(_COHORT_KEY)["symbol"].transform("size")
    df = df[df["cohort_size"] >= _MIN_COHORT_SIZE].copy()
    df["rank_own"] = df.groupby(_COHORT_KEY)["early_pct_own"].rank(pct=True)
    df["rank_1m"] = df.groupby(_COHORT_KEY)["early_pct_1m"].rank(pct=True)
    return df


def _report(df: pd.DataFrame, rank_col: str, label: str) -> None:
    for interval in ["5m", "15m"]:
        idf = df[df["interval"] == interval]
        for sig_type in ["Long", "Short"]:
            sub = idf[idf["signal_type"] == sig_type]
            if len(sub) < 30:
                continue
            rho, p = spearmanr(sub[rank_col], sub["realized_pnl"])
            tercile = pd.qcut(sub[rank_col], 3, labels=["alt", "orta", "üst"], duplicates="drop")
            g = sub.groupby(tercile, observed=True)["realized_pnl"].agg(
                ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean()
            )
            ust = g.loc["üst"] if "üst" in g.index else None
            ust_str = f"üst: n={int(ust['n'])} ort_pnl={ust['ort_pnl']:+.3f} wr={ust['wr']*100:.1f}%" if ust is not None else "üst: yok"
            print(f"  [{label}] {interval} {sig_type}: n={len(sub)}, rho={rho:+.3f} (p={p:.4f})  |  {ust_str}")


def main() -> None:
    print("RSI_Cross(9,24) 5m/15m verisi (hem kendi-TF hem 1m erken fiyatı) çekiliyor...")
    df = _fetch(_EARLY_BARS)
    df = _add_pct(df)
    cohorts = _build_cohorts(df)
    print(f"Kohort içi sinyal: {len(cohorts)}\n")

    print("=== YÖNTEM A: Kendi TF'sinde 5 bar sonrası (15m'de 75dk, 5m'de 25dk bekleme) ===")
    _report(cohorts, "rank_own", "kendi-TF")

    print("\n=== YÖNTEM B: 1m'de 5 bar sonrası (HER interval için sadece 5dk bekleme) ===")
    _report(cohorts, "rank_1m", "1m-alt")

    print("\n=== KARŞILAŞTIRMA: iki sıralamanın kendi arasındaki korelasyonu ===")
    for interval in ["5m", "15m"]:
        idf = cohorts[cohorts["interval"] == interval]
        rho, p = spearmanr(idf["rank_own"], idf["rank_1m"])
        print(f"  {interval}: rank_own vs rank_1m rho={rho:+.3f} (p={p:.4f}) — 1.0'a yakınsa aynı sembolü seçiyorlar demek")


if __name__ == "__main__":
    main()
