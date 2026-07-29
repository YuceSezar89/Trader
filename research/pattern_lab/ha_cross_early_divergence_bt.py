"""
HA_Cross icin erken-ayrisma testi — [[rsi_cross_early_divergence_bt]] ile ayni
metodoloji, HA_Cross'a uyarlanmis. 17 Tem 2026'da HA_Cross'un TF hizalanmasi
olmadan (is_confluence sadece Long+15m'de ve sadece UI yildizi icin var, gercek
filtre degil) uretildigi ve ham performansinin zayif oldugu (win-rate %38.4)
bulundu — bu script, RSI_Cross'ta ise yarayan "kohort icinde en cok ayrisani
sec" yonteminin HA_Cross icin de bir edge saglayip saglamadigini test ediyor.

HA_Cross sadece 5m ve 15m'de uretiliyor (1m/1h/4h yok) — price_data sadece 1m
tuttugu icin, her interval icin ayri bir 1m-offset hesaplaniyor (5m sinyalde
"N bar sonrasi" = N*5 dakika sonrasi, 15m'de N*15 dakika sonrasi).

Kullanım: python -m research.pattern_lab.ha_cross_early_divergence_bt
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
_INTERVAL_MIN = {"5m": 5, "15m": 15}


def _fetch_interval(interval: str, early_bars: int) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    early_offset_1m = early_bars * _INTERVAL_MIN[interval] - 1
    q = """
        SELECT s.symbol, s.signal_type, s.interval, s.opened_at,
               s.open_price, s.realized_pnl, pd.close AS price_early
        FROM signals s
        JOIN LATERAL (
            SELECT p.close
            FROM price_data p
            WHERE p.symbol = s.symbol AND p.interval = '1m'
              AND p.timestamp > s.opened_at
            ORDER BY p.timestamp ASC
            LIMIT 1 OFFSET %s
        ) pd ON true
        WHERE s.indicators = 'HA_Cross' AND s.interval = %s
          AND s.status = 'closed' AND s.realized_pnl IS NOT NULL
          AND s.open_price IS NOT NULL AND s.open_price > 0
    """
    df = pd.read_sql(q, conn, params=(early_offset_1m, interval))
    conn.close()
    return df


def _fetch(early_bars: int) -> pd.DataFrame:
    return pd.concat(
        [_fetch_interval(iv, early_bars) for iv in _INTERVAL_MIN], ignore_index=True
    )


def _add_early_pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw_pct = (df["price_early"] - df["open_price"]) / df["open_price"] * 100.0
    side = np.where(df["signal_type"] == "Long", 1.0, -1.0)
    df["early_pct"] = raw_pct * side
    return df


def _build_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cohort_size"] = df.groupby(_COHORT_KEY)["symbol"].transform("size")
    df = df[df["cohort_size"] >= _MIN_COHORT_SIZE].copy()
    df["early_rank"] = df.groupby(_COHORT_KEY)["early_pct"].rank(pct=True)
    return df


def _placebo(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["early_rank"] = out.groupby(_COHORT_KEY)["early_rank"].transform(
        lambda s: rng.permutation(s.values)
    )
    return out


def _report_correlation(df: pd.DataFrame, label: str) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        if len(sub) < 30:
            continue
        rho, p = spearmanr(sub["early_rank"], sub["realized_pnl"])
        print(f"  [{label}] {sig_type}: n={len(sub)}, rho={rho:+.3f} (p={p:.4f})")


def _report_buckets(df: pd.DataFrame, label: str) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        if len(sub) < 30:
            continue
        tercile = pd.qcut(sub["early_rank"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
        g = sub.groupby(tercile, observed=True)["realized_pnl"].agg(
            ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean()
        )
        print(f"  [{label}] {sig_type}:")
        print(g.to_string().replace("\n", "\n    "))


def main() -> None:
    df = _fetch(_EARLY_BARS)
    df = _add_early_pct(df)
    print(f"Toplam HA_Cross sinyali (kapanmış, {_EARLY_BARS} bar sonrası fiyat mevcut): {len(df)}")

    cohorts = _build_cohorts(df)
    n_cohorts = cohorts.groupby(_COHORT_KEY).ngroups
    print(
        f"Kohort (boyut>={_MIN_COHORT_SIZE}) sayısı: {n_cohorts}, içindeki sinyal: {len(cohorts)}\n"
    )

    print(f"=== ANA TEST (gerçek {_EARLY_BARS}-bar erken ayrışma sıralaması) ===")
    _report_correlation(cohorts, "gerçek")
    _report_buckets(cohorts, "gerçek")

    print("\n=== PLACEBO (rank kohort içinde rastgele karıştırıldı) ===")
    _report_correlation(_placebo(cohorts), "placebo")

    print("\n=== SPLIT-PERIOD (dönem ikiye bölündü) ===")
    mid = cohorts["opened_at"].min() + (cohorts["opened_at"].max() - cohorts["opened_at"].min()) / 2
    for half_name, half_df in [
        ("ilk_yari", cohorts[cohorts["opened_at"] < mid]),
        ("ikinci_yari", cohorts[cohorts["opened_at"] >= mid]),
    ]:
        print(f"-- {half_name} ({len(half_df)} sinyal) --")
        _report_correlation(half_df, half_name)

    print("\n=== INTERVAL'A GÖRE KIRILIM ===")
    for interval, int_df in cohorts.groupby("interval"):
        if len(int_df) < 30:
            continue
        print(f"-- {interval} ({len(int_df)} sinyal) --")
        _report_correlation(int_df, interval)


if __name__ == "__main__":
    main()
