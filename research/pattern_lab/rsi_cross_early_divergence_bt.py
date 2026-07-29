"""
Kohort-içi erken-ayrışma testi — aynı barda RSI_Cross(9,24) alan semboller
arasından, sinyalden sonraki ilk EARLY_BARS barda fiyatı en çok ayrışanı
seçmek gerçekten realized_pnl'i tahmin ediyor mu? [[project_signal_radar_vision]]
kohort fikrinin "hoca" varyantı: statik skor (devisso/cvd/oi, bkz.
cohort_rank_bt.py) yerine sinyal SONRASI erken fiyat hareketi kullanılıyor.

Kohort tanımı: aynı (signal_type, interval, opened_at) — cohort_rank_bt.py
ile aynı yöntem, sadece indicators sabit RSI_Cross(9,24) olduğu için o alan
kohort anahtarından çıkarıldı.

early_pct: sinyal barından price_data'da EARLY_BARS sonraki kapanışa göre
% değişim (Long için düz, Short için ters çevrilmiş — pozitif = sinyal
yönünde ilerleme). Kohort içi percentile rank'i (0-1) alınıp realized_pnl
ile Spearman korelasyonuna bakılıyor.

Bu projenin battle-tested 3 kapısı uygulanıyor (bkz. [[project_pattern_lab]]):
placebo (rank kohort içinde rastgele karıştırılır), split-period (dönem
ikiye bölünür), interval'a göre kırılım.

Kullanım: python -m research.pattern_lab.rsi_cross_early_divergence_bt
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
_EARLY_BARS = 5  # sinyalden kaç bar sonrasına bakılacak - ayarlanabilir


def _fetch(early_bars: int) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    q = """
        SELECT s.symbol, s.signal_type, s.interval, s.opened_at,
               s.open_price, s.realized_pnl, pd.close AS price_early
        FROM signals s
        JOIN LATERAL (
            SELECT p.close
            FROM price_data p
            WHERE p.symbol = s.symbol AND p.interval = s.interval
              AND p.timestamp > s.opened_at
            ORDER BY p.timestamp ASC
            LIMIT 1 OFFSET %s
        ) pd ON true
        WHERE s.indicators = 'RSI_Cross(9,24)'
          AND s.status = 'closed'
          AND s.realized_pnl IS NOT NULL
          AND s.open_price IS NOT NULL AND s.open_price > 0
    """
    df = pd.read_sql(q, conn, params=(early_bars - 1,))
    conn.close()
    return df


def _fetch_btc(intervals: list[str], start, end) -> pd.DataFrame:
    """BTC'nin fiyat serisini tek seferde çeker (sinyal başına ayrı sorgu yerine) —
    sonra pandas tarafında pozisyonel eşleştirme (merge_asof + N bar ötesi) yapılır."""
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    q = """
        SELECT interval, timestamp, close
        FROM price_data
        WHERE symbol = %s AND interval = ANY(%s)
          AND timestamp BETWEEN %s AND %s
        ORDER BY interval, timestamp ASC
    """
    df = pd.read_sql(q, conn, params=(Config.MARKET_REFERENCE_SYMBOL, intervals, start, end))
    conn.close()
    return df


def _add_btc_divergence(df: pd.DataFrame, early_bars: int) -> pd.DataFrame:
    df = df.copy()
    btc = _fetch_btc(
        df["interval"].unique().tolist(),
        df["opened_at"].min(),
        df["opened_at"].max(),
    )

    pieces = []
    for interval, sub in df.groupby("interval"):
        btc_i = btc[btc["interval"] == interval].sort_values("timestamp").reset_index(drop=True)
        if btc_i.empty:
            sub = sub.copy()
            sub["btc_entry"] = np.nan
            sub["btc_early"] = np.nan
            pieces.append(sub)
            continue
        btc_i["pos"] = btc_i.index
        merged = pd.merge_asof(
            sub.sort_values("opened_at"),
            btc_i[["timestamp", "close", "pos"]],
            left_on="opened_at",
            right_on="timestamp",
            direction="forward",
        ).rename(columns={"close": "btc_entry", "pos": "btc_pos"})
        early_pos = merged["btc_pos"] + early_bars
        valid = early_pos < len(btc_i)
        merged["btc_early"] = np.nan
        merged.loc[valid, "btc_early"] = btc_i.loc[early_pos[valid], "close"].values
        pieces.append(merged.drop(columns=["timestamp", "btc_pos"]))

    df = pd.concat(pieces, ignore_index=True)
    side = np.where(df["signal_type"] == "Long", 1.0, -1.0)
    btc_pct = (df["btc_early"] - df["btc_entry"]) / df["btc_entry"] * 100.0 * side
    df["btc_pct"] = btc_pct
    df["divergence"] = df["early_pct"] - df["btc_pct"]
    return df


def _add_early_pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw_pct = (df["price_early"] - df["open_price"]) / df["open_price"] * 100.0
    side = np.where(df["signal_type"] == "Long", 1.0, -1.0)
    df["early_pct"] = raw_pct * side  # pozitif = sinyal yönünde ilerleme
    return df


def _build_cohorts(df: pd.DataFrame, value_col: str, rank_col: str) -> pd.DataFrame:
    df = df.copy()
    df["cohort_size"] = df.groupby(_COHORT_KEY)["symbol"].transform("size")
    df = df[df["cohort_size"] >= _MIN_COHORT_SIZE].copy()
    df[rank_col] = df.groupby(_COHORT_KEY)[value_col].rank(pct=True)
    return df


def _placebo(df: pd.DataFrame, rank_col: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out[rank_col] = out.groupby(_COHORT_KEY)[rank_col].transform(
        lambda s: rng.permutation(s.values)
    )
    return out


def _report_correlation(df: pd.DataFrame, rank_col: str, label: str) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type].dropna(subset=[rank_col])
        if len(sub) < 30:
            continue
        rho, p = spearmanr(sub[rank_col], sub["realized_pnl"])
        print(f"  [{label}] {sig_type}: n={len(sub)}, rho={rho:+.3f} (p={p:.4f})")


def _report_buckets(df: pd.DataFrame, rank_col: str, label: str) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type].dropna(subset=[rank_col])
        if len(sub) < 30:
            continue
        tercile = pd.qcut(sub[rank_col], 3, labels=["alt", "orta", "üst"], duplicates="drop")
        g = sub.groupby(tercile, observed=True)["realized_pnl"].agg(
            ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean()
        )
        print(f"  [{label}] {sig_type}:")
        print(g.to_string().replace("\n", "\n    "))


def main() -> None:
    df = _fetch(_EARLY_BARS)
    df = _add_early_pct(df)
    print(f"Toplam sinyal (RSI_Cross(9,24), kapanmış, {_EARLY_BARS} bar sonrası fiyat mevcut): {len(df)}")

    print("BTC referans serisi çekiliyor ve eşleştiriliyor...")
    df = _add_btc_divergence(df, _EARLY_BARS)

    cohorts = _build_cohorts(df, "early_pct", "early_rank")
    cohorts = _build_cohorts(cohorts, "divergence", "div_rank")
    n_cohorts = cohorts.groupby(_COHORT_KEY).ngroups
    print(
        f"Kohort (boyut>={_MIN_COHORT_SIZE}) sayısı: {n_cohorts}, içindeki sinyal: {len(cohorts)}\n"
    )

    print(f"=== HAM ERKEN AYRIŞMA (BTC düzeltmesi YOK, {_EARLY_BARS} bar) ===")
    _report_correlation(cohorts, "early_rank", "gerçek")
    _report_buckets(cohorts, "early_rank", "gerçek")

    print(f"\n=== BTC-DÜZELTİLMİŞ AYRIŞMA (sembol % - BTC %, {_EARLY_BARS} bar) ===")
    _report_correlation(cohorts, "div_rank", "gerçek")
    _report_buckets(cohorts, "div_rank", "gerçek")

    print("\n=== PLACEBO (div_rank kohort içinde rastgele karıştırıldı) ===")
    _report_correlation(_placebo(cohorts, "div_rank"), "div_rank", "placebo")

    print("\n=== SPLIT-PERIOD (div_rank, dönem ikiye bölündü) ===")
    mid = cohorts["opened_at"].min() + (cohorts["opened_at"].max() - cohorts["opened_at"].min()) / 2
    for half_name, half_df in [
        ("ilk_yari", cohorts[cohorts["opened_at"] < mid]),
        ("ikinci_yari", cohorts[cohorts["opened_at"] >= mid]),
    ]:
        print(f"-- {half_name} ({len(half_df)} sinyal) --")
        _report_correlation(half_df, "div_rank", half_name)


if __name__ == "__main__":
    main()
