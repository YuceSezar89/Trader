"""
rsi_cross_live'in MEVCUT canlı filtresi (all_up + candle_kategori='govde',
paper_trade_manager.py:53-55) üzerine, bugün doğrulanan TF-hizalanma (1h+4h
HA) + kohort-ici erken-ayrisma filtresini EKLEYEREK yeniden test — 18 Tem
2026, kullanici "eski filtreye ek olarak yeniden test edelim" dedi.

[[rsi_cross_tf_alignment_bt]]'ten TEK FARK: _fetch_interval sorgusuna
`AND s.all_up AND s.candle_kategori = 'govde'` eklendi — kohort, artik TUM
RSI_Cross(9,24) evreninden degil, SADECE rsi_cross_live'in zaten actigi
sinyal turunden kuruluyor. Ardindan gercek paper_trades.pnl_usd ile de
dogrulaniyor (bkz. [[rsi_cross_live_real_pnl_bt]]).

Kullanım: python -m research.pattern_lab.rsi_cross_live_filter_tf_alignment_bt
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
_INTERVAL_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
_HTF_TABLES = {"1h": "cagg_1h", "4h": "cagg_4h"}


def _fetch_interval(interval: str, early_bars: int) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    early_offset_1m = early_bars * _INTERVAL_MIN[interval] - 1
    q = """
        SELECT s.id, s.symbol, s.signal_type, s.interval, s.opened_at,
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
        WHERE s.indicators = 'RSI_Cross(9,24)' AND s.interval = %s
          AND s.status = 'closed' AND s.realized_pnl IS NOT NULL
          AND s.open_price IS NOT NULL AND s.open_price > 0
          AND s.all_up AND s.candle_kategori = 'govde'
    """
    df = pd.read_sql(q, conn, params=(early_offset_1m, interval))
    conn.close()
    return df


def _fetch(early_bars: int) -> pd.DataFrame:
    pieces = [_fetch_interval(iv, early_bars) for iv in _INTERVAL_MIN]
    return pd.concat(pieces, ignore_index=True)


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


def _compute_ha(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("bucket").reset_index(drop=True)
    o, h, l, c = (df[x].to_numpy() for x in ["open", "high", "low", "close"])
    n = len(df)
    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty(n)
    if n > 0:
        ha_open[0] = (o[0] + c[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    df["ha_bull"] = ha_close > ha_open
    return df[["bucket", "ha_bull"]]


def _fetch_htf_ha(tf: str, symbols: list[str], start, end) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    table = _HTF_TABLES[tf]
    q = f"SELECT symbol, bucket, open, high, low, close FROM {table} WHERE symbol = ANY(%s) AND bucket BETWEEN %s AND %s ORDER BY symbol, bucket"
    df = pd.read_sql(q, conn, params=(symbols, start, end))
    conn.close()
    pieces = [_compute_ha(sub).assign(symbol=symbol) for symbol, sub in df.groupby("symbol")]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["bucket", "ha_bull", "symbol"])


def _add_htf_alignment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    symbols = df["symbol"].unique().tolist()
    start = df["opened_at"].min() - pd.Timedelta(days=10)
    end = df["opened_at"].max()
    for tf in _HTF_TABLES:
        ha = _fetch_htf_ha(tf, symbols, start, end)
        pieces = []
        for symbol, sub in df.groupby("symbol"):
            ha_sym = ha[ha["symbol"] == symbol].sort_values("bucket")
            if ha_sym.empty:
                sub = sub.copy()
                sub[f"ha_bull_{tf}"] = np.nan
                pieces.append(sub)
                continue
            merged = pd.merge_asof(
                sub.sort_values("opened_at"), ha_sym[["bucket", "ha_bull"]],
                left_on="opened_at", right_on="bucket", direction="backward",
            ).rename(columns={"ha_bull": f"ha_bull_{tf}"})
            pieces.append(merged.drop(columns=["bucket"]))
        df = pd.concat(pieces, ignore_index=True)
    aligned_1h = (df["signal_type"] == "Long") == (df["ha_bull_1h"] == True)  # noqa: E712
    aligned_4h = (df["signal_type"] == "Long") == (df["ha_bull_4h"] == True)  # noqa: E712
    df["aligned_count"] = aligned_1h.astype(int) + aligned_4h.astype(int)
    df.loc[df["ha_bull_1h"].isna() | df["ha_bull_4h"].isna(), "aligned_count"] = np.nan
    return df


def _report_standalone(df: pd.DataFrame) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type].dropna(subset=["aligned_count"])
        if len(sub) < 30:
            print(f"  {sig_type}: yetersiz örnek (n={len(sub)})")
            continue
        print(f"  {sig_type}:")
        for cnt, g in sub.groupby("aligned_count"):
            print(
                f"    aligned_count={int(cnt)}: n={len(g)}, ort_pnl={g['realized_pnl'].mean():+.4f}, "
                f"wr={(g['realized_pnl']>0).mean()*100:.1f}%"
            )
        rho, p = spearmanr(sub["aligned_count"], sub["realized_pnl"])
        print(f"    rho(aligned_count, realized_pnl)={rho:+.3f} (p={p:.4f})")


def _report_combined(df: pd.DataFrame) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[(df["signal_type"] == sig_type) & (df["aligned_count"] == 2)]
        if len(sub) < 30:
            print(f"  {sig_type}: aligned_count=2 icin yetersiz ornek (n={len(sub)})")
            continue
        try:
            tercile = pd.qcut(sub["early_rank"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
        except ValueError:
            tercile = pd.qcut(sub["early_rank"], 2, labels=["alt", "üst"], duplicates="drop")
        g = sub.groupby(tercile, observed=True)["realized_pnl"].agg(
            ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean()
        )
        print(f"  {sig_type} (sadece aligned_count=2, n={len(sub)}):")
        print(g.to_string().replace("\n", "\n    "))


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
    print("all_up+govde ile önceden filtrelenmiş RSI_Cross(9,24) verisi çekiliyor...")
    df = _fetch(_EARLY_BARS)
    df = _add_early_pct(df)
    print(f"Toplam sinyal (all_up+govde, kapanmış): {len(df)}")

    cohorts = _build_cohorts(df)
    print(f"Kohort içi sinyal: {len(cohorts)}")

    print("1h/4h Heikin Ashi hizalanması hesaplanıyor...")
    aligned = _add_htf_alignment(cohorts)

    print("\n=== TEK BAŞINA: aligned_count arttıkça performans (backtest realized_pnl) ===")
    _report_standalone(aligned)

    print("\n=== BİRLEŞİK: TAM hizalı + erken-ayrışma tercile (backtest realized_pnl) ===")
    _report_combined(aligned)

    print("\n=== GERÇEK $ DOĞRULAMASI (paper_trades.pnl_usd ile) ===")
    real = _fetch_real_trades()
    merged = aligned.merge(real, left_on="id", right_on="signal_id", how="inner")
    print(f"Kohort + hizalanma + GERÇEK işlem kesişimi: {len(merged)}")
    for sig_type in ["Long", "Short"]:
        sub_all = merged[merged["signal_type"] == sig_type]
        if len(sub_all) < 10:
            print(f"  {sig_type}: yetersiz örnek (n={len(sub_all)})")
            continue
        is_filtered = (sub_all["aligned_count"] == 2) & (
            sub_all["early_rank"] >= sub_all["early_rank"].quantile(0.667)
        )
        filtered = sub_all[is_filtered]
        rest = sub_all[~is_filtered]
        print(f"  {sig_type} — TÜMÜ (n={len(sub_all)}): ort_$={sub_all['pnl_usd'].mean():+.3f}, "
              f"toplam_$={sub_all['pnl_usd'].sum():+.2f}, wr={(sub_all['pnl_usd']>0).mean()*100:.1f}%")
        if len(filtered) > 0:
            print(f"  {sig_type} — HİZALI+ÇOK-AYRIŞAN (n={len(filtered)}): "
                  f"ort_$={filtered['pnl_usd'].mean():+.3f}, toplam_$={filtered['pnl_usd'].sum():+.2f}, "
                  f"wr={(filtered['pnl_usd']>0).mean()*100:.1f}%")
        else:
            print(f"  {sig_type} — HİZALI+ÇOK-AYRIŞAN: örnek yok")
        if len(rest) > 0:
            print(f"  {sig_type} — GERİ KALANI (n={len(rest)}): "
                  f"ort_$={rest['pnl_usd'].mean():+.3f}, toplam_$={rest['pnl_usd'].sum():+.2f}, "
                  f"wr={(rest['pnl_usd']>0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
