"""
TAM YIĞIN testi — [[rsi_cross_live_filter_tf_alignment_bt]] (all_up+govde +
1h/4h hizalanma + erken-ayrışma) üzerine bugün doğrulanan st_confirmed
(aynı-TF Supertrend onayı, bkz. [[st_confirmed_quick_bt]]) de eklendi — 18 Tem
2026, kullanıcı "ekle ve all up gövde de dahil olsun" dedi.

4 filtre birlikte: all_up + candle_kategori='govde' (mevcut canlı filtre) +
aligned_count==2 (1h+4h HA hizalı) + early_rank üst tercile (kohort içi en
çok ayrışan) + st_confirmed (aynı TF Supertrend onaylı).

Kullanım: python -m research.pattern_lab.rsi_cross_full_stack_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2

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
               s.open_price, s.realized_pnl, s.st_confirmed, pd.close AS price_early
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
          AND s.st_confirmed IS NOT NULL
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
    print(f"Toplam sinyal (all_up+govde, kapanmış, st_confirmed dolu): {len(df)}")

    cohorts = _build_cohorts(df)
    print(f"Kohort içi sinyal: {len(cohorts)}")

    print("1h/4h Heikin Ashi hizalanması hesaplanıyor...")
    aligned = _add_htf_alignment(cohorts)

    print("\n=== BACKTEST realized_pnl: kademeli filtre ekleme ===")
    for sig_type in ["Long", "Short"]:
        sub = aligned[aligned["signal_type"] == sig_type]
        print(f"\n  {sig_type}:")
        print(f"    (1) all_up+govde SADECE:            n={len(sub)}, ort_pnl={sub['realized_pnl'].mean():+.4f}, wr={(sub['realized_pnl']>0).mean()*100:.1f}%")

        s2 = sub[sub["aligned_count"] == 2]
        print(f"    (2) +hizalı (1h+4h):                n={len(s2)}, ort_pnl={s2['realized_pnl'].mean():+.4f}, wr={(s2['realized_pnl']>0).mean()*100:.1f}%" if len(s2) > 0 else "    (2) örnek yok")

        s3 = s2[s2["early_rank"] >= s2["early_rank"].quantile(0.667)] if len(s2) > 0 else s2
        print(f"    (3) +çok-ayrışan (üst 1/3):          n={len(s3)}, ort_pnl={s3['realized_pnl'].mean():+.4f}, wr={(s3['realized_pnl']>0).mean()*100:.1f}%" if len(s3) > 0 else "    (3) örnek yok")

        s4 = s3[s3["st_confirmed"] == True] if len(s3) > 0 else s3  # noqa: E712
        print(f"    (4) +st_confirmed:                   n={len(s4)}, ort_pnl={s4['realized_pnl'].mean():+.4f}, wr={(s4['realized_pnl']>0).mean()*100:.1f}%" if len(s4) > 0 else "    (4) örnek yok")

    print("\n=== GERÇEK $ DOĞRULAMASI (paper_trades.pnl_usd ile) — kademeli filtre ===")
    real = _fetch_real_trades()
    merged = aligned.merge(real, left_on="id", right_on="signal_id", how="inner")
    print(f"Kohort + hizalanma + GERÇEK işlem kesişimi: {len(merged)}")
    for sig_type in ["Long", "Short"]:
        sub_all = merged[merged["signal_type"] == sig_type]
        if len(sub_all) < 5:
            print(f"  {sig_type}: yetersiz örnek (n={len(sub_all)})")
            continue
        print(f"\n  {sig_type}:")
        print(f"    (1) TÜMÜ:                 n={len(sub_all)}, ort_$={sub_all['pnl_usd'].mean():+.3f}, toplam_$={sub_all['pnl_usd'].sum():+.2f}, wr={(sub_all['pnl_usd']>0).mean()*100:.1f}%")

        s2 = sub_all[sub_all["aligned_count"] == 2]
        if len(s2) > 0:
            print(f"    (2) +hizalı:              n={len(s2)}, ort_$={s2['pnl_usd'].mean():+.3f}, toplam_$={s2['pnl_usd'].sum():+.2f}, wr={(s2['pnl_usd']>0).mean()*100:.1f}%")

        s3 = s2[s2["early_rank"] >= s2["early_rank"].quantile(0.667)] if len(s2) > 0 else s2
        if len(s3) > 0:
            print(f"    (3) +çok-ayrışan:         n={len(s3)}, ort_$={s3['pnl_usd'].mean():+.3f}, toplam_$={s3['pnl_usd'].sum():+.2f}, wr={(s3['pnl_usd']>0).mean()*100:.1f}%")

        s4 = s3[s3["st_confirmed"] == True] if len(s3) > 0 else s3  # noqa: E712
        if len(s4) > 0:
            print(f"    (4) +st_confirmed:        n={len(s4)}, ort_$={s4['pnl_usd'].mean():+.3f}, toplam_$={s4['pnl_usd'].sum():+.2f}, wr={(s4['pnl_usd']>0).mean()*100:.1f}%")
        else:
            print("    (4) +st_confirmed: örnek yok")


if __name__ == "__main__":
    main()
