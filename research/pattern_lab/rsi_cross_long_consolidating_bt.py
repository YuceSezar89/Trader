"""
RSI_Cross(9,24) Long icin is_consolidating rejim filtresi — bugun dogrulanan
1h/4h hizalanma + kohort-ici erken-ayrisma ustune, hocanin DevisSoTrader
kodundan (control_decision.py::_volatility_breakout_decision, bkz.
[[project_turtle_traders]] 11 Tem "2 SAGLAM BULGU") gelen ucuncu bir kosul
ekleniyor: is_consolidating<=0 (yani KONSOLIDE DEGIL) — 18 Tem 2026,
kullanici "regime bulgularini ekleyelim" dedi, once RSI_Cross Long parcasi.

is_consolidating: devissotrader_agents_bt.py::_is_consolidating_series ile
BIREBIR AYNI formul (BREAKOUT_PERIODS=20, MIN_CONSOLIDATION_PERIODS=5,
esik<0.05), 15m cagg'den hesaplanip look-ahead-guvenli sekilde (opened_at -
15dk cutoff, backward asof) sinyale eslestiriliyor — orijinal
rsi_cross_volbreakout_regime_bt.py'deki AYNI yontem.

Kullanım: python -m research.pattern_lab.rsi_cross_long_consolidating_bt
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
    _fetch as _fetch_base,
)

BREAKOUT_PERIODS = 20
MIN_CONSOLIDATION_PERIODS = 5
DAYS = 45
BAR_DURATION = pd.Timedelta(minutes=15)


def _fetch_with_id(early_bars: int) -> pd.DataFrame:
    """rsi_cross_tf_alignment_bt._fetch ile ayni, sadece Long + signal id ekli."""
    df = _fetch_base(early_bars)
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    sig_ids = pd.read_sql(
        "SELECT id, symbol, signal_type, interval, opened_at FROM signals WHERE indicators='RSI_Cross(9,24)'",
        conn,
    )
    conn.close()
    df = df.merge(sig_ids, on=["symbol", "signal_type", "interval", "opened_at"], how="inner")
    return df[df["signal_type"] == "Long"].copy()


def _is_consolidating_series(g: pd.DataFrame) -> pd.Series:
    high, low, close = g["high"], g["low"], g["close"]
    n_high = high.rolling(BREAKOUT_PERIODS).max()
    n_low = low.rolling(BREAKOUT_PERIODS).min()
    price_range = (n_high - n_low) / close
    return price_range.rolling(MIN_CONSOLIDATION_PERIODS).mean() < 0.05


def _fetch_regime(symbols: list) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, high, low, close
        FROM cagg_15m
        WHERE bucket > NOW() - INTERVAL '{DAYS} days' AND symbol = ANY(%s)
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn, params=(symbols,))
    conn.close()

    out = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < 50:
            continue
        state = _is_consolidating_series(g)
        out.append(pd.DataFrame({"symbol": sym, "ts": g["ts"], "is_consolidating": state.astype(float)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["symbol", "ts", "is_consolidating"])


def _merge_regime(df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cutoff"] = df["opened_at"] - BAR_DURATION
    merged = pd.merge_asof(
        df.sort_values("cutoff"),
        regime_df.sort_values("ts"),
        left_on="cutoff", right_on="ts", by="symbol", direction="backward",
    )
    return merged


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
    print("RSI_Cross(9,24) Long verisi hazırlanıyor...")
    df = _fetch_with_id(_EARLY_BARS)
    df = _add_early_pct(df)
    cohorts = _build_cohorts(df)
    print(f"Kohort içi Long sinyal: {len(cohorts)}")

    print("1h/4h Heikin Ashi hizalanması hesaplanıyor...")
    aligned = _add_htf_alignment(cohorts)

    print("is_consolidating (15m, hocanın kodu) hesaplanıyor...")
    regime = _fetch_regime(aligned["symbol"].unique().tolist())
    merged = _merge_regime(aligned, regime)
    print(f"Regime eşleşen sinyal: {merged.dropna(subset=['is_consolidating']).shape[0]}")

    print("\n=== BACKTEST realized_pnl: kademeli filtre ===")
    s1 = merged[merged["aligned_count"] == 2]
    print(f"  (1) +hizalı (1h+4h):              n={len(s1)}, ort_pnl={s1['realized_pnl'].mean():+.4f}, wr={(s1['realized_pnl']>0).mean()*100:.1f}%")

    s2 = s1[s1["early_rank"] >= s1["early_rank"].quantile(0.667)] if len(s1) > 0 else s1
    print(f"  (2) +çok-ayrışan (üst 1/3):        n={len(s2)}, ort_pnl={s2['realized_pnl'].mean():+.4f}, wr={(s2['realized_pnl']>0).mean()*100:.1f}%" if len(s2) > 0 else "  (2) örnek yok")

    s3 = s2[s2["is_consolidating"] == 0.0] if len(s2) > 0 else s2
    print(f"  (3) +NOT consolidating:            n={len(s3)}, ort_pnl={s3['realized_pnl'].mean():+.4f}, wr={(s3['realized_pnl']>0).mean()*100:.1f}%" if len(s3) > 0 else "  (3) örnek yok")

    print("\n=== GERÇEK $ DOĞRULAMASI (paper_trades.pnl_usd) — kademeli filtre ===")
    real = _fetch_real_trades()
    rmerged = merged.merge(real, left_on="id", right_on="signal_id", how="inner")
    print(f"Kohort + hizalanma + GERÇEK işlem kesişimi: {len(rmerged)}")

    r1 = rmerged[rmerged["aligned_count"] == 2]
    print(f"  (1) +hizalı:              n={len(r1)}, ort_$={r1['pnl_usd'].mean():+.3f}, toplam_$={r1['pnl_usd'].sum():+.2f}, wr={(r1['pnl_usd']>0).mean()*100:.1f}%" if len(r1) > 0 else "  (1) örnek yok")

    r2 = r1[r1["early_rank"] >= r1["early_rank"].quantile(0.667)] if len(r1) > 0 else r1
    print(f"  (2) +çok-ayrışan:         n={len(r2)}, ort_$={r2['pnl_usd'].mean():+.3f}, toplam_$={r2['pnl_usd'].sum():+.2f}, wr={(r2['pnl_usd']>0).mean()*100:.1f}%" if len(r2) > 0 else "  (2) örnek yok")

    r3 = r2[r2["is_consolidating"] == 0.0] if len(r2) > 0 else r2
    print(f"  (3) +NOT consolidating:   n={len(r3)}, ort_$={r3['pnl_usd'].mean():+.3f}, toplam_$={r3['pnl_usd'].sum():+.2f}, wr={(r3['pnl_usd']>0).mean()*100:.1f}%" if len(r3) > 0 else "  (3) örnek yok")


if __name__ == "__main__":
    main()
