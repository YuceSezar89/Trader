"""
DevisSoTrader'ın volatility_breakout ajanındaki `is_consolidating` durumunu
(devissotrader_agents_bt.py::_is_consolidating_series — hocanın kodu, coin
bazlı sıkışma tespiti) TETİKLEYİCİ değil REJİM/FİLTRE olarak RSI_Cross
sinyallerine uygular. regime_score_bt.py ile AYNI desen: threshold_optimizer'ın
3-kapılı disiplini (IS/OOS + split-period + placebo), tek fark regime_score.py
piyasa-geneli (BTC trend + breadth) iken burada COIN BAZLI bir durum test
ediliyor.

Gerekçe: devissotrader_agents_bt.py'de bu ajanın ham AL/SAT sinyali (kırılım
anı) baseline'ı geçemedi. Ama "şu an sıkışma var mı" durumu, sinyalin
KENDİSİNDEN bağımsız bir soru — RSI_Cross gibi zaten çalışan bir aile,
sıkışma döneminde mi yoksa sıkışma dışında mı daha iyi çalışıyor?

Look-ahead: cagg_15m.bucket bar AÇILIŞ zamanı (bkz. mtf_helpers.py). Bir barın
GERÇEKTEN kapanmış sayılması için bucket+15dk <= opened_at şartı aranıyor —
merge_asof'a opened_at yerine (opened_at - 15dk) veriliyor.
"""
import os
import sys

import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.devissotrader_agents_bt import (  # pylint: disable=wrong-import-position
    _is_consolidating_series,
)
from research.pattern_lab.threshold_optimizer import _run_single_var_on_df  # pylint: disable=wrong-import-position

INDICATOR = "RSI_Cross(9,24)"
DAYS = 60
BAR_DURATION = pd.Timedelta(minutes=15)


def _fetch_signals(indicator: str, direction: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT symbol, opened_at, realized_pnl
        FROM signals
        WHERE indicators = %s AND signal_type = %s
          AND status = 'closed' AND realized_pnl IS NOT NULL
        ORDER BY symbol, opened_at
    """
    df = pd.read_sql(q, conn, params=(indicator, direction))
    conn.close()
    return df


def _fetch_regime(symbols: list, interval: str, state_fn, col_name: str, min_bars: int = 50) -> pd.DataFrame:
    """Jenerik rejim durumu çekici — devissotrader_agents_bt.py'deki herhangi
    bir `_..._series(g) -> pd.Series` durum fonksiyonuyla çalışır (interval'e
    göre cagg_{interval} tablosundan). rsi_cross_trend_meanrev_regime_bt.py'de
    trend_follower (4h) ve mean_reversion (15m) durumları için de kullanılıyor."""
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, open, high, low, close, volume
        FROM cagg_{interval}
        WHERE bucket > NOW() - INTERVAL '{DAYS} days' AND symbol = ANY(%s)
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn, params=(symbols,))
    conn.close()

    out = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < min_bars:
            continue
        state = state_fn(g)
        out.append(pd.DataFrame({"symbol": sym, "ts": g["ts"], col_name: state.astype(float)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["symbol", "ts", col_name])


def _merge_regime(sig_df: pd.DataFrame, regime_df: pd.DataFrame, col_name: str, bar_duration: pd.Timedelta) -> pd.DataFrame:
    sig_df = sig_df.copy()
    sig_df["cutoff"] = sig_df["opened_at"] - bar_duration
    merged = pd.merge_asof(
        sig_df.sort_values("cutoff"),
        regime_df.sort_values("ts"),
        left_on="cutoff", right_on="ts", by="symbol", direction="backward",
    )
    return merged.dropna(subset=[col_name])


def run() -> None:
    for direction in ("Long", "Short"):
        sig_df = _fetch_signals(INDICATOR, direction)
        if len(sig_df) < 50:
            print(f"{INDICATOR} — {direction}: yetersiz sinyal ({len(sig_df)}), atlanıyor")
            continue

        regime_df = _fetch_regime(sig_df["symbol"].unique().tolist(), "15m", _is_consolidating_series, "is_consolidating")
        merged = _merge_regime(sig_df, regime_df, "is_consolidating", BAR_DURATION)
        print(f"{INDICATOR} — {direction}: {len(sig_df):,} sinyal, {len(merged):,} rejim durumuyla eşleşti "
              f"(is_consolidating ort={merged['is_consolidating'].mean():.2f})")

        label = f"{INDICATOR} — {direction} — is_consolidating (volatility_breakout rejimi)"
        _run_single_var_on_df(label, merged, "is_consolidating")


if __name__ == "__main__":
    run()
