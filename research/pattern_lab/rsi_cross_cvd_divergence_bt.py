"""
CVD divergence + RSI_Cross ([[project_cvd_divergence]], bekleyen karar: filtre mi
skor mu). O nottaki formül GEOMETRİK PROXI'ydi (high/low/close'tan tahmin — Tlosx'un
"Proxy (VDS)" dediği, kendi kılavuzunda "absorpsiyonlu barlarda sapar" diye uyardığı
yöntem). Bu script GERÇEK taker buy/sell hacmini kullanıyor (`price_data.buy_volume/
sell_volume` — [[project_directional_volume]], 4 Tem'den beri temiz, Binance'ın
taker_buy_base_asset_volume'undan türetilmiş, tahmin değil).

CVD = cumsum(buy_volume - sell_volume), sembol bazlı, 15m'ye toplanmış 1m veriden.

Divergence mantığı (project_cvd_divergence.md'den birebir, N=20 bar):
  Long (bullish): fiyat son N barın dibine değdi/altına sarktı (%0.2 tolerans) AMA
                   CVD o dipteki seviyesinden şu an daha YÜKSEKte → satıcı tükenmesi.
  Short (bearish): fiyat son N barın zirvesine değdi/üstüne sarktı AMA CVD o
                   zirvedeki seviyesinden şu an daha DÜŞÜKte → alıcı tükenmesi.

Look-ahead: price_data.timestamp bar AÇILIŞ zamanı (cagg'lerle aynı sözleşme).
merge_asof'a opened_at yerine (opened_at - 15dk) veriliyor — bar kapanmadan
divergence durumu kullanılmıyor. Disiplin: threshold_optimizer'ın 3 kapısı
(IS/OOS + split-period + placebo) — regime_matrix_bt.py ile aynı desen.
"""
import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_volbreakout_regime_bt import (  # pylint: disable=wrong-import-position
    INDICATOR, _fetch_signals, _merge_regime,
)
from research.pattern_lab.threshold_optimizer import _run_single_var_on_df  # pylint: disable=wrong-import-position

DAYS = 60
N = 20
TOL = 0.002
BAR_DURATION = pd.Timedelta(minutes=15)


def _fetch_15m_with_volume(symbols: list) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, time_bucket('15 minutes', timestamp) AS ts,
               max(high) AS high, min(low) AS low,
               sum(buy_volume) AS buy_volume, sum(sell_volume) AS sell_volume
        FROM price_data
        WHERE interval = '1m' AND timestamp > NOW() - INTERVAL '{DAYS} days'
          AND symbol = ANY(%s) AND buy_volume IS NOT NULL
        GROUP BY symbol, time_bucket('15 minutes', timestamp)
        ORDER BY symbol, ts
    """
    df = pd.read_sql(q, conn, params=(symbols,))
    conn.close()
    return df


def _divergence_state(g: pd.DataFrame) -> pd.Series:
    """+1 = bullish divergence (Long teyidi), -1 = bearish divergence (Short teyidi), 0 = yok."""
    high, low = g["high"].to_numpy(), g["low"].to_numpy()
    cvd = (g["buy_volume"] - g["sell_volume"]).cumsum().to_numpy()
    n = len(g)

    roll_low_pos = g["low"].rolling(N).apply(lambda x: np.argmin(x), raw=True)
    roll_high_pos = g["high"].rolling(N).apply(lambda x: np.argmax(x), raw=True)

    state = np.zeros(n, dtype=int)
    for i in range(N - 1, n):
        low_pos = roll_low_pos.iloc[i]
        high_pos = roll_high_pos.iloc[i]
        if not np.isnan(low_pos):
            low_idx = i - N + 1 + int(low_pos)
            if low[i] <= low[low_idx] * (1 + TOL) and cvd[i] > cvd[low_idx]:
                state[i] = 1
        if not np.isnan(high_pos):
            high_idx = i - N + 1 + int(high_pos)
            if high[i] >= high[high_idx] * (1 - TOL) and cvd[i] < cvd[high_idx]:
                state[i] = -1
    return pd.Series(state, index=g.index)


def _fetch_cvd_regime(symbols: list) -> pd.DataFrame:
    df = _fetch_15m_with_volume(symbols)
    out = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < 50:
            continue
        state = _divergence_state(g)
        out.append(pd.DataFrame({"symbol": sym, "ts": g["ts"], "cvd_divergence": state.astype(float)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["symbol", "ts", "cvd_divergence"])


def run() -> None:
    for direction in ("Long", "Short"):
        sig_df = _fetch_signals(INDICATOR, direction)
        if len(sig_df) < 50:
            print(f"{INDICATOR} — {direction}: yetersiz sinyal ({len(sig_df)}), atlanıyor")
            continue

        regime_df = _fetch_cvd_regime(sig_df["symbol"].unique().tolist())
        merged = _merge_regime(sig_df, regime_df, "cvd_divergence", BAR_DURATION)

        # Yöne özel ikili bayrak: Long için sadece bullish(+1), Short için sadece
        # bearish(-1) "divergence var" sayılır — -1/0/+1 skalasında percentile arama
        # küçük azınlık gruplarını (%6-8) izole edemiyordu, doğrudan ikili test daha doğru.
        wanted = 1 if direction == "Long" else -1
        merged["divergence_match"] = (merged["cvd_divergence"] == wanted).astype(float)

        print(f"{INDICATOR} — {direction}: {len(sig_df):,} sinyal, {len(merged):,} CVD durumuyla eşleşti "
              f"(bullish={len(merged[merged['cvd_divergence']==1])}, "
              f"bearish={len(merged[merged['cvd_divergence']==-1])}, "
              f"nötr={len(merged[merged['cvd_divergence']==0])}, "
              f"yöne-uygun-divergence={int(merged['divergence_match'].sum())})")

        label = f"{INDICATOR} — {direction} — divergence_match (yöne özel, gerçek taker buy/sell)"
        _run_single_var_on_df(label, merged, "divergence_match")


if __name__ == "__main__":
    run()
