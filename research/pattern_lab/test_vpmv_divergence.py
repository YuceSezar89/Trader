"""
Hacim-Momentum Divergence Testi (küçük örneklem, hoca doktrini madde 4:
"manipülasyon mu gerçek mi — Δmomentum+Δhacim eşliği?").

vol_score ve momentum_score'u AYRI hesaplayıp (birleşik VPMV skoru yerine)
sinyal yönüne göre hizalar:
  - Confluence (gerçek hareket): hacim YÜKSEK + momentum yönle uyumlu YÜKSEK
  - Divergence (olası manipülasyon): hacim YÜKSEK ama momentum yönle uyumsuz/nötr

Kullanım: python -m research.pattern_lab.test_vpmv_divergence [n]
"""
import sys
import warnings

import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config
from indicators.core import calculate_atr, calculate_rsi
from utils.preprocessing import normalize_momentum_0_100, normalize_volume_0_100

_CAGG = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_BARS_NEEDED = 220


def _fetch_bars(cur, symbol: str, interval: str, opened_at) -> pd.DataFrame | None:
    if interval == "1m":
        cur.execute(
            "SELECT timestamp AS open_time, open, high, low, close, volume "
            "FROM price_data WHERE symbol=%s AND interval='1m' AND timestamp <= %s "
            "ORDER BY timestamp DESC LIMIT %s",
            (symbol, opened_at, _BARS_NEEDED),
        )
    else:
        cagg = _CAGG.get(interval)
        if not cagg:
            return None
        cur.execute(
            f"SELECT bucket AS open_time, open, high, low, close, volume "
            f"FROM {cagg} WHERE symbol=%s AND bucket <= %s ORDER BY bucket DESC LIMIT %s",
            (symbol, opened_at, _BARS_NEEDED),
        )
    rows = cur.fetchall()
    if not rows or len(rows) < 60:
        return None
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    return df.iloc[::-1].reset_index(drop=True)


def _classify(df: pd.DataFrame, signal_type: str) -> tuple[str, float, float] | None:
    try:
        rsi_centered = calculate_rsi(df, period=14) - 50
        vol_score = float(normalize_volume_0_100(df["volume"]).iloc[-1])
        momentum_score = float(normalize_momentum_0_100(rsi_centered).iloc[-1])
    except Exception:  # pylint: disable=broad-exception-caught
        return None

    # Yönle hizala: Long için yüksek=bullish momentum, Short için yüksek=bearish momentum
    momentum_aligned = momentum_score if signal_type == "Long" else (100.0 - momentum_score)

    if vol_score > 60 and momentum_aligned > 60:
        cls = "Confluence (gerçek hareket)"
    elif vol_score > 70 and momentum_aligned < 50:
        cls = "Divergence (olası manipülasyon)"
    else:
        cls = "Nötr"
    return cls, vol_score, momentum_aligned


def main(n: int = 5000, close_reason: str | None = None) -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    reason_filter = "AND close_reason = %s" if close_reason else ""
    params = (close_reason, n) if close_reason else (n,)
    cur.execute(
        f"""
        SELECT id, symbol, interval, opened_at, signal_type, realized_pnl
        FROM signals
        WHERE status='closed' AND realized_pnl IS NOT NULL
          AND interval IN ('1m','5m','15m','1h','4h')
          {reason_filter}
        ORDER BY opened_at DESC
        LIMIT %s
        """,
        params,
    )
    rows = cur.fetchall()
    print(f"[test] {len(rows)} sinyal örneklemi çekildi (close_reason={close_reason or 'hepsi'})")

    results = []
    skipped = 0
    for i, (sid, symbol, interval, opened_at, signal_type, pnl) in enumerate(rows, 1):
        df = _fetch_bars(cur, symbol, interval, opened_at)
        if df is None:
            skipped += 1
            continue
        out = _classify(df, signal_type)
        if out is None:
            skipped += 1
            continue
        cls, vol_score, momentum_aligned = out
        results.append({
            "signal_type": signal_type, "cls": cls,
            "vol_score": vol_score, "momentum_aligned": momentum_aligned,
            "pnl": float(pnl),
        })
        if i % 500 == 0:
            print(f"  ... {i}/{len(rows)} işlendi")

    conn.close()
    print(f"[test] tamamlandı — {len(results)} işlendi, {skipped} atlandı\n")

    df_r = pd.DataFrame(results)
    summary = (
        df_r.groupby(["signal_type", "cls"])["pnl"]
        .agg(n="count", ort_pnl="mean", kazanma_pct=lambda s: (s > 0).mean() * 100)
        .round(3)
        .reset_index()
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    _n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    _reason = sys.argv[2] if len(sys.argv) > 2 else None
    main(_n, _reason)
