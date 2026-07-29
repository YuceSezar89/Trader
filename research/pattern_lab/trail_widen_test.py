"""
Trailing mesafesi genişletme testi (20 Tem 2026, kullanıcı iddiası doğrultusunda:
"trail işlemden erken çıkardı" — trailing_exit_timing_bt.py'nin bulgusu:
işlemlerin %66-68'inde kapanıştan sonra fiyat kilitlenenden daha lehte gitmiş).

Gerçek `signals/trailing.py::update_trailing` state machine'ini gerçek kline
verisiyle (bar-bar high/low) yeniden simüle eder — SL/TP AYNI kalır, sadece
trailing mesafesi (dist = ATR × sl_mult × çarpan) genişletilir. Orijinal
mesafe çarpanı 1.0 (kontrol), 1.5/2.0/2.5 test edilir.

Sadece ORİJİNALDE trailing_stop ile kapanmış işlemler kullanılır (TP'ye
ulaşıp trailing aktive olmuş olanlar) — SL'e hiç ulaşmadan trailing'e
girenlerin davranışı test ediliyor.

Kullanım: python -m research.pattern_lab.trail_widen_test
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config

_STRATEGIES = ("rsi_cross_live", "tf_alignment_live")
_WIDEN_FACTORS = (1.0, 1.5, 2.0, 2.5)
_HORIZON_DAYS = 7
_INTERVAL_TABLE = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}


def _fetch_trades(cur, strategy: str) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, signal_type, interval, entry_price, atr,
               stop_loss_price, take_profit_price, opened_at
        FROM paper_trades
        WHERE strategy = %s AND status = 'closed' AND close_reason = 'trailing_stop'
          AND atr IS NOT NULL AND atr > 0 AND take_profit_price IS NOT NULL
          AND stop_loss_price IS NOT NULL
        ORDER BY opened_at ASC
        """,
        (strategy,),
    )
    cols = ["id", "symbol", "signal_type", "interval", "entry_price", "atr",
            "stop_loss_price", "take_profit_price", "opened_at"]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def _fetch_bars(cur, symbol: str, interval: str, start, end) -> pd.DataFrame:
    table = _INTERVAL_TABLE.get(interval, "cagg_5m")
    cur.execute(
        f"SELECT bucket, high, low FROM {table} WHERE symbol=%s AND bucket >= %s AND bucket <= %s ORDER BY bucket ASC",
        (symbol, start, end),
    )
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["bucket", "high", "low"]) if rows else pd.DataFrame()


def _simulate(row, bars: pd.DataFrame, widen: float) -> tuple:
    """Gerçek update_trailing mantığının bar-bar simülasyonu.
    Döner: (exit_pct, reason) — exit_pct girişe göre yön-ayarlı % PnL."""
    entry = row["entry_price"]
    atr = row["atr"]
    sl = row["stop_loss_price"]
    tp = row["take_profit_price"]
    side = row["signal_type"]
    dist = atr * 1.5 * widen  # orijinal formül: current_atr * RISK_SL_MULTIPLIER(1.5), çarpanla genişletildi

    trail = None
    for _, b in bars.iterrows():
        high, low = b["high"], b["low"]
        if side == "Long":
            check_price = low  # SL/trailing önce en kötü fiyatla test edilir (konservatif)
            if trail is None:
                if check_price <= sl:
                    return (sl - entry) / entry * 100.0, "stop_loss"
                if high >= tp:
                    trail = high - dist
                    # Aynı barda trail'e de değinmiş olabilir (yüksek volatilite barı)
                    if low <= trail:
                        return (trail - entry) / entry * 100.0, "trailing_stop"
            else:
                new_trail = high - dist
                if new_trail > trail:
                    trail = new_trail
                if low <= trail:
                    return (trail - entry) / entry * 100.0, "trailing_stop"
        else:  # Short
            check_price = high
            if trail is None:
                if check_price >= sl:
                    return (entry - sl) / entry * 100.0, "stop_loss"
                if low <= tp:
                    trail = low + dist
                    if high >= trail:
                        return (entry - trail) / entry * 100.0, "trailing_stop"
            else:
                new_trail = low + dist
                if new_trail < trail:
                    trail = new_trail
                if high >= trail:
                    return (entry - trail) / entry * 100.0, "trailing_stop"
    return None, "unresolved"


def _stats(rets: list) -> dict:
    rets = np.array([r for r in rets if r is not None])
    if len(rets) == 0:
        return {"n": 0}
    g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean()), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()

    for strategy in _STRATEGIES:
        print(f"\n{'='*70}\n{strategy}\n{'='*70}")
        trades = _fetch_trades(cur, strategy)
        print(f"[fetch] {len(trades)} orijinalde trailing_stop ile kapanmış işlem")

        results = {w: [] for w in _WIDEN_FACTORS}
        unresolved_counts = {w: 0 for w in _WIDEN_FACTORS}

        for i, row in trades.iterrows():
            start = row["opened_at"]
            end = start + pd.Timedelta(days=_HORIZON_DAYS)
            bars = _fetch_bars(cur, row["symbol"], row["interval"], start, end)
            if bars.empty:
                continue
            for w in _WIDEN_FACTORS:
                pct, reason = _simulate(row, bars, w)
                if reason == "unresolved":
                    unresolved_counts[w] += 1
                else:
                    results[w].append(pct)
            if (i + 1) % 100 == 0:
                print(f"  ... {i+1}/{len(trades)}")

        print(f"\n{'çarpan':>8} | {'n':>6} {'unresolved':>10} | {'WR%':>6} {'ort%':>8} {'PF':>6}")
        for w in _WIDEN_FACTORS:
            s = _stats(results[w])
            tag = " (orijinal)" if w == 1.0 else ""
            print(f"{w:>8.1f} | {s.get('n',0):>6} {unresolved_counts[w]:>10} | "
                  f"{s.get('wr',0):>6.1f} {s.get('ort_%',0):>8.3f} {s.get('pf',0):>6.3f}{tag}")

    conn.close()


if __name__ == "__main__":
    main()
