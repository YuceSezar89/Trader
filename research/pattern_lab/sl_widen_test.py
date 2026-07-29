"""
SL genişletme testi (20 Tem 2026, kullanıcı isteği: "paper trade rezillik" —
tf_alignment_live/rsi_cross_live WR ~%25, çıkışların HEPSİ ya stop_loss (WR
%0) ya trailing_stop (WR ~%100) — timeout/reversal hiç yok).

Soru: SL = ATR×1.5 çok mu dar? TP'yi (canlıda kaydedilmiş gerçek
take_profit_price) SABİT tutup SADECE SL mesafesini genişletirsek
(2.0x/2.5x/3.0x) WR/PF nasıl değişir — gerçek geçmiş fiyat barlarıyla
(cagg_5m/cagg_15m high/low) "hangi bariyer önce vuruldu" simülasyonu.

Basitleştirme (şeffaf): orijinal trailing-stop mekaniği (TP'ye değince
kapatmak yerine trail aktive etmek) burada modellenmiyor — TP dokunuşu
doğrudan "kazanç" sayılıyor (gerçek veride trailing_stop çıkışlarının
%99-100'ü zaten kazançla kapanmıştı, bu yaklaşım kabul edilebilir bir
yaklaşıklık). Yeni SL, orijinal SL'den daha dar olamaz (sadece genişletme
test ediliyor) — orijinal SL basit bir ATR×mult çarpanı, mesafe olarak
kullanılıyor.

Kullanım: python -m research.pattern_lab.sl_widen_test
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config

_STRATEGIES = ("tf_alignment_live", "rsi_cross_live")
_SL_MULTS = (1.5, 2.0, 2.5, 3.0)  # 1.5 = orijinal (kontrol)
_HORIZON_DAYS = 7
_INTERVAL_TABLE = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}


def _fetch_trades(cur, strategy: str) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, signal_type, interval, entry_price, atr,
               take_profit_price, opened_at, pnl_pct, close_reason
        FROM paper_trades
        WHERE strategy = %s AND status = 'closed'
          AND atr IS NOT NULL AND atr > 0
          AND take_profit_price IS NOT NULL
        ORDER BY opened_at ASC
        """,
        (strategy,),
    )
    rows = cur.fetchall()
    cols = [
        "id", "symbol", "signal_type", "interval", "entry_price", "atr",
        "take_profit_price", "opened_at", "pnl_pct", "close_reason",
    ]
    return pd.DataFrame(rows, columns=cols)


def _fetch_bars(cur, symbol: str, interval: str, start, end) -> pd.DataFrame:
    table = _INTERVAL_TABLE.get(interval, "cagg_5m")
    cur.execute(
        f"""
        SELECT bucket, high, low FROM {table}
        WHERE symbol = %s AND bucket >= %s AND bucket <= %s
        ORDER BY bucket ASC
        """,
        (symbol, start, end),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["bucket", "high", "low"])


def _simulate_trade(row, bars: pd.DataFrame, sl_mult: float) -> tuple[str, float]:
    """(outcome, pnl_pct) — hangi bariyer önce vuruldu + o bariyerin
    girişe göre yüzde mesafesi (yön-ayarlı, her zaman win için +, loss için -)."""
    entry = row["entry_price"]
    atr = row["atr"]
    tp = row["take_profit_price"]
    side = row["signal_type"]

    sl_dist = atr * sl_mult
    tp_pct = abs(tp - entry) / entry * 100.0
    sl_pct = sl_dist / entry * 100.0

    if side == "Long":
        sl_price = entry - sl_dist
        for _, b in bars.iterrows():
            hit_sl = b["low"] <= sl_price
            hit_tp = b["high"] >= tp
            if hit_sl:
                return "loss", -sl_pct
            if hit_tp:
                return "win", tp_pct
    else:
        sl_price = entry + sl_dist
        for _, b in bars.iterrows():
            hit_sl = b["high"] >= sl_price
            hit_tp = b["low"] <= tp
            if hit_sl:
                return "loss", -sl_pct
            if hit_tp:
                return "win", tp_pct
    return "unresolved", 0.0


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()

    for strategy in _STRATEGIES:
        print(f"\n{'='*60}\n{strategy}\n{'='*60}")
        trades = _fetch_trades(cur, strategy)
        print(f"[fetch] {len(trades)} kapanmış işlem (atr+tp dolu)")

        results = {
            m: {"win": 0, "loss": 0, "unresolved": 0, "win_pnl": 0.0, "loss_pnl": 0.0}
            for m in _SL_MULTS
        }

        for i, row in trades.iterrows():
            start = row["opened_at"]
            end = start + pd.Timedelta(days=_HORIZON_DAYS)
            bars = _fetch_bars(cur, row["symbol"], row["interval"], start, end)
            if bars.empty:
                for m in _SL_MULTS:
                    results[m]["unresolved"] += 1
                continue
            for m in _SL_MULTS:
                outcome, pnl = _simulate_trade(row, bars, m)
                results[m][outcome] += 1
                if outcome == "win":
                    results[m]["win_pnl"] += pnl
                elif outcome == "loss":
                    results[m]["loss_pnl"] += pnl
            if (i + 1) % 100 == 0:
                print(f"  ... {i+1}/{len(trades)}")

        print(
            f"\n{'SL mult':>8} | {'win':>6} {'loss':>6} {'unresolved':>10} | "
            f"{'WR%':>6} | {'ort_kzn%':>9} {'ort_kyp%':>9} | {'PF':>6}"
        )
        for m in _SL_MULTS:
            r = results[m]
            resolved = r["win"] + r["loss"]
            wr = round(r["win"] / resolved * 100, 1) if resolved else float("nan")
            avg_win = round(r["win_pnl"] / r["win"], 3) if r["win"] else float("nan")
            avg_loss = round(r["loss_pnl"] / r["loss"], 3) if r["loss"] else float("nan")
            pf = round(r["win_pnl"] / -r["loss_pnl"], 3) if r["loss_pnl"] else float("nan")
            tag = " (orijinal)" if m == 1.5 else ""
            print(
                f"{m:>8.1f} | {r['win']:>6} {r['loss']:>6} {r['unresolved']:>10} | "
                f"{wr:>6.1f} | {avg_win:>9.3f} {avg_loss:>9.3f} | {pf:>6.3f}{tag}"
            )

    conn.close()


if __name__ == "__main__":
    main()
