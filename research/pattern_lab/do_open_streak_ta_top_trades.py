"""
do_open_streak+TA-kovalama sonucunu taşıyan EN İYİ işlemlerin incelemesi —
yoğunlaşma teşhisi top-5 katkının %92.7 olduğunu gösterdi, bu birkaç
işlemin gerçek mi (büyük ama meşru hareketler) yoksa veri artefaktı mı
olduğunu doğrulamak için (24 Tem 2026, kullanıcı isteği — bkz. 22 Tem'deki
龙虾USDT emsali, project_do_open_streak_22_23tem.md).

Kullanım: python -m research.pattern_lab.do_open_streak_ta_top_trades
(önce do_open_streak_ta_combo_bt.py çalıştırılmış olmalı — cache'i kullanır)
"""

import pandas as pd
import psycopg2

from config import Config
from research.pattern_lab.do_open_streak_ta_combo_bt import _CACHE_PATH

_BEST_TH = 85
_TOP_N = 15


def _fetch_price_context(symbol: str, ts: pd.Timestamp, hours_before: int = 6, hours_after: int = 24) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    cur.execute(
        """
        SELECT bucket, open, high, low, close, volume FROM cagg_15m
        WHERE symbol=%s AND bucket >= %s AND bucket <= %s
        ORDER BY bucket ASC
        """,
        (symbol, ts - pd.Timedelta(hours=hours_before), ts + pd.Timedelta(hours=hours_after)),
    )
    rows = cur.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["bucket", "open", "high", "low", "close", "volume"])


def main() -> None:
    df = pd.read_parquet(_CACHE_PATH)
    df = df.dropna(subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h"]).reset_index(drop=True)
    kov = ((df["pct_1h"] >= _BEST_TH) & (df["slope_1h"] > 0)) | ((df["pct_4h"] >= _BEST_TH) & (df["slope_4h"] > 0))
    group = df[kov].sort_values("pnl_usd", ascending=False).reset_index(drop=True)

    print(f"Grup n={len(group)}, toplam pnl=${group['pnl_usd'].sum():.2f}\n")
    print("=" * 100)
    print(f"EN İYİ {_TOP_N} İŞLEM")
    print("=" * 100)
    top = group.head(_TOP_N)
    for _, row in top.iterrows():
        print(f"{row['symbol']:16} {str(row['ts']):20} pnl=${row['pnl_usd']:>9.2f}  "
              f"gauss={row['gauss_val']:>6.2f}  long_perc={row['long_perc']:>6.2f}%  "
              f"reason={row['reason']:12}  pct1h/4h={row['pct_1h']:.0f}/{row['pct_4h']:.0f}  "
              f"slope1h/4h={row['slope_1h']:+.2f}/{row['slope_4h']:+.2f}")

    print(f"\nToplam n={len(group)}, top-{_TOP_N} katkısı: %{top['pnl_usd'].sum()/group['pnl_usd'].sum()*100:.1f}\n")

    print("=" * 100)
    print("EN İYİ 3 İŞLEMİN FİYAT BAĞLAMI (giriş öncesi 6sa - sonrası 24sa)")
    print("=" * 100)
    for _, row in top.head(3).iterrows():
        sym, ts, pnl = row["symbol"], pd.Timestamp(row["ts"]), row["pnl_usd"]
        print(f"\n--- {sym} @ {ts} (pnl=${pnl:.2f}, gauss={row['gauss_val']:.2f}) ---")
        ctx = _fetch_price_context(sym, ts)
        if ctx.empty:
            print("  fiyat verisi bulunamadı")
            continue
        entry_bar = ctx[ctx["bucket"] >= ts].iloc[0] if (ctx["bucket"] >= ts).any() else None
        print(f"  bağlam: {len(ctx)} bar, fiyat aralığı: {ctx['low'].min():.6g} - {ctx['high'].max():.6g}")
        if entry_bar is not None:
            print(f"  giriş bar açılış/kapanış: {entry_bar['open']:.6g} / {entry_bar['close']:.6g}")
        pre = ctx[ctx["bucket"] < ts]
        post = ctx[ctx["bucket"] >= ts]
        if len(pre) and len(post):
            pre_range = (pre["high"].max() - pre["low"].min()) / pre["low"].min() * 100
            post_move = (post["high"].max() - post["close"].iloc[0]) / post["close"].iloc[0] * 100
            print(f"  giriş ÖNCESİ 6sa dalgalanma: %{pre_range:.2f}  |  giriş SONRASI en yüksek hareket: %{post_move:.2f}")
        print(f"  hacim (ort, 15m bar): {ctx['volume'].mean():.0f}  (max: {ctx['volume'].max():.0f})")


if __name__ == "__main__":
    main()
