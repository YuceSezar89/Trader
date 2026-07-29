"""
"Trailing kârı erken kesiyor mu" testi (20 Tem 2026, kullanıcı iddiası:
"uzaya giden sinyaller erken kapandı, trail işlemden erken çıkardı").

Yöntem: trailing_stop ile kapanan işlemler için, KAPANIŞTAN SONRAKİ 24 saat
içinde fiyat hâlâ lehte gitmeye devam etmiş mi (= trail çok sıkı, erken
çıkardı) yoksa gerçekten dönmüş mü (= trail doğru zamanladı)? Gerçek kline
verisiyle (cagg_5m/15m/1h/4h), look-ahead yok (kapanıştan SONRAsına bakıyoruz,
kapanış kararını etkilemiyor, sadece kararı DEĞERLENDİRİYORUZ).

Kullanım: python -m research.pattern_lab.trailing_exit_timing_bt
"""

import warnings

import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config

_HORIZON = pd.Timedelta(hours=24)
_INTERVAL_TABLE = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_STRATEGIES = ("rsi_cross_live", "tf_alignment_live")


def _fetch_trailing_trades(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, strategy, symbol, signal_type, interval, exit_price, closed_at, pnl_pct
        FROM paper_trades
        WHERE strategy = ANY(%s) AND status = 'closed' AND close_reason = 'trailing_stop'
          AND exit_price IS NOT NULL
        """,
        (list(_STRATEGIES),),
    )
    cols = ["id", "strategy", "symbol", "signal_type", "interval", "exit_price", "closed_at", "pnl_pct"]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def _fetch_bars_after(cur, symbol: str, interval: str, start, end) -> pd.DataFrame:
    table = _INTERVAL_TABLE.get(interval, "cagg_5m")
    cur.execute(
        f"SELECT bucket, high, low FROM {table} WHERE symbol=%s AND bucket > %s AND bucket <= %s ORDER BY bucket ASC",
        (symbol, start, end),
    )
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["bucket", "high", "low"]) if rows else pd.DataFrame()


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    trades = _fetch_trailing_trades(cur)
    print(f"[fetch] {len(trades)} trailing_stop ile kapanmış işlem\n")

    records = []
    for i, row in trades.iterrows():
        bars = _fetch_bars_after(cur, row["symbol"], row["interval"], row["closed_at"], row["closed_at"] + _HORIZON)
        if bars.empty:
            continue
        exit_price = row["exit_price"]
        side = row["signal_type"]
        if side == "Long":
            further_move_pct = (bars["high"].max() - exit_price) / exit_price * 100.0
            reversal_pct = (exit_price - bars["low"].min()) / exit_price * 100.0
        else:
            further_move_pct = (exit_price - bars["low"].min()) / exit_price * 100.0
            reversal_pct = (bars["high"].max() - exit_price) / exit_price * 100.0
        records.append({
            "strategy": row["strategy"], "symbol": row["symbol"], "signal_type": side,
            "pnl_pct_locked": row["pnl_pct"], "further_favorable_pct": further_move_pct,
            "further_adverse_pct": reversal_pct,
        })
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(trades)}")

    conn.close()
    df = pd.DataFrame(records)
    print(f"\n[collect] {len(df)} işlem için 24s sonrası fiyat verisi bulundu\n")

    for strategy in _STRATEGIES:
        sub = df[df["strategy"] == strategy]
        if sub.empty:
            continue
        print(f"{'='*70}\n{strategy} (n={len(sub)})\n{'='*70}")
        print(f"  Ortalama kilitlenen kâr (pnl_pct):             {sub['pnl_pct_locked'].mean():+.3f}%")
        print(f"  Ortalama kapanış-SONRASI lehte hareket (24s):  {sub['further_favorable_pct'].mean():+.3f}%")
        print(f"  Ortalama kapanış-SONRASI aleyhte hareket (24s):{sub['further_adverse_pct'].mean():+.3f}%")
        missed_bigger = (sub["further_favorable_pct"] > sub["pnl_pct_locked"]).mean() * 100
        missed_meaningful = (sub["further_favorable_pct"] > 1.0).mean() * 100
        reversed_meaningful = (sub["further_adverse_pct"] > 1.0).mean() * 100
        print(f"  Kilitlenenden DAHA BÜYÜK lehte hareket olan işlem oranı: %{missed_bigger:.1f}")
        print(f"  24s içinde >1% daha lehte giden işlem oranı:            %{missed_meaningful:.1f}")
        print(f"  24s içinde >1% aleyhte dönen (trail haklı çıktı) oranı: %{reversed_meaningful:.1f}")
        print(f"  Medyan lehte hareket: {sub['further_favorable_pct'].median():+.3f}% | "
              f"Medyan aleyhte hareket: {sub['further_adverse_pct'].median():+.3f}%")
        print()


if __name__ == "__main__":
    main()
