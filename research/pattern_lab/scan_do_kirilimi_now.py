"""
Şu anki canlı veriyle GERÇEK check() fonksiyonunu tüm sembollerde tarar.
BTC filtresi GERÇEK (bypass yok) — canlı sistemin şu an ne göreceğini birebir test eder.

Kullanım: python -m research.pattern_lab.scan_do_kirilimi_now [sembol_limit]
"""
import sys
import warnings

import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config
from signals.do_kirilimi import do_kirilimi_detector, btc_day_context


def main(symbol_limit: int = 150):
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )

    top_symbols = pd.read_sql(
        """SELECT symbol, sum(volume) v FROM cagg_5m
           WHERE bucket >= now() - interval '2 days'
           GROUP BY symbol ORDER BY v DESC LIMIT %s""",
        conn, params=(symbol_limit,),
    )["symbol"].tolist()
    if "BTCUSDT" not in top_symbols:
        top_symbols.append("BTCUSDT")

    btc_df = pd.read_sql(
        "SELECT bucket, open, high, low, close, volume FROM cagg_5m WHERE symbol=%s ORDER BY bucket DESC LIMIT 320",
        conn, params=("BTCUSDT",),
    ).sort_values("bucket").reset_index(drop=True)
    btc_df["open_time"] = (btc_df["bucket"] - pd.Timedelta(hours=3)).astype("int64") // 10**6
    btc_ctx = btc_day_context(btc_df)
    print(f"[BTC ctx] {btc_ctx}")
    print(f"[tarama] {len(top_symbols)} sembol, şu an ({pd.Timestamp.now()})")
    print()

    fired = []
    for symbol in top_symbols:
        df = pd.read_sql(
            "SELECT bucket, open, high, low, close, volume FROM cagg_5m WHERE symbol=%s ORDER BY bucket DESC LIMIT 320",
            conn, params=(symbol,),
        )
        if df.empty:
            continue
        df = df.sort_values("bucket").reset_index(drop=True)
        df["open_time"] = (df["bucket"] - pd.Timedelta(hours=3)).astype("int64") // 10**6
        try:
            result = do_kirilimi_detector.check(symbol, df, btc_ctx)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"  ! {symbol}: hata {exc}")
            continue
        if result:
            fired.append((symbol, result))
            print(f"  ✅ {symbol}: SİNYAL VAR fiyat={result['price']} pattern={result['pattern']} ayrisma={result['ayrisma']}")

    conn.close()
    print()
    print(f"=== SONUÇ: {len(fired)}/{len(top_symbols)} sembolde şu an gerçek filtreyle sinyal var ===")
    if not fired:
        print("Hiçbiri şu an ateşlemiyor — bu normal (sinyal doğası gereği nadir).")


if __name__ == "__main__":
    _limit = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    main(_limit)
