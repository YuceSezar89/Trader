"""
Tek nokta testi: verilen sembol + zamanda do_kirilimi'nin HAM 6-kapı+ADX+ST
sinyali ateşliyor mu? BTC rejim/ayrışma paper filtresi BİLEREK devre dışı
(fake btc_ctx ile bypass) — sadece Pine setup/entry mantığı test ediliyor.

Kullanım: python -m research.pattern_lab.check_signal_at SEMBOL "YYYY-MM-DD HH:MM:SS"
"""
import sys
import warnings

import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config
from signals.do_kirilimi import do_kirilimi_detector

# BTC rejim/ayrışma filtresini bypass eden sahte bağlam — sadece ham setup görülsün
_BYPASS_BTC_CTX = {"day_ret": -999.0, "day_up": True}


def check_signal_at(symbol: str, end_ts: str, n: int = 320) -> dict | None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """SELECT bucket, open, high, low, close, volume FROM cagg_5m
           WHERE symbol=%s AND bucket<=%s ORDER BY bucket DESC LIMIT %s"""
    df = pd.read_sql(q, conn, params=(symbol, end_ts, n))
    conn.close()
    if df.empty:
        print(f"{symbol}: veri yok")
        return None
    df = df.sort_values("bucket").reset_index(drop=True)
    # KRİTİK: check() içinde +3h eklenir, burada -3h ile telafi et (bu geceki ders)
    df["open_time"] = (df["bucket"] - pd.Timedelta(hours=3)).astype("int64") // 10**6

    print(f"{symbol}: pencere {df['bucket'].iloc[0]} -> {df['bucket'].iloc[-1]} ({len(df)} bar)")
    result = do_kirilimi_detector.check(symbol, df, _BYPASS_BTC_CTX)
    if result:
        print(f"  ✅ SİNYAL VAR: fiyat={result['price']} pattern={result['pattern']}")
    else:
        print("  ❌ sinyal yok (6 kapı + ADX + ST birlikte sağlanmadı)")
    return result


if __name__ == "__main__":
    check_signal_at(sys.argv[1], sys.argv[2])
