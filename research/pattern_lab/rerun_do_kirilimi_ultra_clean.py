"""
do_kirilimi ULTRA-MTF testini (look-ahead DÜZELTİLMİŞ hali, mtf_helpers.py'deki
fix zaten kalıcı) daha geniş pencere + hâlâ büyük iç-boşluğu olan sembolleri
dışlayarak tekrar çalıştırır. 13 Tem 2026 gece/sabah backfill'i sonrası birçok
sembolde artık 45 günden çok daha fazla temiz geçmiş var — DAYS'i genişletiyoruz.

Kullanım: python -m research.pattern_lab.rerun_do_kirilimi_ultra_clean
"""

import psycopg2

from config import Config
from research.pattern_lab import do_kirilimi_mtf_alignment_bt as target
from research.pattern_lab import do_open_streak_bt

_GAP_HOURS_THRESHOLD = 200
_NEW_DAYS = 90


def _bad_symbols() -> set[str]:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    cur.execute(
        """
        WITH gaps AS (
            SELECT symbol, EXTRACT(EPOCH FROM (curr_ts-prev_ts))/3600 AS saat
            FROM (
                SELECT symbol, timestamp AS curr_ts,
                       LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) AS prev_ts
                FROM price_data WHERE interval='1m'
            ) t
            WHERE prev_ts IS NOT NULL
        )
        SELECT DISTINCT symbol FROM gaps WHERE saat > %s
        """,
        (_GAP_HOURS_THRESHOLD,),
    )
    bad = {r[0] for r in cur.fetchall()}
    conn.close()
    return bad


def main() -> None:
    bad = _bad_symbols()
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")
    print(f"[filtre] pencere {do_open_streak_bt.DAYS} gün -> {_NEW_DAYS} gün olarak genişletildi\n")

    do_open_streak_bt.DAYS = _NEW_DAYS
    original_fetch = do_open_streak_bt._fetch  # pylint: disable=protected-access

    def _fetch_clean():
        df = original_fetch()
        return df[~df["symbol"].isin(bad)].reset_index(drop=True)

    target._fetch = _fetch_clean  # pylint: disable=protected-access
    target.run()


if __name__ == "__main__":
    main()
