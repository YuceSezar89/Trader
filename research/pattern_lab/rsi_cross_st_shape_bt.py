"""
st_confirmed (bugünün en güçlü/tutarlı bulgusu — PF 1.33-1.82 vs 0.67-0.83,
6/6 kombinasyonda ve 24/24 split-period alt-testte doğru tarafta) ile
mum-şekli filtresini (v2-9, look-ahead'siz — VPMV sıçraması BUGÜN GEÇERSİZ
ilan edildiği için dahil edilmedi) BİRLEŞTİRİR. RSI_Cross'un TÜM sinyalleri
— Long+Short, 5m+15m — `signals` tablosundaki gerçek realized_pnl ile.
"""
import os
import sys

import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_vpmv_jump_bt import INTERVALS, _fetch_symbol_history, _signal_bar_ts  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_candle_shape_bt import _classify  # pylint: disable=wrong-import-position
from research.pattern_lab.do_break_gauss_economic_bt import POSITION_USD, ROUND_TRIP_FEE  # pylint: disable=wrong-import-position

CATEGORIES = ["gövde-baskın", "üst-fitil-baskın", "alt-fitil-baskın"]


def _dollar_stats(df: pd.DataFrame, days_span: float) -> dict:
    if len(df) == 0 or days_span <= 0:
        return {"n": 0}
    pnl = df["realized_pnl"].to_numpy() / 100 * POSITION_USD - ROUND_TRIP_FEE
    total = float(pnl.sum())
    return {
        "n": len(pnl), "wr": round(float((pnl > 0).mean() * 100), 1),
        "avg_usd": round(float(pnl.mean()), 3), "total_usd": round(total, 1),
        "usd_per_month": round(total / days_span * 30, 1),
    }


def _fetch_signals(interval: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT symbol, signal_type, st_confirmed, realized_pnl, opened_at
        FROM signals
        WHERE indicators LIKE '%%RSI_Cross%%'
          AND interval = %s
          AND status = 'closed' AND realized_pnl IS NOT NULL
          AND st_confirmed IS NOT NULL
        ORDER BY opened_at
    """
    df = pd.read_sql(q, conn, params=(interval,))
    conn.close()
    return df


def _pf(df: pd.DataFrame) -> dict:
    return _stats(df["realized_pnl"].to_numpy() / 100)


def run():
    rows = []
    for interval in INTERVALS:
        sigs = _fetch_signals(interval)
        print(f"{interval}: {len(sigs):,} kapanmış RSI_Cross sinyali (st_confirmed dolu)")

        for symbol, sub in sigs.groupby("symbol"):
            hist = _fetch_symbol_history(symbol, interval)
            if hist.empty:
                continue
            hist = hist.sort_values("ts").reset_index(drop=True)
            ts_to_idx = {t: i for i, t in enumerate(hist["ts"])}

            for _, row in sub.iterrows():
                i = ts_to_idx.get(_signal_bar_ts(row["opened_at"], interval))
                if i is None:
                    continue
                kategori = _classify(hist.iloc[i])
                rows.append({
                    "st_confirmed": bool(row["st_confirmed"]),
                    "kategori": kategori,
                    "realized_pnl": row["realized_pnl"],
                    "signal_type": row["signal_type"],
                    "opened_at": row["opened_at"],
                })

    df = pd.DataFrame(rows)
    print(f"\ntoplam eşleşen sinyal: {len(df):,}\n")
    if len(df) < 100:
        print("Örneklem çok küçük.")
        return

    print(f"{'st_confirmed':14} {'kategori':20} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for st in (True, False):
        for kategori in CATEGORIES:
            sub = df[(df["st_confirmed"] == st) & (df["kategori"] == kategori)]
            s = _pf(sub)
            print(f"{str(st):14} {kategori:20} {s.get('n',0):>7} {s.get('wr',0):>6} "
                  f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")

    print("\n── SADECE st_confirmed=True içinde mum-şekli katkısı ──")
    only_true = df[df["st_confirmed"] == True]  # noqa: E712  pylint: disable=singleton-comparison
    s_all = _pf(only_true)
    print(f"{'(hepsi)':20} {s_all.get('n',0):>7} {s_all.get('wr',0):>6} "
          f"{s_all.get('ort_%',0):>8} {s_all.get('pf',0):>7}")
    for kategori in CATEGORIES:
        sub = only_true[only_true["kategori"] == kategori]
        s = _pf(sub)
        print(f"{kategori:20} {s.get('n',0):>7} {s.get('wr',0):>6} "
              f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")
    print("\n(üst-fitil-baskın HARİÇ, st_confirmed=True):")
    sub = only_true[only_true["kategori"] != "üst-fitil-baskın"]
    s = _pf(sub)
    print(f"{'gövde+alt-fitil':20} {s.get('n',0):>7} {s.get('wr',0):>6} "
          f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")

    days_span = (df["opened_at"].max() - df["opened_at"].min()).total_seconds() / 86400
    print(f"\n── Ekonomik etki (${POSITION_USD:.0f} pozisyon, gerçek fee, {days_span:.1f} günlük dönem) ──")
    print(f"{'grup':32} {'n':>7} {'WR%':>6} {'ort $/işlem':>12} {'toplam $':>10} {'$/ay':>10}")
    groups = {
        "baseline (tüm RSI_Cross)": df,
        "st_confirmed=True (hepsi)": only_true,
        "st_confirmed=True + gövde-baskın": only_true[only_true["kategori"] == "gövde-baskın"],
        "st_confirmed=True + gövde/alt-fitil": sub,
        "st_confirmed=False (hepsi)": df[df["st_confirmed"] == False],  # noqa: E712  pylint: disable=singleton-comparison
    }
    for name, g in groups.items():
        s = _dollar_stats(g, days_span)
        if s.get("n", 0) == 0:
            print(f"{name:32} {'0':>7}")
            continue
        print(f"{name:32} {s['n']:>7} {s['wr']:>6} {s['avg_usd']:>12} "
              f"{s['total_usd']:>10} {s['usd_per_month']:>10}")


if __name__ == "__main__":
    run()
