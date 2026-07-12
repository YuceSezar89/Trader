"""
rsi_cross_st_shape_bt.py'nin devamı — mum-şekli sınıflandırmasını
DETAYLANDIRIR. Önceki hali "kazanan kategori" diye 3'e ayırıyordu (gövde/
üst-fitil/alt-fitil-baskın) ama HANGİ KATEGORİ OLURSA OLSUN, o kategorinin
kendi büyüklüğünü (ör. gövde-baskın içinde gövde %35 mi %95 mi) kaybediyordu
— "az gövde-baskın" ile "aşırı gövde-baskın" (Marubozu) aynı kutuya
düşüyordu. Bu script her kategori İÇİNDE kazanan yüzdenin büyüklüğüne göre
3 alt-tercile ayırıyor (9 hücre toplam) — DAHA GÜÇLÜ dominasyonun (ör. saf
Marubozu) daha zayıfından gerçekten farklı performans verip vermediğini
ölçer. st_confirmed (v2-23) ile birlikte de gösteriliyor.
"""
import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_vpmv_jump_bt import INTERVALS, _fetch_symbol_history, _signal_bar_ts  # pylint: disable=wrong-import-position
from research.pattern_lab.do_break_gauss_economic_bt import POSITION_USD, ROUND_TRIP_FEE  # pylint: disable=wrong-import-position

CATEGORIES = ["gövde-baskın", "üst-fitil-baskın", "alt-fitil-baskın"]
MAG_TIERS = ["düşük", "orta", "yüksek"]


def _classify_detailed(row) -> tuple:
    """(kategori, kazanan_yüzdenin_büyüklüğü) döner — büyüklük her zaman
    33.3-100 aralığında (3'e bölünen bir menzilde en büyük parça en az
    %33.3 olmak zorunda)."""
    rng = row["high"] - row["low"]
    if rng <= 0:
        return "belirsiz", np.nan
    upper = max(row["open"], row["close"])
    lower = min(row["open"], row["close"])
    body = abs(row["close"] - row["open"]) / rng * 100
    upper_wick = (row["high"] - upper) / rng * 100
    lower_wick = (lower - row["low"]) / rng * 100
    parts = {"gövde-baskın": body, "üst-fitil-baskın": upper_wick, "alt-fitil-baskın": lower_wick}
    winner = max(parts, key=parts.get)
    return winner, parts[winner]


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
        ORDER BY opened_at
    """
    df = pd.read_sql(q, conn, params=(interval,))
    conn.close()
    return df


def _pf(df: pd.DataFrame) -> dict:
    return _stats(df["realized_pnl"].to_numpy() / 100)


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


def run():
    rows = []
    for interval in INTERVALS:
        sigs = _fetch_signals(interval)
        print(f"{interval}: {len(sigs):,} kapanmış RSI_Cross sinyali")

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
                kategori, magnitude = _classify_detailed(hist.iloc[i])
                if kategori == "belirsiz" or not np.isfinite(magnitude):
                    continue
                rows.append({
                    "kategori": kategori,
                    "magnitude": magnitude,
                    "st_confirmed": row["st_confirmed"],
                    "realized_pnl": row["realized_pnl"],
                    "opened_at": row["opened_at"],
                })

    df = pd.DataFrame(rows)
    print(f"\ntoplam eşleşen sinyal: {len(df):,}\n")
    if len(df) < 200:
        print("Örneklem çok küçük.")
        return

    # Her kategori İÇİNDE, kendi büyüklük değerine göre tercile böl.
    df["mag_tier"] = "?"
    for kategori in CATEGORIES:
        mask = df["kategori"] == kategori
        vals = df.loc[mask, "magnitude"]
        q1, q2 = vals.quantile([0.333, 0.667])
        df.loc[mask, "mag_tier"] = pd.cut(
            vals, bins=[-np.inf, q1, q2, np.inf], labels=MAG_TIERS,
        ).astype(str)

    print(f"{'kategori':20} {'büyüklük':10} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for kategori in CATEGORIES:
        for tier in MAG_TIERS:
            sub = df[(df["kategori"] == kategori) & (df["mag_tier"] == tier)]
            s = _pf(sub)
            print(f"{kategori:20} {tier:10} {s.get('n',0):>7} {s.get('wr',0):>6} "
                  f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")

    print("\n── SADECE st_confirmed=True içinde (gövde-baskın, büyüklüğe göre) ──")
    only_true_govde = df[(df["st_confirmed"] == True) & (df["kategori"] == "gövde-baskın")]  # noqa: E712  pylint: disable=singleton-comparison
    print(f"{'büyüklük':10} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for tier in MAG_TIERS:
        sub = only_true_govde[only_true_govde["mag_tier"] == tier]
        s = _pf(sub)
        print(f"{tier:10} {s.get('n',0):>7} {s.get('wr',0):>6} {s.get('ort_%',0):>8} {s.get('pf',0):>7}")

    days_span = (df["opened_at"].max() - df["opened_at"].min()).total_seconds() / 86400
    print(f"\n── Ekonomik etki (${POSITION_USD:.0f} pozisyon, {days_span:.1f} gün) ──")
    print(f"{'grup':40} {'n':>7} {'WR%':>6} {'ort $/işlem':>12} {'toplam $':>10} {'$/ay':>10}")
    best_govde_true = only_true_govde[only_true_govde["mag_tier"] == "yüksek"]
    groups = {
        "baseline (tüm RSI_Cross)": df,
        "st_confirmed=True + gövde-baskın (hepsi)": df[(df["st_confirmed"] == True) & (df["kategori"] == "gövde-baskın")],  # noqa: E712 pylint: disable=singleton-comparison
        "st_confirmed=True + gövde-baskın + YÜKSEK büyüklük": best_govde_true,
    }
    for name, g in groups.items():
        s = _dollar_stats(g, days_span)
        if s.get("n", 0) == 0:
            print(f"{name:40} {'0':>7}")
            continue
        print(f"{name:40} {s['n']:>7} {s['wr']:>6} {s['avg_usd']:>12} "
              f"{s['total_usd']:>10} {s['usd_per_month']:>10}")


if __name__ == "__main__":
    run()
