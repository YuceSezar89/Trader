"""
Hızlı-reversal gürültüsü — iki düzeltme hipotezi testi (20 Tem 2026).

Kök neden (bulundu, kanıtlı): `signals/signal_filter.py`'nin "önceki ters
sinyali kır" kuralı, dar/gürültülü fiyat hareketinde neredeyse anlamsız
hale geliyor — referans nokta her zaman EN SON ters sinyale sıfırlanıyor,
sakin piyasada bu çok yakın bir seviye oluyor.

İki hipotez:
  A) ATR'ye göreceli minimum kırılma mesafesi — kırılma marjı büyüdükçe
     sonuç iyileşiyor mu? (korelasyon + filtre simülasyonu)
  B) Minimum bekleme süresi — çok hızlı reversal'ları "yok say", pozisyonu
     eşik anına kadar uzat, o andaki fiyattan kapat (karşı-olgusal ikame)

Kullanım: python -m research.pattern_lab.reversal_noise_filter_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_INTERVAL_TABLE = {"1m": None, "5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_HOLD_THRESHOLDS_MIN = (15, 30, 60, 120)
_MARGIN_ATR_THRESHOLDS = (0.0, 0.25, 0.5, 1.0)


def _stats(rets: np.ndarray) -> dict:
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return {"n": 0}
    g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean()), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


# ============================================================
# TEST A: ATR-göreceli kırılma marjı
# ============================================================

def test_a(cur) -> None:
    print(f"\n{'='*70}\nTEST A: ATR-göreceli kırılma marjı\n{'='*70}")
    cur.execute(
        """
        SELECT s.id, s.signal_type, s.atr, s.realized_pnl,
               rf.high AS rev_high, rf.low AS rev_low, rf.bar_time,
               ref.high AS ref_high, ref.low AS ref_low
        FROM signals s
        JOIN signals rev ON rev.id = s.closed_by
        JOIN signal_filter_events rf
          ON rf.symbol = rev.symbol AND rf.interval = rev.interval
         AND rf.indicator = rev.indicators AND rf.signal_type = rev.signal_type
         AND rf.bar_time = rev.opened_at
        JOIN LATERAL (
            SELECT high, low FROM signal_filter_events e2
            WHERE e2.symbol = rev.symbol AND e2.interval = rev.interval
              AND e2.indicator = rev.indicators
              AND e2.signal_type = CASE WHEN rev.signal_type='Long' THEN 'Short' ELSE 'Long' END
              AND e2.bar_time < rf.bar_time
            ORDER BY e2.bar_time DESC LIMIT 1
        ) ref ON true
        WHERE s.status = 'closed' AND s.close_reason = 'reversal'
          AND s.realized_pnl IS NOT NULL AND s.atr IS NOT NULL AND s.atr > 0
        """
    )
    cols = ["id", "signal_type", "atr", "realized_pnl", "rev_high", "rev_low", "bar_time",
            "ref_high", "ref_low"]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    print(f"[fetch] {len(df)} reversal-kapanışı, filtre kaydıyla eşleşti")
    if df.empty:
        return

    # signal_type burada REVERSING (yeni) sinyalin yönü — Long ise margin = rev_high-ref_high
    df["margin"] = np.where(
        df["signal_type"] == "Long", df["rev_high"] - df["ref_high"], df["ref_low"] - df["rev_low"]
    )
    df["margin_atr"] = df["margin"] / df["atr"]

    rho, p = spearmanr(df["margin_atr"], df["realized_pnl"])
    print(f"\nKorelasyon (margin_atr vs realized_pnl): rho={rho:+.3f} (p={p:.4f})")
    tercile = pd.qcut(df["margin_atr"], 4, labels=["1.en dar", "2", "3", "4.en geniş"], duplicates="drop")
    g = df.groupby(tercile, observed=True)["realized_pnl"].agg(
        ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100
    )
    print(g.to_string())

    print("\nFiltre simülasyonu — margin_atr eşiği geçmeyenler elenirse:")
    for thr in _MARGIN_ATR_THRESHOLDS:
        sub = df[df["margin_atr"] >= thr]
        s = _stats(sub["realized_pnl"].to_numpy())
        elenen = len(df) - len(sub)
        print(f"  eşik>={thr:.2f} ATR: kalan n={s.get('n',0):>6} (elenen {elenen:>6}) "
              f"| WR={s.get('wr',0):>5.1f}% ort%={s.get('ort_%',0):>7.3f} PF={s.get('pf',0):>6.3f}")


# ============================================================
# TEST B: minimum bekleme süresi (karşı-olgusal ikame)
# ============================================================

def _fetch_price_at(cur, symbol: str, interval: str, target_time) -> float | None:
    table = _INTERVAL_TABLE.get(interval)
    if table is None:
        return None
    cur.execute(
        f"SELECT close FROM {table} WHERE symbol=%s AND bucket <= %s ORDER BY bucket DESC LIMIT 1",
        (symbol, target_time),
    )
    row = cur.fetchone()
    return float(row[0]) if row else None


def test_b(cur) -> None:
    print(f"\n{'='*70}\nTEST B: minimum bekleme süresi (karşı-olgusal ikame)\n{'='*70}")
    cur.execute(
        """
        SELECT id, symbol, interval, signal_type, open_price, atr, opened_at, closed_at, realized_pnl
        FROM signals
        WHERE status='closed' AND close_reason='reversal' AND realized_pnl IS NOT NULL
          AND EXTRACT(EPOCH FROM (closed_at-opened_at))/60 < 120
        """
    )
    cols = ["id", "symbol", "interval", "signal_type", "open_price", "atr", "opened_at", "closed_at", "realized_pnl"]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    print(f"[fetch] {len(df)} reversal-kapanışı (<120dk sürede kapanmış)")
    if df.empty:
        return

    df["duration_min"] = (df["closed_at"] - df["opened_at"]).dt.total_seconds() / 60.0

    baseline = _stats(df["realized_pnl"].to_numpy())
    print(f"\nBASELINE (hepsi, orijinal realized_pnl): {baseline}")

    for thr in _HOLD_THRESHOLDS_MIN:
        fast = df[df["duration_min"] < thr]
        slow = df[df["duration_min"] >= thr]
        substituted = []
        for _, row in fast.iterrows():
            target = row["opened_at"] + pd.Timedelta(minutes=thr)
            price = _fetch_price_at(cur, row["symbol"], row["interval"], target)
            if price is None:
                continue
            side = 1.0 if row["signal_type"] == "Long" else -1.0
            pct = (price - row["open_price"]) / row["open_price"] * 100.0 * side
            substituted.append(pct)
        combined = np.concatenate([slow["realized_pnl"].to_numpy(), np.array(substituted)])
        s = _stats(combined)
        print(f"  eşik={thr:>4}dk: hızlı(ikame)={len(substituted):>5} + yavaş(orijinal)={len(slow):>6} "
              f"= n={s.get('n',0):>6} | WR={s.get('wr',0):>5.1f}% ort%={s.get('ort_%',0):>7.3f} PF={s.get('pf',0):>6.3f}")


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    test_a(cur)
    test_b(cur)
    conn.close()


if __name__ == "__main__":
    main()
