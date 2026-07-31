"""
evol_trough_bt.py — kullanıcının canlı yarış grafiklerinde gözlemlediği örüntünün
testi: "EVOL dip yapıp yükselmeye döndüğü an, fiyat da yükselmeye başlıyor".

evol_bt.py'den FARKLI: o sinyal anındaki EVOL SEVİYESİni (statik) test ediyordu.
Bu script EVOL'ün YEREL MİNİMUMUNU (dönüş noktasını, zamansal bir örüntü) tespit
edip o andan itibaren fiyatın ne yaptığına bakıyor. do_open_streak'e özgü değil —
geniş bir sembol havuzunda genel piyasa davranışı olarak test ediliyor.

EVOL hesabı canlı kodla (indicators/core.py::calculate_evol) AYNI formül/disiplin
— sadece "son bar" yerine TÜM seri için rolling percentile-rank uygulanıyor
(bkz. _evol_series). Yerel minimum: EVOL[i], [i-CONFIRM_BEFORE, i] aralığının
en düşüğü VE sonraki CONFIRM_AFTER barda net yükseliyor (tek barlık gürültüyü
elemek için basit iki-taraflı konfirmasyon — scipy YOK, mantık denetlenebilir
kalsın diye elle yazıldı).

Kıyaslar:
  (a) Placebo: rastgele noktalar (trough olmayan) aynı sayıda örneklenip
      aynı ileri-getiri ölçülür — trough'un gerçek bir edge mi yoksa piyasanın
      genel driftı mı olduğunu ayırt eder.
  (b) Ters hipotez: EVOL'ün YEREL MAKSİMUMU (tepe) — kullanıcının hipotezinin
      tam tersini de test ederek "aslında EVOL hiç önemli değil, her dönüş
      noktası böyle davranıyor" ihtimalini eler.
  (c) Split-period IS/OOS.

Kullanım: python -m research.pattern_lab.evol_trough_bt
"""

import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # noqa: E402  pylint: disable=wrong-import-position

DAYS = 40
MIN_BARS = 700
RVOL_WINDOW = 20
RANK_WINDOW = 100
CONFIRM_BEFORE = 6  # trough'tan önceki 6 barın en düşüğü olmalı (~1.5 saat @ 15m)
CONFIRM_AFTER = 3  # trough'tan sonraki 3 bar net yükselmeli (~45dk)
FORWARD_BARS = {"1h": 4, "4h": 16, "12h": 48, "24h": 96}
_GAP_HOURS_THRESHOLD = 200
_PLACEBO_SEED = 42


def _conn():
    return psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )


def _bad_symbols(cur) -> set[str]:
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
    return {r[0] for r in cur.fetchall()}


def _fetch(exclude: set[str]) -> pd.DataFrame:
    conn = _conn()
    q = f"""
        SELECT symbol, bucket AS ts, open, high, low, close, volume
        FROM cagg_15m
        WHERE bucket > NOW() - INTERVAL '{DAYS} days'
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn)
    conn.close()
    if exclude:
        df = df[~df["symbol"].isin(exclude)].reset_index(drop=True)
    return df


def _evol_series(g: pd.DataFrame) -> np.ndarray:
    """calculate_evol ile AYNI formül (indicators/core.py) — her bar için
    rolling percentile-rank skoru (0-100), tam seri boyunca."""
    close = g["close"].to_numpy(float)
    volume = g["volume"].to_numpy(float)
    price_pct = np.diff(close, prepend=np.nan) / np.roll(close, 1) * 100.0
    price_pct[0] = np.nan

    vol_s = pd.Series(volume)
    rvol = (volume / vol_s.rolling(RVOL_WINDOW).mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = price_pct / np.where(rvol == 0, np.nan, rvol)

    smoothed = pd.Series(raw).ewm(span=7, adjust=False).mean().to_numpy()

    n = len(smoothed)
    score = np.full(n, np.nan)
    valid_mask = np.isfinite(smoothed)
    valid_idx = np.where(valid_mask)[0]
    valid_vals = smoothed[valid_mask]
    # rolling percentile-rank: valid_vals içindeki konuma göre (son RANK_WINDOW)
    for pos in range(len(valid_vals)):
        if pos < 19:  # calculate_evol'daki "len(valid)<20" eşiğiyle tutarlı
            continue
        start = max(0, pos - RANK_WINDOW + 1)
        window = valid_vals[start : pos + 1]
        current = valid_vals[pos]
        rank = (window < current).sum() / len(window)
        score[valid_idx[pos]] = round(rank * 100.0, 2)
    return score


def _find_troughs_and_peaks(score: np.ndarray) -> tuple[list[int], list[int]]:
    """Trough/peak, konfirmasyonun TAMAMLANDIĞI bar'da (i+CONFIRM_AFTER)
    raporlanır — i'nin kendisi değil. i sadece geçmişe dönük aday; bar i'de
    "bu bir trough" bilgisi henüz yok, o bilgi ancak CONFIRM_AFTER bar sonra
    (giriş anında gerçekten elde bulunan bar) tamamlanıyor. Giriş fiyatı da
    bu yüzden downstream'de c[i] değil c[confirmed_idx] olmalı."""
    n = len(score)
    troughs, peaks = [], []
    for i in range(CONFIRM_BEFORE, n - CONFIRM_AFTER):
        window = score[i - CONFIRM_BEFORE : i + 1]
        after = score[i + 1 : i + 1 + CONFIRM_AFTER]
        if np.isnan(window).any() or np.isnan(after).any():
            continue
        confirmed_idx = i + CONFIRM_AFTER
        if score[i] == window.min() and after[-1] > score[i] and np.all(np.diff(after) >= -1e-9):
            troughs.append(confirmed_idx)
        if score[i] == window.max() and after[-1] < score[i] and np.all(np.diff(after) <= 1e-9):
            peaks.append(confirmed_idx)
    return troughs, peaks


def _forward_returns(c: np.ndarray, idx: int) -> dict[str, float]:
    entry = c[idx]
    out = {}
    for name, bars in FORWARD_BARS.items():
        j = min(idx + bars, len(c) - 1)
        out[name] = (c[j] - entry) / entry * 100.0
    return out


def _stats(vals: np.ndarray) -> dict:
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return {"n": 0}
    return {
        "n": len(vals),
        "ort_%": round(float(vals.mean()), 3),
        "medyan_%": round(float(np.median(vals)), 3),
        "wr": round(float((vals > 0).mean() * 100), 1),
    }


def run() -> None:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    conn.close()
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    df_all = _fetch(bad)
    print(f"{df_all['symbol'].nunique()} sembol, {len(df_all):,} 15m bar ({DAYS} gün)\n")

    trough_records: list[dict] = []
    peak_records: list[dict] = []
    placebo_records: list[dict] = []
    rng = np.random.default_rng(_PLACEBO_SEED)

    n_syms = 0
    for sym, g in df_all.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1
        c = g["close"].to_numpy(float)
        ts = g["ts"]
        score = _evol_series(g)
        troughs, peaks = _find_troughs_and_peaks(score)

        for idx in troughs:
            if idx + max(FORWARD_BARS.values()) >= len(c):
                continue
            rec = {"symbol": sym, "ts": ts.iloc[idx], "evol_at": score[idx]}
            rec.update(_forward_returns(c, idx))
            trough_records.append(rec)

        for idx in peaks:
            if idx + max(FORWARD_BARS.values()) >= len(c):
                continue
            rec = {"symbol": sym, "ts": ts.iloc[idx], "evol_at": score[idx]}
            rec.update(_forward_returns(c, idx))
            peak_records.append(rec)

        # placebo: trough sayısı kadar rastgele (gecerli, kenarlardan uzak) nokta
        valid_range = np.arange(CONFIRM_BEFORE, len(c) - max(FORWARD_BARS.values()))
        if len(valid_range) > 0 and troughs:
            sample = rng.choice(valid_range, size=min(len(troughs), len(valid_range)), replace=False)
            for idx in sample:
                rec = {"symbol": sym, "ts": ts.iloc[idx]}
                rec.update(_forward_returns(c, idx))
                placebo_records.append(rec)

    print(f"[tarama] {n_syms} sembol")
    print(f"Toplam TROUGH (dip) noktası: {len(trough_records)}")
    print(f"Toplam PEAK (tepe) noktası: {len(peak_records)}")
    print(f"Toplam PLACEBO (rastgele) noktası: {len(placebo_records)}\n")

    tdf = pd.DataFrame(trough_records)
    pdf = pd.DataFrame(peak_records)
    plc = pd.DataFrame(placebo_records)

    print("=== TROUGH (EVOL dip yaptı, dönüşe geçti) sonrası ileri getiri ===")
    for name in FORWARD_BARS:
        s = _stats(tdf[name].to_numpy())
        print(f"  {name:4}: {s}")

    print("\n=== PEAK (EVOL tepe yaptı, düşüşe geçti) sonrası ileri getiri (ters hipotez) ===")
    for name in FORWARD_BARS:
        s = _stats(pdf[name].to_numpy())
        print(f"  {name:4}: {s}")

    print("\n=== PLACEBO (rastgele noktalar) sonrası ileri getiri ===")
    for name in FORWARD_BARS:
        s = _stats(plc[name].to_numpy())
        print(f"  {name:4}: {s}")

    print("\n=== SPLIT-PERIOD (TROUGH, 24h) ===")
    mid = tdf["ts"].median()
    for label, sub in [("IS", tdf[tdf["ts"] < mid]), ("OOS", tdf[tdf["ts"] >= mid])]:
        s = _stats(sub["24h"].to_numpy())
        print(f"  {label}: {s}")

    print("\n=== EVOL değeri troughlarda gerçekten düşük mü? (sağlık kontrolü) ===")
    print(f"  trough'larda EVOL ortalaması: {tdf['evol_at'].mean():.1f} (0-100 bandı, düşük olmalı)")
    print(f"  peak'lerde EVOL ortalaması:   {pdf['evol_at'].mean():.1f} (yüksek olmalı)")


if __name__ == "__main__":
    run()
