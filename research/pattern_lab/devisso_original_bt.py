"""
devisso_original_bt.py — Hoca'nın ORİJİNAL Devis'So ALGO Pro Pine Script
formülünün sadık Python portu ve testi.

Kaynak: /Users/yusuf/Downloads/Devis'So ALGO Pro.txt (31 Tem 2026 kullanıcı
tarafından bulundu). Projedeki ERSI (signal_processor.py::_compute_devisso_score)
bu formülün GERÇEK bir portu DEĞİL — pay/payda yerleri ters, RSI farklı bir
seri üzerinde, sinyal mantığı farklı (bkz. memory:
project_devisso_original_formula_mismatch). Bu script ilk kez orijinal
formülü, orijinal sinyal mantığıyla (EMA kesişimi) test ediyor.

Pine → Python birebir çeviri:
    totalAmount: HİÇ SIFIRLANMAYAN kümülatif toplam
        yukarı bar: totalAmount += (close-close[1])/close[1]*100
        aşağı bar:  totalAmount -= (close[1]-close)/close*100
    consecutiveUp/consecutiveDown: ardışık yukarı/aşağı mum sayacı (yön değişince sıfırlanır)
    rsi2 = RSI(totalAmount, 14)                      ← RSI, fiyatın değil, totalAmount'un üzerinde
    DevissoRatio     = safeDivide(Δrsi2, ΔtotalAmount)
    DevissoRatioup   = safeDivide(Δrsi2, ΔconsecutiveUp)
    DevissoRatiodown = safeDivide(Δrsi2, Δ(consecutiveDown*-1))
    ema1/ema2/ema3 = EMA(7) of up/down/main ratio
    bullishPressure = totalUpAmount >= totalDownAmount
    BUY:  ema1 crossover ema3  AND bullishPressure     AND ema3 >  threshold
    SELL: ema2 crossunder ema3 AND NOT bullishPressure AND ema3 < -threshold
    threshold: Medium hassasiyet (Pine varsayılanı) = 1.0

Kullanım: python -m research.pattern_lab.devisso_original_bt
"""

import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # noqa: E402  pylint: disable=wrong-import-position
from indicators.core import calculate_rsi  # noqa: E402  pylint: disable=wrong-import-position

DAYS = 40
MIN_BARS = 700
EMA_LEN = 7
RSI_LEN = 14
THRESHOLD = 1.0  # Pine "Medium" hassasiyet varsayılanı
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


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Pine'ın safeDivide'ı ile birebir: |denom|<=0.0001 ise 0.0 (NaN değil!)."""
    out = np.zeros_like(num, dtype=float)
    mask = np.abs(den) > 0.0001
    out[mask] = num[mask] / den[mask]
    return out


def _ema(x: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(x).ewm(span=length, adjust=False).mean().to_numpy()


def _compute_devisso_original(g: pd.DataFrame) -> dict[str, np.ndarray]:
    """Bir sembolün tam serisi için orijinal Devis'So göstergesini hesaplar."""
    close = g["close"].to_numpy(float)
    n = len(close)
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan

    up_bar = close > prev_close
    down_bar = close < prev_close

    # totalAmount: kümülatif, hiç sıfırlanmaz
    delta_amount = np.zeros(n)
    delta_amount[up_bar] = (close[up_bar] - prev_close[up_bar]) / prev_close[up_bar] * 100.0
    delta_amount[down_bar] = -(prev_close[down_bar] - close[down_bar]) / close[down_bar] * 100.0
    total_amount = np.cumsum(np.nan_to_num(delta_amount, nan=0.0))
    total_amount[0] = np.nan  # ilk bar tanımsız (prev_close yok)

    # totalUpAmount / totalDownAmount: kümülatif, ayrı ayrı (sadece kendi yönünde artar)
    up_amount_delta = np.where(up_bar, delta_amount, 0.0)
    down_amount_delta = np.where(down_bar, -delta_amount, 0.0)  # pozitif tutmak için işaret çevrildi
    total_up_amount = np.cumsum(up_amount_delta)
    total_down_amount = np.cumsum(down_amount_delta)

    # consecutiveUp / consecutiveDown
    consecutive_up = np.zeros(n)
    consecutive_down = np.zeros(n)
    cu = cd = 0.0
    for i in range(n):
        if up_bar[i]:
            cu += 1
            cd = 0.0
        elif down_bar[i]:
            cd += 1
            cu = 0.0
        else:
            cu = 0.0
            cd = 0.0
        consecutive_up[i] = cu
        consecutive_down[i] = cd

    # rsi2 = RSI(totalAmount, 14) — canlı kodun calculate_rsi'ı (Wilder/EMA, TV uyumlu)
    tmp = g.copy()
    tmp["total_amount"] = total_amount
    try:
        rsi2 = calculate_rsi(tmp, period=RSI_LEN, price_col="total_amount").to_numpy(float)
    except Exception:  # pylint: disable=broad-exception-caught
        rsi2 = np.full(n, np.nan)

    rsi_change = np.diff(rsi2, prepend=np.nan)
    amount_change = np.diff(total_amount, prepend=np.nan)
    up_change = np.diff(consecutive_up, prepend=np.nan)
    down_change = np.diff(consecutive_down * -1, prepend=np.nan)

    valid = np.isfinite(rsi_change) & np.isfinite(amount_change)
    devisso_ratio = np.zeros(n)
    devisso_ratio[valid] = _safe_divide(rsi_change[valid], amount_change[valid])

    valid_up = np.isfinite(rsi_change) & np.isfinite(up_change)
    devisso_up = np.zeros(n)
    devisso_up[valid_up] = _safe_divide(rsi_change[valid_up], up_change[valid_up])

    valid_down = np.isfinite(rsi_change) & np.isfinite(down_change)
    devisso_down = np.zeros(n)
    devisso_down[valid_down] = _safe_divide(rsi_change[valid_down], down_change[valid_down])

    ema1 = _ema(devisso_up, EMA_LEN)     # yukarı momentum
    ema2 = _ema(devisso_down, EMA_LEN)   # aşağı momentum
    ema3 = _ema(devisso_ratio, EMA_LEN)  # ana momentum

    bullish_pressure = total_up_amount >= total_down_amount

    return {
        "ema1": ema1, "ema2": ema2, "ema3": ema3,
        "bullish_pressure": bullish_pressure, "rsi2": rsi2,
    }


def _find_signals(ind: dict, burn_in: int) -> tuple[list[int], list[int]]:
    ema1, ema2, ema3 = ind["ema1"], ind["ema2"], ind["ema3"]
    bp = ind["bullish_pressure"]
    n = len(ema3)

    buys, sells = [], []
    for i in range(max(burn_in, 1), n):
        if not np.isfinite(ema1[i]) or not np.isfinite(ema3[i]) or not np.isfinite(ema1[i - 1]) or not np.isfinite(ema3[i - 1]):
            continue
        bull_cross = ema1[i - 1] <= ema3[i - 1] and ema1[i] > ema3[i]
        if bull_cross and bp[i] and ema3[i] > THRESHOLD:
            buys.append(i)

        if not np.isfinite(ema2[i]) or not np.isfinite(ema2[i - 1]):
            continue
        bear_cross = ema2[i - 1] >= ema3[i - 1] and ema2[i] < ema3[i]
        if bear_cross and (not bp[i]) and ema3[i] < -THRESHOLD:
            sells.append(i)

    return buys, sells


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

    buy_records: list[dict] = []
    sell_records: list[dict] = []
    placebo_records: list[dict] = []
    rng = np.random.default_rng(_PLACEBO_SEED)

    n_syms = 0
    burn_in = 200  # totalAmount/RSI'nin oturması için (ilk ~2 gün @15m atlanır)
    for sym, g in df_all.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1
        c = g["close"].to_numpy(float)
        ts = g["ts"]

        ind = _compute_devisso_original(g)
        buys, sells = _find_signals(ind, burn_in)

        for idx in buys:
            if idx + max(FORWARD_BARS.values()) >= len(c):
                continue
            rec = {"symbol": sym, "ts": ts.iloc[idx], "ema3_at": ind["ema3"][idx]}
            rec.update(_forward_returns(c, idx))
            buy_records.append(rec)

        for idx in sells:
            if idx + max(FORWARD_BARS.values()) >= len(c):
                continue
            rec = {"symbol": sym, "ts": ts.iloc[idx], "ema3_at": ind["ema3"][idx]}
            rec.update(_forward_returns(c, idx))
            sell_records.append(rec)

        valid_range = np.arange(burn_in, len(c) - max(FORWARD_BARS.values()))
        if len(valid_range) > 0 and buys:
            sample = rng.choice(valid_range, size=min(len(buys), len(valid_range)), replace=False)
            for idx in sample:
                rec = {"symbol": sym, "ts": ts.iloc[idx]}
                rec.update(_forward_returns(c, idx))
                placebo_records.append(rec)

    print(f"[tarama] {n_syms} sembol")
    print(f"Toplam BUY sinyali: {len(buy_records)}")
    print(f"Toplam SELL sinyali: {len(sell_records)}")
    print(f"Toplam PLACEBO noktası: {len(placebo_records)}\n")

    bdf = pd.DataFrame(buy_records)
    sdf = pd.DataFrame(sell_records)
    plc = pd.DataFrame(placebo_records)

    if bdf.empty:
        print("Hiç BUY sinyali bulunamadı — formül/eşik gözden geçirilmeli.")
        return

    print("=== BUY (ema1 ema3'ü yukarı kesti) sonrası ileri getiri ===")
    for name in FORWARD_BARS:
        print(f"  {name:4}: {_stats(bdf[name].to_numpy())}")

    if not sdf.empty:
        print("\n=== SELL (ema2 ema3'ü aşağı kesti) sonrası ileri getiri ===")
        for name in FORWARD_BARS:
            print(f"  {name:4}: {_stats(sdf[name].to_numpy())}")

    print("\n=== PLACEBO (rastgele noktalar) sonrası ileri getiri ===")
    for name in FORWARD_BARS:
        print(f"  {name:4}: {_stats(plc[name].to_numpy())}")

    print("\n=== SPLIT-PERIOD (BUY, 24h) ===")
    mid = bdf["ts"].median()
    for label, sub in [("IS", bdf[bdf["ts"] < mid]), ("OOS", bdf[bdf["ts"] >= mid])]:
        print(f"  {label}: {_stats(sub['24h'].to_numpy())}")


if __name__ == "__main__":
    run()
