"""
evol_trough_realistic_bt.py — EVOL dip-giriş fikrinin (evol_trough_bt.py'de
sabit-ufuk ileri getiriyle doğrulanan bulgu) GERÇEKÇİ işlem simülasyonu.

do_open_streak'in disipliniyle: SL=ATR çarpanı, TP=ATR çarpanı, maksimum
tutma süresi (timeout), bar-bar replay (SL/TP hangisi önce vurursa),
komisyon dahil $ ekonomisi, split-period IS/OOS, PLACEBO kıyası (rastgele
giriş noktaları, AYNI çıkış mekaniğiyle — trough'un kendisi mi değer katıyor
yoksa bu SL/TP/timeout yapısı her yerde mi kâr ediyor sorusunu ayırt eder).

Long-only (kullanıcı tercihi: "kovalama değil, dipten giriş").

Kullanım: python -m research.pattern_lab.evol_trough_realistic_bt
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from indicators.core import calculate_atr  # noqa: E402  pylint: disable=wrong-import-position

import research.pattern_lab.evol_trough_bt as base  # noqa: E402  pylint: disable=wrong-import-position

SL_ATR_MULT = 3.0
TP_ATR_MULT = 2.0
TIMEOUT_VARIANTS = {"8h": 32, "24h": 96}  # @15m
TARGET_RISK_USD = 100.0
MAX_POSITION_USD = 1000.0
FEE_RATE = 0.0005  # tek yön, iki yönlü uygulanır
_PLACEBO_SEED = 42


def _simulate_exit(
    c: np.ndarray, l: np.ndarray, h: np.ndarray, entry_idx: int,
    entry_price: float, sl_price: float, tp_price: float, max_hold: int,
) -> tuple[float, str, int]:
    """Bar-bar replay: SL/TP hangisi önce vurursa (aynı barda ikisi de
    vurursa SL öncelikli — muhafazakâr varsayım) veya timeout. Döner:
    (pnl_pct, reason, hold_bars)."""
    n = len(c)
    end = min(entry_idx + max_hold, n - 1)
    for j in range(entry_idx + 1, end + 1):
        hit_sl = l[j] <= sl_price
        hit_tp = h[j] >= tp_price
        if hit_sl:
            return (sl_price - entry_price) / entry_price * 100.0, "stop_loss", j - entry_idx
        if hit_tp:
            return (tp_price - entry_price) / entry_price * 100.0, "take_profit", j - entry_idx
    exit_price = c[end]
    return (exit_price - entry_price) / entry_price * 100.0, "timeout", end - entry_idx


def _econ_stats(df: pd.DataFrame, days: float) -> dict:
    if df.empty:
        return {"n": 0}
    n = len(df)
    wr = round(float((df["pnl_usd"] > 0).mean() * 100), 1)
    total = df["pnl_usd"].sum()
    gross_win = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum()
    gross_loss = -df.loc[df["pnl_usd"] < 0, "pnl_usd"].sum()
    pf = round(float(gross_win / gross_loss), 3) if gross_loss > 0 else float("inf")
    per_month = n / days * 30 if days > 0 else 0
    return {
        "n": n, "wr": wr, "pf": pf,
        "toplam_usd": round(float(total), 2),
        "islem_ay": round(per_month),
        "usd_ay": round(per_month * (total / n if n else 0), 2),
        "ort_hold_bar": round(float(df["hold_bars"].mean()), 1),
        "sl_orani": round(float((df["reason"] == "stop_loss").mean() * 100), 1),
        "tp_orani": round(float((df["reason"] == "take_profit").mean() * 100), 1),
        "timeout_orani": round(float((df["reason"] == "timeout").mean() * 100), 1),
    }


def run() -> None:
    conn = base._conn()  # pylint: disable=protected-access
    bad = base._bad_symbols(conn.cursor())  # pylint: disable=protected-access
    conn.close()
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    df_all = base._fetch(bad)  # pylint: disable=protected-access
    print(f"{df_all['symbol'].nunique()} sembol, {len(df_all):,} 15m bar ({base.DAYS} gün)\n")

    results: dict[str, list[dict]] = {f"trough_{k}": [] for k in TIMEOUT_VARIANTS}
    results.update({f"placebo_{k}": [] for k in TIMEOUT_VARIANTS})
    rng = np.random.default_rng(_PLACEBO_SEED)

    n_syms = 0
    n_troughs_total = 0
    for sym, g in df_all.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < base.MIN_BARS:
            continue
        n_syms += 1
        c = g["close"].to_numpy(float)
        l = g["low"].to_numpy(float)
        h = g["high"].to_numpy(float)
        ts = g["ts"]

        atr = calculate_atr(g, period=14).to_numpy(float)
        score = base._evol_series(g)  # pylint: disable=protected-access
        troughs, _peaks = base._find_troughs_and_peaks(score)  # pylint: disable=protected-access
        n_troughs_total += len(troughs)

        max_hold_overall = max(TIMEOUT_VARIANTS.values())
        valid_troughs = [
            idx for idx in troughs
            if idx + max_hold_overall < len(c) and np.isfinite(atr[idx]) and atr[idx] > 0
        ]

        for idx in valid_troughs:
            entry_price = c[idx]
            atr_val = atr[idx]
            sl_price = entry_price - SL_ATR_MULT * atr_val
            tp_price = entry_price + TP_ATR_MULT * atr_val
            sl_dist = entry_price - sl_price
            position_usd = min(TARGET_RISK_USD * entry_price / sl_dist, MAX_POSITION_USD)

            for label, max_hold in TIMEOUT_VARIANTS.items():
                pnl_pct, reason, hold_bars = _simulate_exit(
                    c, l, h, idx, entry_price, sl_price, tp_price, max_hold
                )
                pnl_usd = position_usd * (pnl_pct / 100.0) - position_usd * FEE_RATE * 2
                results[f"trough_{label}"].append(
                    {"symbol": sym, "ts": ts.iloc[idx], "pnl_pct": pnl_pct, "pnl_usd": pnl_usd,
                     "reason": reason, "hold_bars": hold_bars}
                )

        # placebo: aynı sayıda rastgele nokta, AYNI ATR-tabanlı SL/TP/timeout mekaniği
        valid_range = np.arange(20, len(c) - max_hold_overall)
        if len(valid_range) > 0 and valid_troughs:
            sample = rng.choice(
                valid_range, size=min(len(valid_troughs), len(valid_range)), replace=False
            )
            for idx in sample:
                if not (np.isfinite(atr[idx]) and atr[idx] > 0):
                    continue
                entry_price = c[idx]
                atr_val = atr[idx]
                sl_price = entry_price - SL_ATR_MULT * atr_val
                tp_price = entry_price + TP_ATR_MULT * atr_val
                sl_dist = entry_price - sl_price
                position_usd = min(TARGET_RISK_USD * entry_price / sl_dist, MAX_POSITION_USD)
                for label, max_hold in TIMEOUT_VARIANTS.items():
                    pnl_pct, reason, hold_bars = _simulate_exit(
                        c, l, h, idx, entry_price, sl_price, tp_price, max_hold
                    )
                    pnl_usd = position_usd * (pnl_pct / 100.0) - position_usd * FEE_RATE * 2
                    results[f"placebo_{label}"].append(
                        {"symbol": sym, "ts": ts.iloc[idx], "pnl_pct": pnl_pct, "pnl_usd": pnl_usd,
                         "reason": reason, "hold_bars": hold_bars}
                    )

    print(f"[tarama] {n_syms} sembol, {n_troughs_total} ham trough (ATR/pencere filtresi öncesi)\n")
    print(f"SL={SL_ATR_MULT}xATR  TP={TP_ATR_MULT}xATR  risk=${TARGET_RISK_USD} (maks ${MAX_POSITION_USD})  fee={FEE_RATE*2*100:.2f}%\n")

    for label in TIMEOUT_VARIANTS:
        tdf = pd.DataFrame(results[f"trough_{label}"])
        pdf = pd.DataFrame(results[f"placebo_{label}"])
        print(f"=== TIMEOUT={label} ===")
        print(f"  TROUGH : {_econ_stats(tdf, base.DAYS)}")
        print(f"  PLACEBO: {_econ_stats(pdf, base.DAYS)}")

        if not tdf.empty:
            mid = tdf["ts"].median()
            is_df = tdf[tdf["ts"] < mid]
            oos_df = tdf[tdf["ts"] >= mid]
            print(f"  Split IS : {_econ_stats(is_df, base.DAYS / 2)}")
            print(f"  Split OOS: {_econ_stats(oos_df, base.DAYS / 2)}")
        print()


if __name__ == "__main__":
    run()
