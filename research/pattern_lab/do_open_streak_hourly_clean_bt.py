"""
Saatlik-açılış "temizlik oranı" testi (22 Tem 2026, kullanıcı fikri).

Hipotez: DO'nun üstünde olmak / DO kırılımı + 3 yeşil mum TEK BAŞINA yeterli
bir "verimlilik" göstergesi değil. Asıl soru: fiyat, günün BAŞINDAN (Daily
Open'dan) şu ana kadar geçen HER saat başı açılışını hiç kırmadan (altına
sarkmadan) geçmiş mi? Ne kadar çok saatlik açılış "temiz" kalmışsa, hareket
o kadar "verimli" (az pullback, basamak-basamak ilerleme) sayılır.

clean_ratio = (kırılmamış saatlik açılış sayısı) / (toplam saatlik açılış sayısı)

ÖNEMLİ DÜZELTME: Bu script Daily Open'ı CANLI SİSTEMLE UYUMLU şekilde
hesaplıyor (gün UTC 00:00'da başlıyor) — önceki do_open_streak backtest'lerinde
(_daily_open ön-kaydırmasız çağrıldığı için) 3 saatlik bir gecikme hatası
vardı, burada düzeltildi.

Kullanım: python -m research.pattern_lab.do_open_streak_hourly_clean_bt
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from indicators.core import calculate_atr
from research.pattern_lab.do_open_streak_full_clean_bt import (
    DAYS,
    FEE_RATE,
    GAUSS_THRESHOLD,
    LIQ_WINDOW_BARS,
    MAX_POSITION_USD,
    MIN_BARS,
    MIN_LIQUIDITY_USD,
    STREAK_TH,
    TARGET_RISK_USD,
    _bad_symbols,
    _conn,
    _fetch,
    _gauss_sum,
    _pullback_ok,
    _simulate_exit,
    _stats,
)

_PLACEBO_ITER = 300
_REGIME_SPLIT = pd.Timestamp("2026-06-08")


def _day_and_hour_marks(ts: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canlı sistemle uyumlu: gün UTC 00:00'da başlar (ön-kaydırma yok, doğrudan
    UTC takvim tarihi). Saat başı = dakika==0 (whole-hour offset altında
    zaman dilimi kaymasından bağımsız, aynı an)."""
    n = len(ts)
    day_idx = ts.dt.floor("D").astype("int64").to_numpy()
    is_new_day = np.zeros(n, dtype=bool)
    is_new_day[0] = True
    is_new_day[1:] = day_idx[1:] != day_idx[:-1]

    day_start = np.zeros(n, dtype=np.int64)
    for i in range(n):
        day_start[i] = i if is_new_day[i] else (day_start[i - 1] if i > 0 else 0)

    is_hour_mark = (ts.dt.minute == 0).to_numpy()
    return is_new_day, is_hour_mark, day_start


def _do_break_gate(o: np.ndarray, c: np.ndarray, daily_open: np.ndarray) -> np.ndarray:
    n = len(c)
    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan
    do_break = (c > daily_open) & (prev_c <= daily_open) & np.isfinite(daily_open)
    is_long = c > o
    gate = np.zeros(n, dtype=bool)
    active = False
    for i in range(n):
        if do_break[i]:
            active = True
        elif not is_long[i]:
            active = False
        gate[i] = active
    return gate


def _detect_events(o, c, gate) -> list[tuple[int, int]]:
    n = len(c)
    is_long = c > o
    count_long = 0
    streak_start = -1
    fired = False
    events = []
    for i in range(n):
        if is_long[i]:
            count_long += 1
            if count_long == 1:
                streak_start = i
                fired = False
        else:
            count_long = 0
            streak_start = -1
            fired = False
            continue
        if not gate[i]:
            continue
        if count_long == STREAK_TH and not fired:
            fired = True
            events.append((streak_start, i))
    return events


def _clean_ratio(o: np.ndarray, low: np.ndarray, is_hour_mark: np.ndarray, day_start: int, trigger_idx: int):
    hour_marks = [i for i in range(day_start, trigger_idx + 1) if is_hour_mark[i]]
    if not hour_marks:
        return None
    clean = 0
    for hm in hour_marks:
        seg = low[hm + 1: trigger_idx + 1]
        if len(seg) == 0 or np.all(seg >= o[hm]):
            clean += 1
    return clean / len(hour_marks), len(hour_marks)


def _collect() -> pd.DataFrame:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    conn.close()
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    df_all = _fetch(bad)
    print(f"{df_all['symbol'].nunique()} sembol, {len(df_all):,} 15m bar ({DAYS} gün)\n")

    records = []
    n_syms = 0
    for sym, g in df_all.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1

        ts = g["ts"]
        o = g["open"].to_numpy(float)
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)
        vol = g["volume"].to_numpy(float)
        usd_vol = vol * c

        is_new_day, is_hour_mark, day_start_arr = _day_and_hour_marks(ts)
        daily_open = np.where(is_new_day, o, np.nan)
        daily_open = pd.Series(daily_open).ffill().to_numpy()

        gate = _do_break_gate(o, c, daily_open)
        events = _detect_events(o, c, gate)
        if not events:
            continue
        atr = calculate_atr(g[["high", "low", "close"]], period=14).to_numpy()

        for streak_start, trig in events:
            if trig + 1 >= len(c):
                continue
            bar1, bar2, bar3 = streak_start, streak_start + 1, streak_start + 2
            if bar3 != trig:
                continue
            start_low = l[bar1]
            long_perc = (h[trig] - start_low) / start_low * 100.0
            gauss_val = _gauss_sum(round(long_perc, 2))
            if gauss_val < GAUSS_THRESHOLD:
                continue
            if not _pullback_ok(h, l, bar1, bar2, bar3):
                continue
            liq_start = max(0, trig - LIQ_WINDOW_BARS)
            avg_liq = float(usd_vol[liq_start:trig].mean()) if trig > liq_start else 0.0
            if avg_liq < MIN_LIQUIDITY_USD:
                continue
            atr_val = atr[trig]
            if not np.isfinite(atr_val) or atr_val <= 0:
                continue

            cr = _clean_ratio(o, l, is_hour_mark, day_start_arr[trig], trig)
            if cr is None:
                continue
            clean_ratio, n_hours = cr

            entry_price = c[trig]
            sl_price = entry_price - 3.0 * atr_val
            sl_dist = entry_price - sl_price
            if sl_dist <= 0:
                continue
            pos = min(TARGET_RISK_USD * entry_price / sl_dist, MAX_POSITION_USD)
            pnl_pct, reason, hold_bars = _simulate_exit(c, l, trig, entry_price, sl_price)
            fee = pos * FEE_RATE * 2
            pnl_usd = pnl_pct / 100 * pos - fee
            records.append({
                "symbol": sym, "ts": ts.iloc[trig], "clean_ratio": clean_ratio, "n_hours": n_hours,
                "pnl_usd": pnl_usd, "pnl_pct": pnl_pct, "reason": reason,
            })

    print(f"analize giren sembol: {n_syms}, toplam olay: {len(records)}\n")
    return pd.DataFrame(records)


def main() -> None:
    df = _collect()
    if df.empty:
        print("Olay yok.")
        return

    print("=== [0] n_hours (streak'e kadar geçen saatlik kontrol noktası) dağılımı ===")
    print(df["n_hours"].describe().to_string())

    print("\n=== [1] Korelasyon (clean_ratio, ham) ===")
    rho, p = spearmanr(df["clean_ratio"], df["pnl_usd"])
    print(f"  rho={rho:+.4f} (p={p:.4f})")

    print("\n=== [2] Çeyrek kırılımı ===")
    try:
        q = pd.qcut(df["clean_ratio"], 4, labels=["1.düşük", "2", "3", "4.yüksek(tam temiz)"], duplicates="drop")
        g = df.groupby(q, observed=True)["pnl_usd"].agg(ort="mean", n="count", toplam="sum")
        print(g.to_string())
    except ValueError:
        print("  çeyrek hesaplanamadı (tekrarlı değer)")

    print("\n=== [3] Tam temiz (clean_ratio==1.0) vs en az bir kırılma (clean_ratio<1.0) ===")
    perfect = df[df["clean_ratio"] >= 0.999]
    broken = df[df["clean_ratio"] < 0.999]
    print(f"  tam temiz : n={len(perfect)}  toplam=${perfect['pnl_usd'].sum():.2f}  ort=${perfect['pnl_usd'].mean():.2f}  "
          f"WR%={_stats(perfect['pnl_pct'].to_numpy()).get('wr','-')}")
    print(f"  kırılmış  : n={len(broken)}  toplam=${broken['pnl_usd'].sum():.2f}  ort=${broken['pnl_usd'].mean():.2f}  "
          f"WR%={_stats(broken['pnl_pct'].to_numpy()).get('wr','-')}")

    real_gap = perfect["pnl_usd"].mean() - broken["pnl_usd"].mean() if len(perfect) and len(broken) else float("nan")
    rng = np.random.default_rng(42)
    labels = (df["clean_ratio"] >= 0.999).to_numpy()
    target = df["pnl_usd"].to_numpy()
    count_ge = 0
    for _ in range(_PLACEBO_ITER):
        shuffled = rng.permutation(labels)
        d = target[shuffled].mean() - target[~shuffled].mean()
        if abs(d) >= abs(real_gap):
            count_ge += 1
    print(f"  placebo: gerçek ort-$ farkı ({real_gap:+.3f}) rastgelede %{count_ge/_PLACEBO_ITER*100:.1f} sıklıkta çıktı")

    if len(perfect) >= 30:
        p_sorted = perfect.sort_values("ts")
        mid = p_sorted["ts"].iloc[len(p_sorted)//2]
        fh = _stats(p_sorted[p_sorted["ts"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(p_sorted[p_sorted["ts"] >= mid]["pnl_usd"].to_numpy())
        print(f"  split-period (tam temiz içi): ilk yarı {fh} | ikinci yarı {sh}")
    else:
        print("  split-period için örneklem yetersiz")


if __name__ == "__main__":
    main()
