"""
do_open_streak + TA-kovalama birleşimi, TA döngüsü 24 SAAT ve GÜNÜN AÇILIŞIYLA
(UTC 00:00 = gece 3'te İstanbul yerel saati) hizalı — kullanıcı önerisi
(24 Tem 2026): "open streak ile TA birleştirsek, sıfırlama gece 3'te olsa."

Neden UTC 00:00: do_open_streak'in KENDİ Daily Open sınırı da UTC 00:00
(canlı sistemde ts +3h ön-kaydırılıp DO_HOUR=3 çıkarılınca net UTC-gece
yarısına iptal oluyor — [[project_do_open_streak_22_23tem]]). Yani TA'yı
24h döngüyle UTC 00:00'da sıfırlamak, do_open_streak'in kendi "gün"
tanımıyla TAM HİZALI oluyor — bu script'te BİLEREK bu hizalama kullanıldı.

DİKKAT: do_open_streak_percentile_bt.py'nin KENDİSİ hâlâ eski 3-saatlik
Daily-Open bug'ını taşıyordu (_daily_open ham UTC ts ile çağrılıyordu,
gerçek sınır UTC 03:00 çıkıyordu) — burada DÜZELTİLMİŞ günlük-açılış
mantığı (do_open_streak_hourly_clean_bt.py::_day_and_hour_marks ile aynı,
UTC 00:00) kullanıldı.

Kullanım: python -m research.pattern_lab.do_open_streak_ta_combo_bt
"""

import os

import numpy as np
import pandas as pd

from research.pattern_lab.concentration_diagnostics import summarize
from research.pattern_lab.do_open_streak_full_clean_bt import (
    DAYS,
    FEE_RATE,
    LIQ_WINDOW_BARS,
    MAX_POSITION_USD,
    MIN_BARS,
    MIN_LIQUIDITY_USD,
    TARGET_RISK_USD,
    _bad_symbols,
    _conn,
    _detect_events,
    _do_break_gate,
    _econ,
    _fetch,
    _gauss_sum,
    _pullback_ok,
    _simulate_exit,
    _stats,
)
from indicators.core import calculate_atr
from research.pattern_lab.rsi_cross_ta_percentile_bt import _percentile_at, _slope_at

_CYCLE_MS_24H = 24 * 3600 * 1000  # UTC 00:00 = gece 3'te İstanbul yerel
_TA_TFS = ["1h", "4h"]
_TA_TABLE = {"1h": "cagg_1h", "4h": "cagg_4h"}
_PLACEBO_ITER = 300
_KOVALAMA_THRESHOLDS = [55, 65, 75, 80, 85, 90]
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "_cache_do_open_streak_ta_combo.parquet")


def _day_marks_utc_midnight(ts: pd.Series) -> np.ndarray:
    """Düzeltilmiş günlük-açılış sınırı: UTC 00:00 (canlı sistemle uyumlu,
    do_open_streak_hourly_clean_bt.py::_day_and_hour_marks ile aynı)."""
    day_idx = ts.dt.floor("D").astype("int64").to_numpy()
    is_new_day = np.zeros(len(ts), dtype=bool)
    is_new_day[0] = True
    is_new_day[1:] = day_idx[1:] != day_idx[:-1]
    return is_new_day


def _total_amount_24h(ts: pd.Series, close: np.ndarray) -> np.ndarray:
    t_ms = ts.astype("int64").to_numpy() // 10**6
    cycle = (t_ms // _CYCLE_MS_24H) * _CYCLE_MS_24H
    n = len(close)
    net = np.zeros(n)
    for i in range(1, n):
        if cycle[i] != cycle[i - 1]:
            net[i] = 0.0
        else:
            p = close[i - 1]
            net[i] = net[i - 1] + ((close[i] - p) / p * 100.0 if p else 0.0)
    return net


def _collect_events_with_ta() -> pd.DataFrame:
    conn = _conn()
    cur = conn.cursor()
    bad = _bad_symbols(cur)
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    df_all = _fetch(bad)
    print(f"{df_all['symbol'].nunique()} sembol, {len(df_all):,} 15m bar ({DAYS} gün)\n")

    records = []
    n_syms = 0
    for sym, g in df_all.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue

        ts = g["ts"]
        o = g["open"].to_numpy(float)
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)
        vol = g["volume"].to_numpy(float)
        usd_vol = vol * c

        is_new_day = _day_marks_utc_midnight(ts)
        daily_open = np.where(is_new_day, o, np.nan)
        daily_open = pd.Series(daily_open).ffill().to_numpy()

        gate = _do_break_gate(o, c, daily_open)
        events = _detect_events(o, c, gate)
        if not events:
            continue
        atr = calculate_atr(g[["high", "low", "close"]], period=14).to_numpy()

        # TA (1h/4h, 24h döngü) serilerini bu sembol için bir kez çek
        ta_series = {}
        ok_series = True
        for tf in _TA_TFS:
            cur.execute(
                f"SELECT bucket, close FROM {_TA_TABLE[tf]} WHERE symbol=%s ORDER BY bucket ASC", (sym,)
            )
            rows = cur.fetchall()
            if not rows:
                ok_series = False
                break
            dta = pd.DataFrame(rows, columns=["bucket", "close"]).astype({"close": float})
            net_arr = _total_amount_24h(dta["bucket"], dta["close"].to_numpy())
            ta_series[tf] = (dta["bucket"].to_numpy(), net_arr)
        if not ok_series:
            continue
        n_syms += 1

        for streak_start, trig in events:
            if trig + 1 >= len(c):
                continue
            bar1, bar2, bar3 = streak_start, streak_start + 1, streak_start + 2
            if bar3 != trig:
                continue
            start_low = l[bar1]
            long_perc = (h[trig] - start_low) / start_low * 100.0
            gauss_val = _gauss_sum(round(long_perc, 2))
            if gauss_val < 1.0:
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
            entry_price = c[trig]
            sl_price = entry_price - 3.0 * atr_val
            sl_dist = entry_price - sl_price
            if sl_dist <= 0:
                continue
            pos = min(TARGET_RISK_USD * entry_price / sl_dist, MAX_POSITION_USD)
            pnl_pct, reason, hold_bars = _simulate_exit(c, l, trig, entry_price, sl_price)
            fee = pos * FEE_RATE * 2
            pnl_usd = pnl_pct / 100 * pos - fee

            trig_ts = np.datetime64(ts.iloc[trig])
            rec = {
                "symbol": sym, "ts": ts.iloc[trig], "gauss_val": gauss_val,
                "long_perc": long_perc, "pnl_usd": pnl_usd, "pnl_pct": pnl_pct, "reason": reason,
            }
            ok = True
            for tf, (b_arr, net_arr) in ta_series.items():
                j = np.searchsorted(b_arr, trig_ts, side="right") - 1
                if j < 0:
                    ok = False
                    break
                rec[f"pct_{tf}"] = _percentile_at(net_arr, j)
                rec[f"slope_{tf}"] = _slope_at(net_arr, j)
            if not ok:
                continue
            records.append(rec)

    conn.close()
    print(f"analize giren sembol: {n_syms}, toplam olay: {len(records)}\n")
    return pd.DataFrame(records)


def _collect_cached() -> pd.DataFrame:
    if os.path.exists(_CACHE_PATH):
        print(f"[cache] {_CACHE_PATH} kullanılıyor")
        return pd.read_parquet(_CACHE_PATH)
    df = _collect_events_with_ta()
    if not df.empty:
        df.to_parquet(_CACHE_PATH)
    return df


def main() -> None:
    df = _collect_cached()
    if df.empty:
        print("Olay yok.")
        return
    df = df.dropna(subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h"]).reset_index(drop=True)
    print(f"[TA hesaplanabilir popülasyon] n={len(df)}\n")

    days_span = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400

    print("=== [0] Baseline: mevcut sistem (GAUSS_THRESHOLD>=4.5, TA'sız) ===")
    base = df[df["gauss_val"] >= 4.5]
    e = _econ(base, days_span)
    print(f"  n={e['n']}  WR%={e['wr']}  toplam=${e['toplam_usd']}  işlem/ay={e['islem_ay']}  $/ay={e['usd_ay']}")

    print("\n=== [1] TA-kovalama eşiği taraması (gauss>=4.5 popülasyonu İÇİNDE) ===")
    print(f"{'esik':>6} | {'n':>6} {'WR%':>6} {'toplam$':>10} {'$/ay':>10}")
    print("-" * 50)
    for th in _KOVALAMA_THRESHOLDS:
        kov = ((base["pct_1h"] >= th) & (base["slope_1h"] > 0)) | ((base["pct_4h"] >= th) & (base["slope_4h"] > 0))
        sub = base[kov]
        es = _econ(sub, days_span)
        print(f"{th:>6} | {es['n']:>6} {es['wr'] if es['wr'] is not None else '-':>6} "
              f"{es['toplam_usd'] if es['toplam_usd'] is not None else '-':>10} "
              f"{es['usd_ay'] if es['usd_ay'] is not None else '-':>10}")

    print("\n=== [2] Aynı TA-kovalama filtresi, gauss eşiği OLMADAN (sadece pullback+likidite+TA) ===")
    for th in _KOVALAMA_THRESHOLDS:
        kov = ((df["pct_1h"] >= th) & (df["slope_1h"] > 0)) | ((df["pct_4h"] >= th) & (df["slope_4h"] > 0))
        sub = df[kov]
        es = _econ(sub, days_span)
        print(f"{th:>6} | {es['n']:>6} {es['wr'] if es['wr'] is not None else '-':>6} "
              f"{es['toplam_usd'] if es['toplam_usd'] is not None else '-':>10} "
              f"{es['usd_ay'] if es['usd_ay'] is not None else '-':>10}")

    print("\n=== [3] En iyi görünen kombinasyonun tam derin doğrulaması ===")
    best_th, best_usd = None, -1e18
    for th in _KOVALAMA_THRESHOLDS:
        kov = ((df["pct_1h"] >= th) & (df["slope_1h"] > 0)) | ((df["pct_4h"] >= th) & (df["slope_4h"] > 0))
        sub = df[kov]
        es = _econ(sub, days_span)
        if es.get("n", 0) >= 20 and es.get("usd_ay") is not None and es["usd_ay"] > best_usd:
            best_usd = es["usd_ay"]
            best_th = th

    if best_th is None:
        print("  Hiçbir eşik minimum örneklem şartını sağlamadı.")
        return

    kov = ((df["pct_1h"] >= best_th) & (df["slope_1h"] > 0)) | ((df["pct_4h"] >= best_th) & (df["slope_4h"] > 0))
    group, rest = df[kov], df[~kov]
    print(f"  seçilen eşik: {best_th} (n={len(group)}, ${best_usd}/ay)")
    print(f"  grup   : {_stats(group['pnl_usd'].to_numpy())}")
    print(f"  geri kalan: {_stats(rest['pnl_usd'].to_numpy())}")

    real_gap = group["pnl_usd"].mean() - rest["pnl_usd"].mean() if len(rest) else float("nan")
    rng = np.random.default_rng(42)
    labels = kov.to_numpy()
    target = df["pnl_usd"].to_numpy()
    count_ge = 0
    for _ in range(_PLACEBO_ITER):
        shuffled = rng.permutation(labels)
        d = target[shuffled].mean() - target[~shuffled].mean()
        if abs(d) >= abs(real_gap):
            count_ge += 1
    print(f"  placebo: gerçek ort-$ farkı ({real_gap:+.3f}) rastgelede %{count_ge/_PLACEBO_ITER*100:.1f} sıklıkta çıktı")

    if len(group) >= 30:
        g_sorted = group.sort_values("ts")
        mid = g_sorted["ts"].iloc[len(g_sorted)//2]
        fh = _stats(g_sorted[g_sorted["ts"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(g_sorted[g_sorted["ts"] >= mid]["pnl_usd"].to_numpy())
        print(f"  split-period: ilk yarı {fh} | ikinci yarı {sh}")

    if len(group) >= 10:
        summarize(f"do_open_streak + TA-kovalama(24h,eşik={best_th})", group["pnl_usd"].to_numpy(), group["ts"], days_span)


if __name__ == "__main__":
    main()
