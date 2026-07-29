"""
do_open_streak sinyallerine "all_up" filtresi (22 Tem 2026, kullanıcı isteği).

Bugünün en sağlam bulgusu olan all_up mantığını (RSI_Cross için doğrulanmış:
vol/mom/volat/price skorlarının HEPSİ bir önceki AYNI SEMBOL sinyaline göre
artmış mı) do_open_streak'in kendi sinyal akışına uyguluyoruz.

utils/vpmv.py::compute_components ile BİREBİR aynı canlı formül (yönlü
buy_volume/sell_volume — cagg_15m'de zaten var, ekstra 1m-agregasyon
gerekmiyor çünkü do_open_streak zaten 15m'in kendi verisiyle çalışıyor).

Kullanım: python -m research.pattern_lab.do_open_streak_all_up_bt
"""

import numpy as np
import pandas as pd

from indicators.core import calculate_atr
from research.pattern_lab.do_open_streak_full_clean_bt import (
    DAYS,
    FEE_RATE,
    GAUSS_THRESHOLD,
    LIQ_WINDOW_BARS,
    MAX_POSITION_USD,
    MIN_BARS,
    MIN_LIQUIDITY_USD,
    TARGET_RISK_USD,
    _bad_symbols,
    _conn,
    _fetch,
    _gauss_sum,
    _pullback_ok,
    _simulate_exit,
    _stats,
)
from research.pattern_lab.do_open_streak_hourly_clean_bt import _day_and_hour_marks, _do_break_gate, _detect_events
from utils.vpmv import compute_components

_PLACEBO_ITER = 300
_COMPONENT_WINDOW = 60


def _fetch_with_buy_volume(exclude: set[str]) -> pd.DataFrame:
    conn = _conn()
    q = f"""
        SELECT symbol, bucket AS ts, open, high, low, close, volume, buy_volume
        FROM cagg_15m
        WHERE bucket > NOW() - INTERVAL '{DAYS} days'
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn)
    conn.close()
    if exclude:
        df = df[~df["symbol"].isin(exclude)].reset_index(drop=True)
    return df


def _collect() -> pd.DataFrame:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    conn.close()
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    df_all = _fetch_with_buy_volume(bad)
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
        buy_vol = g["buy_volume"].fillna(0).to_numpy(float)
        sell_vol = vol - buy_vol
        usd_vol = vol * c

        is_new_day, _, _ = _day_and_hour_marks(ts)
        daily_open = np.where(is_new_day, o, np.nan)
        daily_open = pd.Series(daily_open).ffill().to_numpy()

        gate = _do_break_gate(o, c, daily_open)
        events = _detect_events(o, c, gate)
        if not events:
            continue
        atr = calculate_atr(g[["high", "low", "close"]], period=14).to_numpy()

        for streak_start, trig in events:
            if trig + 1 >= len(c) or trig < _COMPONENT_WINDOW:
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

            window = g.iloc[trig - _COMPONENT_WINDOW + 1: trig + 1].copy()
            window["buy_volume"] = buy_vol[trig - _COMPONENT_WINDOW + 1: trig + 1]
            window["sell_volume"] = sell_vol[trig - _COMPONENT_WINDOW + 1: trig + 1]
            try:
                vol_s, mom_s, vlt_s, prc_s = compute_components(window, "Long", volume_mode="real")
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            if any(pd.isna(v) for v in (vol_s, mom_s, vlt_s, prc_s)):
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
            records.append({
                "symbol": sym, "ts": ts.iloc[trig],
                "vol": vol_s, "mom": mom_s, "volat": vlt_s, "price": prc_s,
                "pnl_usd": pnl_usd, "pnl_pct": pnl_pct, "reason": reason,
            })

    print(f"analize giren sembol: {n_syms}, toplam olay: {len(records)}\n")
    return pd.DataFrame(records)


def _add_all_up(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    for col in ("vol", "mom", "volat", "price"):
        df[f"d_{col}"] = df.groupby("symbol")[col].diff()
    df = df.dropna(subset=["d_vol", "d_mom", "d_volat", "d_price"])
    df["all_up"] = (df["d_vol"] > 0) & (df["d_mom"] > 0) & (df["d_volat"] > 0) & (df["d_price"] > 0)
    return df


def _deep_validate(label: str, group: pd.DataFrame, rest: pd.DataFrame) -> None:
    print(f"\n  -- {label} (n={len(group)}) vs geri kalan (n={len(rest)}) --")
    if len(group) == 0:
        print("    örneklem yok")
        return
    s_g = _stats(group["pnl_usd"].to_numpy())
    s_r = _stats(rest["pnl_usd"].to_numpy())
    print(f"    {label}: n={s_g['n']} toplam=${group['pnl_usd'].sum():.2f} ort=${group['pnl_usd'].mean():.2f} WR%={s_g.get('wr','-')}")
    print(f"    geri kalan: n={s_r['n']} toplam=${rest['pnl_usd'].sum():.2f} ort=${rest['pnl_usd'].mean():.2f} WR%={s_r.get('wr','-')}")

    real_gap = group["pnl_usd"].mean() - rest["pnl_usd"].mean()
    combo = pd.concat([group.assign(_g=True), rest.assign(_g=False)])
    rng = np.random.default_rng(42)
    labels = combo["_g"].to_numpy()
    target = combo["pnl_usd"].to_numpy()
    count_ge = 0
    for _ in range(_PLACEBO_ITER):
        shuffled = rng.permutation(labels)
        d = target[shuffled].mean() - target[~shuffled].mean()
        if abs(d) >= abs(real_gap):
            count_ge += 1
    print(f"    placebo: gerçek ort-$ farkı ({real_gap:+.3f}) rastgelede %{count_ge/_PLACEBO_ITER*100:.1f} sıklıkta çıktı")

    if len(group) >= 30:
        g_sorted = group.sort_values("ts")
        mid = g_sorted["ts"].iloc[len(g_sorted)//2]
        fh = _stats(g_sorted[g_sorted["ts"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(g_sorted[g_sorted["ts"] >= mid]["pnl_usd"].to_numpy())
        print(f"    split-period: ilk yarı {fh} | ikinci yarı {sh}")
    else:
        print("    split-period için örneklem yetersiz")


def main() -> None:
    raw = _collect()
    if raw.empty:
        print("Olay yok.")
        return
    df = _add_all_up(raw)
    print(f"[delta hesaplandı] {len(df)} sinyal (bir önceki aynı sembol olayı olanlar)\n")

    _deep_validate("all_up=True", df[df["all_up"]], df[~df["all_up"]])


if __name__ == "__main__":
    main()
