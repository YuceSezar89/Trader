"""
do_open_streak — 22 Tem 2026 güncellemelerinin (pullback şartı + pozisyon
tavanı + minimum likidite) TAM ve TEMİZ testi.

`signals/do_open_streak.py` ile BİREBİR AYNI mantık:
  - DO kırılımı → EN AZ 3 ardışık yeşil mum (backtest'te tam replay olduğu
    için "en az 3" pratikte "ilk ulaştığı an" ile aynı — canlıdaki fark
    sadece kaçırılan tarama turlarını tolare etmek, backtest'i etkilemez)
  - Streak'in ilk 3 mumunda pullback şartı: 2. mum 1. mumun fitil-dahil
    yarısının altına sarkmasın, 3. mum 2. mumun altına sarkmasın
  - Gauss-ağırlıklı büyüklük eşiği (4.5)
  - YENİ: MIN_LIQUIDITY_USD ($50k, olay anındaki trailing ~1 günlük ort.
    15m USD hacmi) + MAX_POSITION_USD ($1000) tavanlı volatilite-ayarlı boyut

Çıkış simülasyonu GERÇEKÇİ (temiz sabit-ufuk DEĞİL) — bar-bar SL/timeout
replay: SL=entry-3×ATR (TP yok), 96-bar (24h @ 15m) timeout. Bu, bugün
rsi_cross_live'da SL kaymasını ölçtüğümüz yöntemin aynısı — gerçek $ sonucu
görmek istiyoruz, teorik ileri-getiri değil.

ESKİ mantık (pullback/tavan/likidite YOK, sadece tam-3 + Gauss) ile YAN YANA
karşılaştırılıyor — değişikliklerin gerçek etkisini izole etmek için.

Kullanım: python -m research.pattern_lab.do_open_streak_full_clean_bt
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import psycopg2  # pylint: disable=wrong-import-position

from config import Config  # pylint: disable=wrong-import-position
from indicators.core import calculate_atr  # pylint: disable=wrong-import-position
from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position

DAYS = 90
MIN_BARS = 700
STREAK_TH = 3
GAUSS_THRESHOLD = 4.5
SL_ATR_MULT = 3.0
MAX_HOLD_BARS = 96  # 24h @ 15m
TARGET_RISK_USD = 100.0
MAX_POSITION_USD = 1000.0
MIN_LIQUIDITY_USD = 50_000.0
LIQ_WINDOW_BARS = 96  # trailing ~1 gün
FEE_RATE = 0.0005  # tek yön, iki yönlü uygulanır
_GAP_HOURS_THRESHOLD = 200
_PLACEBO_ITER = 300


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
    """(streak_start_idx, trigger_idx) — trigger_idx = streak 3'e ulaştığı bar."""
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


def _pullback_ok(h, l, bar1, bar2, bar3) -> bool:
    mid1 = (h[bar1] + l[bar1]) / 2.0
    if l[bar2] < mid1:
        return False
    mid2 = (h[bar2] + l[bar2]) / 2.0
    if l[bar3] < mid2:
        return False
    return True


def _gauss_sum(x: float) -> float:
    return x * (x + 1) / 2.0


def _simulate_exit(c, l, entry_idx, entry_price, sl_price) -> tuple[float, str, int]:
    """Bar-bar replay: SL'e (low<=sl_price) veya timeout'a kadar. Döner: (pnl_pct, reason, hold_bars)."""
    n = len(c)
    end = min(entry_idx + MAX_HOLD_BARS, n - 1)
    for j in range(entry_idx + 1, end + 1):
        if l[j] <= sl_price:
            pnl_pct = (sl_price - entry_price) / entry_price * 100.0
            return pnl_pct, "stop_loss", j - entry_idx
    exit_price = c[end]
    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
    return pnl_pct, "timeout", end - entry_idx


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


def _econ(df: pd.DataFrame, days: float) -> dict:
    s = _stats(df["pnl_usd"].to_numpy())
    n = s.get("n", 0)
    per_month = n / days * 30 if days > 0 else 0
    total = df["pnl_usd"].sum()
    return {
        "n": n, "wr": _stats(df["pnl_pct"].to_numpy() if "pnl_pct" in df else df["pnl_usd"].to_numpy()).get("wr", 0),
        "toplam_usd": round(float(total), 2),
        "islem_ay": round(per_month),
        "usd_ay": round(per_month * (total / n if n else 0), 2),
    }


def run() -> None:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    conn.close()
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    df_all = _fetch(bad)
    print(f"{df_all['symbol'].nunique()} sembol, {len(df_all):,} 15m bar ({DAYS} gün)\n")

    new_records = []
    old_records = []
    n_syms = 0
    n_events_raw = 0
    n_pullback_rejected = 0
    n_liquidity_rejected = 0

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

        daily_open, _ = _daily_open(ts, o)
        gate = _do_break_gate(o, c, daily_open)
        events = _detect_events(o, c, gate)

        atr_series = calculate_atr(g[["high", "low", "close"]], period=14)
        atr = atr_series.to_numpy()

        for streak_start, trig in events:
            n_events_raw += 1
            if trig + 1 >= len(c):
                continue
            bar1, bar2, bar3 = streak_start, streak_start + 1, streak_start + 2
            if bar3 != trig:
                continue  # güvenlik: tetikleyici bar streak'in 3.'sü olmalı

            start_low = l[bar1]
            long_perc = (h[trig] - start_low) / start_low * 100.0
            gauss_val = _gauss_sum(round(long_perc, 2))
            if gauss_val < GAUSS_THRESHOLD:
                continue

            atr_val = atr[trig]
            if not np.isfinite(atr_val) or atr_val <= 0:
                continue
            entry_price = c[trig]
            sl_price = entry_price - SL_ATR_MULT * atr_val
            pnl_pct, reason, hold_bars = _simulate_exit(c, l, trig, entry_price, sl_price)

            # ---- ESKİ mantık: pullback/tavan/likidite YOK ----
            sl_dist = entry_price - sl_price
            old_pos = TARGET_RISK_USD * entry_price / sl_dist if sl_dist > 0 else 0
            old_fee = old_pos * FEE_RATE * 2
            old_pnl_usd = pnl_pct / 100 * old_pos - old_fee
            old_records.append({
                "symbol": sym, "ts": ts.iloc[trig], "pnl_pct": pnl_pct,
                "pnl_usd": old_pnl_usd, "position_usd": old_pos, "reason": reason,
            })

            # ---- YENİ mantık: pullback + tavan + likidite ----
            if not _pullback_ok(h, l, bar1, bar2, bar3):
                n_pullback_rejected += 1
                continue
            liq_start = max(0, trig - LIQ_WINDOW_BARS)
            avg_liq = float(usd_vol[liq_start:trig].mean()) if trig > liq_start else 0.0
            if avg_liq < MIN_LIQUIDITY_USD:
                n_liquidity_rejected += 1
                continue
            new_pos = min(old_pos, MAX_POSITION_USD) if sl_dist > 0 else 0
            new_fee = new_pos * FEE_RATE * 2
            new_pnl_usd = pnl_pct / 100 * new_pos - new_fee
            new_records.append({
                "symbol": sym, "ts": ts.iloc[trig], "pnl_pct": pnl_pct,
                "pnl_usd": new_pnl_usd, "position_usd": new_pos, "reason": reason,
                "avg_liquidity_usd": avg_liq, "hold_bars": hold_bars,
            })

    print(f"analize giren sembol: {n_syms}")
    print(f"ham olay (do_break+3-streak+Gauss geçen): {n_events_raw}")
    print(f"pullback şartıyla elenen: {n_pullback_rejected}")
    print(f"likidite eşiğiyle elenen: {n_liquidity_rejected}\n")

    old_df = pd.DataFrame(old_records)
    new_df = pd.DataFrame(new_records)

    days_span = (df_all["ts"].max() - df_all["ts"].min()).total_seconds() / 86400

    print("=" * 78)
    print("ESKİ mantık (tam-3, pullback/tavan/likidite YOK) — gerçekçi SL/timeout replay")
    print("=" * 78)
    if not old_df.empty:
        e = _econ(old_df, days_span)
        print(f"  n={e['n']}  WR%={e['wr']}  toplam=${e['toplam_usd']}  işlem/ay={e['islem_ay']}  $/ay={e['usd_ay']}")
    else:
        print("  veri yok")

    print("\n" + "=" * 78)
    print("YENİ mantık (pullback + $1000 tavan + $50k likidite eşiği)")
    print("=" * 78)
    if new_df.empty:
        print("  veri yok — hiçbir olay yeni filtreleri geçemedi")
        return
    e = _econ(new_df, days_span)
    print(f"  n={e['n']}  WR%={e['wr']}  toplam=${e['toplam_usd']}  işlem/ay={e['islem_ay']}  $/ay={e['usd_ay']}")

    print("\n  -- kapanış sebebine göre --")
    for reason, sub in new_df.groupby("reason"):
        s = _stats(sub["pnl_usd"].to_numpy())
        print(f"    {reason:12} n={s.get('n',0):>4}  ort_usd={sub['pnl_usd'].mean():>8.2f}  toplam=${sub['pnl_usd'].sum():>9.2f}")

    if len(new_df) >= 30:
        print("\n=== Split-period (kronolojik yarı-yarı) ===")
        nd_sorted = new_df.sort_values("ts")
        mid = nd_sorted["ts"].iloc[len(nd_sorted)//2]
        first = nd_sorted[nd_sorted["ts"] < mid]
        second = nd_sorted[nd_sorted["ts"] >= mid]
        print(f"  ilk yarı  : n={len(first)}  toplam=${first['pnl_usd'].sum():.2f}  ort=${first['pnl_usd'].mean():.2f}")
        print(f"  ikinci yarı: n={len(second)}  toplam=${second['pnl_usd'].sum():.2f}  ort=${second['pnl_usd'].mean():.2f}")

        print("\n=== Placebo: YENİ filtrenin geçtiği olaylar rastgele bir alt-küme olsaydı ===")
        rng = np.random.default_rng(42)
        real_mean = new_df["pnl_usd"].mean()
        pool = old_df["pnl_usd"].to_numpy()  # tüm ham (pullback/likidite öncesi) evren
        n_new = len(new_df)
        if len(pool) >= n_new:
            count_ge = 0
            for _ in range(_PLACEBO_ITER):
                sample = rng.choice(pool, size=n_new, replace=False)
                if sample.mean() >= real_mean:
                    count_ge += 1
            print(f"  gerçek ort. $ ({real_mean:+.2f}) — rastgele aynı-boyutlu altkümenin bunu "
                  f"eşitleme/geçme sıklığı: %{count_ge/_PLACEBO_ITER*100:.1f}")
    else:
        print("\n  split-period/placebo için örneklem yetersiz")


if __name__ == "__main__":
    run()
