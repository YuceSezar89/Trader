"""
do_open_streak + TF Hizalanma (1h+4h HA) karşılaştırması — 29 Tem 2026.

Giriş anında (SADECE kapanmış barlarla, look-ahead yok — signals/tf_alignment_gate.py
ile AYNI closed-bar disiplini) 1h VE 4h Heikin Ashi'nin ikisi de Long yönüyle
uyumluysa "hizalı", değilse "hizasız" etiketi. Sabit N-bar ileri getiri
(SL/TP mekaniği yok — feedback_clean_forward_return_target prensibi: hedef
PnL değil sabit ileri getiri) + giriş anı VPMV bileşenleri (utils/vpmv.py::
compute_components, canlı sistemle BİREBİR aynı fonksiyon) karşılaştırılır.

Kullanım: python -m research.pattern_lab.do_open_streak_tf_alignment_bt
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import psycopg2  # pylint: disable=wrong-import-position

from config import Config  # pylint: disable=wrong-import-position
from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position
from utils.vpmv import compute_components  # pylint: disable=wrong-import-position

DAYS = 35
HTF_LOOKBACK_DAYS = DAYS + 15  # HA rengi stabilize olsun diye ekstra geçmiş
MIN_BARS = 700
STREAK_TH = 3
GAUSS_THRESHOLD = 4.5
_GAP_HOURS_THRESHOLD = 200
_FORWARD_BARS = {"4h": 16, "12h": 48, "24h": 96}


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


def _fetch(view: str, days: int, exclude: set[str]) -> pd.DataFrame:
    conn = _conn()
    q = f"""
        SELECT symbol, bucket AS ts, open, high, low, close, volume, buy_volume, sell_volume
        FROM {view}
        WHERE bucket > NOW() - INTERVAL '{days} days'
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


def _ha_bull_series(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Her bar için (bucket KAPANIŞ zamanı ms, HA boğa mı) — HA rengi bar'ın
    kendisi ve öncesine bağlı (recursive-backward), bu yüzden tam seriyi bir
    kere hesaplamak sonradan-bakma sayılmaz: bar[i]'nin rengi bar[i]'den
    sonraki hiçbir veriye bağlı değil."""
    o = g["open"].to_numpy(float)
    h = g["high"].to_numpy(float)
    l = g["low"].to_numpy(float)
    c = g["close"].to_numpy(float)
    n = len(c)
    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty(n)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    bull = ha_close > ha_open
    close_ts_ms = g["ts"].astype("int64").to_numpy() // 1_000_000 + _bucket_ms(g)
    return close_ts_ms, bull


def _bucket_ms(g: pd.DataFrame) -> int:
    ts = g["ts"].astype("int64").to_numpy() // 1_000_000
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    return int(np.median(diffs)) if len(diffs) else 0


def _last_closed_bull(close_ts_ms: np.ndarray, bull: np.ndarray, at_ms: int) -> "bool | None":
    idx = np.searchsorted(close_ts_ms, at_ms, side="right") - 1
    if idx < 0:
        return None
    return bool(bull[idx])


def run() -> None:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    conn.close()
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    df15 = _fetch("cagg_15m", DAYS, bad)
    df1h = _fetch("cagg_1h", HTF_LOOKBACK_DAYS, bad)
    df4h = _fetch("cagg_4h", HTF_LOOKBACK_DAYS, bad)
    print(f"{df15['symbol'].nunique()} sembol, {len(df15):,} 15m bar ({DAYS} gün)\n")

    ha1h = {sym: _ha_bull_series(g.sort_values("ts").reset_index(drop=True)) for sym, g in df1h.groupby("symbol")}
    ha4h = {sym: _ha_bull_series(g.sort_values("ts").reset_index(drop=True)) for sym, g in df4h.groupby("symbol")}

    records = []
    n_syms = 0
    n_events_raw = 0
    n_pullback_rejected = 0

    for sym, g in df15.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS or sym not in ha1h or sym not in ha4h:
            continue
        n_syms += 1

        ts = g["ts"]
        ts_ms = ts.astype("int64").to_numpy() // 1_000_000
        o = g["open"].to_numpy(float)
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)

        daily_open, _ = _daily_open(pd.to_datetime(ts) + pd.Timedelta(hours=3), o)
        gate = _do_break_gate(o, c, daily_open)
        events = _detect_events(o, c, gate)
        n_events_raw += len(events)

        close1h, bull1h = ha1h[sym]
        close4h, bull4h = ha4h[sym]

        for streak_start, trigger_idx in events:
            bar1, bar2, bar3 = streak_start, streak_start + 1, streak_start + 2
            if bar3 >= len(g):
                continue
            if not _pullback_ok(h, l, bar1, bar2, bar3):
                n_pullback_rejected += 1
                continue

            start_low = l[bar1]
            long_perc = (h[trigger_idx] - start_low) / start_low * 100.0
            if _gauss_sum(round(long_perc, 2)) < GAUSS_THRESHOLD:
                continue

            entry_idx = trigger_idx
            entry_ts_ms = int(ts_ms[entry_idx])
            entry_price = c[entry_idx]

            b1h = _last_closed_bull(close1h, bull1h, entry_ts_ms)
            b4h = _last_closed_bull(close4h, bull4h, entry_ts_ms)
            if b1h is None or b4h is None:
                continue
            aligned = bool(b1h and b4h)

            window = g.iloc[: entry_idx + 1].tail(500)
            if len(window) < 60:
                continue
            try:
                vol_s, mom_s, vlt_s, prc_s = compute_components(window, "Long")
            except Exception:  # pylint: disable=broad-exception-caught
                continue

            rec = {
                "symbol": sym, "ts": ts.iloc[entry_idx], "entry_idx": entry_idx,
                "aligned": aligned, "vol": vol_s, "mom": mom_s, "vlt": vlt_s, "prc": prc_s,
                "combined": (vol_s + mom_s + vlt_s + prc_s) / 4.0,
            }
            for name, bars in _FORWARD_BARS.items():
                j = min(entry_idx + bars, len(c) - 1)
                rec[f"fwd_{name}"] = (c[j] - entry_price) / entry_price * 100.0
            records.append(rec)

    print(f"[tarama] {n_syms} sembol, {n_events_raw} ham olay, {n_pullback_rejected} pullback'te elendi")

    res = pd.DataFrame(records)
    if res.empty:
        print("Hiç işlem bulunamadı.")
        return
    print(f"\nToplam do_open_streak sinyali: {len(res)}  ({res['symbol'].nunique()} sembol)\n")

    for label, grp in res.groupby("aligned"):
        tag = "HİZALI (1h+4h boğa)" if label else "HİZASIZ"
        print(f"=== {tag} — n={len(grp)} ===")
        for name in _FORWARD_BARS:
            col = f"fwd_{name}"
            wr = (grp[col] > 0).mean() * 100
            print(f"  {name:4} ileri getiri: ort={grp[col].mean():+.2f}%  medyan={grp[col].median():+.2f}%  WR={wr:.1f}%")
        print(
            f"  VPMV girişte: vol={grp['vol'].mean():.1f} mom={grp['mom'].mean():.1f} "
            f"vlt={grp['vlt'].mean():.1f} prc={grp['prc'].mean():.1f} combined={grp['combined'].mean():.1f}"
        )
        print()

    # Split-period (tarih ortası IS/OOS)
    mid = res["ts"].median()
    res["period"] = np.where(res["ts"] < mid, "IS", "OOS")
    print("--- Split-period (24h ileri getiri, WR) ---")
    for (label, period), grp in res.groupby(["aligned", "period"]):
        tag = "HİZALI" if label else "HİZASIZ"
        wr = (grp["fwd_24h"] > 0).mean() * 100
        print(f"  {tag:8} {period}: n={len(grp):4}  ort={grp['fwd_24h'].mean():+.2f}%  WR={wr:.1f}%")


if __name__ == "__main__":
    run()
