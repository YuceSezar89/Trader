"""
Eğim (slope) hesabındaki "2 bar" gerçekten en iyi mi? — HTML kaynağından
hiç sorgulanmadan alınmış bir parametre (24 Tem 2026, kullanıcı isteği).
slope_N = net_ta[şimdi] - net_ta[N bar önce], N in [1,2,3,4,5,7,10] taranıyor
— TA-base(pct>=55) + kovalama(pct>=90 & slope_N>0) kombinasyonunun PF'si
IS/OOS disiplinli karşılaştırılıyor.

Kullanım: python -m research.pattern_lab.rsi_cross_slope_lookback_bt
"""

import os

import numpy as np
import pandas as pd

from research.pattern_lab.concentration_diagnostics import summarize
from research.pattern_lab.do_open_streak_full_clean_bt import _bad_symbols, _conn
from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up, _classify_candle
from research.pattern_lab.rsi_cross_ta_alignment_bt import _deep_validate, _total_amount_series
from research.pattern_lab.rsi_cross_ta_percentile_bt import _percentile_at
from utils.vpmv import compute_components

_FORWARD_BARS = 24
_COMPONENT_WINDOW = 60
_TFS = ["1h", "4h"]
_TABLE = {"1h": "cagg_1h", "4h": "cagg_4h"}
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "_cache_rsi_cross_slope_lookback.parquet")
_LOOKBACKS = [1, 2, 3, 4, 5, 7, 10]
_TA_BASE_TH = 55
_KOVALAMA_TH = 90
_MIN_N = 30
_PLACEBO_ITER = 300


def _fetch_signals(cur, bad: set[str]) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, opened_at, open_price
        FROM signals
        WHERE indicators = 'RSI_Cross(9,24)' AND signal_type = 'Long' AND interval = '5m'
          AND open_price IS NOT NULL AND open_price > 0
        ORDER BY symbol, opened_at ASC
        """
    )
    cols = ["id", "symbol", "opened_at", "open_price"]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    if bad:
        df = df[~df["symbol"].isin(bad)].reset_index(drop=True)
    return df


def _fetch_5m_with_vol(cur, symbol: str) -> pd.DataFrame:
    cur.execute(
        "SELECT bucket, open, high, low, close, volume, buy_volume FROM cagg_5m WHERE symbol=%s ORDER BY bucket ASC",
        (symbol,),
    )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["bucket", "open", "high", "low", "close", "volume", "buy_volume"])
    for c in ("open", "high", "low", "close", "volume", "buy_volume"):
        df[c] = df[c].astype(float)
    return df


def _fetch_ta_series(cur, symbol: str, tf: str) -> pd.DataFrame:
    cur.execute(f"SELECT bucket, close FROM {_TABLE[tf]} WHERE symbol=%s ORDER BY bucket ASC", (symbol,))
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["bucket", "close"]).astype({"close": float})
    if not df.empty:
        df["net_ta"] = _total_amount_series(df["bucket"], df["close"].to_numpy())
    return df


def _collect() -> pd.DataFrame:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    cur = conn.cursor()
    signals = _fetch_signals(cur, bad)
    print(f"[fetch] {len(signals)} RSI_Cross(9,24) Long / 5m sinyali\n")

    records = []
    symbols = signals["symbol"].unique()
    print(f"{len(symbols)} sembol işlenecek")
    for si, sym in enumerate(symbols):
        df5 = _fetch_5m_with_vol(cur, sym)
        if df5.empty or len(df5) < _COMPONENT_WINDOW:
            continue
        ta_series = {}
        for tf in _TFS:
            d = _fetch_ta_series(cur, sym, tf)
            if d.empty:
                ta_series = None
                break
            ta_series[tf] = (d["bucket"].to_numpy(), d["net_ta"].to_numpy())
        if ta_series is None:
            continue

        b5 = df5["bucket"].to_numpy()
        o5 = df5["open"].to_numpy()
        h5 = df5["high"].to_numpy()
        l5 = df5["low"].to_numpy()
        c5 = df5["close"].to_numpy()
        vol5 = df5["volume"].to_numpy()
        buy5 = df5["buy_volume"].fillna(0).to_numpy()

        sub = signals[signals["symbol"] == sym]
        for _, row in sub.iterrows():
            idx = np.searchsorted(b5, np.datetime64(row["opened_at"]), side="right") - 1
            if idx < _COMPONENT_WINDOW - 1 or idx + _FORWARD_BARS - 1 >= len(c5):
                continue

            window = pd.DataFrame({
                "open": o5[idx - _COMPONENT_WINDOW + 1: idx + 1],
                "high": h5[idx - _COMPONENT_WINDOW + 1: idx + 1],
                "low": l5[idx - _COMPONENT_WINDOW + 1: idx + 1],
                "close": c5[idx - _COMPONENT_WINDOW + 1: idx + 1],
                "volume": vol5[idx - _COMPONENT_WINDOW + 1: idx + 1],
                "buy_volume": buy5[idx - _COMPONENT_WINDOW + 1: idx + 1],
            })
            window["sell_volume"] = window["volume"] - window["buy_volume"]
            try:
                vol_s, mom_s, vlt_s, prc_s = compute_components(window, "Long", volume_mode="real")
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            if any(pd.isna(v) for v in (vol_s, mom_s, vlt_s, prc_s)):
                continue

            fwd_price = c5[idx + _FORWARD_BARS - 1]
            fwd_ret = (fwd_price - row["open_price"]) / row["open_price"] * 100.0
            kategori = _classify_candle(pd.Series({"open": o5[idx], "high": h5[idx], "low": l5[idx], "close": c5[idx]}))

            rec = {
                "symbol": sym, "opened_at": row["opened_at"],
                "vol": vol_s, "mom": mom_s, "volat": vlt_s, "price": prc_s, "kategori": kategori,
                "fwd_ret": fwd_ret,
            }
            ok = True
            for tf, (b_arr, net_arr) in ta_series.items():
                j = np.searchsorted(b_arr, np.datetime64(row["opened_at"]), side="right") - 1
                if j < 0:
                    ok = False
                    break
                rec[f"pct_{tf}"] = _percentile_at(net_arr, j)
                for lb in _LOOKBACKS:
                    rec[f"slope_{tf}_{lb}"] = (net_arr[j] - net_arr[j - lb]) if j >= lb else np.nan
            if not ok:
                continue
            records.append(rec)
        if (si + 1) % 100 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol, {len(records)} kayıt")

    conn.close()
    return pd.DataFrame(records)


def _collect_cached() -> pd.DataFrame:
    if os.path.exists(_CACHE_PATH):
        print(f"[cache] {_CACHE_PATH} kullanılıyor")
        return pd.read_parquet(_CACHE_PATH)
    df = _collect()
    if not df.empty:
        df.to_parquet(_CACHE_PATH)
    return df


def _stats(rets: np.ndarray) -> dict:
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return {"n": 0, "wr": None, "pf": None}
    g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {"n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
            "pf": round(float(g / l), 3) if l > 0 else float("inf")}


def main() -> None:
    raw = _collect_cached()
    print(f"\n[collect] {len(raw)} sinyal toplandı\n")
    if raw.empty:
        return

    df = _add_all_up(raw)
    df = df[df["all_up"]].dropna(subset=["pct_1h", "pct_4h"]).reset_index(drop=True)
    ta_base = (df["pct_1h"] >= _TA_BASE_TH) & (df["pct_4h"] >= _TA_BASE_TH)
    df = df[ta_base].reset_index(drop=True)
    print(f"[all_up=True + TA-base>={_TA_BASE_TH} popülasyonu] n={len(df)}\n")

    df = df.sort_values("opened_at").reset_index(drop=True)
    mid = df["opened_at"].iloc[len(df) // 2]
    is_df = df[df["opened_at"] < mid]
    oos_df = df[df["opened_at"] >= mid]
    print(f"IS: n={len(is_df)}  OOS: n={len(oos_df)}\n")

    def _kovalama(d: pd.DataFrame, lb: int) -> pd.Series:
        s1, s4 = f"slope_1h_{lb}", f"slope_4h_{lb}"
        return ((d["pct_1h"] >= _KOVALAMA_TH) & (d[s1] > 0)) | ((d["pct_4h"] >= _KOVALAMA_TH) & (d[s4] > 0))

    print("=" * 78)
    print("EĞİM BAR-SAYISI (lookback) TARAMASI — SADECE IS")
    print("=" * 78)
    print(f"{'lookback':>9} | {'IS n':>6} {'IS WR%':>7} {'IS PF':>8} | {'OOS n':>6} {'OOS WR%':>7} {'OOS PF':>8}")
    print("-" * 78)
    best_lb, best_is_pf = None, -1.0
    for lb in _LOOKBACKS:
        is_sub = is_df[_kovalama(is_df, lb)]
        oos_sub = oos_df[_kovalama(oos_df, lb)]
        is_s = _stats(is_sub["fwd_ret"].to_numpy())
        oos_s = _stats(oos_sub["fwd_ret"].to_numpy())
        marker = " <- mevcut (2)" if lb == 2 else ""
        print(f"{lb:>9} | {is_s['n']:>6} {is_s['wr'] if is_s['wr'] is not None else '-':>7} "
              f"{is_s['pf'] if is_s['pf'] is not None else '-':>8} | "
              f"{oos_s['n']:>6} {oos_s['wr'] if oos_s['wr'] is not None else '-':>7} "
              f"{oos_s['pf'] if oos_s['pf'] is not None else '-':>8}{marker}")
        if is_s["n"] >= _MIN_N and is_s["pf"] is not None and isinstance(is_s["pf"], float) and is_s["pf"] > best_is_pf:
            best_is_pf = is_s["pf"]
            best_lb = lb

    print(f"\n=== IS'te en iyi lookback: {best_lb} bar (PF={best_is_pf}, SADECE IS verisiyle seçildi) ===")
    oos_best = oos_df[_kovalama(oos_df, best_lb)]
    print(f"  OOS performansı: {_stats(oos_best['fwd_ret'].to_numpy())}")

    print(f"\n=== Karşılaştırma: mevcut (2 bar) vs seçilen ({best_lb} bar), TÜM veri ===")
    current_group = df[_kovalama(df, 2)]
    best_group = df[_kovalama(df, best_lb)]
    print(f"  2 bar (mevcut)   : {_stats(current_group['fwd_ret'].to_numpy())}")
    print(f"  {best_lb} bar (yeni aday): {_stats(best_group['fwd_ret'].to_numpy())}")

    if best_lb != 2:
        rest = df[~_kovalama(df, best_lb)]
        df["_g"] = _kovalama(df, best_lb)
        _deep_validate(f"slope lookback={best_lb}", best_group, rest, df)
        days_span = (best_group["opened_at"].max() - best_group["opened_at"].min()).total_seconds() / 86400
        summarize(f"slope lookback={best_lb} — fwd_ret% serisi", best_group["fwd_ret"].to_numpy(), best_group["opened_at"], days_span)


if __name__ == "__main__":
    main()
