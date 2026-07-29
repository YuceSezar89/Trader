"""
Madde 7: TA-uyumu (1h+4h) şimdiye kadar sadece RSI_Cross(9,24)/5m'de test
edildi — burada AYNI yöntem RSI_Cross(9,24)/15m sinyallerinde tekrarlanıyor
(24 Tem 2026, kullanıcı isteği). TA 1h/4h'de bağımsız hesaplandığı için
sinyal aralığından etkilenmiyor, ama "24 bar ileri getiri" artık 15m
bar biriminde (24×15dk=6 saat, 5m testinde 24×5dk=2 saatti — proje
standardı: her interval kendi bar biriminde sabit 24-bar).

Kovalama eşiği GÜNCEL bulguyla (90, [[project_rsi_cross_kovalama_threshold_sweep_24tem]])
kullanılıyor, 80 değil.

Kullanım: python -m research.pattern_lab.rsi_cross_ta_15m_bt
"""

import os

import numpy as np
import pandas as pd

from research.pattern_lab.concentration_diagnostics import summarize
from research.pattern_lab.do_open_streak_full_clean_bt import _bad_symbols, _conn
from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up, _classify_candle
from research.pattern_lab.rsi_cross_ta_alignment_bt import _deep_validate, _total_amount_series
from research.pattern_lab.rsi_cross_ta_ha_overlap_bt import _ha_bull_series
from research.pattern_lab.rsi_cross_ta_percentile_bt import _decompose, _percentile_at, _slope_at
from utils.vpmv import compute_components

_FORWARD_BARS = 24
_COMPONENT_WINDOW = 60
_TFS = ["1h", "4h"]
_TABLE = {"1h": "cagg_1h", "4h": "cagg_4h"}
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "_cache_rsi_cross_ta_15m.parquet")
_BASE_TH = 55
_KOVALAMA_TH = 90  # güncel bulgu, [[project_rsi_cross_kovalama_threshold_sweep_24tem]]


def _fetch_signals(cur, bad: set[str]) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, opened_at, open_price
        FROM signals
        WHERE indicators = 'RSI_Cross(9,24)' AND signal_type = 'Long' AND interval = '15m'
          AND open_price IS NOT NULL AND open_price > 0
        ORDER BY symbol, opened_at ASC
        """
    )
    cols = ["id", "symbol", "opened_at", "open_price"]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    if bad:
        df = df[~df["symbol"].isin(bad)].reset_index(drop=True)
    return df


def _fetch_15m_with_vol(cur, symbol: str) -> pd.DataFrame:
    cur.execute(
        "SELECT bucket, open, high, low, close, volume, buy_volume FROM cagg_15m WHERE symbol=%s ORDER BY bucket ASC",
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


def _fetch_ohlc_series(cur, symbol: str, tf: str) -> pd.DataFrame:
    cur.execute(f"SELECT bucket, open, high, low, close FROM {_TABLE[tf]} WHERE symbol=%s ORDER BY bucket ASC", (symbol,))
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["bucket", "open", "high", "low", "close"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df


def _collect() -> pd.DataFrame:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    cur = conn.cursor()
    signals = _fetch_signals(cur, bad)
    print(f"[fetch] {len(signals)} RSI_Cross(9,24) Long / 15m sinyali\n")

    records = []
    symbols = signals["symbol"].unique()
    print(f"{len(symbols)} sembol işlenecek")
    for si, sym in enumerate(symbols):
        df15 = _fetch_15m_with_vol(cur, sym)
        if df15.empty or len(df15) < _COMPONENT_WINDOW:
            continue
        ta_series, ha_series = {}, {}
        ok_series = True
        for tf in _TFS:
            d = _fetch_ta_series(cur, sym, tf)
            dohlc = _fetch_ohlc_series(cur, sym, tf)
            if d.empty or dohlc.empty:
                ok_series = False
                break
            ta_series[tf] = (d["bucket"].to_numpy(), d["net_ta"].to_numpy())
            bull = _ha_bull_series(dohlc["open"].to_numpy(), dohlc["high"].to_numpy(),
                                    dohlc["low"].to_numpy(), dohlc["close"].to_numpy())
            ha_series[tf] = (dohlc["bucket"].to_numpy(), bull)
        if not ok_series:
            continue

        b15 = df15["bucket"].to_numpy()
        o15 = df15["open"].to_numpy()
        h15 = df15["high"].to_numpy()
        l15 = df15["low"].to_numpy()
        c15 = df15["close"].to_numpy()
        vol15 = df15["volume"].to_numpy()
        buy15 = df15["buy_volume"].fillna(0).to_numpy()

        sub = signals[signals["symbol"] == sym]
        for _, row in sub.iterrows():
            idx = np.searchsorted(b15, np.datetime64(row["opened_at"]), side="right") - 1
            if idx < _COMPONENT_WINDOW - 1 or idx + _FORWARD_BARS - 1 >= len(c15):
                continue

            window = pd.DataFrame({
                "open": o15[idx - _COMPONENT_WINDOW + 1: idx + 1],
                "high": h15[idx - _COMPONENT_WINDOW + 1: idx + 1],
                "low": l15[idx - _COMPONENT_WINDOW + 1: idx + 1],
                "close": c15[idx - _COMPONENT_WINDOW + 1: idx + 1],
                "volume": vol15[idx - _COMPONENT_WINDOW + 1: idx + 1],
                "buy_volume": buy15[idx - _COMPONENT_WINDOW + 1: idx + 1],
            })
            window["sell_volume"] = window["volume"] - window["buy_volume"]
            try:
                vol_s, mom_s, vlt_s, prc_s = compute_components(window, "Long", volume_mode="real")
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            if any(pd.isna(v) for v in (vol_s, mom_s, vlt_s, prc_s)):
                continue

            fwd_price = c15[idx + _FORWARD_BARS - 1]
            fwd_ret = (fwd_price - row["open_price"]) / row["open_price"] * 100.0
            kategori = _classify_candle(pd.Series({"open": o15[idx], "high": h15[idx], "low": l15[idx], "close": c15[idx]}))

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
                rec[f"slope_{tf}"] = _slope_at(net_arr, j)

                hb_arr, bull_arr = ha_series[tf]
                jh = np.searchsorted(hb_arr, np.datetime64(row["opened_at"]), side="right") - 1
                if jh < 0:
                    ok = False
                    break
                rec[f"ha_bull_{tf}"] = float(bull_arr[jh])
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
        return {"n": 0, "wr": None, "ort_%": None, "pf": None}
    g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean()), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def main() -> None:
    raw = _collect_cached()
    print(f"\n[collect] {len(raw)} sinyal toplandı\n")
    if raw.empty:
        return

    df = _add_all_up(raw)
    df = df[df["all_up"]].dropna(
        subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h", "ha_bull_1h", "ha_bull_4h"]
    ).reset_index(drop=True)
    print(f"[all_up=True + bileşenler (HA dahil)] n={len(df)}\n")

    ta_base = (df["pct_1h"] >= _BASE_TH) & (df["pct_4h"] >= _BASE_TH)
    kovalama = ((df["pct_1h"] >= _KOVALAMA_TH) & (df["slope_1h"] > 0)) | \
               ((df["pct_4h"] >= _KOVALAMA_TH) & (df["slope_4h"] > 0))
    ha_aligned = (df["ha_bull_1h"] > 0.5) & (df["ha_bull_4h"] > 0.5)

    print("=" * 78)
    print("ÖZET TABLO — sırayla ekleme etkisi (15m, 5m'deki tam merdiven)")
    print("=" * 78)
    steps = [
        ("all_up (filtresiz)", np.ones(len(df), dtype=bool)),
        (f"+ TA-base (pct>={_BASE_TH})", ta_base),
        (f"+ kovalama (pct>={_KOVALAMA_TH}&tırmanıyor)", ta_base & kovalama),
        ("+ HA-hizalı (üçlü)", ta_base & kovalama & ha_aligned),
    ]
    for label, mask in steps:
        s = _stats(df[mask]["fwd_ret"].to_numpy())
        print(f"  {label:38}: n={s['n']:>5}  WR%={s.get('wr','-'):>6}  PF={s.get('pf','-')}")

    double_mask = ta_base & kovalama
    triple_mask = ta_base & kovalama & ha_aligned
    double, triple = df[double_mask], df[triple_mask]
    double_pf = _stats(double["fwd_ret"].to_numpy())["pf"]
    triple_pf = _stats(triple["fwd_ret"].to_numpy())["pf"] if len(triple) >= 30 else -1
    best_mask, best_label = (triple_mask, "üçlü (+HA)") if triple_pf >= double_pf else (double_mask, "ikili (HA'sız)")
    final, rest = df[best_mask], df[~best_mask]

    print(f"\n=== En iyi ({best_label}) tam derin doğrulama (n={len(final)}) ===")
    if len(final) < 30:
        print("  Örneklem çok küçük, derin doğrulama atlanıyor.")
        return

    df["_g"] = best_mask
    _deep_validate(f"15m: {best_label}", final, rest, df)
    print(f"\n  PF şüphesi (medyan): {_decompose(final['fwd_ret'].to_numpy())}")

    days_span = (final["opened_at"].max() - final["opened_at"].min()).total_seconds() / 86400
    summarize(f"15m {best_label} — fwd_ret% serisi", final["fwd_ret"].to_numpy(), final["opened_at"], days_span)


if __name__ == "__main__":
    main()
