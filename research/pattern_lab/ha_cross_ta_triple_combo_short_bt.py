"""
HA_Cross Short tarafı — şimdiye kadar hiç dokunulmamış tek kombinasyon
(24 Tem 2026, kullanıcı isteği). rsi_cross_ta_triple_combo_short_bt.py +
ha_cross_ta_triple_combo_bt.py'nin birleşimi: HA_Cross(Heikin-Ashi crossover)
Short/5m sinyalleri üzerinde all_up + TA-kovalama(1h/4h) + HA-hizalanması
zinciri.

Short için "güçlü" TA/HA (Long'un aynası):
  - TA-base: percentile(net_ta) <=45 (net_ta olağandışı NEGATİF)
  - kovalama: percentile<=20 VE hâlâ DÜŞÜYOR (slope<0)
  - HA-hizalı: 1h VE 4h Heikin-Ashi mumu İKİSİ DE BEARISH

Kullanım: python -m research.pattern_lab.ha_cross_ta_triple_combo_short_bt
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
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "_cache_ha_cross_ta_triple_short.parquet")
_BASE_TH = 45
_EXTREME_TH = 20


def _fetch_signals(cur, bad: set[str]) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, opened_at, open_price
        FROM signals
        WHERE indicators = 'HA_Cross' AND signal_type = 'Short' AND interval = '5m'
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
    print(f"[fetch] {len(signals)} HA_Cross Short / 5m sinyali\n")

    records = []
    symbols = signals["symbol"].unique()
    print(f"{len(symbols)} sembol işlenecek")
    for si, sym in enumerate(symbols):
        df5 = _fetch_5m_with_vol(cur, sym)
        if df5.empty or len(df5) < _COMPONENT_WINDOW:
            continue
        ta_series, ha_series = {}, {}
        ok_series = True
        for tf in _TFS:
            dta = _fetch_ta_series(cur, sym, tf)
            dohlc = _fetch_ohlc_series(cur, sym, tf)
            if dta.empty or dohlc.empty:
                ok_series = False
                break
            ta_series[tf] = (dta["bucket"].to_numpy(), dta["net_ta"].to_numpy())
            bull = _ha_bull_series(dohlc["open"].to_numpy(), dohlc["high"].to_numpy(),
                                    dohlc["low"].to_numpy(), dohlc["close"].to_numpy())
            ha_series[tf] = (dohlc["bucket"].to_numpy(), bull)
        if not ok_series:
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
                vol_s, mom_s, vlt_s, prc_s = compute_components(window, "Short", volume_mode="real")
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            if any(pd.isna(v) for v in (vol_s, mom_s, vlt_s, prc_s)):
                continue

            fwd_price = c5[idx + _FORWARD_BARS - 1]
            fwd_ret = (row["open_price"] - fwd_price) / row["open_price"] * 100.0  # Short: ters yön
            kategori = _classify_candle(pd.Series({"open": o5[idx], "high": h5[idx], "low": l5[idx], "close": c5[idx]}))

            rec = {
                "symbol": sym, "opened_at": row["opened_at"],
                "vol": vol_s, "mom": mom_s, "volat": vlt_s, "price": prc_s, "kategori": kategori,
                "fwd_ret": fwd_ret,
            }
            ok = True
            for tf in _TFS:
                b_arr, net_arr = ta_series[tf]
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
    print(f"[all_up=True + tüm bileşenler popülasyonu] n={len(df)}\n")

    ta_base = (df["pct_1h"] <= _BASE_TH) & (df["pct_4h"] <= _BASE_TH)
    kovalama = ((df["pct_1h"] <= _EXTREME_TH) & (df["slope_1h"] < 0)) | \
               ((df["pct_4h"] <= _EXTREME_TH) & (df["slope_4h"] < 0))
    ha_aligned = (df["ha_bull_1h"] < 0.5) & (df["ha_bull_4h"] < 0.5)

    print("=" * 78)
    print("ÖZET TABLO — sırayla ekleme etkisi (HA_Cross Short)")
    print("=" * 78)
    steps = [
        ("all_up (filtresiz)", np.ones(len(df), dtype=bool)),
        (f"+ TA-base (pct<={_BASE_TH})", ta_base),
        ("+ kovalama (pct<=20&düşüyor)", ta_base & kovalama),
        ("+ HA-hizalı (üçlü)", ta_base & kovalama & ha_aligned),
    ]
    for label, mask in steps:
        s = _stats(df[mask]["fwd_ret"].to_numpy())
        print(f"  {label:32}: n={s['n']:>5}  WR%={s.get('wr','-'):>6}  PF={s.get('pf','-')}")

    triple_mask = ta_base & kovalama & ha_aligned
    triple = df[triple_mask]
    rest = df[~triple_mask]

    print("\n" + "=" * 78)
    print("EN İYİ İKİ ADAY: (TA+kovalama, HA'sız) vs (TA+kovalama+HA, üçlü) — hangisi daha iyi?")
    print("=" * 78)
    double_mask = ta_base & kovalama
    double = df[double_mask]
    print(f"  ikili (TA+kovalama)      : {_stats(double['fwd_ret'].to_numpy())}")
    print(f"  üçlü (TA+kovalama+HA)    : {_stats(triple['fwd_ret'].to_numpy())}")

    best_mask, best_label = (triple_mask, "üçlü") if len(triple) >= 30 and _stats(triple["fwd_ret"].to_numpy())["pf"] >= _stats(double["fwd_ret"].to_numpy())["pf"] else (double_mask, "ikili")
    best_group, best_rest = df[best_mask], df[~best_mask]

    print(f"\n=== Seçilen ({best_label}) tam derin doğrulama ===")
    if len(best_group) < 30:
        print(f"  Örneklem çok küçük (n={len(best_group)}), derin doğrulama atlanıyor.")
        return

    df["_g"] = best_mask
    _deep_validate(f"HA_Cross Short {best_label}", best_group, best_rest, df)
    print(f"\n  PF şüphesi (medyan): {_decompose(best_group['fwd_ret'].to_numpy())}")

    days_span = (best_group["opened_at"].max() - best_group["opened_at"].min()).total_seconds() / 86400
    summarize(f"HA_Cross Short {best_label} — fwd_ret% serisi", best_group["fwd_ret"].to_numpy(), best_group["opened_at"], days_span)


if __name__ == "__main__":
    main()
