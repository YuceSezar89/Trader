"""
rsi_cross_ta_multiframe_combo_bt.py'nin SHORT tarafı (23 Tem 2026).

Aynı test — RSI_Cross(9,24) Short/5m + all_up=True + TA'nın 4 TF'de (5m/15m/
1h/4h) BAĞIMSIZ hesaplanmış net%'inin sinyal yönüyle (Short: net<0) uyumu.

Kullanım: python -m research.pattern_lab.rsi_cross_ta_multiframe_combo_short_bt
"""

import itertools

import numpy as np
import pandas as pd

from research.pattern_lab.do_open_streak_full_clean_bt import _bad_symbols, _conn
from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up, _classify_candle
from research.pattern_lab.rsi_cross_ta_alignment_bt import _deep_validate, _total_amount_series
from utils.vpmv import compute_components

_FORWARD_BARS = 24
_COMPONENT_WINDOW = 60
_TFS = ["5m", "15m", "1h", "4h"]
_TABLE = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_PLACEBO_ITER = 300


def _fetch_signals(cur, bad: set[str]) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, opened_at, open_price
        FROM signals
        WHERE indicators = 'RSI_Cross(9,24)' AND signal_type = 'Short' AND interval = '5m'
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
    table = _TABLE[tf]
    cur.execute(f"SELECT bucket, close FROM {table} WHERE symbol=%s ORDER BY bucket ASC", (symbol,))
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["bucket", "close"]).astype({"close": float})
    if not df.empty:
        df["net_ta"] = _total_amount_series(df["bucket"], df["close"].to_numpy())
    return df


def _collect() -> pd.DataFrame:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    cur = conn.cursor()
    signals = _fetch_signals(cur, bad)
    print(f"[fetch] {len(signals)} RSI_Cross(9,24) Short / 5m sinyali\n")

    records = []
    symbols = signals["symbol"].unique()
    print(f"{len(symbols)} sembol işlenecek")
    for si, sym in enumerate(symbols):
        df5 = _fetch_5m_with_vol(cur, sym)
        if df5.empty or len(df5) < _COMPONENT_WINDOW:
            continue
        ta_series = {}
        for tf in _TFS:
            if tf == "5m":
                d = df5.copy()
                d["net_ta"] = _total_amount_series(d["bucket"], d["close"].to_numpy())
            else:
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
            for tf, (b_arr, net_arr) in ta_series.items():
                j = np.searchsorted(b_arr, np.datetime64(row["opened_at"]), side="right") - 1
                if j < 0:
                    ok = False
                    break
                rec[f"match_{tf}"] = net_arr[j] < 0  # Short: bearish momentum hemfikir
            if not ok:
                continue
            records.append(rec)
        if (si + 1) % 100 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol, {len(records)} kayıt")

    conn.close()
    return pd.DataFrame(records)


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


def main() -> None:
    raw = _collect()
    print(f"\n[collect] {len(raw)} sinyal toplandı\n")
    if raw.empty:
        return

    df = _add_all_up(raw)
    df = df[df["all_up"]].reset_index(drop=True)
    print(f"[all_up=True popülasyonu] n={len(df)}\n")

    print("=" * 78)
    print("AŞAMA 1 — TF ikilisi/üçlüsü taraması (hepsi all_up=True İÇİNDE, SHORT)")
    print("=" * 78)
    results = []
    combos = list(itertools.combinations(_TFS, 2)) + list(itertools.combinations(_TFS, 3))
    for combo in combos:
        mask = np.ones(len(df), dtype=bool)
        for tf in combo:
            mask &= df[f"match_{tf}"].to_numpy()
        group = df[mask]
        rest = df[~mask]
        s = _stats(group["fwd_ret"].to_numpy())
        label = "+".join(combo)
        results.append((label, s.get("n", 0), s.get("wr"), s.get("pf"), group, rest))
        print(f"  {label:16} n={s.get('n',0):>5}  WR%={s.get('wr','-'):>6}  PF={s.get('pf','-')}")

    results_valid = [r for r in results if r[1] >= 30 and isinstance(r[3], (int, float))]
    results_valid.sort(key=lambda r: r[3], reverse=True)

    print("\n" + "=" * 78)
    print("AŞAMA 2 — en iyi 3 kombinasyonun tam derin doğrulaması (SHORT)")
    print("=" * 78)
    for label, n, wr, pf, group, rest in results_valid[:3]:
        df["_g"] = df.index.isin(group.index)
        _deep_validate(label, group, rest, df)


if __name__ == "__main__":
    main()
