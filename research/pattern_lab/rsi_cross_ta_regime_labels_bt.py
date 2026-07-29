"""
Madde 2: süleyman özçelik.html'in TAM rejim etiket seti (taObserveAt) —
biz şu ana kadar sadece basitleştirilmiş "kovalama" (high&rising) ayrımını
test etmiştik (Hipotez B, [[project_rsi_cross_ta_hypothesis_b_23tem]]).
Burada TÜM 6 etiket (Long tarafı) birebir JS formülüyle uygulanıp fwd_ret
ile karşılaştırılıyor (24 Tem 2026, kullanıcı isteği).

JS formülü (taObserveAt, birebir):
  slope = net[i]-net[i-2], prevSlope = net[i-1]-net[i-3], eps=.02
  rising = slope>eps, falling = slope<-eps
  turnedUp = rising & prevSlope<=eps, turnedDown = falling & prevSlope>=-eps
  low = pct<=20, high = pct>=80  (z-score OR'u atlandı, henüz test edilmedi — Madde 4)
  LONG etiketleri (ilk eşleşen kazanır):
    low & turnedUp        -> DİP DÖNÜŞ        (HTML bucket: DESTEK)
    low & rising           -> DİPTEN TOPARLANMA (DESTEK)
    low & falling           -> DÜŞEN BIÇAK      (RİSK)
    high & rising           -> LONG KOVALAMA    (RİSK)  [Hipotez B'de TERSİ bulunmuştu]
    net>0 & rising           -> LONG DEVAM       (DESTEK)
    net<=0 & rising          -> TOPARLANMA       (DESTEK)
    (hiçbiri)                 -> NÖTR

Kullanım: python -m research.pattern_lab.rsi_cross_ta_regime_labels_bt
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

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "_cache_rsi_cross_ta_regime_labels.parquet")

_FORWARD_BARS = 24
_COMPONENT_WINDOW = 60
_TFS = ["1h", "4h"]
_TABLE = {"1h": "cagg_1h", "4h": "cagg_4h"}
_EXTREME_PCT = 20
_EPS = 0.02
_MIN_N = 20
_PLACEBO_ITER = 300

_LABELS_ORDER = ["DİP DÖNÜŞ", "DİPTEN TOPARLANMA", "DÜŞEN BIÇAK", "LONG KOVALAMA", "LONG DEVAM", "TOPARLANMA", "NÖTR"]


def _label_at(net_arr: np.ndarray, j: int) -> str:
    if j < 3:
        return "NÖTR"
    net = net_arr[j]
    slope = net_arr[j] - net_arr[j - 2]
    prev_slope = net_arr[j - 1] - net_arr[j - 3]
    rising, falling = slope > _EPS, slope < -_EPS
    turned_up = rising and prev_slope <= _EPS
    pct = _percentile_at(net_arr, j)
    if np.isnan(pct):
        return "NÖTR"
    low, high = pct <= _EXTREME_PCT, pct >= (100 - _EXTREME_PCT)

    if low and turned_up:
        return "DİP DÖNÜŞ"
    if low and rising:
        return "DİPTEN TOPARLANMA"
    if low and falling:
        return "DÜŞEN BIÇAK"
    if high and rising:
        return "LONG KOVALAMA"
    if net > 0 and rising:
        return "LONG DEVAM"
    if net <= 0 and rising:
        return "TOPARLANMA"
    return "NÖTR"


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
                rec[f"label_{tf}"] = _label_at(net_arr, j)
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
        return {"n": 0, "wr": None, "ort_%": None, "medyan_%": None, "pf": None}
    g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean()), 3), "medyan_%": round(float(np.median(rets)), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def _collect_cached() -> pd.DataFrame:
    if os.path.exists(_CACHE_PATH):
        print(f"[cache] {_CACHE_PATH} kullanılıyor")
        return pd.read_parquet(_CACHE_PATH)
    df = _collect()
    if not df.empty:
        df.to_parquet(_CACHE_PATH)
    return df


def main() -> None:
    raw = _collect_cached()
    print(f"\n[collect] {len(raw)} sinyal toplandı\n")
    if raw.empty:
        return

    df = _add_all_up(raw)
    df = df[df["all_up"]].reset_index(drop=True)
    print(f"[all_up=True popülasyonu] n={len(df)}\n")

    for tf in _TFS:
        print("=" * 78)
        print(f"{tf.upper()} REJİM ETİKETİ — fwd_ret dağılımı (HTML'in kendi RİSK/DESTEK etiketiyle)")
        print("=" * 78)
        bucket_map = {
            "DİP DÖNÜŞ": "DESTEK", "DİPTEN TOPARLANMA": "DESTEK", "DÜŞEN BIÇAK": "RİSK",
            "LONG KOVALAMA": "RİSK", "LONG DEVAM": "DESTEK", "TOPARLANMA": "DESTEK", "NÖTR": "NÖTR",
        }
        counts = df[f"label_{tf}"].value_counts()
        for label in _LABELS_ORDER:
            sub = df[df[f"label_{tf}"] == label]
            s = _stats(sub["fwd_ret"].to_numpy())
            html_bucket = bucket_map[label]
            print(f"  {label:20} (HTML: {html_bucket:6}) n={s['n']:>5}  WR%={s.get('wr','-'):>6}  "
                  f"medyan%={s.get('medyan_%','-'):>7}  PF={s.get('pf','-')}")

    print("\n" + "=" * 78)
    print("DERİN DOĞRULAMA — en ilginç etiketler (n>=30 olanlar)")
    print("=" * 78)
    for tf in _TFS:
        for label in _LABELS_ORDER:
            if label == "NÖTR":
                continue
            mask = df[f"label_{tf}"] == label
            group, rest = df[mask], df[~mask]
            if len(group) < _MIN_N:
                continue
            df["_g"] = mask
            _deep_validate(f"{tf.upper()}: {label}", group, rest, df)


if __name__ == "__main__":
    main()
