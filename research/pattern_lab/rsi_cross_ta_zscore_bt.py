"""
Madde 4: percentile yerine Z-SCORE ile TA uyumu — süleyman özçelik.html'in
zscoreAt() fonksiyonu birebir (24 Tem 2026, kullanıcı isteği).

zscoreAt: (cur - ort) / std, 200-bar pencere (percentile'dan farklı olarak
CARİ barı da pencereye dahil ediyor — percentileAt hariç tutuyordu, JS
kaynağında böyle). Percentile dağılım-bağımsız SIRALAMA'dır (0-100 arası
sınırlı); z-score ise ortalamadan kaç STANDART SAPMA uzakta olduğunu ölçer,
sınırsızdır (uç değerlerde çok büyük/küçük olabilir) — teorik olarak farklı
bir bilgi taşıyabilir.

Test:
  1. Hipotez A ile birebir aynı disiplin (IS/OOS eşik taraması) ama
     percentile yerine z-score ile.
  2. HTML'in kendi "extreme" tanımı aslında OR: low=(pct<=20)|(z<=-1),
     high=(pct>=80)|(z>=1) — z eklemek kovalama popülasyonunu büyütüyor mu,
     ekleneni ayrı test ediyoruz.

Kullanım: python -m research.pattern_lab.rsi_cross_ta_zscore_bt
"""

import os

import numpy as np
import pandas as pd

from research.pattern_lab.concentration_diagnostics import summarize
from research.pattern_lab.do_open_streak_full_clean_bt import _bad_symbols, _conn
from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up, _classify_candle
from research.pattern_lab.rsi_cross_ta_alignment_bt import _deep_validate, _total_amount_series
from research.pattern_lab.rsi_cross_ta_percentile_bt import _percentile_at, _slope_at
from utils.vpmv import compute_components

_FORWARD_BARS = 24
_COMPONENT_WINDOW = 60
_TFS = ["1h", "4h"]
_TABLE = {"1h": "cagg_1h", "4h": "cagg_4h"}
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "_cache_rsi_cross_ta_zscore.parquet")
_LOOKBACK = 200
_MIN_LOOKBACK = 50
_MIN_N = 30
_THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
_EXTREME_PCT = 80
_EXTREME_Z = 1.0


def _zscore_at(net_arr: np.ndarray, j: int) -> float:
    s = max(0, j - _LOOKBACK + 1)
    q = net_arr[s:j + 1]  # HTML: cari bar dahil
    if len(q) < _MIN_LOOKBACK:
        return np.nan
    m = q.mean()
    sd = q.std()
    return float((net_arr[j] - m) / sd) if sd > 0 else 0.0


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
                rec[f"z_{tf}"] = _zscore_at(net_arr, j)
                rec[f"slope_{tf}"] = _slope_at(net_arr, j)
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
        return {"n": 0, "wr": None, "ort_%": None, "medyan_%": None, "pf": None}
    g, l = rets[rets > 0].sum(), -rets[rets < 0].sum()
    return {
        "n": len(rets), "wr": round(float((rets > 0).mean() * 100), 1),
        "ort_%": round(float(rets.mean()), 3), "medyan_%": round(float(np.median(rets)), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def main() -> None:
    raw = _collect_cached()
    print(f"\n[collect] {len(raw)} sinyal toplandı\n")
    if raw.empty:
        return

    df = _add_all_up(raw)
    df = df[df["all_up"]].dropna(subset=["pct_1h", "pct_4h", "z_1h", "z_4h"]).reset_index(drop=True)
    print(f"[all_up=True + bileşenler hesaplanabilir] n={len(df)}\n")

    df = df.sort_values("opened_at").reset_index(drop=True)
    mid = df["opened_at"].iloc[len(df) // 2]
    is_df = df[df["opened_at"] < mid]
    oos_df = df[df["opened_at"] >= mid]
    print(f"IS: n={len(is_df)}  OOS: n={len(oos_df)}\n")

    print("=" * 78)
    print("AŞAMA 1 — Z-SCORE eşiği taraması (SADECE IS verisiyle), percentile eşdeğeriyle karşılaştırma")
    print("=" * 78)
    print(f"{'esik':>6} | {'IS n':>6} {'IS PF':>8} | {'OOS n':>6} {'OOS PF':>8}")
    print("-" * 60)
    best_th, best_is_pf = None, -1.0
    for th in _THRESHOLDS:
        is_sub = is_df[(is_df["z_1h"] >= th) & (is_df["z_4h"] >= th)]
        oos_sub = oos_df[(oos_df["z_1h"] >= th) & (oos_df["z_4h"] >= th)]
        is_s = _stats(is_sub["fwd_ret"].to_numpy())
        oos_s = _stats(oos_sub["fwd_ret"].to_numpy())
        print(f"{th:>6} | {is_s['n']:>6} {is_s['pf'] if is_s['pf'] is not None else '-':>8} | "
              f"{oos_s['n']:>6} {oos_s['pf'] if oos_s['pf'] is not None else '-':>8}")
        if is_s["n"] >= _MIN_N and is_s["pf"] is not None and isinstance(is_s["pf"], float) and is_s["pf"] > best_is_pf:
            best_is_pf = is_s["pf"]
            best_th = th

    print(f"\n=== IS'te en iyi z-eşiği: {best_th} (PF={best_is_pf}) ===")
    oos_best = oos_df[(oos_df["z_1h"] >= best_th) & (oos_df["z_4h"] >= best_th)]
    print(f"  OOS performansı: {_stats(oos_best['fwd_ret'].to_numpy())}")

    z_mask = (df["z_1h"] >= best_th) & (df["z_4h"] >= best_th)
    z_group, z_rest = df[z_mask], df[~z_mask]
    df["_g"] = z_mask
    _deep_validate(f"Z-SCORE>={best_th} (1h+4h)", z_group, z_rest, df)

    print("\n" + "=" * 78)
    print("AŞAMA 2 — KARŞILAŞTIRMA: Z-SCORE filtresi vs PERCENTILE≥55 filtresi (aynı veri)")
    print("=" * 78)
    pct_mask = (df["pct_1h"] >= 55) & (df["pct_4h"] >= 55)
    pct_group = df[pct_mask]
    print(f"  percentile>=55 : {_stats(pct_group['fwd_ret'].to_numpy())}")
    print(f"  z-score>={best_th}   : {_stats(z_group['fwd_ret'].to_numpy())}")
    overlap = (pct_mask & z_mask).sum()
    print(f"  örtüşme: percentile grubunun %{overlap/max(1,pct_mask.sum())*100:.1f}'i z-score grubunda da var")
    print(f"           z-score grubunun %{overlap/max(1,z_mask.sum())*100:.1f}'i percentile grubunda da var")

    print("\n" + "=" * 78)
    print("AŞAMA 3 — HTML'in KENDİ tanımı: extreme = percentile>=80 OR z>=1 (kovalama'yı büyütüyor mu?)")
    print("=" * 78)
    high_pct_only = (df["pct_1h"] >= _EXTREME_PCT) | (df["pct_4h"] >= _EXTREME_PCT)
    high_or = ((df["pct_1h"] >= _EXTREME_PCT) | (df["z_1h"] >= _EXTREME_Z)) | \
              ((df["pct_4h"] >= _EXTREME_PCT) | (df["z_4h"] >= _EXTREME_Z))
    print(f"  sadece percentile>=80 : n={high_pct_only.sum()}  {_stats(df[high_pct_only]['fwd_ret'].to_numpy())}")
    print(f"  percentile>=80 OR z>=1: n={high_or.sum()}  {_stats(df[high_or]['fwd_ret'].to_numpy())}")
    extra = high_or & (~high_pct_only)
    print(f"  SADECE z ile eklenen ekstra kayıtlar: n={extra.sum()}  {_stats(df[extra]['fwd_ret'].to_numpy())}")

    days_span = (df["opened_at"].max() - df["opened_at"].min()).total_seconds() / 86400
    summarize(f"z-score>={best_th} — fwd_ret% serisi", z_group["fwd_ret"].to_numpy(), z_group["opened_at"], days_span)


if __name__ == "__main__":
    main()
