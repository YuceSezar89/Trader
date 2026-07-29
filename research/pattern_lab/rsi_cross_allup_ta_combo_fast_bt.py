"""
all_up + TA-uyumu KOMBİNASYONU — HIZLI versiyon (22 Tem 2026).

rsi_cross_allup_ta_combo_bt.py ile AYNI test ama 1m-agregasyon YOK —
cagg_5m zaten buy_volume/sell_volume içeriyor, doğrudan kullanılıyor
(do_open_streak_all_up_bt.py'deki gibi). Sadece interval='5m' sinyaller.

Kullanım: python -m research.pattern_lab.rsi_cross_allup_ta_combo_fast_bt
"""

import numpy as np
import pandas as pd

from research.pattern_lab.do_open_streak_full_clean_bt import _bad_symbols, _conn
from research.pattern_lab.rsi_cross_allup_candleshape_clean_bt import _add_all_up, _classify_candle
from research.pattern_lab.rsi_cross_ta_alignment_bt import _deep_validate, _total_amount_series
from utils.vpmv import compute_components

_FORWARD_BARS = 24
_COMPONENT_WINDOW = 60


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


def _fetch_5m_1h(cur, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cur.execute(
        "SELECT bucket, open, high, low, close, volume, buy_volume FROM cagg_5m WHERE symbol=%s ORDER BY bucket ASC",
        (symbol,),
    )
    rows5 = cur.fetchall()
    df5 = pd.DataFrame(rows5, columns=["bucket", "open", "high", "low", "close", "volume", "buy_volume"])
    for c in ("open", "high", "low", "close", "volume", "buy_volume"):
        df5[c] = df5[c].astype(float)
    if not df5.empty:
        df5["net_ta"] = _total_amount_series(df5["bucket"], df5["close"].to_numpy())

    cur.execute("SELECT bucket, close FROM cagg_1h WHERE symbol=%s ORDER BY bucket ASC", (symbol,))
    rows1h = cur.fetchall()
    df1h = pd.DataFrame(rows1h, columns=["bucket", "close"]).astype({"close": float})
    if not df1h.empty:
        df1h["net_ta"] = _total_amount_series(df1h["bucket"], df1h["close"].to_numpy())
    return df5, df1h


def _collect() -> pd.DataFrame:
    conn = _conn()
    bad = _bad_symbols(conn.cursor())
    print(f"[filtre] {len(bad)} sembol büyük iç-boşluk taşıyor, teste dahil edilmiyor")

    cur = conn.cursor()
    signals = _fetch_signals(cur, bad)
    print(f"[fetch] {len(signals)} RSI_Cross(9,24) Long / 5m sinyali\n")

    records = []
    symbols = signals["symbol"].unique()
    print(f"{len(symbols)} sembol işlenecek")
    for si, sym in enumerate(symbols):
        df5, df1h = _fetch_5m_1h(cur, sym)
        if df5.empty or len(df5) < _COMPONENT_WINDOW or df1h.empty:
            continue
        b5 = df5["bucket"].to_numpy()
        o5 = df5["open"].to_numpy()
        h5 = df5["high"].to_numpy()
        l5 = df5["low"].to_numpy()
        c5 = df5["close"].to_numpy()
        vol5 = df5["volume"].to_numpy()
        buy5 = df5["buy_volume"].fillna(0).to_numpy() if "buy_volume" in df5 else np.zeros(len(df5))
        net5 = df5["net_ta"].to_numpy()
        b1h = df1h["bucket"].to_numpy()
        net1h = df1h["net_ta"].to_numpy()

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

            idx1h = np.searchsorted(b1h, np.datetime64(row["opened_at"]), side="right") - 1
            if idx1h < 0:
                continue
            match5 = net5[idx] > 0
            match1h = net1h[idx1h] > 0
            align = "UYUMLU" if (match5 and match1h) else ("KISMİ" if (match5 or match1h) else "UYUMSUZ")

            records.append({
                "symbol": sym, "opened_at": row["opened_at"],
                "vol": vol_s, "mom": mom_s, "volat": vlt_s, "price": prc_s, "kategori": kategori,
                "ta_align": align, "fwd_ret": fwd_ret,
            })
        if (si + 1) % 100 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol, {len(records)} kayıt")

    conn.close()
    return pd.DataFrame(records)


def main() -> None:
    raw = _collect()
    print(f"\n[collect] {len(raw)} sinyal toplandı\n")
    if raw.empty:
        return

    df = _add_all_up(raw)
    print(f"[all_up hesaplandı] n={len(df)} (bir önceki aynı sembol sinyali olanlar)\n")

    print("=== [1] all_up × ta_align çapraz tablo ===")
    g = df.groupby(["all_up", "ta_align"])["fwd_ret"].agg(ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100)
    print(g.to_string())

    df["_g"] = df["all_up"] & (df["ta_align"] == "UYUMLU")
    print("\n=== [2] KOMBO: all_up=True VE ta_align=UYUMLU ===")
    _deep_validate("all_up + TA UYUMLU", df[df["_g"]], df[~df["_g"]], df)

    df["_g"] = df["all_up"]
    print("\n=== [3] Referans: sadece all_up=True (TA'sız) ===")
    _deep_validate("sadece all_up=True", df[df["all_up"]], df[~df["all_up"]], df)

    df["_g"] = df["ta_align"] == "UYUMLU"
    print("\n=== [4] Referans: sadece ta_align=UYUMLU (all_up'sız) ===")
    _deep_validate("sadece TA UYUMLU", df[df["ta_align"] == "UYUMLU"], df[df["ta_align"] != "UYUMLU"], df)


if __name__ == "__main__":
    main()
