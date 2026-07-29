"""
"TA" (Total Amount, süleyman özçelik.html'den) çapraz-TF uyum fikri —
RSI_Cross(9,24) Long / 5m üzerinde TEMİZ yöntemle test (22 Tem 2026).

TA formülü BİREBİR (JS'ten): 12 saatte bir sıfırlanan (UTC 00:00/12:00)
kümülatif bar-bar yüzde değişim toplamı. 5m ve 1H'de BAĞIMSIZ hesaplanıp,
ikisi de sinyal yönüyle (Long: net>0) uyuşuyor mu diye bakılıyor:
  UYUMLU  : hem 5m hem 1H net>0
  KISMİ   : sadece biri
  UYUMSUZ : hiçbiri

Hedef: sabit 24-bar (5m) ileri getiri, gerçek do_open_streak/all_up
testlerindeki aynı disiplin (placebo, split-period).

Kullanım: python -m research.pattern_lab.rsi_cross_ta_alignment_bt
"""

import numpy as np
import pandas as pd
import psycopg2

from config import Config

_FORWARD_BARS = 24
_PLACEBO_ITER = 300
_REGIME_SPLIT = pd.Timestamp("2026-06-08")
_CYCLE_MS = 12 * 3600 * 1000


def _conn():
    return psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )


def _total_amount_series(ts: pd.Series, close: np.ndarray) -> np.ndarray:
    t_ms = ts.astype("int64").to_numpy() // 10**6
    cycle = (t_ms // _CYCLE_MS) * _CYCLE_MS
    n = len(close)
    net = np.zeros(n)
    for i in range(1, n):
        if cycle[i] != cycle[i - 1]:
            net[i] = 0.0
        else:
            p = close[i - 1]
            if p != 0:
                pct = (close[i] - p) / p * 100.0
                net[i] = net[i - 1] + pct
            else:
                net[i] = net[i - 1]
    return net


def _fetch_series(cur, symbol: str, table: str) -> pd.DataFrame:
    cur.execute(f"SELECT bucket, close FROM {table} WHERE symbol=%s ORDER BY bucket ASC", (symbol,))
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["bucket", "close"]).astype({"close": float})
    df["net_ta"] = _total_amount_series(df["bucket"], df["close"].to_numpy())
    return df


def _fetch_signals(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, opened_at, open_price
        FROM signals
        WHERE indicators = 'RSI_Cross(9,24)' AND signal_type = 'Long' AND interval = '5m'
          AND open_price IS NOT NULL AND open_price > 0
        ORDER BY opened_at ASC
        """
    )
    cols = ["id", "symbol", "opened_at", "open_price"]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def _fetch_forward_price(cur, symbol: str, after, n_bars: int) -> float | None:
    cur.execute(
        "SELECT close FROM cagg_5m WHERE symbol=%s AND bucket >= %s ORDER BY bucket ASC LIMIT 1 OFFSET %s",
        (symbol, after, n_bars - 1),
    )
    row = cur.fetchone()
    return float(row[0]) if row else None


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


def _deep_validate(label: str, group: pd.DataFrame, rest: pd.DataFrame, full_df: pd.DataFrame) -> None:
    print(f"\n  -- {label} (n={len(group)}) vs geri kalan (n={len(rest)}) --")
    if len(group) == 0:
        print("    örneklem yok")
        return
    print(f"    {label}: {_stats(group['fwd_ret'].to_numpy())}")
    print(f"    geri kalan: {_stats(rest['fwd_ret'].to_numpy())}")

    real_gap = group["fwd_ret"].mean() - rest["fwd_ret"].mean()
    rng = np.random.default_rng(42)
    labels = full_df["_g"].to_numpy()
    target = full_df["fwd_ret"].to_numpy()
    count_ge = 0
    for _ in range(_PLACEBO_ITER):
        shuffled = rng.permutation(labels)
        d = target[shuffled].mean() - target[~shuffled].mean()
        if abs(d) >= abs(real_gap):
            count_ge += 1
    print(f"    placebo: gerçek ort% farkı ({real_gap:+.4f}) rastgelede %{count_ge/_PLACEBO_ITER*100:.1f} sıklıkta çıktı")

    if len(group) >= 40:
        g_sorted = group.sort_values("opened_at")
        mid = g_sorted["opened_at"].iloc[len(g_sorted)//2]
        fh = _stats(g_sorted[g_sorted["opened_at"] < mid]["fwd_ret"].to_numpy())
        sh = _stats(g_sorted[g_sorted["opened_at"] >= mid]["fwd_ret"].to_numpy())
        print(f"    split-period: ilk yarı {fh} | ikinci yarı {sh}")
    else:
        print("    split-period için örneklem yetersiz")


def main() -> None:
    conn = _conn()
    cur = conn.cursor()
    signals = _fetch_signals(cur)
    print(f"[fetch] {len(signals)} RSI_Cross(9,24) Long / 5m sinyali")

    records = []
    symbols = signals["symbol"].unique()
    print(f"{len(symbols)} sembol için 5m+1H TA serisi hesaplanacak")
    for si, sym in enumerate(symbols):
        df5 = _fetch_series(cur, sym, "cagg_5m")
        df1h = _fetch_series(cur, sym, "cagg_1h")
        if df5.empty or df1h.empty:
            continue
        b5 = df5["bucket"].to_numpy()
        net5 = df5["net_ta"].to_numpy()
        b1h = df1h["bucket"].to_numpy()
        net1h = df1h["net_ta"].to_numpy()

        sub = signals[signals["symbol"] == sym]
        for _, row in sub.iterrows():
            idx5 = np.searchsorted(b5, np.datetime64(row["opened_at"]), side="right") - 1
            if idx5 < 0:
                continue
            idx1h = np.searchsorted(b1h, np.datetime64(row["opened_at"]), side="right") - 1
            if idx1h < 0:
                continue
            n5 = net5[idx5]
            n1h = net1h[idx1h]
            fwd_price = _fetch_forward_price(cur, sym, row["opened_at"], _FORWARD_BARS)
            if fwd_price is None:
                continue
            fwd_ret = (fwd_price - row["open_price"]) / row["open_price"] * 100.0
            match5 = n5 > 0
            match1h = n1h > 0
            if match5 and match1h:
                align = "UYUMLU"
            elif match5 or match1h:
                align = "KISMİ"
            else:
                align = "UYUMSUZ"
            records.append({
                "symbol": sym, "opened_at": row["opened_at"], "net5": n5, "net1h": n1h,
                "align": align, "fwd_ret": fwd_ret,
            })
        if (si + 1) % 100 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol, {len(records)} kayıt")

    conn.close()
    df = pd.DataFrame(records)
    print(f"\n[collect] {len(df)} sinyal için TA uyumu + {_FORWARD_BARS}-bar ileri getiri\n")
    if df.empty:
        return

    print("=== [1] Uyum grubuna göre sabit-ileri-getiri ===")
    g = df.groupby("align")["fwd_ret"].agg(ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100)
    print(g.to_string())

    df["_g"] = df["align"] == "UYUMLU"
    _deep_validate("UYUMLU (5m+1H hemfikir)", df[df["_g"]], df[~df["_g"]], df)


if __name__ == "__main__":
    main()
