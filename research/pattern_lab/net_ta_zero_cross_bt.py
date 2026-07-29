"""
net_ta'nın 0'ı YUKARI kestiği an — Long giriş sinyali olarak test. (24 Tem
2026, kullanıcı fikri: "döngü-açılış seviyesini yeniden ele geçirme" anı,
gerçek dip değil ama dipten sonraki teyitli toparlanma anı — bkz. konuşma.)

net_ta = signals/ta_kovalama_gate.py::_net_ta_series ile BİREBİR aynı
formül (12 saatte bir UTC 00:00/12:00'de sıfırlanan kümülatif bar-bar %
değişim), burada 1h bar'da hesaplanıyor. Sinyal: net_ta[i-1]<=0 VE
net_ta[i]>0 (aşağıdan yukarı sıfır kesişimi).

Kontrol grubu: AYNI sembol evreninde RASTGELE barlar (aynı büyüklükte
örneklem) — "0-kesişimi zamanlaması gerçekten mi iyi, yoksa herhangi bir
an girmekle aynı mı" sorusuna cevap.

Çıkış: gerçekçi SL/TP/trailing replay (SL=1.5xATR, TP=3xATR, trailing —
rsi_cross_combo_realistic_replay_bt.py ile aynı disiplin, 1h bar'da).

Kullanım: python -m research.pattern_lab.net_ta_zero_cross_bt
"""

import numpy as np
import pandas as pd

from indicators.core import calculate_atr
from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.rsi_cross_combo_realistic_replay_bt import _pnl_usd, _simulate, _stats

_LOOKBACK_DAYS = 120
_MIN_BARS = 400  # ~120 gün 1h icin fazlasiyla yeterli, gevsek tutuldu
_MAX_SYMBOLS = 250
_CYCLE_MS = 12 * 3600 * 1000
_PLACEBO_ITER = 300


def _fetch_symbols(cur) -> list[str]:
    cur.execute(
        """
        SELECT symbol, count(*) c FROM cagg_1h
        WHERE bucket > NOW() - INTERVAL '%s days'
        GROUP BY symbol HAVING count(*) >= %s
        ORDER BY c DESC LIMIT %s
        """,
        (_LOOKBACK_DAYS, _MIN_BARS, _MAX_SYMBOLS),
    )
    return [r[0] for r in cur.fetchall()]


def _fetch_1h(cur, symbol: str) -> pd.DataFrame:
    cur.execute(
        """
        SELECT bucket, open, high, low, close FROM cagg_1h
        WHERE symbol=%s AND bucket > NOW() - INTERVAL '%s days'
        ORDER BY bucket ASC
        """,
        (symbol, _LOOKBACK_DAYS),
    )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["bucket", "open", "high", "low", "close"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df


def _net_ta_1h(bucket: np.ndarray, close: np.ndarray) -> np.ndarray:
    t_ms = bucket.astype("datetime64[ms]").astype("int64")
    cycle = (t_ms // _CYCLE_MS) * _CYCLE_MS
    n = len(close)
    net = np.zeros(n)
    for i in range(1, n):
        if cycle[i] != cycle[i - 1]:
            net[i] = 0.0
        else:
            p = close[i - 1]
            net[i] = net[i - 1] + ((close[i] - p) / p * 100.0 if p else 0.0)
    return net


def main() -> None:
    conn = _conn()
    cur = conn.cursor()
    symbols = _fetch_symbols(cur)
    print(f"analiz edilecek sembol: {len(symbols)}")

    signal_records = []
    random_records = []
    rng = np.random.default_rng(7)

    for si, sym in enumerate(symbols):
        df = _fetch_1h(cur, sym)
        if len(df) < _MIN_BARS:
            continue
        b = df["bucket"].to_numpy()
        h = df["high"].to_numpy()
        l = df["low"].to_numpy()
        c = df["close"].to_numpy()
        atr = calculate_atr(df, period=14).to_numpy()
        net_ta = _net_ta_1h(b, c)
        n = len(c)

        cross_up = np.zeros(n, dtype=bool)
        cross_up[1:] = (net_ta[:-1] <= 0) & (net_ta[1:] > 0)
        cross_idxs = np.where(cross_up)[0]

        valid_start = 20  # ATR ısınma payı
        for idx in cross_idxs:
            if idx < valid_start or idx + 1 >= n or np.isnan(atr[idx]) or atr[idx] <= 0:
                continue
            entry_price = c[idx]
            exit_price, reason, bars_held = _simulate(h, l, c, idx + 1, "Long", entry_price, float(atr[idx]))
            pnl = _pnl_usd("Long", entry_price, exit_price)
            signal_records.append(
                {"symbol": sym, "opened_at": b[idx], "pnl_usd": pnl, "reason": reason}
            )

        # kontrol: aynı sembolde, aynı SAYIDA rastgele bar (ATR gecerli olanlardan)
        valid_idxs = np.where((~np.isnan(atr)) & (atr > 0))[0]
        valid_idxs = valid_idxs[(valid_idxs >= valid_start) & (valid_idxs + 1 < n)]
        n_rand = min(len(cross_idxs), len(valid_idxs))
        if n_rand > 0:
            rand_idxs = rng.choice(valid_idxs, size=n_rand, replace=False)
            for idx in rand_idxs:
                entry_price = c[idx]
                exit_price, reason, bars_held = _simulate(h, l, c, idx + 1, "Long", entry_price, float(atr[idx]))
                pnl = _pnl_usd("Long", entry_price, exit_price)
                random_records.append(
                    {"symbol": sym, "opened_at": b[idx], "pnl_usd": pnl, "reason": reason}
                )

        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()

    sig_df = pd.DataFrame(signal_records)
    rand_df = pd.DataFrame(random_records)

    print(f"\n0-kesişimi sinyal sayısı: {len(sig_df)}")
    print(f"Rastgele kontrol grubu sayısı: {len(rand_df)}")

    print(f"\n0-YUKARI-KESİŞİMİ (net_ta): {_stats(sig_df['pnl_usd'].to_numpy())}")
    print(f"   kapanış nedeni: {sig_df['reason'].value_counts().to_dict()}")
    print(f"RASTGELE BAR (kontrol): {_stats(rand_df['pnl_usd'].to_numpy())}")
    print(f"   kapanış nedeni: {rand_df['reason'].value_counts().to_dict()}")

    rng2 = np.random.default_rng(42)
    pool = rand_df["pnl_usd"].to_numpy()
    real_mean = sig_df["pnl_usd"].mean()
    n_sig = len(sig_df)
    if len(pool) >= n_sig:
        ge = sum(1 for _ in range(_PLACEBO_ITER) if rng2.choice(pool, size=n_sig, replace=False).mean() >= real_mean)
        print(f"\nplacebo (rastgele kontrolden aynı büyüklükte örnek, sinyal ort'unu eşitleme/geçme sıklığı): "
              f"%{ge/_PLACEBO_ITER*100:.1f}")

    if len(sig_df) >= 40:
        d = sig_df.sort_values("opened_at")
        mid = d["opened_at"].iloc[len(d) // 2]
        fh = _stats(d[d["opened_at"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(d[d["opened_at"] >= mid]["pnl_usd"].to_numpy())
        print(f"\nsplit-period: ilk yarı {fh} | ikinci yarı {sh}")


if __name__ == "__main__":
    main()
