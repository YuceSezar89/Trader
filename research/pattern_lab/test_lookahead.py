"""
Look-ahead sızıntı testi.

İddia: extract_features t0 sonrası veriyi GÖREMEZ.
Kanıt: t0 sonrasına kasıtlı uçuk barlar (100× fiyat, 1000× hacim) eklenmiş
veriyle sonuç, eklenmemişle bit-bit aynı olmalı. Ayrıca RankProvider'ın
rank_pct(t)'si de t sonrası ızgara satırlarından etkilenmemeli.
"""
import numpy as np
import pandas as pd

from research.pattern_lab.features import extract_features
from research.pattern_lab.rank import RankProvider


def _synth(n=3000, seed=7):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-06-20 03:00", periods=n, freq="5min")
    close = 1 + np.abs(np.cumsum(rng.normal(0, 0.002, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    vol = rng.uniform(50, 150, n)
    buy = vol * rng.uniform(0.3, 0.7, n)
    return pd.DataFrame({
        "ts": ts, "open": close, "high": high, "low": low, "close": close,
        "volume": vol, "buy_volume": buy, "sell_volume": vol - buy,
    })


def test_features_no_lookahead():
    df = _synth()
    t0 = df["ts"].iloc[2500]
    btc = pd.Series(df["close"].to_numpy(), index=df["ts"])

    base = extract_features(df, btc, t0)
    assert base, "özellikler boş dönmemeli"

    poisoned = df.copy()
    after = poisoned["ts"] >= t0
    poisoned.loc[after, ["open", "high", "low", "close"]] *= 100.0
    poisoned.loc[after, "volume"] *= 1000.0
    poisoned.loc[after, "buy_volume"] = poisoned.loc[after, "volume"]
    poisoned.loc[after, "sell_volume"] = 0.0
    btc_poisoned = btc.copy()
    btc_poisoned[btc_poisoned.index >= t0] *= 100.0

    poisoned_out = extract_features(poisoned, btc_poisoned, t0)

    assert base.keys() == poisoned_out.keys()
    for k in base:
        a, b = base[k], poisoned_out[k]
        if isinstance(a, float) and np.isnan(a):
            assert np.isnan(b), f"{k}: NaN beklerken {b}"
        else:
            assert a == b, f"{k}: {a} != {b} — GELECEK SIZDI!"


def test_rank_no_lookahead():
    rng = np.random.default_rng(3)
    ts = pd.date_range("2026-06-25 03:00", periods=600, freq="5min")
    frames = []
    for i in range(200):
        frames.append(pd.DataFrame({
            "ts": ts, "symbol": f"S{i}",
            "close": 1 + np.abs(np.cumsum(rng.normal(0, 0.002, 600))),
        }))
    layer_b = pd.concat(frames, ignore_index=True)
    t = ts[400]

    r1 = RankProvider(layer_b).rank_pct("S0", t, 12)

    poisoned = layer_b.copy()
    poisoned.loc[poisoned["ts"] > t, "close"] *= 50.0
    r2 = RankProvider(poisoned).rank_pct("S0", t, 12)

    assert r1 == r2, f"rank sızdırdı: {r1} != {r2}"


if __name__ == "__main__":
    test_features_no_lookahead()
    test_rank_no_lookahead()
    print("LOOK-AHEAD TESTLERİ GEÇTİ — gelecek verisi hiçbir özelliğe sızmıyor")
