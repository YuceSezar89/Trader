"""
Entropy pencere/kutu duyarlılık taraması (24 Tem 2026) — 50 bar/5 kutu
keyfi mi seçilmişti (entropy_clean_forward_return_bt.py'nin "en umut
verici kombinasyonu"ydu, ama o ham/evrensel popülasyonda; kovalama
kombosunda hiç taranmadı). Long kovalama popülasyonunda (n=266) pencere
(20/30/50/100) × kutu (5/10) taranıyor, veri BİR KEZ çekilip tüm
kombinasyonlarda yeniden kullanılıyor (hız için).

Kullanım: python -m research.pattern_lab.rsi_cross_ta_triple_entropy_sensitivity_bt
"""

import numpy as np
import pandas as pd

from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.entropy_clean_forward_return_bt import _entropy
from research.pattern_lab.rsi_cross_combo_realistic_replay_bt import (
    _BASE_TH_LONG,
    _fetch_5m_full,
    _fetch_signal_open_prices,
    _pnl_usd,
    _simulate,
    _stats,
)
from research.pattern_lab.rsi_cross_ta_ha_overlap_bt import _HA_CACHE_PATH

_LIVE_KOVALAMA_TH_LONG = 90
_WINDOWS = [20, 30, 50, 100]
_BIN_COUNTS = [5, 10]


def main() -> None:
    long_df = pd.read_parquet(_HA_CACHE_PATH)
    long_df = long_df.dropna(
        subset=["pct_1h", "pct_4h", "slope_1h", "slope_4h", "ha_bull_1h", "ha_bull_4h"]
    )
    ta_base = (long_df["pct_1h"] >= _BASE_TH_LONG) & (long_df["pct_4h"] >= _BASE_TH_LONG)
    kovalama = (
        (long_df["pct_1h"] >= _LIVE_KOVALAMA_TH_LONG) & (long_df["slope_1h"] > 0)
    ) | ((long_df["pct_4h"] >= _LIVE_KOVALAMA_TH_LONG) & (long_df["slope_4h"] > 0))
    ha = (long_df["ha_bull_1h"] > 0.5) & (long_df["ha_bull_4h"] > 0.5)
    qualifying = long_df[ta_base & kovalama & ha][["symbol", "opened_at"]].reset_index(drop=True)

    conn = _conn()
    cur = conn.cursor()
    open_prices = _fetch_signal_open_prices(cur, "Long")
    merged = qualifying.merge(open_prices, on=["symbol", "opened_at"], how="inner")
    print(f"açılış fiyatı eşleşen n={len(merged)}")

    # Her sembol için veri + PnL'i BİR KEZ hesapla (window/bin'e bağlı değil),
    # log_ret'i sakla, sadece entropy hesabı kombinasyona göre değişecek.
    per_signal = []  # (symbol, idx, log_ret_window_kaynağı, opened_at, pnl_usd)
    max_window = max(_WINDOWS)
    symbols = merged["symbol"].unique()
    for si, sym in enumerate(symbols):
        df5 = _fetch_5m_full(cur, sym)
        if df5.empty:
            continue
        b = df5["bucket"].to_numpy()
        h = df5["high"].to_numpy()
        l = df5["low"].to_numpy()
        c = df5["close"].to_numpy()
        atr = df5["atr"].to_numpy()
        log_ret = np.diff(np.log(c))

        sub = merged[merged["symbol"] == sym]
        for _, row in sub.iterrows():
            idx = np.searchsorted(b, np.datetime64(row["opened_at"]), side="right") - 1
            if idx < max(14, max_window) or idx + 1 >= len(c) or np.isnan(atr[idx]) or atr[idx] <= 0:
                continue
            entry_price = float(row["open_price"])
            atr_entry = float(atr[idx])
            exit_price, reason, bars_held = _simulate(h, l, c, idx + 1, "Long", entry_price, atr_entry)
            pnl = _pnl_usd("Long", entry_price, exit_price)
            per_signal.append(
                {"symbol": sym, "idx": idx, "opened_at": row["opened_at"], "pnl_usd": pnl,
                 "log_ret_full": log_ret}
            )
        if (si + 1) % 50 == 0:
            print(f"  ... {si+1}/{len(symbols)} sembol")
    conn.close()
    print(f"toplam geçerli işlem: {len(per_signal)}")

    print("\n" + "=" * 90)
    print(f"{'pencere':>8} {'kutu':>5} {'n':>5} {'korelasyon':>11} {'placebo%':>9} {'PF(<=p25)':>10} {'PF(>=p75)':>10}")
    print("=" * 90)
    rng = np.random.default_rng(42)
    for window in _WINDOWS:
        for bins in _BIN_COUNTS:
            ents, pnls = [], []
            for rec in per_signal:
                lr = rec["log_ret_full"]
                idx = rec["idx"]
                e = _entropy(lr[idx - window : idx], bins)
                if e is None:
                    continue
                ents.append(e)
                pnls.append(rec["pnl_usd"])
            ents = np.array(ents)
            pnls = np.array(pnls)
            if len(ents) < 30:
                continue
            corr = np.corrcoef(ents, pnls)[0, 1]
            shuffled = [np.corrcoef(rng.permutation(ents), pnls)[0, 1] for _ in range(200)]
            pb = float(np.mean(np.abs(shuffled) >= abs(corr)) * 100)

            p25 = np.quantile(ents, 0.25)
            p75 = np.quantile(ents, 0.75)
            low_pnl = pnls[ents <= p25]
            high_pnl = pnls[ents >= p75]
            pf_low = low_pnl[low_pnl > 0].sum() / -low_pnl[low_pnl < 0].sum() if (low_pnl < 0).any() else float("inf")
            pf_high = high_pnl[high_pnl > 0].sum() / -high_pnl[high_pnl < 0].sum() if (high_pnl < 0).any() else float("inf")

            print(f"{window:>8} {bins:>5} {len(ents):>5} {corr:>11.3f} {pb:>9.1f} {pf_low:>10.2f} {pf_high:>10.2f}")


if __name__ == "__main__":
    main()
