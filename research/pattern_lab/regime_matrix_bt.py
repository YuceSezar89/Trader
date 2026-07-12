"""
Kapsamlı matris: DevisSoTrader'ın 3 ajan durumunu (is_consolidating,
trend_state, meanrev_state — devissotrader_agents_bt.py) TÜM sinyal
ailelerine (RSI_Cross, HA_Cross, MA200_Cross, Supertrend — threshold_optimizer.py
ile aynı liste) hem TEK TEK hem KOMBİNE (açgözlü/greedy stepwise, threshold_optimizer'ın
alpha→beta 2-adımlı aramasının 3 değişkene genellenmişi) uygular.

rsi_cross_volbreakout_regime_bt.py + rsi_cross_trend_meanrev_regime_bt.py'de
sadece RSI_Cross üzerinde bulunanları (is_consolidating→Long, trend_state→Short)
diğer 3 aileye ve birleşik kurallara genelliyor — "tek tek/toplu/hepsini test
edelim" isteği.

Disiplin threshold_optimizer.py ile birebir aynı: kronolojik IS/OOS + split-period
+ placebo (IS realized_pnl karıştırılıp aynı arama tekrarlanır).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.pattern_lab.devissotrader_agents_bt import (  # pylint: disable=wrong-import-position
    _is_consolidating_series, _mean_reversion_state_series, _trend_state_series,
)
from research.pattern_lab.rsi_cross_volbreakout_regime_bt import (  # pylint: disable=wrong-import-position
    _fetch_regime, _fetch_signals, _merge_regime,
)
from research.pattern_lab.threshold_optimizer import (  # pylint: disable=wrong-import-position
    MIN_N, N_PLACEBO, _apply_rule, _best_single_threshold, _pf, _run_single_var_on_df,
)

INDICATORS = ["RSI_Cross(9,24)", "HA_Cross", "MA200_Cross", "Supertrend(10,3.0)"]
DIRECTIONS = ["Long", "Short"]

STATES = {
    "is_consolidating": ("15m", _is_consolidating_series, pd.Timedelta(minutes=15)),
    "trend_state": ("4h", _trend_state_series, pd.Timedelta(hours=4)),
    "meanrev_state": ("15m", _mean_reversion_state_series, pd.Timedelta(minutes=15)),
}


def _greedy_stepwise(df: pd.DataFrame, cols: list) -> tuple:
    """threshold_optimizer._stepwise_search'ün (sabit alpha→beta sırası) N
    değişkene genellenmişi — her adımda IS PF'yi en çok artıran değişken
    seçilir, uygulanır, kalan değişkenlerle devam edilir."""
    remaining = list(cols)
    applied = []
    cur = df
    while remaining:
        best = None
        for c in remaining:
            rule = _best_single_threshold(cur, c)
            if rule and (best is None or rule["pf"] > best[1]["pf"]):
                best = (c, rule)
        if best is None:
            break
        c, rule = best
        applied.append(rule)
        cur = _apply_rule(cur, rule)
        remaining.remove(c)
    return applied, cur


def _rules_text(rules: list) -> str:
    if not rules:
        return "(kural yok — hiçbir değişken min_n şartını geçemedi)"
    return " AND ".join(f"{r['col']}{r['op']}{r['threshold']:.2f}" for r in rules)


def _run_combined(label: str, df: pd.DataFrame, cols: list) -> None:
    print(f"\n{'='*70}\n{label}  (n={len(df):,})\n{'='*70}")
    if len(df) < MIN_N * 4:
        print("Örneklem çok küçük, atlanıyor.")
        return

    t_min, t_max = df["opened_at"].min(), df["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    is_df = df[df["opened_at"] < mid].reset_index(drop=True)
    oos_df = df[df["opened_at"] >= mid].reset_index(drop=True)
    print(f"dönem: {t_min} .. {t_max} | IS: {len(is_df)} | OOS: {len(oos_df)}")

    baseline_is = _pf(is_df)
    print(f"\nIS baseline PF={baseline_is.get('pf',0):.3f} (n={baseline_is.get('n',0)})")

    rules, is_filtered = _greedy_stepwise(is_df, cols)
    is_stats = _pf(is_filtered) if len(is_filtered) > 0 else {"pf": 0, "n": 0}
    print(f"Bulunan kombine kural: {_rules_text(rules)}")
    print(f"IS filtreli PF={is_stats.get('pf',0):.3f} (n={is_stats.get('n',0)})")

    def _apply_all(d, rls):
        for r in rls:
            d = _apply_rule(d, r)
        return d

    oos_baseline = _pf(oos_df)
    oos_filtered = _apply_all(oos_df, rules)
    oos_stats = _pf(oos_filtered) if len(oos_filtered) > 0 else {"pf": 0, "n": 0, "wr": 0}

    print(f"\n── OOS (sabit kombine kuralla) ──")
    print(f"baseline:  PF={oos_baseline.get('pf',0):.3f} WR%={oos_baseline.get('wr',0)} n={oos_baseline.get('n',0)}")
    print(f"filtreli:  PF={oos_stats.get('pf',0):.3f} WR%={oos_stats.get('wr',0)} n={oos_stats.get('n',0)}")

    oos_mid = mid + (t_max - mid) / 2
    oos_first = _apply_all(oos_df[oos_df["opened_at"] < oos_mid], rules)
    oos_second = _apply_all(oos_df[oos_df["opened_at"] >= oos_mid], rules)
    s1 = _pf(oos_first) if len(oos_first) > 0 else {"pf": 0, "n": 0}
    s2 = _pf(oos_second) if len(oos_second) > 0 else {"pf": 0, "n": 0}
    print(f"\n── OOS split-period sağlamlık ──")
    print(f"ilk yarı:    PF={s1.get('pf',0):.3f} (n={s1.get('n',0)})")
    print(f"ikinci yarı: PF={s2.get('pf',0):.3f} (n={s2.get('n',0)})")

    pnl_vals = is_df["realized_pnl"].to_numpy().copy()
    placebo_pfs = []
    rng = np.random.default_rng(42)
    for _ in range(N_PLACEBO):
        shuffled = is_df.copy()
        shuffled["realized_pnl"] = rng.permutation(pnl_vals)
        _, filt = _greedy_stepwise(shuffled, cols)
        s = _pf(filt) if len(filt) > 0 else {"pf": 0, "n": 0}
        if s.get("n", 0) >= MIN_N:
            placebo_pfs.append(s.get("pf", 0))

    if placebo_pfs:
        placebo_arr = np.array(placebo_pfs)
        rank = float((placebo_arr < is_stats.get("pf", 0)).mean() * 100)
        print(f"\n── Placebo (n={len(placebo_pfs)} karıştırma) ──")
        print(f"placebo PF ort={placebo_arr.mean():.3f} p90={np.percentile(placebo_arr,90):.3f} "
              f"max={placebo_arr.max():.3f} | gerçek IS PF={is_stats.get('pf',0):.3f} "
              f"(placebo'nun %{rank:.0f}'ini geçiyor)")
        verdict = "GEÇERLİ ADAY" if rank >= 90 and oos_stats.get("pf", 0) > oos_baseline.get("pf", 0) \
            and s1.get("pf", 0) > 1 and s2.get("pf", 0) > 1 else "GÜVENİLMEZ/ZAYIF"
        print(f"\n>>> SONUÇ: {verdict}")


def _build_merged(indicator: str, direction: str) -> pd.DataFrame | None:
    sig_df = _fetch_signals(indicator, direction)
    if len(sig_df) < 200:
        print(f"{indicator} — {direction}: yetersiz sinyal ({len(sig_df)}), atlanıyor")
        return None

    symbols = sig_df["symbol"].unique().tolist()
    merged = sig_df.copy()
    for col_name, (interval, state_fn, bar_dur) in STATES.items():
        regime_df = _fetch_regime(symbols, interval, state_fn, col_name)
        merged = _merge_regime(merged, regime_df, col_name, bar_dur)
        merged = merged.drop(columns=["ts"], errors="ignore")
    return merged


def run() -> None:
    for indicator in INDICATORS:
        for direction in DIRECTIONS:
            merged = _build_merged(indicator, direction)
            if merged is None:
                continue

            print(f"\n{'#'*74}\n# {indicator} — {direction}  "
                  f"({len(merged):,} sinyal, 3 rejim durumuyla eşleşti)\n{'#'*74}")

            for col_name in STATES:
                label = f"{indicator} — {direction} — {col_name} (tekil)"
                _run_single_var_on_df(label, merged, col_name)

            _run_combined(
                f"{indicator} — {direction} — KOMBİNE (3 durum, açgözlü stepwise)",
                merged, list(STATES.keys()),
            )


if __name__ == "__main__":
    run()
