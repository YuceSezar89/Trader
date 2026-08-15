"""
Trajectory Lab — HIZLI SAĞLAMA (resmi Stage'lere GİRMEDEN önce), 11 Ağustos.

Soru: iki farklı "momentum" tanımından hangisi return_t5_pct (temiz,
execution-bağımsız hedef) ile daha güçlü ilişkili?

  A) BİZİM (utils/vpmv.py::compute_series ile aynı): rsi.diff() * side,
     sinyal barındaki (t_offset=0) değeri.
  B) KORPUS %VPM (Hoca Telegram külliyatı, S02): son 75 barda
     log(volume) * |priceChange%| kümülatif toplamı, yükselen barlarda
     "buy", düşen barlarda "sell" tarafına yazılıp net (buy-sell) alınır,
     side ile işaretlenir.

Bu SADECE bir ön-sağlama — Cohen's d + Pearson/Spearman + iki adayın
birbirine redundancy'si (r). Placebo/bootstrap/walk-forward YOK (Stage 2
işi). Sonuç "resmi Stage'lere sokmaya değer mi" sorusuna karar vermek
için.

Kullanım:
    python -m research.trajectory_lab.momentum_formula_compare
"""

from __future__ import annotations

import asyncio
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database.engine import async_engine  # noqa: E402
from research.trajectory_lab import config as C  # noqa: E402
from research.trajectory_lab import metrics as M  # noqa: E402
from research.trajectory_lab.corpus_builder import (  # noqa: E402
    _fetch_case_signals,
    _fetch_window_klines,
)

SEED = 42
SAMPLE_PER_FAMILY = 200  # aile (indicator,direction) başına — hızlı olsun diye
LOOKBACK = 75  # korpus %VPM'in kendi lookback varsayılanı (S02)

FAMILIES = [
    ("HA_Cross", "Long"),
    ("HA_Cross", "Short"),
    ("RSI_Cross(9,24)", "Long"),
    ("RSI_Cross(9,24)", "Short"),
    ("Supertrend(10,3.0)", "Long"),
    ("Supertrend(10,3.0)", "Short"),
]


def _corpus_vpm_momentum(klines: pd.DataFrame, side: float) -> float | None:
    """t_offset==0 barındaki, son LOOKBACK barlık log(hacim)x|fiyat değişim%|
    kümülatif net momentum (buy-sell), side ile işaretli."""
    window = klines[(klines["t_offset"] > -LOOKBACK) & (klines["t_offset"] <= 0)]
    if len(window) < LOOKBACK // 2:
        return None
    price_change_pct = window["close"].pct_change().fillna(0.0) * 100.0
    log_vol = np.log1p(window["volume"].clip(lower=0))
    buy_mom = (log_vol * price_change_pct.clip(lower=0)).sum()
    sell_mom = (log_vol * (-price_change_pct.clip(upper=0))).sum()
    return float((buy_mom - sell_mom) * side)


async def _collect() -> pd.DataFrame:
    rows = []
    for indicator, direction in FAMILIES:
        side = 1.0 if direction == "Long" else -1.0
        signals_df = await _fetch_case_signals(indicator, direction)
        if signals_df.empty:
            continue
        sample = signals_df.sample(
            n=min(SAMPLE_PER_FAMILY, len(signals_df)), random_state=SEED
        )
        for _, sig in sample.iterrows():
            klines = await _fetch_window_klines(sig["symbol"], sig["interval"], sig["opened_at"])
            if klines.empty or len(klines) < C.WARMUP_BARS:
                continue

            corpus_mom = _corpus_vpm_momentum(klines, side)
            if corpus_mom is None:
                continue

            comp = M.vpmv_components_series(klines, direction)
            at_t0 = klines.index[klines["t_offset"] == 0]
            if len(at_t0) == 0:
                continue
            our_mom = comp["mom"].loc[at_t0[0]]
            if pd.isna(our_mom):
                continue

            rows.append(
                {
                    "indicator": indicator,
                    "direction": direction,
                    "signal_id": sig["signal_id"],
                    "outcome_value": sig["outcome_value"],
                    "our_mom": float(our_mom),
                    "corpus_mom": corpus_mom,
                }
            )
        print(f"{indicator} {direction}: {len(sample)} sinyal denendi, kümülatif toplam satır={len(rows)}")
    return pd.DataFrame(rows)


def _cohens_d(values: pd.Series, outcome: pd.Series) -> float:
    winners = values[outcome >= C.WINNER_THRESHOLD]
    losers = values[outcome <= C.LOSER_THRESHOLD]
    if len(winners) < 5 or len(losers) < 5:
        return float("nan")
    pooled_std = np.sqrt((winners.std() ** 2 + losers.std() ** 2) / 2)
    if pooled_std == 0:
        return float("nan")
    return float((winners.mean() - losers.mean()) / pooled_std)


def _report(df: pd.DataFrame) -> None:
    print(f"\nToplam örnek: {len(df)}")

    for col in ("our_mom", "corpus_mom"):
        pear = df[col].corr(df["outcome_value"], method="pearson")
        spear = df[col].corr(df["outcome_value"], method="spearman")
        d = _cohens_d(df[col], df["outcome_value"])
        print(f"\n{col} vs {C.OUTCOME_METRIC}:")
        print(f"  Pearson r  = {pear:+.4f}")
        print(f"  Spearman r = {spear:+.4f}")
        print(f"  Cohen's d (winner-loser) = {d:+.4f}")

    r_between = df["our_mom"].corr(df["corpus_mom"], method="pearson")
    print(f"\nour_mom <-> corpus_mom Pearson r (redundancy) = {r_between:+.4f}")

    # Basit kısmi korelasyon: corpus_mom'un our_mom kontrol edilince kalan katkısı.
    # (11 Ağu düzeltmesi: scipy.stats.linregress bazı numpy/scipy sürüm
    # kombinasyonlarında np.cov'un weights yoluna girip 'float has no shape'
    # hatası veriyordu — np.polyfit ile aynı OLS artığı, dtype'a duyarlı değil.)
    def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        slope_x, intercept_x = np.polyfit(z, x, 1)
        slope_y, intercept_y = np.polyfit(z, y, 1)
        resid_x = x - (slope_x * z + intercept_x)
        resid_y = y - (slope_y * z + intercept_y)
        return float(np.corrcoef(resid_x, resid_y)[0, 1])

    pc = _partial_corr(df["corpus_mom"].to_numpy(), df["outcome_value"].to_numpy(), df["our_mom"].to_numpy())
    print(f"\npartial(outcome, corpus_mom | our_mom) = {pc:+.4f}  (our_mom kontrol edilince corpus_mom'un kalan katkısı)")

    pc2 = _partial_corr(df["our_mom"].to_numpy(), df["outcome_value"].to_numpy(), df["corpus_mom"].to_numpy())
    print(f"partial(outcome, our_mom | corpus_mom) = {pc2:+.4f}  (corpus_mom kontrol edilince our_mom'un kalan katkısı)")


async def _main() -> None:
    df = await _collect()
    if df.empty:
        print("Hiç veri toplanamadı.")
        return
    out_path = os.path.join(C.CORPUS_DIR, "momentum_formula_compare.parquet")
    os.makedirs(C.CORPUS_DIR, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nHam veri kaydedildi: {out_path}")
    _report(df)


if __name__ == "__main__":
    asyncio.run(_main())
