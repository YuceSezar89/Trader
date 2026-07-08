"""
Teori C: Hacim ivmesi — EVOL'den bağımsız, ham hacim trendi.

Hipotez: Belki önemli olan fiyat-hacim VERİMLİLİĞİ (EVOL) değil, hacmin
KENDİSİNİN ivmesi — son birkaç barda hacim (fiyat ne kadar hareket ettiğinden
bağımsız olarak) artıyor mu? Bu, "ilgi/katılım artıyor mu" sorusuna EVOL'den
farklı, daha ham bir cevap.

vol_accel = ortalama(son 5 bar hacmi) / ortalama(önceki 15 bar hacmi)
>1: hacim son barlarda hızlanıyor (ilgi artıyor)
<1: hacim son barlarda yavaşlıyor (ilgi azalıyor)

Kullanım: python -m research.pattern_lab.vol_accel_bt
"""
import warnings

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config
from research.pattern_lab.evol_bt import _fetch_bars, _fetch_signals

_SHORT_WINDOW = 5
_LONG_WINDOW = 20


def _compute_vol_accel(df: pd.DataFrame) -> "float | None":
    try:
        if len(df) < _LONG_WINDOW + 5:
            return None
        volume = df["volume"].astype(float)
        recent = volume.iloc[-_SHORT_WINDOW:].mean()
        baseline = volume.iloc[-_LONG_WINDOW:-_SHORT_WINDOW].mean()
        if baseline <= 0:
            return None
        return float(recent / baseline)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _pf(pnls: pd.Series) -> float:
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    return float(wins / losses) if losses > 0 else float("inf")


def _report_correlation(df: pd.DataFrame, label: str) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        if len(sub) < 30:
            continue
        rho, p = spearmanr(sub["vol_accel"], sub["realized_pnl"])
        print(f"  [{label}] {sig_type}: n={len(sub)}, rho={rho:+.3f} (p={p:.4f})")


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    signals = _fetch_signals(cur)
    print(f"Toplam kapanmış sinyal (5m/15m): {len(signals)}")

    rows = []
    for i, sig in enumerate(signals, 1):
        bars = _fetch_bars(cur, sig["symbol"], sig["interval"], sig["opened_at"])
        if bars is None or len(bars) < _LONG_WINDOW + 5:
            continue
        accel = _compute_vol_accel(bars)
        if accel is None:
            continue
        rows.append({
            "symbol": sig["symbol"], "signal_type": sig["signal_type"],
            "opened_at": sig["opened_at"], "vol_accel": accel,
            "realized_pnl": sig["realized_pnl"],
        })
        if i % 5000 == 0:
            print(f"  [{i}/{len(signals)}] işlendi, {len(rows)} geçerli satır")

    conn.close()
    df = pd.DataFrame(rows)
    print(f"\nGeçerli satır: {len(df)}\n")

    print("=== ANA TEST: hacim ivmesi vs realized_pnl ===")
    _report_correlation(df, "gerçek")

    print("\n=== PF/WR: ivme>1 (hızlanan) vs ivme<=1 (yavaşlayan) ===")
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        fast = sub[sub["vol_accel"] > 1]
        slow = sub[sub["vol_accel"] <= 1]
        print(f"  {sig_type}:")
        print(f"    ivme>1 (hızlanan) : n={len(fast)}, PF={_pf(fast['realized_pnl']):.3f}, "
              f"WR={(fast['realized_pnl'] > 0).mean() * 100:.1f}%")
        print(f"    ivme<=1 (yavaşlayan): n={len(slow)}, PF={_pf(slow['realized_pnl']):.3f}, "
              f"WR={(slow['realized_pnl'] > 0).mean() * 100:.1f}%")

    print("\n=== PLACEBO (ivme rastgele karıştırıldı) ===")
    rng = np.random.default_rng(42)
    placebo = df.copy()
    placebo["vol_accel"] = rng.permutation(placebo["vol_accel"].values)
    _report_correlation(placebo, "placebo")

    print("\n=== SPLIT-PERIOD ===")
    mid = df["opened_at"].min() + (df["opened_at"].max() - df["opened_at"].min()) / 2
    for half_name, half_df in [
        ("ilk_yari", df[df["opened_at"] < mid]),
        ("ikinci_yari", df[df["opened_at"] >= mid]),
    ]:
        print(f"-- {half_name} ({len(half_df)}) --")
        _report_correlation(half_df, half_name)


if __name__ == "__main__":
    main()
