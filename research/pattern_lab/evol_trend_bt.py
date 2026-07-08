"""
Teori B: "sessiz birikim → patlama" (Wyckoff tarzı) — EVOL'ün SEVİYESİ değil,
TRENDİ (değişim hızı) testi.

Hipotez: Gerçek bir Long kırılımından ÖNCE sessiz bir birikim dönemi olur —
hacim orta-yüksek ama fiyat pek hareket etmez, yani bu dönemde EVOL DÜŞÜK
görünür. Kırılım anında verimlilik artmaya başlar. `evol_bt.py`'nin tek-barlık
anlık EVOL ölçümü bu İKİ AŞAMALI (durgunluktan harekete geçiş) örüntüyü
kaçırıyor olabilir — sadece "şu an ne kadar verimli" ölçüyor, "durgunluktan mı
geliyor, zaten hareketliyken mi sönümleniyor" ölçmüyor.

Bu script `evol_trend` = sinyal anındaki EVOL percentile rank - N bar önceki
EVOL percentile rank (AYNI referans dağılımına göre, aynı `_compute_evol`
mekaniği). Pozitif değer = "uyanan" (durgunluktan harekete geçen) sinyal,
negatif = "zaten hareketliyken sönümlenen" sinyal.

Kullanım: python -m research.pattern_lab.evol_trend_bt
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

_RVOL_WINDOW = 20
_TREND_LOOKBACK = 10  # kaç bar öncesine göre trend ölçülecek


def _compute_evol_trend(df: pd.DataFrame, lookback: int = _TREND_LOOKBACK) -> "float | None":
    """_compute_evol (evol_bt.py) ile AYNI seri — ama tek skaler yerine
    'şu anki percentile rank - lookback bar önceki percentile rank' döner."""
    try:
        if len(df) < 30 + lookback:
            return None
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        price_pct = close.pct_change() * 100.0
        rvol = volume / volume.rolling(_RVOL_WINDOW).mean()
        raw = price_pct / rvol.replace(0.0, np.nan)
        smoothed = raw.ewm(span=7, adjust=False).mean()
        valid = smoothed.dropna()
        if len(valid) < 20 + lookback:
            return None
        recent = valid.iloc[-100:]
        current = float(valid.iloc[-1])
        prior = float(valid.iloc[-1 - lookback])
        rank_current = float((recent < current).sum()) / len(recent)
        rank_prior = float((recent < prior).sum()) / len(recent)
        return round((rank_current - rank_prior) * 100.0, 2)
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
        rho, p = spearmanr(sub["evol_trend"], sub["realized_pnl"])
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
        if bars is None or len(bars) < 30 + _TREND_LOOKBACK:
            continue
        trend = _compute_evol_trend(bars)
        if trend is None:
            continue
        rows.append({
            "symbol": sig["symbol"], "signal_type": sig["signal_type"],
            "opened_at": sig["opened_at"], "evol_trend": trend,
            "realized_pnl": sig["realized_pnl"],
        })
        if i % 5000 == 0:
            print(f"  [{i}/{len(signals)}] işlendi, {len(rows)} geçerli satır")

    conn.close()
    df = pd.DataFrame(rows)
    print(f"\nGeçerli satır: {len(df)}\n")

    print("=== ANA TEST: EVOL trendi vs realized_pnl ===")
    _report_correlation(df, "gerçek")

    print("\n=== PF/WR: trend pozitif (uyanan) vs negatif (sönümlenen) ===")
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        waking = sub[sub["evol_trend"] > 0]
        fading = sub[sub["evol_trend"] <= 0]
        print(f"  {sig_type}:")
        print(f"    trend>0 (uyanan)     : n={len(waking)}, PF={_pf(waking['realized_pnl']):.3f}, "
              f"WR={(waking['realized_pnl'] > 0).mean() * 100:.1f}%")
        print(f"    trend<=0 (sönümlenen): n={len(fading)}, PF={_pf(fading['realized_pnl']):.3f}, "
              f"WR={(fading['realized_pnl'] > 0).mean() * 100:.1f}%")

    print("\n=== PLACEBO (trend rastgele karıştırıldı) ===")
    rng = np.random.default_rng(42)
    placebo = df.copy()
    placebo["evol_trend"] = rng.permutation(placebo["evol_trend"].values)
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
