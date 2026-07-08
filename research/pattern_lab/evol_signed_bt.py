"""
EVOL — yöne göre ayarlanmış (signed) versiyon testi.

Teori: mevcut EVOL (`evol_bt.py`) ham `ΔPrice%` kullanıyor — sinyalin yönünü
bilmiyor, sadece "fiyat yukarı mı gidiyor" ölçüyor. Bu yüzden Long'da zayıf
çıktı (Short'a göre daha az ayırt edici, split-period'da kararsız, Nötr
kohortunda hiç fark yaratmıyor) — çünkü "yüksek EVOL" hep "yukarı trend" demek,
bu da zaten Long sinyaliyle örtüşen bir bilgi, ek katkısı az.

Bu script `price_pct`'i sinyalin yönüyle çarpıyor (`* side`, tıpkı
`signal_processor.py::_compute_vol_momentum_scores`'daki price_score gibi) —
"fiyat SİNYALİN LEHİNE yönde ne kadar verimli hareket ediyor" ölçen simetrik
bir EVOL. Beklenti: Long ve Short'ta AYNI yönde (yüksek=iyi) davranan, daha
kararlı bir metrik.

Kullanım: python -m research.pattern_lab.evol_signed_bt
"""
import warnings

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config
from research.pattern_lab.evol_bt import _compute_evol, _fetch_bars, _fetch_signals

_RVOL_WINDOW = 20


def _compute_evol_signed(df: pd.DataFrame, signal_type: str) -> "float | None":
    """_compute_evol ile BİREBİR AYNI metodoloji — tek fark: price_pct sinyal
    yönüyle çarpılıyor, böylece yüksek skor HER ZAMAN 'sinyalin lehine' demek."""
    try:
        if len(df) < 30:
            return None
        side = 1.0 if signal_type == "Long" else -1.0
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        price_pct = close.pct_change() * 100.0 * side
        rvol = volume / volume.rolling(_RVOL_WINDOW).mean()
        raw = price_pct / rvol.replace(0.0, np.nan)
        smoothed = raw.ewm(span=7, adjust=False).mean()
        valid = smoothed.dropna()
        if len(valid) < 20:
            return None
        recent = valid.iloc[-100:]
        current = float(valid.iloc[-1])
        rank = float((recent < current).sum()) / len(recent)
        return round(rank * 100.0, 2)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _pf(pnls: pd.Series) -> float:
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    return float(wins / losses) if losses > 0 else float("inf")


def _report_correlation(df: pd.DataFrame, col: str, label: str) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        if len(sub) < 30:
            continue
        rho, p = spearmanr(sub[col], sub["realized_pnl"])
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
        if bars is None or len(bars) < 30:
            continue
        evol_unsigned = _compute_evol(bars)
        evol_signed = _compute_evol_signed(bars, sig["signal_type"])
        if evol_unsigned is None or evol_signed is None:
            continue
        rows.append({
            "symbol": sig["symbol"], "signal_type": sig["signal_type"],
            "opened_at": sig["opened_at"],
            "evol_unsigned": evol_unsigned, "evol_signed": evol_signed,
            "realized_pnl": sig["realized_pnl"],
        })
        if i % 5000 == 0:
            print(f"  [{i}/{len(signals)}] işlendi, {len(rows)} geçerli satır")

    conn.close()
    df = pd.DataFrame(rows)
    print(f"\nGeçerli satır: {len(df)}\n")

    print("=== KARŞILAŞTIRMA: ham (unsigned) vs yöne-ayarlı (signed) EVOL ===")
    print("-- Korelasyon --")
    _report_correlation(df, "evol_unsigned", "ham")
    _report_correlation(df, "evol_signed", "signed")

    print("\n-- Placebo (signed EVOL rastgele karıştırıldı) --")
    rng = np.random.default_rng(42)
    placebo = df.copy()
    placebo["evol_signed"] = rng.permutation(placebo["evol_signed"].values)
    _report_correlation(placebo, "evol_signed", "placebo-signed")

    print("\n-- Split-period (signed EVOL) --")
    mid = df["opened_at"].min() + (df["opened_at"].max() - df["opened_at"].min()) / 2
    for half_name, half_df in [
        ("ilk_yari", df[df["opened_at"] < mid]),
        ("ikinci_yari", df[df["opened_at"] >= mid]),
    ]:
        print(f"  -- {half_name} ({len(half_df)}) --")
        _report_correlation(half_df, "evol_signed", half_name)

    print("\n-- PF/WR bantları (signed EVOL, >=65 iyi, <35 kötü — HER İKİ yön için AYNI kural) --")
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        baseline = sub
        good = sub[sub["evol_signed"] >= 65]
        bad = sub[sub["evol_signed"] < 35]
        print(f"  {sig_type}:")
        print(f"    baseline     : n={len(baseline)}, PF={_pf(baseline['realized_pnl']):.3f}, "
              f"WR={(baseline['realized_pnl']>0).mean()*100:.1f}%")
        print(f"    signed>=65   : n={len(good)}, PF={_pf(good['realized_pnl']):.3f}, "
              f"WR={(good['realized_pnl']>0).mean()*100:.1f}%")
        print(f"    signed<35    : n={len(bad)}, PF={_pf(bad['realized_pnl']):.3f}, "
              f"WR={(bad['realized_pnl']>0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
