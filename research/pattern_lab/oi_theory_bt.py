"""
OI (Open Interest) teorisi — taze pozisyon mu, karşıt yön kapanışı mı?

Teori: Long sinyalinde OI ARTIYORSA taze/kaldıraçlı long pozisyonları açılıyor
demektir — sağlam, kendi ayakları üzerinde duran bir hareket. OI AZALIYORSA
short'lar pozisyon kapatıyor (squeeze) demektir — geçici, kırılgan, squeeze
bitince geri dönmeye yatkın. Short'ta simetrik: OI artışı = taze short (sağlam),
OI azalışı = long kapanışı/kapitülasyon (kırılgan).

EVOL'den farkı: hacim "ne kadar işlem oldu" der ama "kim aldı" demez — OI bu
ayrımı (yeni kaldıraçlı pozisyon mu, mevcut pozisyon kapanışı mı) doğrudan
ölçer. [[project_signal_radar_vision]]'daki Long-özel teorilerden biri.

Veri: signals.oi_data (JSON: oi/prev_oi/change_pct/ts) her sinyal anında zaten
yakalanmış — YENİ backfill/fetch gerekmiyor, EVOL testlerinin aksine per-sinyal
bar çekmeye gerek yok, tek SQL sorgusu yeterli (çok daha hızlı).

Kullanım: python -m research.pattern_lab.oi_theory_bt
"""
import json
import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config


def _pf(pnls: pd.Series) -> float:
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    return float(wins / losses) if losses > 0 else float("inf")


def _report_correlation(df: pd.DataFrame, label: str) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        if len(sub) < 30:
            continue
        rho, p = spearmanr(sub["oi_change"], sub["realized_pnl"])
        print(f"  [{label}] {sig_type}: n={len(sub)}, rho={rho:+.3f} (p={p:.4f})")


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT symbol, signal_type, interval, opened_at, oi_data, realized_pnl
        FROM signals
        WHERE status='closed' AND realized_pnl IS NOT NULL
          AND interval IN ('5m', '15m') AND oi_data IS NOT NULL
    """)
    rows = cur.fetchall()
    conn.close()
    print(f"Toplam sinyal (oi_data dolu): {len(rows)}")

    data = []
    for symbol, signal_type, interval, opened_at, oi_data, pnl in rows:
        try:
            oi_change = json.loads(oi_data).get("change_pct")
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        if oi_change is None:
            continue
        data.append({
            "symbol": symbol, "signal_type": signal_type, "interval": interval,
            "opened_at": opened_at, "oi_change": float(oi_change), "realized_pnl": pnl,
        })
    df = pd.DataFrame(data)
    print(f"Geçerli OI verisi: {len(df)}\n")

    print("=== ANA TEST: OI değişimi vs realized_pnl (korelasyon) ===")
    _report_correlation(df, "gerçek")

    print("\n=== PF/WR: OI ARTAN (taze pozisyon) vs OI AZALAN (karşıt kapanış) ===")
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        rising = sub[sub["oi_change"] > 0]
        falling = sub[sub["oi_change"] <= 0]
        print(f"  {sig_type}:")
        print(f"    OI artıyor  (n={len(rising)}): PF={_pf(rising['realized_pnl']):.3f}, "
              f"WR={(rising['realized_pnl'] > 0).mean() * 100:.1f}%")
        print(f"    OI azalıyor (n={len(falling)}): PF={_pf(falling['realized_pnl']):.3f}, "
              f"WR={(falling['realized_pnl'] > 0).mean() * 100:.1f}%")

    print("\n=== PLACEBO (OI değişimi rastgele karıştırıldı) ===")
    rng = np.random.default_rng(42)
    placebo = df.copy()
    placebo["oi_change"] = rng.permutation(placebo["oi_change"].values)
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
