"""
ADX(14) — Hurst/FDI ile AYNI temiz yöntemle test (21 Tem 2026).

Not: ADX daha önce İKİ farklı bağlamda denenmişti ama ikisi de bu soruya
tam cevap vermiyordu:
  - rsi_cross_adx_regime_bt.py (11 Tem): BTC'nin GENEL ADX'i, "piyasa
    çökerken Long neden zayıf" sorusu için — kısmi iyileşme, yetersiz.
  - adx_trend_strength_bt.py (18 Tem): sembolün KENDİ ADX'i, sinyal anında
    — ama realized_pnl hedefliydi (kirli etiket) VE sonucu hiç kaydedilmemiş.

Bu script: sembolün KENDİ ADX'i, sinyal açılış anında, çıkış mekanizmasından
BAĞIMSIZ sabit 24-bar ileri getiri hedefiyle (indicators.core.calculate_adx,
projenin gerçek/canlı ADX fonksiyonu — yeniden icat edilmedi).

Hipotez: yüksek ADX (gerçek trend) sinyalleri, düşük ADX'e (yatay/sıkışık)
göre daha iyi forward-return vermeli — bu doğruysa reversal-noise filtresi
olarak kullanılabilir.

Kullanım: python -m research.pattern_lab.adx_clean_forward_return_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config
from indicators.core import calculate_adx

_INTERVAL_TABLE = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_ADX_LEN = 14
_HISTORY_BARS = 100  # ADX(14) için warmup + stabilite payı
_FORWARD_BARS = 24
_MAX_SIGNALS = 6000
_PLACEBO_ITER = 300
_REGIME_SPLIT = pd.Timestamp("2026-06-08")


def _fetch_signals(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, interval, signal_type, opened_at, open_price
        FROM signals
        WHERE interval IN ('5m','15m','1h') AND open_price IS NOT NULL AND open_price > 0
        ORDER BY random() LIMIT %s
        """,
        (_MAX_SIGNALS,),
    )
    cols = ["id", "symbol", "interval", "signal_type", "opened_at", "open_price"]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def _fetch_hlc_before(cur, symbol: str, interval: str, before, n: int) -> pd.DataFrame:
    table = _INTERVAL_TABLE[interval]
    cur.execute(
        f"SELECT high, low, close FROM {table} WHERE symbol=%s AND bucket < %s ORDER BY bucket DESC LIMIT %s",
        (symbol, before, n),
    )
    rows = cur.fetchall()
    if len(rows) < n:
        return pd.DataFrame()
    rows = list(reversed(rows))
    return pd.DataFrame(rows, columns=["high", "low", "close"])


def _fetch_forward_price(cur, symbol: str, interval: str, after, n_bars: int) -> float | None:
    table = _INTERVAL_TABLE[interval]
    cur.execute(
        f"SELECT close FROM {table} WHERE symbol=%s AND bucket >= %s ORDER BY bucket ASC LIMIT 1 OFFSET %s",
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


def main() -> None:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    signals = _fetch_signals(cur)
    print(f"[fetch] {len(signals)} sinyal örneklendi (status/close_reason önemsiz)")

    records = []
    for i, row in signals.iterrows():
        hlc = _fetch_hlc_before(cur, row["symbol"], row["interval"], row["opened_at"], _HISTORY_BARS)
        if len(hlc) < _ADX_LEN * 3:
            continue
        try:
            adx, plus_di, minus_di = calculate_adx(hlc, adxlen=_ADX_LEN, dilen=_ADX_LEN)
        except Exception:
            continue
        adx_val = adx.iloc[-1]
        if pd.isna(adx_val):
            continue
        fwd_price = _fetch_forward_price(cur, row["symbol"], row["interval"], row["opened_at"], _FORWARD_BARS)
        if fwd_price is None:
            continue
        side = 1.0 if row["signal_type"] == "Long" else -1.0
        fwd_ret = (fwd_price - row["open_price"]) / row["open_price"] * 100.0 * side
        records.append({"adx": float(adx_val), "fwd_ret": fwd_ret, "opened_at": row["opened_at"]})
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(signals)}")

    conn.close()
    df = pd.DataFrame(records)
    print(f"\n[collect] {len(df)} sinyal için ADX + {_FORWARD_BARS}-bar ileri getiri hesaplandı\n")
    if df.empty:
        return

    print(f"ADX dağılımı: min={df['adx'].min():.1f} medyan={df['adx'].median():.1f} "
          f"max={df['adx'].max():.1f} std={df['adx'].std():.1f}")
    print(f"ADX<20 (yatay say.) oranı: %{(df['adx']<20).mean()*100:.1f} | "
          f"ADX>=25 (trend say.) oranı: %{(df['adx']>=25).mean()*100:.1f}")

    print(f"\n=== [1] Korelasyon (ADX vs {_FORWARD_BARS}-bar ileri getiri) ===")
    rho, p = spearmanr(df["adx"], df["fwd_ret"])
    print(f"  rho={rho:+.4f} (p={p:.4f})")

    tercile = pd.qcut(df["adx"], 4, labels=["1.en düşük", "2", "3", "4.en yüksek"], duplicates="drop")
    g = df.groupby(tercile, observed=True)["fwd_ret"].agg(
        ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100
    )
    print("\n  ADX dörtte-biri:")
    print(g.to_string())

    print("\n=== [2] Placebo (ADX karıştırma) ===")
    rng = np.random.default_rng(42)
    vals = df["adx"].to_numpy()
    target = df["fwd_ret"].to_numpy()
    count_ge = sum(
        1 for _ in range(_PLACEBO_ITER)
        if abs(spearmanr(rng.permutation(vals), target)[0]) >= abs(rho)
    )
    print(f"  gerçek rho={rho:+.4f} — rastgele karıştırmada aynı/daha büyük |rho| sıklığı: %{count_ge/_PLACEBO_ITER*100:.1f}")

    print("\n=== [3] Split-period (çöküş vs toparlanma) ===")
    crash = df[df["opened_at"] < _REGIME_SPLIT]
    recovery = df[df["opened_at"] >= _REGIME_SPLIT]
    for label, sub in (("Çöküş+öncesi", crash), ("Toparlanma", recovery)):
        if len(sub) >= 100:
            r, p2 = spearmanr(sub["adx"], sub["fwd_ret"])
            print(f"  {label} (n={len(sub)}): rho={r:+.4f} (p={p2:.4f})")
        else:
            print(f"  {label}: yetersiz örnek (n={len(sub)})")

    print("\n=== [4] Filtre simülasyonu (endüstri standardı eşikler) ===")
    for thr in (15, 20, 25, 30):
        sub = df[df["adx"] >= thr]
        s = _stats(sub["fwd_ret"].to_numpy())
        elenen = len(df) - len(sub)
        print(f"  ADX>={thr}: kalan n={s.get('n',0):>5} (elenen {elenen:>5}) | "
              f"WR={s.get('wr',0):>5.1f}% ort%={s.get('ort_%',0):>7.3f} PF={s.get('pf',0):>6.3f}")

    print("\n=== [5] ADX>=30 eşiğinin ÖZEL doğrulaması ===")
    real_group = df[df["adx"] >= 30]
    n_group = len(real_group)
    real_stats = _stats(real_group["fwd_ret"].to_numpy())
    real_mean = real_group["fwd_ret"].mean()
    print(f"  Gerçek ADX>=30 grubu: n={n_group}, {real_stats}")

    print(f"\n  [5a] Permütasyon testi: {n_group} büyüklüğünde 1000 RASTGELE altküme "
          f"ortalaması, gerçek ortalamayı (={real_mean:+.4f}) kaç kez geçiyor/eşitliyor?")
    rng2 = np.random.default_rng(7)
    all_vals = df["fwd_ret"].to_numpy()
    count_ge2 = 0
    random_means = []
    for _ in range(1000):
        sample = rng2.choice(all_vals, size=n_group, replace=False)
        m = sample.mean()
        random_means.append(m)
        if m >= real_mean:
            count_ge2 += 1
    random_means = np.array(random_means)
    print(f"    rastgele altkümelerin ortalaması: medyan={np.median(random_means):+.4f}, "
          f"std={random_means.std():.4f}")
    print(f"    gerçek ortalamayı geçen/eşitleyen rastgele altküme oranı: %{count_ge2/1000*100:.1f} "
          f"(düşükse: ADX>=30 seçimi tesadüften iyi)")

    print(f"\n  [5b] Kronolojik yarı-yarı tutarlılık (ADX>=30 grubu İÇİNDE):")
    real_group_sorted = real_group.sort_values("opened_at")
    mid = real_group_sorted["opened_at"].iloc[len(real_group_sorted) // 2]
    first_half = real_group_sorted[real_group_sorted["opened_at"] < mid]
    second_half = real_group_sorted[real_group_sorted["opened_at"] >= mid]
    print(f"    ilk yarı  (n={len(first_half)}):  {_stats(first_half['fwd_ret'].to_numpy())}")
    print(f"    ikinci yarı (n={len(second_half)}): {_stats(second_half['fwd_ret'].to_numpy())}")


if __name__ == "__main__":
    main()
