"""
SMC Yapı-Uyum (market_structure) — TEMİZ yöntemle yeniden doğrulama (21 Tem 2026).

`signals.market_structure` sinyal açılışında BİR KEZ hesaplanıp donuyor
(bkz. signals/market_context.py::compute_smc_market_structure) — entry
öncesi 50 bar'daki son BOS/CHoCH yönü, sinyal yönüyle karşılaştırılıyor:
  Uyum↑/Uyum↓  — sinyal mevcut yapıyla aynı yönde (trend devamı)
  Karşı↑/Karşı↓ — sinyal yapıya karşı (dönüş denemesi)
Bu alan look-ahead içermez (geçmiş bar'lardan, sinyal anında donduruluyor).

12 Tem'de ham ortalama-PnL karşılaştırmasıyla "Uyum > Karşı" yönü
doğrulanmıştı ama resmi 4/5-kapılı test hiç yapılmamıştı — bugün standart
(sabit 24-bar ileri getiri, çıkış mekanizmasından bağımsız) ile test
ediyoruz. FVG aynı ham-karşılaştırma tuzağına düşüp disiplinli testte
çürümüştü (12 Tem) — aynı riski burada da arıyoruz.

Verimlilik: market_structure zaten DB'de hazır olduğundan (bar'dan yeniden
hesaplama YOK), sinyal başına sorgu yerine (symbol, interval) grubu başına
TEK sorguyla tüm close/bucket serisi çekilip searchsorted ile ileri-bar
indeksi bulunuyor — Madde-4'teki sinyal-başına-sorgu düzenine göre çok
daha hızlı.

Kullanım: python -m research.pattern_lab.smc_clean_forward_return_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_INTERVAL_TABLE = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}
_FORWARD_BARS = 24
_PLACEBO_ITER = 300
_REGIME_SPLIT = pd.Timestamp("2026-06-08")
_UYUM = {"Uyum↑", "Uyum↓"}
_KARSI = {"Karşı↑", "Karşı↓"}


def _fetch_signals(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT id, symbol, interval, signal_type, opened_at, open_price, market_structure
        FROM signals
        WHERE interval IN ('5m','15m','1h','4h')
          AND open_price IS NOT NULL AND open_price > 0
          AND market_structure IN ('Uyum↑','Uyum↓','Karşı↑','Karşı↓')
        ORDER BY opened_at ASC
        """
    )
    cols = ["id", "symbol", "interval", "signal_type", "opened_at", "open_price", "market_structure"]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def _fetch_series(cur, symbol: str, interval: str) -> tuple[np.ndarray, np.ndarray]:
    table = _INTERVAL_TABLE[interval]
    cur.execute(f"SELECT bucket, close FROM {table} WHERE symbol=%s ORDER BY bucket ASC", (symbol,))
    rows = cur.fetchall()
    if not rows:
        return np.array([]), np.array([])
    buckets = np.array([r[0] for r in rows])
    closes = np.array([float(r[1]) for r in rows])
    return buckets, closes


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
    print(f"[fetch] {len(signals)} sinyal (market_structure dolu, status/close_reason önemsiz)")

    records = []
    groups = signals.groupby(["symbol", "interval"])
    n_groups = len(groups)
    for gi, ((symbol, interval), sub) in enumerate(groups):
        buckets, closes = _fetch_series(cur, symbol, interval)
        if len(buckets) < _FORWARD_BARS + 1:
            continue
        for _, row in sub.iterrows():
            idx0 = np.searchsorted(buckets, np.datetime64(row["opened_at"]), side="left")
            fwd_idx = idx0 + (_FORWARD_BARS - 1)
            if fwd_idx >= len(closes):
                continue
            side = 1.0 if row["signal_type"] == "Long" else -1.0
            fwd_ret = (closes[fwd_idx] - row["open_price"]) / row["open_price"] * 100.0 * side
            grp = "Uyum" if row["market_structure"] in _UYUM else "Karşı"
            records.append({
                "grp": grp, "sub_label": row["market_structure"],
                "fwd_ret": fwd_ret, "opened_at": row["opened_at"],
            })
        if (gi + 1) % 200 == 0:
            print(f"  ... {gi+1}/{n_groups} grup")

    conn.close()
    df = pd.DataFrame(records)
    print(f"\n[collect] {len(df)} sinyal için SMC yapı-uyum + {_FORWARD_BARS}-bar ileri getiri\n")
    if df.empty:
        return

    print("=== [1] Uyum vs Karşı — sabit-ileri-getiri (TEMİZ hedef) ===")
    g = df.groupby("grp")["fwd_ret"].agg(ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100)
    print(g.to_string())
    uyum = df[df["grp"] == "Uyum"]["fwd_ret"]
    karsi = df[df["grp"] == "Karşı"]["fwd_ret"]
    print(f"\n  Uyum (n={len(uyum)}): {_stats(uyum.to_numpy())}")
    print(f"  Karşı (n={len(karsi)}): {_stats(karsi.to_numpy())}")

    print("\n  Alt kırılım (4 etiket ayrı):")
    g2 = df.groupby("sub_label")["fwd_ret"].agg(ort="mean", n="count", wr=lambda s: (s > 0).mean() * 100)
    print(g2.to_string())

    print("\n=== [2] Placebo (grup etiketini karıştır) ===")
    rng = np.random.default_rng(42)
    real_diff = uyum.mean() - karsi.mean()
    labels = df["grp"].to_numpy()
    target = df["fwd_ret"].to_numpy()
    count_ge = 0
    for _ in range(_PLACEBO_ITER):
        shuffled = rng.permutation(labels)
        fake_uyum = target[shuffled == "Uyum"]
        fake_karsi = target[shuffled == "Karşı"]
        fake_diff = fake_uyum.mean() - fake_karsi.mean()
        if abs(fake_diff) >= abs(real_diff):
            count_ge += 1
    print(f"  gerçek Uyum-Karşı farkı={real_diff:+.4f} — rastgele etiketlemede "
          f"aynı/daha büyük fark sıklığı: %{count_ge/_PLACEBO_ITER*100:.1f}")

    print("\n=== [3] Split-period (çöküş vs toparlanma) ===")
    crash = df[df["opened_at"] < _REGIME_SPLIT]
    recovery = df[df["opened_at"] >= _REGIME_SPLIT]
    for label, sub in (("Çöküş+öncesi", crash), ("Toparlanma", recovery)):
        if len(sub) >= 100:
            u = sub[sub["grp"] == "Uyum"]["fwd_ret"]
            k = sub[sub["grp"] == "Karşı"]["fwd_ret"]
            print(f"  {label} (n={len(sub)}): Uyum ort={u.mean():+.4f}(n={len(u)}) "
                  f"| Karşı ort={k.mean():+.4f}(n={len(k)})")
        else:
            print(f"  {label}: yetersiz örnek (n={len(sub)})")

    print("\n=== [4] Kronolojik yarı-yarı (Uyum grubu İÇİNDE) ===")
    uyum_sorted = df[df["grp"] == "Uyum"].sort_values("opened_at")
    if len(uyum_sorted) >= 50:
        mid = uyum_sorted["opened_at"].iloc[len(uyum_sorted)//2]
        fh = _stats(uyum_sorted[uyum_sorted["opened_at"] < mid]["fwd_ret"].to_numpy())
        sh = _stats(uyum_sorted[uyum_sorted["opened_at"] >= mid]["fwd_ret"].to_numpy())
        print(f"  ilk yarı: {fh}")
        print(f"  ikinci yarı: {sh}")
    else:
        print("  yetersiz örnek")

    print("\n=== [5] Kronolojik yarı-yarı (Karşı grubu İÇİNDE) ===")
    karsi_sorted = df[df["grp"] == "Karşı"].sort_values("opened_at")
    if len(karsi_sorted) >= 50:
        mid = karsi_sorted["opened_at"].iloc[len(karsi_sorted)//2]
        fh = _stats(karsi_sorted[karsi_sorted["opened_at"] < mid]["fwd_ret"].to_numpy())
        sh = _stats(karsi_sorted[karsi_sorted["opened_at"] >= mid]["fwd_ret"].to_numpy())
        print(f"  ilk yarı: {fh}")
        print(f"  ikinci yarı: {sh}")
    else:
        print("  yetersiz örnek")


if __name__ == "__main__":
    main()
