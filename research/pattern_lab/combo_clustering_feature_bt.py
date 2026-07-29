"""
Madde 1 (ML-yaklaşımı öncesi hızlı test): "sinyal kalabalığı" (aynı anda
kaç sembol aynı yönde sinyal veriyor) gerçekten sonucu tahmin ediyor mu?
[[project_short_split_period_weakness_24tem]]'de bir haftada 93 sembolün
aynı anda Short tetiklediği ve o dönemde WR'nin düştüğü bulunmuştu — ama bu
haftalık bir gözlemdi, burada İŞLEM BAZINDA (her sinyalin kendi anındaki
kalabalık boyutu) sürekli bir özellik olarak test ediliyor (24 Tem 2026,
kullanıcı isteği: "ML açısından değerlendirelim" öncesi hızlı ön-test).

4 gerçekçi replay sonucunu (RSI_Cross+HA_Cross, Long+Short) birleştirip,
her sinyal için "concurrent_count" = aynı YÖNDE, o sinyalden önceki N saat
içinde açılmış BAŞKA kaç sinyal var (tüm semboller, iki indikatör birlikte)
hesaplanıyor. Korelasyon + çeyreklik kırılım.

Kullanım: python -m research.pattern_lab.combo_clustering_feature_bt
"""

import os

import numpy as np
import pandas as pd

_FILES = {
    ("RSI_Cross", "Long"): "_cache_replay_long.parquet",
    ("RSI_Cross", "Short"): "_cache_replay_short.parquet",
    ("HA_Cross", "Long"): "_cache_replay_ha_long.parquet",
    ("HA_Cross", "Short"): "_cache_replay_ha_short.parquet",
}
_WINDOWS_HOURS = [1, 2, 4, 6]


def _load() -> pd.DataFrame:
    frames = []
    for (indicator, direction), fname in _FILES.items():
        path = os.path.join(os.path.dirname(__file__), fname)
        df = pd.read_parquet(path)
        df["indicator"] = indicator
        df["direction"] = direction
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["opened_at"] = pd.to_datetime(df["opened_at"])
    return df.sort_values("opened_at").reset_index(drop=True)


def _concurrent_count(times: np.ndarray, window_hours: float) -> np.ndarray:
    """Her zaman damgası için, KENDİSİNDEN ÖNCEKİ window_hours içinde açılmış
    (kendisi hariç) kaç tane var — sadece geçmişe bakıyor (look-ahead yok)."""
    window_ns = window_hours * 3600 * 1e9
    counts = np.zeros(len(times), dtype=int)
    left = 0
    for i in range(len(times)):
        while times[i] - times[left] > window_ns:
            left += 1
        counts[i] = i - left  # [left, i) aralığındaki sayı (i hariç kendisi)
    return counts


def main() -> None:
    df = _load()
    print(f"Toplam n={len(df)} işlem\n")

    for direction in ["Short", "Long"]:
        sub = df[df["direction"] == direction].sort_values("opened_at").reset_index(drop=True)
        times = sub["opened_at"].to_numpy().astype("datetime64[ns]").astype(np.int64)

        print("=" * 78)
        print(f"YÖN: {direction} (n={len(sub)})")
        print("=" * 78)
        for wh in _WINDOWS_HOURS:
            cnt = _concurrent_count(times, wh)
            sub[f"conc_{wh}h"] = cnt
            corr = np.corrcoef(cnt, sub["pnl_usd"].to_numpy())[0, 1]
            print(f"  pencere={wh}sa: ort. kalabalık={cnt.mean():.1f} (max={cnt.max()}), "
                  f"korelasyon(kalabalık, pnl_usd) = {corr:+.3f}")

        best_window = 4
        cnt = sub[f"conc_{best_window}h"]
        q = pd.qcut(cnt, 4, duplicates="drop")
        print(f"\n  Çeyreklik kırılım (pencere={best_window}sa kalabalık boyutuna göre):")
        g = sub.groupby(q, observed=True).agg(
            n=("pnl_usd", "size"), wr=("pnl_usd", lambda x: round((x > 0).mean() * 100, 1)),
            toplam=("pnl_usd", "sum"), ort=("pnl_usd", "mean"),
        )
        print(g)
        print()


if __name__ == "__main__":
    main()
