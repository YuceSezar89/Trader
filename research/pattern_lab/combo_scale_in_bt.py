"""
Madde 1: aynı sembolde AÇIK POZİSYON varken yeni bir kalifiye sinyal
gelirse — şu anki davranış (SESSİZCE ATLA, paper_trade_manager.py::
_has_open_position) yerine POZİSYONU BÜYÜT (2. giriş, kendi SL/TP'siyle,
"2 TP 2 SL") daha mı iyi? (24 Tem 2026, kullanıcı isteği)

4 gerçekçi replay sonucunu (RSI_Cross Long/Short, HA_Cross Long/Short —
hepsi kendi SL/TP/trailing'iyle BAĞIMSIZ simüle edilmişti) birleştirip aynı
sembolde zaman-çakışan sinyalleri buluyor, iki politikayı karşılaştırıyor:

  A) ATLA (mevcut canlı davranış): sembolde AÇIK pozisyon varsa (yön fark
     etmez), yeni sinyal reddedilir.
  B) BÜYÜT (önerilen): sembolde AYNI YÖNDE açık pozisyon varsa, yeni sinyal
     de AYRI bir pozisyon olarak açılır (kendi bağımsız SL/TP'siyle — "2 TP
     2 SL"). TERS yönde açık pozisyon varsa yine ATLA (çakışan yöne
     büyütme yapılmıyor — bu script'in kendi varsayımı, tartışmaya açık).

Kullanım: python -m research.pattern_lab.combo_scale_in_bt
(önce rsi_cross_combo_realistic_replay_bt.py VE ha_cross_combo_realistic_replay_bt.py
çalıştırılmış olmalı — cache'lerini kullanır)
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
_POSITION_USD = 25.0


def _load() -> pd.DataFrame:
    frames = []
    for (indicator, direction), fname in _FILES.items():
        path = os.path.join(os.path.dirname(__file__), fname)
        if not os.path.exists(path):
            print(f"[uyarı] {fname} bulunamadı, atlanıyor")
            continue
        df = pd.read_parquet(path)
        df["indicator"] = indicator
        df["direction"] = direction
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["opened_at"] = pd.to_datetime(df["opened_at"])
    df["closed_at"] = df["opened_at"] + pd.to_timedelta(df["bars_held"] * 5, unit="min")
    return df.sort_values("opened_at").reset_index(drop=True)


def _stats(pnl: np.ndarray) -> dict:
    if len(pnl) == 0:
        return {"n": 0, "wr": None, "toplam_$": None, "pf": None}
    g, l = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "n": len(pnl), "wr": round(float((pnl > 0).mean() * 100), 1),
        "toplam_$": round(float(pnl.sum()), 2),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def _simulate(df: pd.DataFrame, allow_scale_in: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """Sembol bazlı: açık pozisyon listesini (yön + kapanış zamanı) takip eder.
    allow_scale_in=False -> Politika A (Atla): herhangi bir açık pozisyon varsa reddet.
    allow_scale_in=True  -> Politika B (Büyüt): AYNI yönde açıksa kabul et (2. bacak),
    TERS yönde açıksa yine reddet."""
    taken_rows = []
    max_concurrent_per_symbol: dict[str, int] = {}
    open_positions: dict[str, list[tuple[pd.Timestamp, str]]] = {}

    for _, row in df.iterrows():
        sym = row["symbol"]
        positions = open_positions.get(sym, [])
        positions = [(t, d) for t, d in positions if t > row["opened_at"]]

        same_dir_open = [d for _, d in positions if d == row["direction"]]
        opp_dir_open = [d for _, d in positions if d != row["direction"]]

        accept = False
        if not positions:
            accept = True
        elif allow_scale_in and not opp_dir_open and same_dir_open:
            accept = True
        # else: reddedilir (Politika A: her durumda; Politika B: ters yön açıkken)

        if accept:
            positions.append((row["closed_at"], row["direction"]))
            taken_rows.append(row)
        open_positions[sym] = positions
        max_concurrent_per_symbol[sym] = max(max_concurrent_per_symbol.get(sym, 0), len(positions))

    taken = pd.DataFrame(taken_rows)
    concurrency = np.array(list(max_concurrent_per_symbol.values())) if max_concurrent_per_symbol else np.array([0])
    return taken, concurrency


def main() -> None:
    df = _load()
    print(f"Toplam n={len(df)} sinyal (4 strateji birleşik), {df['opened_at'].min()} .. {df['opened_at'].max()}\n")
    print(df.groupby(["indicator", "direction"]).size())
    print()

    taken_a, conc_a = _simulate(df, allow_scale_in=False)
    taken_b, conc_b = _simulate(df, allow_scale_in=True)

    print("=" * 78)
    print("POLİTİKA A — ATLA (mevcut canlı davranış: açık pozisyon varsa reddet)")
    print("=" * 78)
    print(f"  {_stats(taken_a['pnl_usd'].to_numpy())}")
    print(f"  sembol-başına zirve eşzamanlı pozisyon: ort={conc_a.mean():.2f} zirve={conc_a.max()}")

    print("\n" + "=" * 78)
    print("POLİTİKA B — BÜYÜT (aynı yönde açıksa 2. bacak aç, kendi SL/TP'siyle)")
    print("=" * 78)
    print(f"  {_stats(taken_b['pnl_usd'].to_numpy())}")
    print(f"  sembol-başına zirve eşzamanlı pozisyon: ort={conc_b.mean():.2f} zirve={conc_b.max()}")

    extra_n = len(taken_b) - len(taken_a)
    extra_pnl = taken_b["pnl_usd"].sum() - taken_a["pnl_usd"].sum()
    print(f"\n=== FARK: B, A'ya göre {extra_n} EK işlem alıyor, ek toplam PnL: ${extra_pnl:.2f} ===")

    # sadece B'de alınan (A'da reddedilen) "2. bacak" işlemlerin kendi performansı
    merged = taken_b.merge(taken_a[["symbol", "opened_at"]], on=["symbol", "opened_at"], how="left", indicator=True)
    scale_in_only = merged[merged["_merge"] == "left_only"]
    print(f"\n=== SADECE 2. bacak (scale-in) işlemlerinin kendi performansı ===")
    print(f"  {_stats(scale_in_only['pnl_usd'].to_numpy())}")

    print("\n=== Yön bazında scale-in işlemlerinin dağılımı ===")
    print(scale_in_only.groupby(["indicator", "direction"]).agg(
        n=("pnl_usd", "size"), toplam=("pnl_usd", "sum"), wr=("pnl_usd", lambda x: round((x > 0).mean() * 100, 1))
    ))

    print("\n=== Sermaye/risk karşılaştırması (zirve eşzamanlı pozisyon dağılımı) ===")
    print(f"  A: p90={np.percentile(conc_a,90):.1f} p99={np.percentile(conc_a,99):.1f} zirve={conc_a.max()} (${conc_a.max()*_POSITION_USD:.0f} tek sembolde)")
    print(f"  B: p90={np.percentile(conc_b,90):.1f} p99={np.percentile(conc_b,99):.1f} zirve={conc_b.max()} (${conc_b.max()*_POSITION_USD:.0f} tek sembolde)")


if __name__ == "__main__":
    main()
