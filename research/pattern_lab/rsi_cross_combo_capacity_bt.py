"""
Madde 1: eşzamanlı pozisyon / kapasite analizi (24 Tem 2026, kullanıcı
isteği) — Long üçlü + Short ikili filtrelerinin CANLIDA kaç eşzamanlı
pozisyon ürettiği, $1000/$2000 sermayenin yeterli olup olmadığı, kapasite
tavanı konursa kaç sinyalin (ve ne kadar kârın) kaçırılacağı.

Girdi: rsi_cross_combo_realistic_replay_bt.py'nin kaydettiği trade-level
sonuçlar (_cache_replay_long.parquet / _cache_replay_short.parquet —
opened_at, bars_held, pnl_usd, symbol).

20 Tem'deki "ACE" bug'ının aynısını burada SİMÜLE ediyoruz: kapasite
dolduğunda sinyal sessizce atlanır, sonra tekrar denenmez (tf_alignment_
gate.py'nin FIFO mantığıyla aynı) — [[project_tf_alignment_early_divergence]].

Kullanım: python -m research.pattern_lab.rsi_cross_combo_capacity_bt
(önce rsi_cross_combo_realistic_replay_bt.py çalıştırılmış olmalı)
"""

import os

import numpy as np
import pandas as pd

_LONG_CACHE = os.path.join(os.path.dirname(__file__), "_cache_replay_long.parquet")
_SHORT_CACHE = os.path.join(os.path.dirname(__file__), "_cache_replay_short.parquet")
_POSITION_USD = 25.0
_CAPS = [10, 20, 40, 80, 160]  # $250 / $500 / $1000 / $2000 / $4000 sermayeye karşılık


def _load() -> pd.DataFrame:
    long_df = pd.read_parquet(_LONG_CACHE)
    long_df["strategy"] = "Long üçlü"
    short_df = pd.read_parquet(_SHORT_CACHE)
    short_df["strategy"] = "Short ikili"
    df = pd.concat([long_df, short_df], ignore_index=True)
    df["opened_at"] = pd.to_datetime(df["opened_at"])
    df["closed_at"] = df["opened_at"] + pd.to_timedelta(df["bars_held"] * 5, unit="min")
    return df.sort_values("opened_at").reset_index(drop=True)


def _concurrency_profile(df: pd.DataFrame) -> None:
    events = []
    for _, row in df.iterrows():
        events.append((row["opened_at"], 1))
        events.append((row["closed_at"], -1))
    events.sort(key=lambda x: (x[0], -x[1]))  # aynı anda açılış kapanıştan önce sayılsın (muhafazakâr)

    concurrent = 0
    series = []
    for ts, delta in events:
        concurrent += delta
        series.append(concurrent)
    arr = np.array(series)

    print(f"  n işlem={len(df)}  gözlenen zaman aralığı: {df['opened_at'].min()} .. {df['closed_at'].max()}")
    print(f"  eşzamanlı pozisyon: ort={arr.mean():.2f}  medyan={np.median(arr):.0f}  "
          f"p90={np.percentile(arr,90):.0f}  p95={np.percentile(arr,95):.0f}  p99={np.percentile(arr,99):.0f}  "
          f"ZİRVE={arr.max()}")
    print(f"  zirve sermaye ihtiyacı: ${arr.max()*_POSITION_USD:.0f}  (p95 sermaye: ${np.percentile(arr,95)*_POSITION_USD:.0f})")


def _simulate_cap(df: pd.DataFrame, cap: int) -> dict:
    """FIFO kapasite simülasyonu: sinyal geldiğinde açık pozisyon sayısı
    cap'e ulaşmışsa sinyal SESSİZCE ATLANIR (tekrar denenmez) — canlı
    sistemin gerçek davranışı (paper_trade_manager.py:MAX_OPEN kontrolü)."""
    df = df.sort_values("opened_at").reset_index(drop=True)
    open_until: list[pd.Timestamp] = []  # açık pozisyonların kapanış zamanları
    taken_pnl = 0.0
    skipped_pnl = 0.0
    n_taken = 0
    n_skipped = 0
    for _, row in df.iterrows():
        open_until = [t for t in open_until if t > row["opened_at"]]
        if len(open_until) < cap:
            open_until.append(row["closed_at"])
            taken_pnl += row["pnl_usd"]
            n_taken += 1
        else:
            skipped_pnl += row["pnl_usd"]
            n_skipped += 1
    return {
        "cap": cap, "sermaye": cap * _POSITION_USD, "n_alinan": n_taken, "n_kacan": n_skipped,
        "alinan_pnl": round(taken_pnl, 2), "kacan_pnl": round(skipped_pnl, 2),
        "kacan_pct": round(n_skipped / len(df) * 100, 1) if len(df) else 0,
    }


def main() -> None:
    df = _load()
    days_span = (df["closed_at"].max() - df["opened_at"].min()).total_seconds() / 86400
    print(f"Toplam n={len(df)} işlem, {days_span:.1f} günlük pencere\n")

    print("=" * 78)
    print("BİRLEŞİK (Long üçlü + Short ikili aynı hesapta) — eşzamanlı pozisyon profili")
    print("=" * 78)
    _concurrency_profile(df)

    for strat in ["Long üçlü", "Short ikili"]:
        print(f"\n  -- sadece {strat} --")
        _concurrency_profile(df[df["strategy"] == strat])

    print("\n" + "=" * 78)
    print("KAPASİTE TAVANI SİMÜLASYONU — tavan konursa kaç sinyal/ne kadar kâr kaçırılır")
    print("=" * 78)
    print(f"{'tavan':>6} {'sermaye':>9} | {'alınan':>7} {'kaçan':>7} {'kaçan%':>7} | {'alınan $':>10} {'kaçan $':>10}")
    print("-" * 70)
    for cap in _CAPS:
        r = _simulate_cap(df, cap)
        print(f"{r['cap']:>6} {'$'+str(int(r['sermaye'])):>9} | {r['n_alinan']:>7} {r['n_kacan']:>7} "
              f"{r['kacan_pct']:>6}% | {'$'+str(r['alinan_pnl']):>10} {'$'+str(r['kacan_pnl']):>10}")


if __name__ == "__main__":
    main()
