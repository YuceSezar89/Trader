"""
Hacim kuruması → teyit teorisi ("hoca doktrini", 19 Tem 2026) — GERÇEK açık
pozisyon `vol_score` seyriyle test (24 Tem 2026, kullanıcı isteği).

19 Tem'deki ilk deneme (`vol_dryup_confirmation_bt.py`) geçmiş kline'dan
zorlama pencere/eşik tahmin ediyordu — placebo %16 (yetersiz), eşik
sıkılaştırma p-hacking tuzağına düştü. Karar: `trade_snapshots` (5dk/1dk
periyodik, işlem ömrü boyunca vol_score kaydı) birikince GERÇEK trajektoriyle
tekrar test edilecekti — şimdi 46.772 kayıt (tf_alignment_live, 1536 işlem)
birikmiş, o test bu.

Yöntem: her işlem için İLK 2 snapshot (giriş-yakını) = vol_pre, sonraki 4
snapshot (bir süre sonra) = vol_post. kuruma_var = vol_pre düşük (<eşik),
teyit_geldi = vol_post ARTMIŞ (vol_pre'den anlamlı yüksek). pnl_usd
karşılaştırması, subset-placebo + split-period.

Kullanım: python -m research.pattern_lab.trade_snapshot_vol_dryup_bt
"""

import numpy as np
import pandas as pd

from research.pattern_lab.do_open_streak_full_clean_bt import _conn


def _fetch(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT ts.trade_id, pt.strategy, pt.pnl_usd, pt.opened_at, ts.taken_at, ts.vol_score
        FROM trade_snapshots ts
        JOIN paper_trades pt ON pt.id = ts.trade_id
        WHERE pt.status = 'closed' AND pt.pnl_usd IS NOT NULL AND ts.vol_score IS NOT NULL
        ORDER BY ts.trade_id, ts.taken_at ASC
        """
    )
    rows = cur.fetchall()
    return pd.DataFrame(
        rows, columns=["trade_id", "strategy", "pnl_usd", "opened_at", "taken_at", "vol_score"]
    )


def _stats(pnl: np.ndarray) -> dict:
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) == 0:
        return {"n": 0}
    g, l = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "n": len(pnl), "wr": round(float((pnl > 0).mean() * 100), 1),
        "toplam_$": round(float(pnl.sum()), 2), "ort_$": round(float(pnl.mean()), 3),
        "pf": round(float(g / l), 3) if l > 0 else float("inf"),
    }


def _subset_placebo(pool: np.ndarray, sub: np.ndarray, iters: int = 1000) -> float:
    rng = np.random.default_rng(42)
    real_mean = sub.mean()
    n = len(sub)
    ge = sum(1 for _ in range(iters) if rng.choice(pool, size=n, replace=False).mean() >= real_mean)
    return ge / iters * 100.0


def main() -> None:
    conn = _conn()
    cur = conn.cursor()
    df = _fetch(cur)
    conn.close()
    print(f"toplam snapshot satırı: {len(df)}, işlem sayısı: {df['trade_id'].nunique()}")
    print(f"strateji dağılımı:\n{df.groupby('strategy')['trade_id'].nunique()}")

    records = []
    for tid, g in df.groupby("trade_id"):
        g = g.sort_values("taken_at").reset_index(drop=True)
        if len(g) < 6:
            continue
        vol_pre = g["vol_score"].iloc[:2].mean()
        vol_post = g["vol_score"].iloc[2:6].mean()
        pnl = g["pnl_usd"].iloc[0]
        strategy = g["strategy"].iloc[0]
        records.append({"trade_id": tid, "strategy": strategy, "pnl_usd": pnl,
                         "vol_pre": vol_pre, "vol_post": vol_post, "opened_at": g["opened_at"].iloc[0]})
    rdf = pd.DataFrame(records)
    print(f"\nyeterli snapshot'ı (>=6) olan işlem: {len(rdf)}")

    print(f"\nvol_pre dağılımı: min={rdf['vol_pre'].min():.1f} medyan={rdf['vol_pre'].median():.1f} max={rdf['vol_pre'].max():.1f}")
    print(f"vol_post dağılımı: min={rdf['vol_post'].min():.1f} medyan={rdf['vol_post'].median():.1f} max={rdf['vol_post'].max():.1f}")

    print(f"\nGenel baseline: {_stats(rdf['pnl_usd'].to_numpy())}")
    pool = rdf["pnl_usd"].to_numpy()

    print("\n" + "=" * 78)
    print("Hipotez: vol_pre DÜŞÜK (kuruma) + vol_post - vol_pre YÜKSELDİ (teyit) → daha iyi mi?")
    print("=" * 78)
    for pre_th, post_delta_th in [(30, 10), (30, 20), (40, 15), (50, 20)]:
        sub = rdf[(rdf["vol_pre"] < pre_th) & ((rdf["vol_post"] - rdf["vol_pre"]) >= post_delta_th)]
        if len(sub) < 10:
            print(f"  vol_pre<{pre_th} & Δvol>={post_delta_th}: n={len(sub)} (yetersiz)")
            continue
        pb = _subset_placebo(pool, sub["pnl_usd"].to_numpy())
        print(f"  vol_pre<{pre_th} & Δvol>={post_delta_th}: {_stats(sub['pnl_usd'].to_numpy())}  subset-placebo=%{pb:.1f}")

    print("\n" + "=" * 78)
    print("Karşı test: vol_pre DÜŞÜK ama teyit GELMEDİ (Δvol küçük/negatif)")
    print("=" * 78)
    for pre_th in [30, 40, 50]:
        sub = rdf[(rdf["vol_pre"] < pre_th) & ((rdf["vol_post"] - rdf["vol_pre"]) < 5)]
        if len(sub) < 10:
            continue
        pb = _subset_placebo(pool, sub["pnl_usd"].to_numpy())
        print(f"  vol_pre<{pre_th} & teyitsiz: {_stats(sub['pnl_usd'].to_numpy())}  subset-placebo=%{pb:.1f}")

    corr = np.corrcoef(rdf["vol_post"] - rdf["vol_pre"], rdf["pnl_usd"])[0, 1]
    print(f"\nΔvol (post-pre) vs pnl_usd korelasyonu: {corr:.3f}")
    rng = np.random.default_rng(42)
    delta = (rdf["vol_post"] - rdf["vol_pre"]).to_numpy()
    shuffled = [np.corrcoef(rng.permutation(delta), rdf["pnl_usd"].to_numpy())[0, 1] for _ in range(300)]
    pct_ge = float(np.mean(np.abs(shuffled) >= abs(corr)) * 100)
    print(f"genel korelasyon placebo: %{pct_ge:.1f}")

    best_sub = rdf[(rdf["vol_pre"] < 40) & ((rdf["vol_post"] - rdf["vol_pre"]) >= 15)].sort_values("opened_at")
    if len(best_sub) >= 30:
        mid = best_sub["opened_at"].iloc[len(best_sub) // 2]
        fh = _stats(best_sub[best_sub["opened_at"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(best_sub[best_sub["opened_at"] >= mid]["pnl_usd"].to_numpy())
        print(f"\nvol_pre<40 & Δvol>=15 split-period: ilk yarı {fh} | ikinci yarı {sh}")
    else:
        print(f"\nsplit-period için yetersiz (n={len(best_sub)})")


if __name__ == "__main__":
    main()
