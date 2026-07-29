"""
TF-hizalanma + erken-ayrışma — TAM STRATEJİ simülasyonu (20 Tem 2026).

rsi_cross_tf_alignment_bt.py'nin bulgusu ("üst tercil realized_pnl ort. iyi")
bir SEÇİM ETKİSİ ölçüyordu — alt 2/3'ün erken-kapatılsa ne kaybedeceği hiç
ölçülmemişti (kullanıcı sorusu: "SL sayısı ve zarar artmaz mı?").

Bu script TAM stratejiyi simüle ediyor — kohorttaki HER üyeye (üst tercil DAHİL
alt 2/3 DAHİL) uygulanacak gerçekçi kural:
  üst tercil (early_rank >= 2/3)  → realized_pnl (doğal SL/TP/reversal ile biter)
  alt 2/3 (early_rank < 2/3)      → early_pct (N-bar'da erken kapatılır — bu
                                     zaten mevcut, N-bar sonraki fiyattan hesaplı)

Baseline (hiç filtre yok, hepsi realized_pnl) ile karşılaştırılıyor.

Kullanım: python -m research.pattern_lab.tf_alignment_full_strategy_bt
"""

from research.pattern_lab.rsi_cross_tf_alignment_bt import (
    _add_early_pct,
    _add_htf_alignment,
    _build_cohorts,
    _fetch,
)

_EARLY_BARS_LIVE = 3  # tf_alignment_gate.py'nin CANLIDA kullandığı değer


def _stats(pnl_series) -> dict:
    n = len(pnl_series)
    if n == 0:
        return {"n": 0, "wr": 0.0, "ort": 0.0, "toplam": 0.0}
    wins = pnl_series[pnl_series > 0].sum()
    losses = -pnl_series[pnl_series < 0].sum()
    return {
        "n": n,
        "wr": round(float((pnl_series > 0).mean() * 100), 1),
        "ort": round(float(pnl_series.mean()), 4),
        "toplam": round(float(pnl_series.sum()), 2),
        "pf": round(float(wins / losses), 3) if losses > 0 else float("inf"),
    }


def main() -> None:
    df = _fetch(_EARLY_BARS_LIVE)
    df = _add_early_pct(df)
    cohorts = _build_cohorts(df)
    print(f"Kohort içi sinyal: {len(cohorts)} (_EARLY_BARS={_EARLY_BARS_LIVE}, canlıdaki değer)\n")

    aligned = _add_htf_alignment(cohorts)

    for sig_type in ["Long", "Short"]:
        sub = aligned[(aligned["signal_type"] == sig_type) & (aligned["aligned_count"] == 2)].copy()
        if len(sub) < 30:
            print(f"{sig_type}: yetersiz örnek (n={len(sub)})\n")
            continue

        tercile_cut = sub["early_rank"].quantile(0.667)
        sub["ust_tercil"] = sub["early_rank"] >= tercile_cut

        # TAM STRATEJİ: üst tercil -> realized_pnl, alt 2/3 -> early_pct (erken kapatılmış gibi)
        sub["full_strategy_pnl"] = sub["early_pct"].where(~sub["ust_tercil"], sub["realized_pnl"])

        baseline = _stats(sub["realized_pnl"])
        idealized_top_only = _stats(sub.loc[sub["ust_tercil"], "realized_pnl"])
        bottom_if_held_full = _stats(sub.loc[~sub["ust_tercil"], "realized_pnl"])
        bottom_if_closed_early = _stats(sub.loc[~sub["ust_tercil"], "early_pct"])
        full_strategy = _stats(sub["full_strategy_pnl"])

        print(f"=== {sig_type} (aligned_count=2, n={len(sub)}) ===")
        print(f"  BASELINE (filtre yok, hepsi doğal biter):        {baseline}")
        print(f"  Sadece üst tercil (mevcut rapor, n={idealized_top_only['n']}):     {idealized_top_only}")
        print(f"  Alt 2/3 — TAM tutulsaydı (doğal biter):          {bottom_if_held_full}")
        print(f"  Alt 2/3 — ERKEN kapatılsaydı ({_EARLY_BARS_LIVE} bar'da):        {bottom_if_closed_early}")
        print(f"  >>> TAM STRATEJİ (üst=tam tutma + alt=erken kapat): {full_strategy}")
        print()


if __name__ == "__main__":
    main()
