"""
Kohort-içi göreceli sıralama testi — aynı anda aynı sinyali alan semboller
arasından hangisinin gerçekten hareket ettiğini ERSI+CVD+OI kohort-içi
rank'i ile ayırt edebiliyor muyuz? [[project_signal_radar_vision]]'daki
"kesitsel sentez" önerisinin Pattern Lab doğrulaması.

Neden gerekli: devisso_score'un MUTLAK değeri realized_pnl ile ilişkisiz
çıkmıştı (|r|<0.05, bkz. [[project_devisso_ersi]]) — ama bu KESİTSEL/göreceli
(aynı kohort içindeki sıralama) farklı bir soru, otomatik geçersiz kılınmıyor.

Veri: signals tablosunda devisso_score/cvd_slope/oi_data zaten dolu, yeni
backfill gerekmiyor. cvd_slope 2026-06-26 öncesi NULL (özellik sonradan
eklendi) — bu yüzden opened_at bu tarihten itibaren alınıyor.

Kohort tanımı: aynı (indicators, signal_type, interval, opened_at) — yani
aynı bar kapanışında, aynı indikatör+yönde tetiklenen semboller.

Bu projenin battle-tested 3 kapısı uygulanıyor (bkz. [[project_pattern_lab]]):
placebo (rank kohort içinde rastgele karıştırılır), split-period (dönem
ikiye bölünür), indikatör ailesine göre kırılım.

Kullanım: python -m research.pattern_lab.cohort_rank_bt
"""
import json
import warnings

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import Config

_MIN_COHORT_SIZE = 2
_CVD_START = "2026-06-26"
_COHORT_KEY = ["indicators", "signal_type", "interval", "opened_at"]
_COMPONENTS = ["devisso_score", "cvd_slope", "oi_delta"]


def _fetch() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT symbol, indicators, signal_type, interval, opened_at,
               devisso_score, cvd_slope, oi_data, realized_pnl
        FROM signals
        WHERE status='closed' AND realized_pnl IS NOT NULL
          AND devisso_score IS NOT NULL AND cvd_slope IS NOT NULL AND oi_data IS NOT NULL
          AND opened_at >= %s
    """
    df = pd.read_sql(q, conn, params=(_CVD_START,))
    conn.close()
    df["oi_delta"] = df["oi_data"].apply(lambda s: json.loads(s).get("change_pct", np.nan))
    return df.dropna(subset=_COMPONENTS)


def _build_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """>=2 üyeli kohortları bulur, her üye için kohort-içi percentile rank (0-1)
    ve üçünün ortalaması olan composite_rank'i hesaplar."""
    df = df.copy()
    df["cohort_size"] = df.groupby(_COHORT_KEY)["symbol"].transform("size")
    df = df[df["cohort_size"] >= _MIN_COHORT_SIZE].copy()

    for col in _COMPONENTS:
        df[f"{col}_rank"] = df.groupby(_COHORT_KEY)[col].rank(pct=True)

    df["composite_rank"] = df[[f"{c}_rank" for c in _COMPONENTS]].mean(axis=1)
    return df


def _placebo(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Kohort üyeliğini koruyup composite_rank'i kohort içinde rastgele karıştırır
    — gerçek etki varsa placebo'da kaybolmalı (aksi halde etki kohort-yapısının
    kendisinden değil rastgele bir artefakttan geliyor demektir)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["composite_rank"] = (
        out.groupby(_COHORT_KEY)["composite_rank"].transform(lambda s: rng.permutation(s.values))
    )
    return out


def _report_correlation(df: pd.DataFrame, label: str) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        if len(sub) < 30:
            continue
        rho, p = spearmanr(sub["composite_rank"], sub["realized_pnl"])
        print(f"  [{label}] {sig_type}: n={len(sub)}, rho={rho:+.3f} (p={p:.4f})")


def _report_buckets(df: pd.DataFrame, label: str) -> None:
    for sig_type in ["Long", "Short"]:
        sub = df[df["signal_type"] == sig_type]
        if len(sub) < 30:
            continue
        tercile = pd.qcut(sub["composite_rank"], 3, labels=["alt", "orta", "üst"], duplicates="drop")
        g = sub.groupby(tercile, observed=True)["realized_pnl"].agg(
            ort_pnl="mean", n="count", wr=lambda s: (s > 0).mean()
        )
        print(f"  [{label}] {sig_type}:")
        print(g.to_string().replace("\n", "\n    "))


def main() -> None:
    df = _fetch()
    print(f"Toplam sinyal (cvd_slope>={_CVD_START}, tüm alanlar dolu): {len(df)}")

    cohorts = _build_cohorts(df)
    n_cohorts = cohorts.groupby(_COHORT_KEY).ngroups
    print(f"Kohort (boyut>={_MIN_COHORT_SIZE}) sayısı: {n_cohorts}, içindeki sinyal: {len(cohorts)}\n")

    print("=== ANA TEST (gerçek kohort-içi sıralama) ===")
    _report_correlation(cohorts, "gerçek")
    _report_buckets(cohorts, "gerçek")

    print("\n=== PLACEBO (rank kohort içinde rastgele karıştırıldı) ===")
    _report_correlation(_placebo(cohorts), "placebo")

    print("\n=== SPLIT-PERIOD (dönem ikiye bölündü) ===")
    mid = cohorts["opened_at"].min() + (cohorts["opened_at"].max() - cohorts["opened_at"].min()) / 2
    for half_name, half_df in [
        ("ilk_yari", cohorts[cohorts["opened_at"] < mid]),
        ("ikinci_yari", cohorts[cohorts["opened_at"] >= mid]),
    ]:
        print(f"-- {half_name} ({len(half_df)} sinyal) --")
        _report_correlation(half_df, half_name)

    print("\n=== İNDİKATÖR AİLESİNE GÖRE KIRILIM ===")
    for ind, ind_df in cohorts.groupby("indicators"):
        if len(ind_df) < 30:
            continue
        print(f"-- {ind} ({len(ind_df)} sinyal) --")
        _report_correlation(ind_df, ind)


if __name__ == "__main__":
    main()
