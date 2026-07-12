"""
Otomatik eşik optimizasyonu — Alpha/Beta filtreleri için (project_pending_topics
memory'sinde 9 Tem'de tasarlanıp koda dökülmemiş fikir, 11 Tem'de somutlaştırıldı).

Kullanılan veri: `signals` tablosundaki GERÇEK kapanmış sinyaller (alpha/beta
CAPM-tipi geriye-dönük rolling hesap — indicators/financial_metrics.py, look-ahead
YOK, bugünkü VPMV dersinden etkilenmiyor).

Yöntem (3 kapı, bugünün disiplinine uygun):
1. Kronolojik bölünme (rastgele değil) — ilk yarı kalibrasyon (in-sample),
   ikinci yarı test (out-of-sample).
2. İleri-adımlı arama — TAM α×β grid'i taramak yerine önce α'nın tek başına
   en iyi eşiği bulunur (min-n şartıyla), sonra β o eşiğin ÜZERİNE eklenir.
   Hem "yüksek X iyi" hem "düşük X iyi" hipotezleri percentile adaylarıyla
   test edilir.
3. OOS'a SABİT uygulama + OOS'un kendi içinde split-period + placebo
   (IS PnL karıştırılıp aynı arama tekrarlanır — gerçek sonucun saf
   gürültüden bulunabilecek en iyi PF'den anlamlı şekilde iyi olup
   olmadığını ölçer).

Long ve Short AYRI ayrı optimize edilir (bugün RSI_Cross'ta görülen asimetri
nedeniyle aynı kurala tabi tutulmazlar).
"""
import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _stats  # pylint: disable=wrong-import-position

MIN_N = 50  # eşik adayının kabul edilmesi için gereken minimum örneklem
PERCENTILE_CANDIDATES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
N_PLACEBO = 30
INDICATORS = ["RSI_Cross(9,24)", "HA_Cross", "MA200_Cross", "Supertrend(10,3.0)"]
DIRECTIONS = ["Long", "Short"]


def _fetch_signals(indicator: str, direction: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT alpha, beta, realized_pnl, opened_at
        FROM signals
        WHERE indicators = %s AND signal_type = %s
          AND status = 'closed' AND realized_pnl IS NOT NULL
          AND alpha IS NOT NULL AND beta IS NOT NULL
        ORDER BY opened_at
    """
    df = pd.read_sql(q, conn, params=(indicator, direction))
    conn.close()
    return df


def _pf(df: pd.DataFrame) -> dict:
    return _stats(df["realized_pnl"].to_numpy() / 100)


def _best_single_threshold(df: pd.DataFrame, col: str, min_n: int = MIN_N, percentiles=None):
    """Hem 'yüksek X iyi' (>=eşik tut) hem 'düşük X iyi' (<=eşik tut)
    hipotezlerini percentile adaylarıyla dener. En iyi PF'yi veren
    (op, threshold, pf, n) döner; hiçbiri min_n'i geçemezse None."""
    if len(df) < min_n:
        return None
    values = df[col].to_numpy()
    best = None
    for pct in (percentiles or PERCENTILE_CANDIDATES):
        th = float(np.percentile(values, pct))
        for op in (">=", "<="):
            sub = df[df[col] >= th] if op == ">=" else df[df[col] <= th]
            if len(sub) < min_n:
                continue
            pf = _pf(sub).get("pf", 0)
            if best is None or pf > best[2]:
                best = (op, th, pf, len(sub))
    if best is None:
        return None
    return {"col": col, "op": best[0], "threshold": best[1], "pf": best[2], "n": best[3]}


def _apply_rule(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    if rule is None:
        return df
    col, op, th = rule["col"], rule["op"], rule["threshold"]
    return df[df[col] >= th] if op == ">=" else df[df[col] <= th]


def _stepwise_search(is_df: pd.DataFrame) -> tuple:
    """Döner: (alpha_rule, beta_rule, is_combined_pf, is_combined_n)."""
    alpha_rule = _best_single_threshold(is_df, "alpha")
    after_alpha = _apply_rule(is_df, alpha_rule)
    beta_rule = _best_single_threshold(after_alpha, "beta")
    combined = _apply_rule(after_alpha, beta_rule)
    combined_stats = _pf(combined) if len(combined) > 0 else {"pf": 0, "n": 0}
    return alpha_rule, beta_rule, combined_stats.get("pf", 0), combined_stats.get("n", 0)


def _placebo_distribution(is_df: pd.DataFrame, n_shuffles: int = N_PLACEBO) -> list:
    """IS realized_pnl'i karıştırıp (alpha/beta ile ilişkisini kırıp) AYNI
    aramayı tekrarlar — saf gürültüden bulunabilecek en iyi PF dağılımı."""
    pnl_vals = is_df["realized_pnl"].to_numpy().copy()
    placebo_pfs = []
    rng = np.random.default_rng(42)
    for _ in range(n_shuffles):
        shuffled = is_df.copy()
        shuffled["realized_pnl"] = rng.permutation(pnl_vals)
        _, _, pf, n = _stepwise_search(shuffled)
        if n >= MIN_N:
            placebo_pfs.append(pf)
    return placebo_pfs


def _single_var_search(is_df: pd.DataFrame, col: str, percentiles=None) -> tuple:
    """Tek değişkenli (stepwise DEĞİL) arama — sadece `col` için en iyi eşik.
    Döner: (rule, is_pf, is_n)."""
    rule = _best_single_threshold(is_df, col, percentiles=percentiles)
    filtered = _apply_rule(is_df, rule)
    stats = _pf(filtered) if len(filtered) > 0 else {"pf": 0, "n": 0}
    return rule, stats.get("pf", 0), stats.get("n", 0)


def _single_var_placebo_distribution(is_df: pd.DataFrame, col: str, n_shuffles: int = N_PLACEBO, percentiles=None) -> list:
    pnl_vals = is_df["realized_pnl"].to_numpy().copy()
    placebo_pfs = []
    rng = np.random.default_rng(42)
    for _ in range(n_shuffles):
        shuffled = is_df.copy()
        shuffled["realized_pnl"] = rng.permutation(pnl_vals)
        _, pf, n = _single_var_search(shuffled, col, percentiles=percentiles)
        if n >= MIN_N:
            placebo_pfs.append(pf)
    return placebo_pfs


def run_single_var(indicator: str, direction: str, col: str) -> None:
    """Sadece TEK değişken (ör. alpha) için arama — beta'nın gürültüsü yok,
    arama uzayı 36'dan 18'e düşüyor (overfitting riski daha az)."""
    df = _fetch_signals(indicator, direction)
    label = f"{indicator} — {direction} — SADECE {col.upper()}"
    _run_single_var_on_df(label, df, col)


def _run_single_var_on_df(label: str, df: pd.DataFrame, col: str, percentiles=None) -> None:
    """run_single_var'ın çekirdeği — dışarıdan hazır bir df (ör. rejim skoru
    gibi `signals` tablosunda olmayan bir kolonla zenginleştirilmiş) alır,
    böylece bu rapor mantığı tekrar yazılmadan yeniden kullanılabilir."""
    print(f"\n{'='*70}\n{label}  (n={len(df):,})\n{'='*70}")
    if len(df) < MIN_N * 4:
        print("Örneklem çok küçük, atlanıyor.")
        return

    t_min, t_max = df["opened_at"].min(), df["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    is_df = df[df["opened_at"] < mid].reset_index(drop=True)
    oos_df = df[df["opened_at"] >= mid].reset_index(drop=True)
    print(f"dönem: {t_min} .. {t_max} | IS: {len(is_df)} | OOS: {len(oos_df)}")

    baseline_is = _pf(is_df)
    print(f"\nIS baseline PF={baseline_is.get('pf',0):.3f} (n={baseline_is.get('n',0)})")

    rule, is_pf, is_n = _single_var_search(is_df, col, percentiles=percentiles)
    print(f"Bulunan kural: {_rule_text(rule)}")
    print(f"IS filtreli PF={is_pf:.3f} (n={is_n})")

    oos_baseline = _pf(oos_df)
    oos_filtered = _apply_rule(oos_df, rule)
    oos_stats = _pf(oos_filtered) if len(oos_filtered) > 0 else {"pf": 0, "n": 0, "wr": 0}

    print(f"\n── OOS (sabit eşikle) ──")
    print(f"baseline:  PF={oos_baseline.get('pf',0):.3f} WR%={oos_baseline.get('wr',0)} n={oos_baseline.get('n',0)}")
    print(f"filtreli:  PF={oos_stats.get('pf',0):.3f} WR%={oos_stats.get('wr',0)} n={oos_stats.get('n',0)}")

    oos_mid = mid + (t_max - mid) / 2
    oos_first = _apply_rule(oos_df[oos_df["opened_at"] < oos_mid], rule)
    oos_second = _apply_rule(oos_df[oos_df["opened_at"] >= oos_mid], rule)
    s1 = _pf(oos_first) if len(oos_first) > 0 else {"pf": 0, "n": 0}
    s2 = _pf(oos_second) if len(oos_second) > 0 else {"pf": 0, "n": 0}
    print(f"\n── OOS split-period sağlamlık ──")
    print(f"ilk yarı:    PF={s1.get('pf',0):.3f} (n={s1.get('n',0)})")
    print(f"ikinci yarı: PF={s2.get('pf',0):.3f} (n={s2.get('n',0)})")

    placebo_pfs = _single_var_placebo_distribution(is_df, col, percentiles=percentiles)
    if placebo_pfs:
        placebo_arr = np.array(placebo_pfs)
        rank = float((placebo_arr < is_pf).mean() * 100)
        print(f"\n── Placebo (n={len(placebo_pfs)} karıştırma) ──")
        print(f"placebo PF ort={placebo_arr.mean():.3f} p90={np.percentile(placebo_arr,90):.3f} "
              f"max={placebo_arr.max():.3f} | gerçek IS PF={is_pf:.3f} (placebo'nun %{rank:.0f}'ini geçiyor)")
        verdict = "GEÇERLİ ADAY" if rank >= 90 and oos_stats.get("pf", 0) > oos_baseline.get("pf", 0) \
            and s1.get("pf", 0) > 1 and s2.get("pf", 0) > 1 else "GÜVENİLMEZ/ZAYIF"
        print(f"\n>>> SONUÇ: {verdict}")


def _rule_text(rule: dict) -> str:
    if rule is None:
        return "(filtre yok — hiçbir eşik min-n şartını geçemedi)"
    return f"{rule['col']} {rule['op']} {rule['threshold']:.3f} (IS PF={rule['pf']:.3f}, n={rule['n']})"


def run_one(indicator: str, direction: str) -> None:
    df = _fetch_signals(indicator, direction)
    print(f"\n{'='*70}\n{indicator} — {direction}  (n={len(df):,})\n{'='*70}")
    if len(df) < MIN_N * 4:
        print("Örneklem çok küçük, atlanıyor.")
        return

    t_min, t_max = df["opened_at"].min(), df["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    is_df = df[df["opened_at"] < mid].reset_index(drop=True)
    oos_df = df[df["opened_at"] >= mid].reset_index(drop=True)
    print(f"dönem: {t_min} .. {t_max} | IS: {len(is_df)} | OOS: {len(oos_df)}")

    baseline_is = _pf(is_df)
    print(f"\nIS baseline PF={baseline_is.get('pf',0):.3f} (n={baseline_is.get('n',0)})")

    alpha_rule, beta_rule, is_combined_pf, is_combined_n = _stepwise_search(is_df)
    print(f"Bulunan alpha kuralı: {_rule_text(alpha_rule)}")
    print(f"Bulunan beta kuralı:  {_rule_text(beta_rule)}")
    print(f"IS BİRLEŞİK PF={is_combined_pf:.3f} (n={is_combined_n})")

    # OOS'a SABİT uygula
    oos_baseline = _pf(oos_df)
    oos_after_alpha = _apply_rule(oos_df, alpha_rule)
    oos_combined = _apply_rule(oos_after_alpha, beta_rule)
    oos_stats = _pf(oos_combined) if len(oos_combined) > 0 else {"pf": 0, "n": 0, "wr": 0, "ort_%": 0}

    print(f"\n── OOS (sabit eşiklerle) ──")
    print(f"baseline:  PF={oos_baseline.get('pf',0):.3f} WR%={oos_baseline.get('wr',0)} n={oos_baseline.get('n',0)}")
    print(f"filtreli:  PF={oos_stats.get('pf',0):.3f} WR%={oos_stats.get('wr',0)} n={oos_stats.get('n',0)}")

    # OOS split-period
    oos_mid = mid + (t_max - mid) / 2
    oos_first = _apply_rule(_apply_rule(oos_df[oos_df["opened_at"] < oos_mid], alpha_rule), beta_rule)
    oos_second = _apply_rule(_apply_rule(oos_df[oos_df["opened_at"] >= oos_mid], alpha_rule), beta_rule)
    s1 = _pf(oos_first) if len(oos_first) > 0 else {"pf": 0, "n": 0}
    s2 = _pf(oos_second) if len(oos_second) > 0 else {"pf": 0, "n": 0}
    print(f"\n── OOS split-period sağlamlık ──")
    print(f"ilk yarı:    PF={s1.get('pf',0):.3f} (n={s1.get('n',0)})")
    print(f"ikinci yarı: PF={s2.get('pf',0):.3f} (n={s2.get('n',0)})")

    # Placebo
    placebo_pfs = _placebo_distribution(is_df)
    if placebo_pfs:
        placebo_arr = np.array(placebo_pfs)
        rank = float((placebo_arr < is_combined_pf).mean() * 100)
        print(f"\n── Placebo (n={len(placebo_pfs)} karıştırma) ──")
        print(f"placebo PF ort={placebo_arr.mean():.3f} p90={np.percentile(placebo_arr,90):.3f} "
              f"max={placebo_arr.max():.3f} | gerçek IS PF={is_combined_pf:.3f} "
              f"(placebo'nun %{rank:.0f}'ini geçiyor)")
        verdict = "GEÇERLİ ADAY" if rank >= 90 and oos_stats.get("pf", 0) > oos_baseline.get("pf", 0) \
            and s1.get("pf", 0) > 1 and s2.get("pf", 0) > 1 else "GÜVENİLMEZ/ZAYIF"
        print(f"\n>>> SONUÇ: {verdict}")


def run():
    for indicator in INDICATORS:
        for direction in DIRECTIONS:
            run_one(indicator, direction)


def run_long_alpha_only():
    for indicator in INDICATORS:
        run_single_var(indicator, "Long", "alpha")


if __name__ == "__main__":
    run()
