"""
Verimlilik (ERSI/devisso_score) — RSI_Cross sinyalleri üzerinde, LOOK-AHEAD'SİZ.

ÖNEMLİ METODOLOJİ NOTU: v2-11'deki (ma200_efficiency_jump_bt.py) "sıçrama"
yöntemi (post[i+1]-pre[i-1]) v2-21'de VPMV için look-ahead içerdiği
kanıtlanan AYNI kalıbı kullanıyordu — post[i+1] sinyal anında bilinmez.
Bu script o hatayı TEKRARLAMAZ: `devisso_delta`/`devisso_ratio` kolonları
zaten signal_lifecycle_manager.py'de AYNI sembol+interval+indicator+yön için
BİR ÖNCEKİ SİNYALE göre hesaplanıp kaydediliyor (bar'a göre değil, sinyale
göre) — mevcut sinyal açılırken önceki sinyal zaten kapanmış/bilinen bir
olay, dolayısıyla look-ahead yok (all_up mantığıyla aynı disiplin).

Üç şey test ediliyor (RSI_Cross, Long/Short ayrı):
  1. devisso_score (mutlak değer) terciline göre PF — zaten genel havuzda
     (28.878 karışık sinyal) çürümüştü, burada RSI_Cross'a izole tekrar.
  2. devisso_delta (önceki sinyale göre değişim) terciline göre PF — YENİ.
  3. devisso_ratio terciline göre PF — YENİ.
Hepsi split-period ile doğrulanıyor.

Kullanım: python -m research.pattern_lab.rsi_cross_efficiency_bt
"""

import numpy as np
import pandas as pd
import psycopg2

from config import Config
from research.pattern_lab.vol_exhaustion_bt import _stats

CUTOFF = "2026-07-03 19:22:16"  # commit e81aa34 — temiz ters-sinyal/timeout rejimi başlangıcı
DIRECTIONS = ["Long", "Short"]


def _fetch(direction: str) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
    )
    q = """
        SELECT opened_at, realized_pnl, devisso_score, devisso_delta, devisso_ratio
        FROM signals
        WHERE indicators = 'RSI_Cross(9,24)'
          AND signal_type = %s
          AND status = 'closed'
          AND realized_pnl IS NOT NULL
          AND opened_at >= %s
        ORDER BY opened_at
    """
    df = pd.read_sql(q, conn, params=(direction, CUTOFF))
    conn.close()
    return df


def _tercile_table(df: pd.DataFrame, col: str) -> None:
    sub = df.dropna(subset=[col, "realized_pnl"])
    if len(sub) < 60:
        print(f"  [{col}] örneklem çok küçük (n={len(sub)}), atlanıyor")
        return
    q1, q2 = sub[col].quantile([0.333, 0.667])

    def bucket(v):
        return "düşük" if v < q1 else ("orta" if v < q2 else "yüksek")

    sub = sub.copy()
    sub["tercil"] = sub[col].apply(bucket)
    print(f"  -- {col} terciline göre (q1={q1:.2f}, q2={q2:.2f}) --")
    print(f"  {'tercil':10} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    for name in ("düşük", "orta", "yüksek"):
        rets = sub.loc[sub["tercil"] == name, "realized_pnl"].to_numpy() / 100
        s = _stats(rets)
        print(
            f"  {name:10} {s.get('n',0):>7} {s.get('wr',0):>6} "
            f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}"
        )
    return q1, q2


def _split_check(df: pd.DataFrame, col: str, q1: float, q2: float) -> None:
    sub = df.dropna(subset=[col, "realized_pnl"])
    if len(sub) < 60:
        return
    t_min, t_max = sub["opened_at"].min(), sub["opened_at"].max()
    mid = t_min + (t_max - t_min) / 2
    first = sub[sub["opened_at"] < mid]
    second = sub[sub["opened_at"] >= mid]

    def bucket(v):
        return "düşük" if v < q1 else ("orta" if v < q2 else "yüksek")

    print(f"  -- split-period ({col}, eşikler tüm dönemden sabit) --")
    for name, part in (("ilk_yari", first), ("ikinci_yari", second)):
        if len(part) < 20:
            print(f"  {name}: örneklem çok küçük")
            continue
        part = part.copy()
        part["tercil"] = part[col].apply(bucket)
        yuksek = part.loc[part["tercil"] == "yüksek", "realized_pnl"].to_numpy() / 100
        dusuk = part.loc[part["tercil"] == "düşük", "realized_pnl"].to_numpy() / 100
        s_y, s_d = _stats(yuksek), _stats(dusuk)
        print(
            f"  {name:12} yüksek: n={s_y.get('n',0):>5} PF={s_y.get('pf',0):>6}  |  "
            f"düşük: n={s_d.get('n',0):>5} PF={s_d.get('pf',0):>6}"
        )


def run():
    for direction in DIRECTIONS:
        df = _fetch(direction)
        print(f"\n{'='*70}\nRSI_Cross {direction} (n={len(df):,}, {CUTOFF} sonrası)\n{'='*70}")

        s = _stats(df["realized_pnl"].to_numpy() / 100)
        print(
            f"  baseline (tümü): n={s.get('n',0)} WR%={s.get('wr',0)} "
            f"ort%={s.get('ort_%',0)} PF={s.get('pf',0)}\n"
        )

        for col in ("devisso_score", "devisso_delta", "devisso_ratio"):
            result = _tercile_table(df, col)
            if result:
                q1, q2 = result
                _split_check(df, col, q1, q2)
            print()


if __name__ == "__main__":
    run()
