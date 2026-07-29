"""
GERCEK SL/TP'li paper trade sonuclariyla TF-hizalanma + erken-ayrisma testi —
[[rsi_cross_tf_alignment_bt]]'nin signals.realized_pnl (ham, kaldiraçsiz, kendi
kapanis mantigi) yerine paper_trades.pnl_usd (gercek SL/trailing_stop ile
kapanmis, komisyon dahil) kullanan versiyonu.

18 Tem 2026: kullanici "pnl hesabi farkli olabilir" dedi, dogrulandi —
signals.realized_pnl ile paper_trades.pnl_pct korelasyonu sadece 0.56 (1710
eslesen kayit). Ayrica rsi_cross_live CANLI stratejisi (filtresiz, 1543 gercek
islem, 14-18 Tem) net ZARARDA (-$267.38 toplam, -$0.17/islem ort.) — bu
script bizim filtremizin (hizali+cok-ayrisan) GERCEK islemlerde fark yaratip
yaratmadigini test ediyor.

Kullanım: python -m research.pattern_lab.rsi_cross_live_real_pnl_bt
"""

import warnings

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

from config import Config
from research.pattern_lab.rsi_cross_tf_alignment_bt import (
    _EARLY_BARS,
    _add_early_pct,
    _add_htf_alignment,
    _build_cohorts,
    _fetch,
)


def _fetch_real_trades() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT pt.signal_id, pt.pnl_usd, pt.pnl_pct, pt.close_reason, pt.fee_usd
        FROM paper_trades pt
        WHERE pt.strategy = 'rsi_cross_live' AND pt.status = 'closed'
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def _fetch_signal_ids() -> pd.DataFrame:
    """opened_at bazlı join yerine signal.id ile net eşleştirme için."""
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = """
        SELECT id, symbol, signal_type, interval, opened_at
        FROM signals
        WHERE indicators = 'RSI_Cross(9,24)'
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def main() -> None:
    print("RSI_Cross(9,24) kohort verisi hazırlanıyor (bu adım birkaç dakika sürebilir)...")
    df = _fetch(_EARLY_BARS)
    df = _add_early_pct(df)

    print("Sinyal id eşlemesi çekiliyor...")
    sig_ids = _fetch_signal_ids()
    # df'de id yok (early_divergence fetch'i içermiyor), opened_at+symbol+interval ile eşle
    df = df.merge(sig_ids, on=["symbol", "signal_type", "interval", "opened_at"], how="inner")

    cohorts = _build_cohorts(df)
    print(f"Kohort içi sinyal: {len(cohorts)}")

    print("1h/4h Heikin Ashi hizalanması hesaplanıyor...")
    aligned = _add_htf_alignment(cohorts)

    print("Gerçek paper_trades (rsi_cross_live) verisi çekiliyor...")
    real = _fetch_real_trades()
    print(f"Gerçek işlem sayısı: {len(real)}")

    merged = aligned.merge(real, left_on="id", right_on="signal_id", how="inner")
    print(f"Kohort + hizalanma + GERÇEK işlem kesişimi: {len(merged)}\n")

    print("=== GERÇEK $ SONUÇLARI: hizalı+çok-ayrışan vs geri kalanı ===")
    for sig_type in ["Long", "Short"]:
        sub_all = merged[merged["signal_type"] == sig_type]
        if len(sub_all) < 10:
            print(f"  {sig_type}: yetersiz örnek (n={len(sub_all)})")
            continue

        is_filtered = (sub_all["aligned_count"] == 2) & (
            sub_all["early_rank"] >= sub_all["early_rank"].quantile(0.667)
        )
        filtered = sub_all[is_filtered]
        rest = sub_all[~is_filtered]

        print(f"  {sig_type} — TÜMÜ (n={len(sub_all)}): ort_$={sub_all['pnl_usd'].mean():+.3f}, "
              f"toplam_$={sub_all['pnl_usd'].sum():+.2f}, wr={(sub_all['pnl_usd']>0).mean()*100:.1f}%")
        print(f"  {sig_type} — HİZALI+ÇOK-AYRIŞAN (n={len(filtered)}): "
              f"ort_$={filtered['pnl_usd'].mean():+.3f}" if len(filtered) > 0 else f"  {sig_type} — HİZALI+ÇOK-AYRIŞAN: örnek yok")
        if len(filtered) > 0:
            print(f"      toplam_$={filtered['pnl_usd'].sum():+.2f}, wr={(filtered['pnl_usd']>0).mean()*100:.1f}%")
        print(f"  {sig_type} — GERİ KALANI (n={len(rest)}): "
              f"ort_$={rest['pnl_usd'].mean():+.3f}" if len(rest) > 0 else f"  {sig_type} — GERİ KALANI: örnek yok")
        if len(rest) > 0:
            print(f"      toplam_$={rest['pnl_usd'].sum():+.2f}, wr={(rest['pnl_usd']>0).mean()*100:.1f}%")
        print()


if __name__ == "__main__":
    main()
