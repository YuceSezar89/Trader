"""
signal_repository.py — Signal/PaperTrade ORM nesnelerinin KURULUŞ mantığını
tek bir yerde toplamaya başlangıç (2 Ağu 2026, Fable 5 mimari denetimi:
"repository katmanı yok" bulgusu — Senaryo A'da bulunan 3 "sessizce kaybolan
veri" bug'ının kök nedeni tam olarak buydu, yazma noktaları 4 dosyaya/8 yere
dağılmıştı).

Kademe 1 (bugün): SADECE `build_signal()` — signal_lifecycle_manager.py'nin
en sık kullanılan yolu (yeni sinyal açma). Session/transaction yönetimine
DOKUNULMADI, davranış birebir aynı — bu fonksiyon session.add()/commit()
yapmıyor, sadece HENÜZ session'a eklenmemiş bir Signal nesnesi kurup dönüyor.
Çağıran taraf kendi transaction sınırlarını (flush/commit sırası, aynı
transaction içinde başka satırları da güncelleme gibi) aynen koruyor.

Sıradaki kademeler (henüz yapılmadı): open_paper_trade(), close_paper_trade(),
record_snapshot() — her biri ayrı, kendi başına doğrulanan bir adım olacak.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from database.models import Signal


def build_signal(
    signal_data: dict[str, Any],
    *,
    devisso_delta: Optional[float] = None,
    devisso_ratio: Optional[float] = None,
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    sl_mult: Optional[float] = None,
    tp_mult: Optional[float] = None,
    status: str = "active",
) -> Signal:
    """enriched_signal dict'ini bir Signal ORM nesnesine eşler — dönen nesne
    HENÜZ session'a eklenmemiştir (session.add/flush/commit çağıranın işi).

    devisso_delta/ratio (önceki sinyale göre fark, DB sorgusu gerektiriyor)
    ve sl/tp fiyat+çarpanları (risk politikası hesabı gerektiriyor)
    signal_data içinde YOK — çağıran taraf ayrıca hesaplayıp geçirir.
    """
    return Signal(
        symbol=signal_data["symbol"],
        interval=signal_data["interval"],
        indicators=signal_data["indicators"],
        signal_type=signal_data["signal_type"],
        opened_at=signal_data.get("opened_at", datetime.now()),
        open_price=float(signal_data["open_price"]),
        status=status,
        vpms_score=signal_data.get("vpms_score"),
        mtf_score=signal_data.get("mtf_score"),
        st_confirmed=signal_data.get("st_confirmed"),
        rsi=signal_data.get("rsi"),
        strength=signal_data.get("strength"),
        atr=signal_data.get("atr"),
        alpha=signal_data.get("alpha"),
        beta=signal_data.get("beta"),
        sharpe_ratio=signal_data.get("sharpe_ratio"),
        sortino_ratio=signal_data.get("sortino_ratio"),
        calmar_ratio=signal_data.get("calmar_ratio"),
        information_ratio=signal_data.get("information_ratio"),
        oi_data=signal_data.get("oi_data"),
        stop_loss_price=sl_price,
        take_profit_price=tp_price,
        sl_multiplier=sl_mult,
        tp_multiplier=tp_mult,
        z_score_entry=signal_data.get("z_score_entry"),
        is_confluence=signal_data.get("is_confluence", False),
        ha_ultra_confirm=signal_data.get("ha_ultra_confirm"),
        vpmv_pre_avg=signal_data.get("vpmv_pre_avg"),
        vpmv_pre_proxy=signal_data.get("vpmv_pre_proxy"),
        vpmv_pre_total=signal_data.get("vpmv_pre_total"),
        vpmv_slope=signal_data.get("vpmv_slope"),
        vpmv_ratio=signal_data.get("vpmv_ratio"),
        cvd_slope=signal_data.get("cvd_slope"),
        vp_buy_avg=signal_data.get("vp_buy_avg"),
        vp_sell_avg=signal_data.get("vp_sell_avg"),
        vp_score=signal_data.get("vp_score"),
        vp_score_real=signal_data.get("vp_score_real"),
        devisso_score=signal_data.get("devisso_score"),
        devisso_delta=devisso_delta,
        devisso_ratio=devisso_ratio,
        pd_zone=signal_data.get("pd_zone"),
        market_structure=signal_data.get("market_structure"),
        fvg_tfs=signal_data.get("fvg_tfs"),
        candle_pattern=signal_data.get("candle_pattern"),
        rank_score=signal_data.get("rank_score"),
        vs_btc=signal_data.get("vs_btc"),
        rank_combined=signal_data.get("rank_combined"),
        rank_rsi_cross=signal_data.get("rank_rsi_cross"),
        rank_z_confluence=signal_data.get("rank_z_confluence"),
        rank_r_score=signal_data.get("rank_r_score"),
        rank_aligned=signal_data.get("rank_aligned"),
        rank_alignment_count=signal_data.get("rank_alignment_count"),
        vol_score=signal_data.get("vol_score"),
        mom_score=signal_data.get("mom_score"),
        volat_score=signal_data.get("volat_score"),
        price_score=signal_data.get("price_score"),
        candle_kategori=signal_data.get("candle_kategori"),
        all_up=signal_data.get("all_up"),
        regime_trend=signal_data.get("regime_trend"),
        volatility_regime=signal_data.get("volatility_regime"),
        btc_z_score=signal_data.get("btc_z_score"),
        btc_trend=signal_data.get("btc_trend"),
        funding_rate=signal_data.get("funding_rate"),
        hour_utc=signal_data.get("hour_utc"),
        day_of_week=signal_data.get("day_of_week"),
    )
