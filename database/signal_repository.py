"""
signal_repository.py — Signal/PaperTrade ORM nesnelerinin KURULUŞ mantığını
tek bir yerde toplamaya başlangıç (2 Ağu 2026, Fable 5 mimari denetimi:
"repository katmanı yok" bulgusu — Senaryo A'da bulunan 3 "sessizce kaybolan
veri" bug'ının kök nedeni tam olarak buydu, yazma noktaları 4 dosyaya/8 yere
dağılmıştı).

Kademe 1: `build_signal()` — signal_lifecycle_manager.py'nin en sık kullanılan
yolu (yeni sinyal açma).
Kademe 2: `build_paper_trade()` — paper_trade_manager.py::on_new_signal'ın
ana açılış yolu (_do_open).
Kademe 3 (toparlama): `close_signal()`, `build_trade_snapshot()`,
`build_paper_trade_direct()` — geri kalan yazma noktaları. `close_signal()`
GERÇEK bir kopya kodu birleştiriyor (signal_lifecycle_manager.py VE
risk_manager.py'de BİREBİR AYNI 5 satırlık kapatma bloğu vardı); diğer ikisi
(TradeSnapshot, _do_open_direct'in PaperTrade'i) tek-siteli, sadece
tutarlılık için taşındı — bug-önleme değerleri daha düşük.

Hepsi aynı disiplinde: session/transaction yönetimine DOKUNULMADI, davranış
birebir aynı — fonksiyonlar session.add()/commit() yapmıyor, sadece HENÜZ
session'a eklenmemiş/mutasyona uğramış bir ORM nesnesi dönüyor ya da
(close_signal için) var olan nesneyi yerinde mutasyona uğratıyor. Çağıran
taraf kendi transaction sınırlarını aynen koruyor.

Bilerek TAŞINMADI: signal_lifecycle_manager.py::_update_scores — koşullu
alan güncellemesi (3 satır, "eğer data'da bu key varsa güncelle"), tek site,
"build_X" deseniyle uyuşmayacak kadar basit/farklı — taşımak indirection
katardı, değer katmazdı.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from database.models import PaperTrade, Signal, TradeSnapshot


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
        vol_raw=signal_data.get("vol_raw"),
        mom_raw=signal_data.get("mom_raw"),
        volat_raw=signal_data.get("volat_raw"),
        price_raw=signal_data.get("price_raw"),
        vol_change_pct=signal_data.get("vol_change_pct"),
        mom_change_pct=signal_data.get("mom_change_pct"),
        volat_change_pct=signal_data.get("volat_change_pct"),
        price_change_pct=signal_data.get("price_change_pct"),
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
        body_pct=signal_data.get("body_pct"),
        wick_pct=signal_data.get("wick_pct"),
        signal_rsi_change=signal_data.get("signal_rsi_change"),
        signal_mfi_change=signal_data.get("signal_mfi_change"),
        signal_macd_change=signal_data.get("signal_macd_change"),
        signal_obv=signal_data.get("signal_obv"),
        all_up=signal_data.get("all_up"),
        regime_trend=signal_data.get("regime_trend"),
        volatility_regime=signal_data.get("volatility_regime"),
        btc_z_score=signal_data.get("btc_z_score"),
        btc_trend=signal_data.get("btc_trend"),
        funding_rate=signal_data.get("funding_rate"),
        hour_utc=signal_data.get("hour_utc"),
        day_of_week=signal_data.get("day_of_week"),
    )


def build_paper_trade(
    signal_data: dict[str, Any],
    *,
    signal_id: Optional[int],
    strategy: str,
    position_usd: float,
    entry_price: float,
    opened_at: datetime,
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    btc_z_score: Optional[float] = None,
    btc_trend: Optional[str] = None,
    funding_rate: Optional[float] = None,
    recent_win_rate: Optional[float] = None,
    regime_trend: Optional[str] = None,
    volatility_regime: Optional[str] = None,
    rank_at_entry: Optional[int] = None,
    devisso_score: Optional[float] = None,
    devisso_delta: Optional[float] = None,
    devisso_ratio: Optional[float] = None,
    sl_multiplier: Optional[float] = None,
    tp_multiplier: Optional[float] = None,
) -> PaperTrade:
    """enriched_signal dict'ini bir PaperTrade ORM nesnesine eşler — dönen
    nesne HENÜZ session'a eklenmemiştir (session.add/commit çağıranın işi).

    Diğer parametreler signal_data içinde YOK — DB sorgusu (recent_win_rate,
    sig.devisso_*, sig.sl/tp_multiplier), risk politikası hesabı (sl/tp_price)
    veya _pt_kwargs üzerinden ayrıca geçirilen bağlam (btc_z_score vb.)
    gerektiriyor, çağıran taraf hesaplayıp geçirir.
    """
    return PaperTrade(
        signal_id=signal_id,
        strategy=strategy,
        symbol=signal_data.get("symbol", ""),
        signal_type=signal_data.get("signal_type", ""),
        interval=signal_data.get("interval", ""),
        position_usd=position_usd,
        entry_price=entry_price,
        stop_loss_price=sl_price,
        take_profit_price=tp_price,
        status="open",
        opened_at=opened_at,
        btc_z_score=btc_z_score,
        btc_trend=btc_trend,
        hour_utc=opened_at.hour if opened_at else None,
        day_of_week=opened_at.weekday() if opened_at else None,
        funding_rate=funding_rate,
        recent_win_rate=recent_win_rate,
        vpms_score=signal_data.get("vpms_score"),
        z_score_entry=signal_data.get("z_score_entry"),
        mtf_score=signal_data.get("mtf_score"),
        atr=signal_data.get("atr"),
        rank_at_entry=rank_at_entry,
        regime_trend=regime_trend,
        volatility_regime=volatility_regime,
        vpmv_pre_avg=signal_data.get("vpmv_pre_avg"),
        vpmv_slope=signal_data.get("vpmv_slope"),
        vpmv_ratio=signal_data.get("vpmv_ratio"),
        vol_raw=signal_data.get("vol_raw"),
        mom_raw=signal_data.get("mom_raw"),
        volat_raw=signal_data.get("volat_raw"),
        price_raw=signal_data.get("price_raw"),
        vol_change_pct=signal_data.get("vol_change_pct"),
        mom_change_pct=signal_data.get("mom_change_pct"),
        volat_change_pct=signal_data.get("volat_change_pct"),
        price_change_pct=signal_data.get("price_change_pct"),
        devisso_score=devisso_score,
        devisso_delta=devisso_delta,
        devisso_ratio=devisso_ratio,
        cvd_slope=signal_data.get("cvd_slope"),
        vp_buy_avg=signal_data.get("vp_buy_avg"),
        vp_sell_avg=signal_data.get("vp_sell_avg"),
        vp_score=signal_data.get("vp_score"),
        alpha=signal_data.get("alpha"),
        beta=signal_data.get("beta"),
        sharpe_ratio=signal_data.get("sharpe_ratio"),
        sortino_ratio=signal_data.get("sortino_ratio"),
        calmar_ratio=signal_data.get("calmar_ratio"),
        information_ratio=signal_data.get("information_ratio"),
        vpmv_pre_proxy=signal_data.get("vpmv_pre_proxy"),
        vpmv_pre_total=signal_data.get("vpmv_pre_total"),
        vp_score_real=signal_data.get("vp_score_real"),
        market_structure=signal_data.get("market_structure"),
        fvg_tfs=signal_data.get("fvg_tfs"),
        candle_pattern=signal_data.get("candle_pattern"),
        ha_ultra_confirm=signal_data.get("ha_ultra_confirm"),
        vol_score=signal_data.get("vol_score"),
        mom_score=signal_data.get("mom_score"),
        volat_score=signal_data.get("volat_score"),
        price_score=signal_data.get("price_score"),
        candle_kategori=signal_data.get("candle_kategori"),
        body_pct=signal_data.get("body_pct"),
        wick_pct=signal_data.get("wick_pct"),
        signal_rsi_change=signal_data.get("signal_rsi_change"),
        signal_mfi_change=signal_data.get("signal_mfi_change"),
        signal_macd_change=signal_data.get("signal_macd_change"),
        signal_obv=signal_data.get("signal_obv"),
        all_up=signal_data.get("all_up"),
        sl_multiplier=sl_multiplier,
        tp_multiplier=tp_multiplier,
        rank_score=signal_data.get("rank_score"),
        vs_btc=signal_data.get("vs_btc"),
        rank_combined=signal_data.get("rank_combined"),
        rank_rsi_cross=signal_data.get("rank_rsi_cross"),
        rank_z_confluence=signal_data.get("rank_z_confluence"),
        rank_r_score=signal_data.get("rank_r_score"),
        rank_aligned=signal_data.get("rank_aligned"),
        rank_alignment_count=signal_data.get("rank_alignment_count"),
        # 19 Tem 2026: giriş anındaki TÜM enriched_signal dict'i (VPMV
        # bileşenleri, SMC, finansal oranlar, vb. — yukarıdaki kolonlarda
        # olmayan her şey). json round-trip ile datetime/numpy tiplerini
        # (JSONB'nin serileştiremeyeceği) güvenli string'e çeviriyoruz.
        entry_features=json.loads(json.dumps(signal_data, default=str)),
    )


def close_signal(sig: Signal, *, close_price: float, reason: str, realized_pnl: float) -> None:
    """Bir Signal'ı kapanmış duruma çevirir — VAR OLAN nesneyi yerinde
    mutasyona uğratır (yeni nesne kurmaz), session.add()/commit() çağıranın
    işi. realized_pnl çağıran tarafça (_calc_pnl ile) önceden hesaplanmış
    olmalı — bu fonksiyon PnL hesabı yapmaz, sadece atar.

    signal_lifecycle_manager.py VE risk_manager.py'de BİREBİR AYNI 5 satır
    kopyalanmıştı (2 Ağu 2026, Fable 5 mimari denetimi) — tek yere toplandı.
    """
    sig.status = "closed"
    sig.closed_at = datetime.now()
    sig.close_price = close_price
    sig.close_reason = reason
    sig.realized_pnl = realized_pnl


def build_trade_snapshot(
    *,
    trade_id: int,
    symbol: str,
    price: Optional[float],
    cvd_slope: Optional[float],
    vp_buy: Optional[float],
    vp_sell: Optional[float],
    vp_score_real: Optional[float],
    vol_score: Optional[float],
    mom_score: Optional[float],
    volat_score: Optional[float],
    price_score: Optional[float],
    price_since_entry_pct: Optional[float],
    vpmv_combined: Optional[float],
    smc_market_structure: Optional[str],
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    sharpe_ratio: Optional[float] = None,
    sortino_ratio: Optional[float] = None,
    calmar_ratio: Optional[float] = None,
    treynor_ratio: Optional[float] = None,
    information_ratio: Optional[float] = None,
    signal_rsi_change: Optional[float] = None,
    signal_mfi_change: Optional[float] = None,
    signal_macd_change: Optional[float] = None,
    signal_obv: Optional[float] = None,
    body_pct: Optional[float] = None,
    wick_pct: Optional[float] = None,
) -> TradeSnapshot:
    """Açık bir paper trade için periyodik piyasa-bağlamı anlık görüntüsü
    kurar — dönen nesne HENÜZ session'a eklenmemiştir. trade_snapshot.py'nin
    tek yazma noktasıydı, kopya riski yoktu — sadece tutarlılık için taşındı.

    17 Ağu 2026 (migration 036): alpha/beta/finansal oranlar/mum oranları
    eklendi — hepsi opsiyonel (varsayılan None), eski çağrı yerleri bozulmaz."""
    # Orijinal kod SADECE vp_buy is not None kontrolü yapıyordu (vp_sell'i
    # değil) — birebir aynı davranış korunuyor, "iyileştirme" yapılmadı.
    vp_score = round(vp_buy - vp_sell, 2) if vp_buy is not None else None
    return TradeSnapshot(
        trade_id=trade_id,
        symbol=symbol,
        price=price,
        cvd_slope=cvd_slope,
        vp_buy=vp_buy,
        vp_sell=vp_sell,
        vp_score=vp_score,
        vp_score_real=vp_score_real,
        vol_score=round(vol_score, 2) if vol_score is not None else None,
        mom_score=round(mom_score, 2) if mom_score is not None else None,
        volat_score=round(volat_score, 2) if volat_score is not None else None,
        price_score=round(price_score, 2) if price_score is not None else None,
        price_since_entry_pct=price_since_entry_pct,
        vpmv_combined=round(vpmv_combined, 2) if vpmv_combined is not None else None,
        smc_market_structure=smc_market_structure,
        alpha=round(alpha, 4) if alpha is not None else None,
        beta=round(beta, 4) if beta is not None else None,
        sharpe_ratio=round(sharpe_ratio, 4) if sharpe_ratio is not None else None,
        sortino_ratio=round(sortino_ratio, 4) if sortino_ratio is not None else None,
        calmar_ratio=round(calmar_ratio, 4) if calmar_ratio is not None else None,
        treynor_ratio=round(treynor_ratio, 4) if treynor_ratio is not None else None,
        information_ratio=(round(information_ratio, 4) if information_ratio is not None else None),
        signal_rsi_change=(round(signal_rsi_change, 4) if signal_rsi_change is not None else None),
        signal_mfi_change=(round(signal_mfi_change, 4) if signal_mfi_change is not None else None),
        signal_macd_change=(
            round(signal_macd_change, 4) if signal_macd_change is not None else None
        ),
        signal_obv=round(signal_obv, 2) if signal_obv is not None else None,
        body_pct=round(body_pct, 2) if body_pct is not None else None,
        wick_pct=round(wick_pct, 2) if wick_pct is not None else None,
    )


def build_paper_trade_direct(
    *,
    strategy: str,
    symbol: str,
    signal_type: str,
    interval: str,
    position_usd: float,
    entry_price: float,
    sl_price: Optional[float],
    tp_price: Optional[float],
    atr: Optional[float],
    source: Optional[str] = None,
) -> PaperTrade:
    """Sinyal tablosundan BAĞIMSIZ pozisyon açan yol (dedektör-tabanlı
    stratejiler: do_kirilimi/do_open_streak) için PaperTrade kurar — dönen
    nesne HENÜZ session'a eklenmemiştir. build_paper_trade()'den farklı
    olarak enriched_signal dict'i YOK, sadece ham parametreler var — tek
    yazma noktasıydı, sadece tutarlılık için taşındı."""
    return PaperTrade(
        signal_id=None,
        strategy=strategy,
        # PaperTrade.source VARCHAR(20) — 15 Ağu 2026 bugfix: tek yazma noktası
        # olduğu için burada kesiliyor, çağıranın karakter sayısını hatırlamasına
        # güvenilmiyor. 14 Ağu'da bu limit aşılınca (StringDataRightTruncationError)
        # ~19.5 saat boyunca totalamount_rank1'in HER açma denemesi sessizce
        # başarısız olmuştu — aynı hata bir daha hiçbir çağıranda tekrarlanamaz.
        source=source[:20] if source else None,
        symbol=symbol,
        signal_type=signal_type,
        interval=interval,
        position_usd=position_usd,
        entry_price=entry_price,
        stop_loss_price=sl_price,
        take_profit_price=tp_price,
        status="open",
        opened_at=datetime.now(),
        atr=atr,
    )
