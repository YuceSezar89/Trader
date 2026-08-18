"""ORM modelleri — SQLAlchemy 2.0 Mapped[]/mapped_column() stili (2 Ağu 2026
refaktörü, bkz. commit serisi 1/6-6/6). Şema/DDL davranışı eski Column()
stiliyle birebir aynı, sadece statik tip bilgisi eklendi.

pylint (4.0.5/astroid 4.0.4, bu depoda kullanılan sürüm) bu dosyanın tam
karmaşıklığında (çok sayıda Mapped[] alanı + __table_args__ birlikte)
'Mapped' unsubscriptable-object false-positive'i üretiyor — minimal/orta
ölçekli tekrar-üretim denemelerinde HİÇ çıkmıyor, mypy database/models.py
tamamen temiz, tüm testler geçiyor. Astroid'in Mapped[] generic'ini bu
ölçekte çözümleyememesi, gerçek bir kod hatası değil."""

# pylint: disable=unsubscriptable-object

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    PrimaryKeyConstraint,
    SmallInteger,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# SQLAlchemy 2.0 stili, mypy ve linter uyumluluğu için
class Base(DeclarativeBase):
    def to_dict(self):
        """Converts the model instance to a dictionary."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# Modelinize to_dict gibi ortak metodlar eklemek isterseniz,
# özel bir Base sınıfı oluşturup bunu kullanabilirsiniz.
# class CustomBase:
#     def to_dict(self):
#         return {c.name: getattr(self, c.name) for c in self.__table__.columns}
#
# Base = declarative_base(cls=CustomBase)


class PriceData(Base):
    __tablename__ = "price_data"

    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    interval: Mapped[str | None] = mapped_column(String, nullable=True)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    buy_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    sell_volume: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("symbol", "timestamp"),
        UniqueConstraint("symbol", "interval", "timestamp"),
    )


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    interval: Mapped[str] = mapped_column(String, nullable=False)
    indicators: Mapped[str] = mapped_column(String, nullable=False)
    signal_type: Mapped[str] = mapped_column(String, nullable=False)

    opened_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    open_price: Mapped[float] = mapped_column(Float, nullable=False)

    vpms_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mtf_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    st_confirmed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    rsi: Mapped[float | None] = mapped_column(Float, nullable=True)
    strength: Mapped[int | None] = mapped_column(Integer, nullable=True)
    atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    alpha: Mapped[float | None] = mapped_column(Float, nullable=True)
    beta: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    close_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    close_reason: Mapped[str | None] = mapped_column(String(20), nullable=True)
    closed_by: Mapped[int | None] = mapped_column(Integer, nullable=True)

    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    oi_data: Mapped[str | None] = mapped_column(String, nullable=True)

    stop_loss_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    sl_multiplier: Mapped[float | None] = mapped_column(Float, nullable=True)
    tp_multiplier: Mapped[float | None] = mapped_column(Float, nullable=True)

    z_score_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_confluence: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    trailing_stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)

    sortino_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    calmar_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    information_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    vpmv_pre_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    vpmv_pre_proxy: Mapped[float | None] = mapped_column(Float, nullable=True)
    vpmv_pre_total: Mapped[float | None] = mapped_column(Float, nullable=True)
    vpmv_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    vpmv_slope: Mapped[float | None] = mapped_column(Float, nullable=True)
    vpmv_post_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    vpmv_post_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 13 Ağu 2026 (migration 033, önceki denemeler 031/032'yi düzeltir):
    # 4 ham bileşenin (hacim/momentum-RSI seviyesi/ATR/kapanış fiyatı —
    # 0-100'e normalize edilMEmiş) BU sinyaldeki değeri + BİR ÖNCEKİ aynı
    # (symbol, interval, indicators, signal_type) sinyaline göre yüzdesel
    # değişimi — Sinyal Mumu Ratioları'ndaki `volume/lastSignalVolume-1`
    # mantığıyla aynı baseline (5-bar pencere ortalaması DEĞİL). Bkz.
    # utils/vpmv.py::compute_raw_components + signal_processor.py::
    # _compute_component_change_pct.
    vol_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    mom_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    volat_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    vol_change_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    mom_change_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    volat_change_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_change_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    cvd_slope: Mapped[float | None] = mapped_column(Float, nullable=True)
    vp_buy_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    vp_sell_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    vp_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    vp_score_real: Mapped[float | None] = mapped_column(Float, nullable=True)
    devisso_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    devisso_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    devisso_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    pd_zone: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_structure: Mapped[str | None] = mapped_column(String(10), nullable=True)
    fvg_tfs: Mapped[str | None] = mapped_column(String(40), nullable=True)
    candle_pattern: Mapped[str | None] = mapped_column(String(100), nullable=True)
    rank_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    vs_btc: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank_combined: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank_rsi_cross: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank_z_confluence: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank_r_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank_aligned: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    rank_alignment_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ha_ultra_confirm: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    cross_indicator_close: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    vol_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mom_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    volat_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    candle_kategori: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # 13 Ağu 2026 (migration 034): candle_kategori'nin sayısal kırılımı —
    # önceden sadece hangi kısmın (gövde/üst fitil/alt fitil) baskın olduğu
    # tutulup ham yüzdeler atılıyordu. wick_pct = üst+alt fitil toplamı
    # (=100-body_pct) — Sinyal Mumu Ratioları'ndaki Body Ratio/Wick Length.
    body_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    wick_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 13 Ağu 2026 (migration 035): Sinyal Mumu Ratioları'nın kalan
    # metrikleri — RSI(14)/MFI(14)/RSI-of-MACD(12,26,9,14) sinyal barındaki
    # bar-to-bar değişimi + OBV seviyesi. "signal_" önekiyle, df üzerindeki
    # farklı-amaçlı rsi_change/macd_change kolonlarıyla (indicators/core.py)
    # karışmasın diye.
    signal_rsi_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_mfi_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_macd_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_obv: Mapped[float | None] = mapped_column(Float, nullable=True)
    all_up: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # 2 Ağu 2026 (migration 029): paper_trades'te zaten hesaplanan ama
    # signals'a hiç yazılmayan piyasa bağlamı/ML alanları — simetri
    regime_trend: Mapped[str | None] = mapped_column(String(20), nullable=True)
    volatility_regime: Mapped[str | None] = mapped_column(String(20), nullable=True)
    btc_z_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    btc_trend: Mapped[str | None] = mapped_column(String(20), nullable=True)
    funding_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    hour_utc: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    day_of_week: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)

    paper_trades: Mapped[list["PaperTrade"]] = relationship(
        "PaperTrade", back_populates="signal", lazy="noload"
    )


class PaperTrade(Base):
    __tablename__ = "paper_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("signals.id", ondelete="SET NULL"), nullable=True
    )
    strategy: Mapped[str] = mapped_column(String(50), nullable=False, default="conf_100")
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="signal")

    symbol: Mapped[str] = mapped_column(String(30), nullable=False)
    signal_type: Mapped[str] = mapped_column(String(10), nullable=False)
    interval: Mapped[str] = mapped_column(String(10), nullable=False)
    position_usd: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    trailing_stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)

    fee_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    balance_after: Mapped[float | None] = mapped_column(Float, nullable=True)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="open")
    close_reason: Mapped[str | None] = mapped_column(String(50), nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # ML snapshot
    btc_z_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    btc_trend: Mapped[str | None] = mapped_column(String(20), nullable=True)
    hour_utc: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    day_of_week: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    funding_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    recent_win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Denormalized signal features
    vpms_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    z_score_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    mtf_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank_at_entry: Mapped[int | None] = mapped_column(Integer, nullable=True)
    regime_trend: Mapped[str | None] = mapped_column(String(20), nullable=True)
    volatility_regime: Mapped[str | None] = mapped_column(String(20), nullable=True)

    vpmv_pre_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    vpmv_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    vpmv_slope: Mapped[float | None] = mapped_column(Float, nullable=True)
    vpmv_post_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    vpmv_post_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 13 Ağu 2026 (migration 033): Signal ile aynı — bkz. yukarıdaki açıklama.
    vol_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    mom_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    volat_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    vol_change_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    mom_change_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    volat_change_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_change_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    cvd_slope: Mapped[float | None] = mapped_column(Float, nullable=True)
    vp_buy_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    vp_sell_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    vp_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 20 Tem 2026: "kolaylık" — ERSI (ΔFiyat%/ΔRSI) skoru + bir önceki aynı
    # sembol/TF/yön sinyaline göre fark/oran. Bilgi/izleme amaçlı (bkz.
    # memory/project_devisso_ersi.md — PnL ile korelasyonu yok, filtre değil).
    devisso_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    devisso_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    devisso_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 19 Tem 2026: giriş anındaki TÜM enriched_signal dict'i (VPMV bileşenleri,
    # SMC, finansal oranlar, all_up, vb. — yukarıdaki kolonlarda yer almayan
    # her şey) — yeni metrik eklemek artık migration gerektirmiyor.
    entry_features: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # 2 Ağu 2026 (migration 029): signals'ta zaten hesaplanan ama
    # paper_trades'e hiç kopyalanmayan sinyal kalite/sıralama alanları —
    # simetri. paper_trade_manager.py zaten signal_id ile Signal satırını
    # tekrar okuyor (devisso_score gibi), aynı okumadan doldurulacak.
    alpha: Mapped[float | None] = mapped_column(Float, nullable=True)
    beta: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    sortino_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    calmar_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    information_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    vpmv_pre_proxy: Mapped[float | None] = mapped_column(Float, nullable=True)
    vpmv_pre_total: Mapped[float | None] = mapped_column(Float, nullable=True)
    vp_score_real: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_structure: Mapped[str | None] = mapped_column(String(10), nullable=True)
    fvg_tfs: Mapped[str | None] = mapped_column(String(40), nullable=True)
    candle_pattern: Mapped[str | None] = mapped_column(String(100), nullable=True)
    rank_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    vs_btc: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank_combined: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank_rsi_cross: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank_z_confluence: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank_r_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank_aligned: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    rank_alignment_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ha_ultra_confirm: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    vol_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mom_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    volat_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    candle_kategori: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # 13 Ağu 2026 (migration 034): candle_kategori'nin sayısal kırılımı —
    # önceden sadece hangi kısmın (gövde/üst fitil/alt fitil) baskın olduğu
    # tutulup ham yüzdeler atılıyordu. wick_pct = üst+alt fitil toplamı
    # (=100-body_pct) — Sinyal Mumu Ratioları'ndaki Body Ratio/Wick Length.
    body_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    wick_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 13 Ağu 2026 (migration 035): Sinyal Mumu Ratioları'nın kalan
    # metrikleri — RSI(14)/MFI(14)/RSI-of-MACD(12,26,9,14) sinyal barındaki
    # bar-to-bar değişimi + OBV seviyesi. "signal_" önekiyle, df üzerindeki
    # farklı-amaçlı rsi_change/macd_change kolonlarıyla (indicators/core.py)
    # karışmasın diye.
    signal_rsi_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_mfi_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_macd_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_obv: Mapped[float | None] = mapped_column(Float, nullable=True)
    all_up: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    sl_multiplier: Mapped[float | None] = mapped_column(Float, nullable=True)
    tp_multiplier: Mapped[float | None] = mapped_column(Float, nullable=True)

    signal: Mapped["Signal | None"] = relationship(
        "Signal", back_populates="paper_trades", lazy="noload"
    )


class SignalPerformance(Base):
    """Sinyalin T+3/T+5/T+10 bar sonraki getirisi + MFE/MAE — trade'in
    GERÇEKTEN nasıl kapandığından bağımsız, sabit-ufuk analitiği
    (signal_performance_analyzer.py yazar). 2 Ağu 2026'ya kadar ORM
    modeli yoktu, ham SQL ile erişiliyordu (migration 029 ile eklendi,
    tabloya/veriye dokunulmadı — sadece SQLAlchemy şema senkronizasyonu)."""

    __tablename__ = "signal_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("signals.id", ondelete="CASCADE"), nullable=False
    )
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_timestamp: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    atr_at_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    interval: Mapped[str | None] = mapped_column(String(10), nullable=True)

    return_t3_atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_t5_atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_t10_atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_t3_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_t5_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_t10_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    mfe_atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae_atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    mfe_bar_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mae_bar_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    is_calculated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    calculation_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    calculated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class TradeSnapshot(Base):
    """Açık bir paper trade'in ömrü boyunca periyodik (5dk) alınan CVD/VP/
    VPMV/SMC anlık görüntüsü — 19 Tem 2026, bkz. migration 022. signals
    tablosuyla aynı desende bir TimescaleDB hypertable (taken_at partition
    key)."""

    __tablename__ = "trade_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trade_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("paper_trades.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(30), nullable=False)
    taken_at: Mapped[datetime] = mapped_column(
        DateTime, primary_key=True, nullable=False, default=datetime.now
    )
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    cvd_slope: Mapped[float | None] = mapped_column(Float, nullable=True)
    vp_buy: Mapped[float | None] = mapped_column(Float, nullable=True)
    vp_sell: Mapped[float | None] = mapped_column(Float, nullable=True)
    vp_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    vp_score_real: Mapped[float | None] = mapped_column(Float, nullable=True)
    vol_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mom_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    volat_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_since_entry_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    vpmv_combined: Mapped[float | None] = mapped_column(Float, nullable=True)
    smc_market_structure: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # 17 Ağu 2026 (migration 036): alpha/beta + finansal oranlar (HAM, normalize
    # edilmemiş — normalized_composite/scaled_avg_normalized'ın "50'ye yapışma"
    # bug'ı burada YOK) + Sinyal Mumu Ratioları. totalamount_rank1 (signal_id'siz
    # açılan tek aktif strateji) için bu ailenin TEK kaynağı — bkz.
    # memory/project_financial_metrics_pine_mismatch_10agu.md.
    alpha: Mapped[float | None] = mapped_column(Float, nullable=True)
    beta: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    sortino_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    calmar_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    treynor_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    information_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_rsi_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_mfi_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_macd_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_obv: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    wick_pct: Mapped[float | None] = mapped_column(Float, nullable=True)


class PaperPortfolio(Base):
    __tablename__ = "paper_portfolio"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy: Mapped[str] = mapped_column(
        String(50), nullable=False, unique=True, default="conf_100"
    )
    balance: Mapped[float] = mapped_column(Float, nullable=False, default=10000.0)
    initial_balance: Mapped[float] = mapped_column(Float, nullable=False, default=10000.0)
    peak_balance: Mapped[float] = mapped_column(Float, nullable=False, default=10000.0)
    max_drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_pnl_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now)
