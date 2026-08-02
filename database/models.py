from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
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

    symbol = Column(String, primary_key=True)
    timestamp = Column(DateTime, primary_key=True)
    interval = Column(String, nullable=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    buy_volume = Column(Float, nullable=True)
    sell_volume = Column(Float, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("symbol", "timestamp"),
        UniqueConstraint("symbol", "interval", "timestamp"),
    )


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    interval = Column(String, nullable=False)
    indicators = Column(String, nullable=False)
    signal_type = Column(String, nullable=False)

    opened_at = Column(DateTime, nullable=False, default=datetime.now)
    open_price = Column(Float, nullable=False)

    vpms_score = Column(Float, nullable=True)
    mtf_score = Column(Float, nullable=True)
    st_confirmed = Column(Boolean, nullable=True)
    rsi = Column(Float, nullable=True)
    strength = Column(Integer, nullable=True)
    atr = Column(Float, nullable=True)
    alpha = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)

    status = Column(String(20), nullable=False, default="active")
    closed_at = Column(DateTime, nullable=True)
    close_price = Column(Float, nullable=True)
    close_reason = Column(String(20), nullable=True)
    closed_by = Column(Integer, nullable=True)

    realized_pnl = Column(Float, nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    oi_data = Column(String, nullable=True)

    stop_loss_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    sl_multiplier = Column(Float, nullable=True)
    tp_multiplier = Column(Float, nullable=True)

    z_score_entry = Column(Float, nullable=True)
    is_confluence = Column(Boolean, nullable=True, default=False)
    trailing_stop_price = Column(Float, nullable=True)

    sortino_ratio = Column(Float, nullable=True)
    calmar_ratio = Column(Float, nullable=True)
    information_ratio = Column(Float, nullable=True)
    vpmv_pre_avg = Column(Float, nullable=True)
    vpmv_pre_proxy = Column(Float, nullable=True)
    vpmv_pre_total = Column(Float, nullable=True)
    vpmv_ratio = Column(Float, nullable=True)
    vpmv_slope = Column(Float, nullable=True)
    vpmv_post_avg = Column(Float, nullable=True)
    vpmv_post_delta = Column(Float, nullable=True)
    cvd_slope = Column(Float, nullable=True)
    vp_buy_avg = Column(Float, nullable=True)
    vp_sell_avg = Column(Float, nullable=True)
    vp_score = Column(Float, nullable=True)
    vp_score_real = Column(Float, nullable=True)
    devisso_score = Column(Float, nullable=True)
    devisso_delta = Column(Float, nullable=True)
    devisso_ratio = Column(Float, nullable=True)
    pd_zone = Column(Float, nullable=True)
    market_structure = Column(String(10), nullable=True)
    fvg_tfs = Column(String(40), nullable=True)
    candle_pattern = Column(String(100), nullable=True)
    rank_score = Column(Float, nullable=True)
    vs_btc = Column(Float, nullable=True)
    rank_combined = Column(Float, nullable=True)
    rank_rsi_cross = Column(Float, nullable=True)
    rank_z_confluence = Column(Float, nullable=True)
    rank_r_score = Column(Float, nullable=True)
    rank_aligned = Column(Boolean, nullable=True)
    rank_alignment_count = Column(Integer, nullable=True)
    ha_ultra_confirm = Column(SmallInteger, nullable=True)
    cross_indicator_close = Column(Boolean, nullable=True)
    vol_score = Column(Float, nullable=True)
    mom_score = Column(Float, nullable=True)
    volat_score = Column(Float, nullable=True)
    price_score = Column(Float, nullable=True)
    candle_kategori = Column(String(20), nullable=True)
    all_up = Column(Boolean, nullable=True)

    # 2 Ağu 2026 (migration 029): paper_trades'te zaten hesaplanan ama
    # signals'a hiç yazılmayan piyasa bağlamı/ML alanları — simetri
    regime_trend = Column(String(20), nullable=True)
    volatility_regime = Column(String(20), nullable=True)
    btc_z_score = Column(Float, nullable=True)
    btc_trend = Column(String(20), nullable=True)
    funding_rate = Column(Float, nullable=True)
    hour_utc = Column(SmallInteger, nullable=True)
    day_of_week = Column(SmallInteger, nullable=True)

    paper_trades = relationship("PaperTrade", back_populates="signal", lazy="noload")


class PaperTrade(Base):
    __tablename__ = "paper_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(Integer, ForeignKey("signals.id", ondelete="SET NULL"), nullable=True)
    strategy = Column(String(50), nullable=False, default="conf_100")
    source = Column(String(20), nullable=False, default="signal")

    symbol = Column(String(30), nullable=False)
    signal_type = Column(String(10), nullable=False)
    interval = Column(String(10), nullable=False)
    position_usd = Column(Float, nullable=False, default=100.0)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    stop_loss_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    trailing_stop_price = Column(Float, nullable=True)

    fee_usd = Column(Float, nullable=True)
    pnl_usd = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    balance_after = Column(Float, nullable=True)

    status = Column(String(20), nullable=False, default="open")
    close_reason = Column(String(50), nullable=True)
    opened_at = Column(DateTime, nullable=False, default=datetime.now)
    closed_at = Column(DateTime, nullable=True)

    # ML snapshot
    btc_z_score = Column(Float, nullable=True)
    btc_trend = Column(String(20), nullable=True)
    hour_utc = Column(SmallInteger, nullable=True)
    day_of_week = Column(SmallInteger, nullable=True)
    funding_rate = Column(Float, nullable=True)
    recent_win_rate = Column(Float, nullable=True)

    # Denormalized signal features
    vpms_score = Column(Float, nullable=True)
    z_score_entry = Column(Float, nullable=True)
    mtf_score = Column(Float, nullable=True)
    atr = Column(Float, nullable=True)
    rank_at_entry = Column(Integer, nullable=True)
    regime_trend = Column(String(20), nullable=True)
    volatility_regime = Column(String(20), nullable=True)

    vpmv_pre_avg = Column(Float, nullable=True)
    vpmv_ratio = Column(Float, nullable=True)
    vpmv_slope = Column(Float, nullable=True)
    vpmv_post_avg = Column(Float, nullable=True)
    vpmv_post_delta = Column(Float, nullable=True)
    cvd_slope = Column(Float, nullable=True)
    vp_buy_avg = Column(Float, nullable=True)
    vp_sell_avg = Column(Float, nullable=True)
    vp_score = Column(Float, nullable=True)

    # 20 Tem 2026: "kolaylık" — ERSI (ΔFiyat%/ΔRSI) skoru + bir önceki aynı
    # sembol/TF/yön sinyaline göre fark/oran. Bilgi/izleme amaçlı (bkz.
    # memory/project_devisso_ersi.md — PnL ile korelasyonu yok, filtre değil).
    devisso_score = Column(Float, nullable=True)
    devisso_delta = Column(Float, nullable=True)
    devisso_ratio = Column(Float, nullable=True)

    # 19 Tem 2026: giriş anındaki TÜM enriched_signal dict'i (VPMV bileşenleri,
    # SMC, finansal oranlar, all_up, vb. — yukarıdaki kolonlarda yer almayan
    # her şey) — yeni metrik eklemek artık migration gerektirmiyor.
    entry_features = Column(JSONB, nullable=True)

    # 2 Ağu 2026 (migration 029): signals'ta zaten hesaplanan ama
    # paper_trades'e hiç kopyalanmayan sinyal kalite/sıralama alanları —
    # simetri. paper_trade_manager.py zaten signal_id ile Signal satırını
    # tekrar okuyor (devisso_score gibi), aynı okumadan doldurulacak.
    alpha = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    calmar_ratio = Column(Float, nullable=True)
    information_ratio = Column(Float, nullable=True)
    vpmv_pre_proxy = Column(Float, nullable=True)
    vpmv_pre_total = Column(Float, nullable=True)
    vp_score_real = Column(Float, nullable=True)
    market_structure = Column(String(10), nullable=True)
    fvg_tfs = Column(String(40), nullable=True)
    candle_pattern = Column(String(100), nullable=True)
    rank_score = Column(Float, nullable=True)
    vs_btc = Column(Float, nullable=True)
    rank_combined = Column(Float, nullable=True)
    rank_rsi_cross = Column(Float, nullable=True)
    rank_z_confluence = Column(Float, nullable=True)
    rank_r_score = Column(Float, nullable=True)
    rank_aligned = Column(Boolean, nullable=True)
    rank_alignment_count = Column(Integer, nullable=True)
    ha_ultra_confirm = Column(SmallInteger, nullable=True)
    vol_score = Column(Float, nullable=True)
    mom_score = Column(Float, nullable=True)
    volat_score = Column(Float, nullable=True)
    price_score = Column(Float, nullable=True)
    candle_kategori = Column(String(20), nullable=True)
    all_up = Column(Boolean, nullable=True)
    sl_multiplier = Column(Float, nullable=True)
    tp_multiplier = Column(Float, nullable=True)

    signal = relationship("Signal", back_populates="paper_trades", lazy="noload")


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
    last_calculation_error: Mapped[str | None] = mapped_column(String, nullable=True)

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
