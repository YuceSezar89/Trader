from sqlalchemy import (
    Column,
    Integer,
    SmallInteger,
    String,
    Float,
    Boolean,
    DateTime,
    PrimaryKeyConstraint,
    UniqueConstraint,
    ForeignKey,
    ForeignKeyConstraint,
)
from sqlalchemy.dialects.postgresql import TIMESTAMP as PG_TIMESTAMP
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime


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

    __table_args__ = (
        PrimaryKeyConstraint("symbol", "timestamp"),
        UniqueConstraint("symbol", "interval", "timestamp"),
    )


class Signal(Base):
    __tablename__ = "signals"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    symbol       = Column(String, nullable=False)
    interval     = Column(String, nullable=False)
    indicators   = Column(String, nullable=False)
    signal_type  = Column(String, nullable=False)

    opened_at    = Column(DateTime, nullable=False, default=datetime.now)
    open_price   = Column(Float, nullable=False)

    vpms_score   = Column(Float, nullable=True)
    mtf_score    = Column(Float, nullable=True)
    st_confirmed = Column(Boolean, nullable=True)
    rsi          = Column(Float, nullable=True)
    strength     = Column(Integer, nullable=True)
    atr          = Column(Float, nullable=True)
    alpha        = Column(Float, nullable=True)
    beta         = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)

    status       = Column(String(20), nullable=False, default='active')
    closed_at    = Column(DateTime, nullable=True)
    close_price  = Column(Float, nullable=True)
    close_reason = Column(String(20), nullable=True)
    closed_by    = Column(Integer, nullable=True)

    realized_pnl     = Column(Float, nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    oi_data          = Column(String, nullable=True)

    stop_loss_price   = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    sl_multiplier     = Column(Float, nullable=True)
    tp_multiplier     = Column(Float, nullable=True)

    z_score_entry      = Column(Float, nullable=True)
    is_confluence      = Column(Boolean, nullable=True, default=False)
    trailing_stop_price = Column(Float, nullable=True)

    sortino_ratio   = Column(Float, nullable=True)
    calmar_ratio    = Column(Float, nullable=True)
    vpmv_pre_avg    = Column(Float, nullable=True)
    vpmv_ratio      = Column(Float, nullable=True)
    vpmv_slope      = Column(Float, nullable=True)
    vpmv_post_avg   = Column(Float, nullable=True)
    vpmv_post_delta = Column(Float, nullable=True)

    paper_trades = relationship("PaperTrade", back_populates="signal", lazy="noload")


class PaperTrade(Base):
    __tablename__ = "paper_trades"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    signal_id       = Column(Integer, ForeignKey("signals.id", ondelete="SET NULL"), nullable=True)
    strategy        = Column(String(50), nullable=False, default="conf_100")

    symbol          = Column(String(30), nullable=False)
    signal_type     = Column(String(10), nullable=False)
    interval        = Column(String(10), nullable=False)
    position_usd    = Column(Float, nullable=False, default=100.0)
    entry_price     = Column(Float, nullable=False)
    exit_price      = Column(Float, nullable=True)
    stop_loss_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    trailing_stop_price = Column(Float, nullable=True)

    fee_usd         = Column(Float, nullable=True)
    pnl_usd         = Column(Float, nullable=True)
    pnl_pct         = Column(Float, nullable=True)
    balance_after   = Column(Float, nullable=True)

    status          = Column(String(20), nullable=False, default="open")
    close_reason    = Column(String(50), nullable=True)
    opened_at       = Column(PG_TIMESTAMP(timezone=True), nullable=False, default=datetime.now)
    closed_at       = Column(PG_TIMESTAMP(timezone=True), nullable=True)

    # ML snapshot
    btc_z_score     = Column(Float, nullable=True)
    btc_trend       = Column(String(20), nullable=True)
    hour_utc        = Column(SmallInteger, nullable=True)
    day_of_week     = Column(SmallInteger, nullable=True)
    funding_rate    = Column(Float, nullable=True)
    recent_win_rate = Column(Float, nullable=True)

    # Denormalized signal features
    vpms_score      = Column(Float, nullable=True)
    z_score_entry   = Column(Float, nullable=True)
    mtf_score       = Column(Float, nullable=True)
    atr             = Column(Float, nullable=True)
    rank_at_entry     = Column(Integer, nullable=True)
    regime_trend      = Column(String(20), nullable=True)
    volatility_regime = Column(String(20), nullable=True)

    vpmv_pre_avg    = Column(Float, nullable=True)
    vpmv_ratio      = Column(Float, nullable=True)
    vpmv_slope      = Column(Float, nullable=True)
    vpmv_post_avg   = Column(Float, nullable=True)
    vpmv_post_delta = Column(Float, nullable=True)

    signal = relationship("Signal", back_populates="paper_trades", lazy="noload")


class PaperPortfolio(Base):
    __tablename__ = "paper_portfolio"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    strategy         = Column(String(50), nullable=False, unique=True, default="conf_100")
    balance          = Column(Float, nullable=False, default=10000.0)
    initial_balance  = Column(Float, nullable=False, default=10000.0)
    peak_balance     = Column(Float, nullable=False, default=10000.0)
    max_drawdown_pct = Column(Float, nullable=False, default=0.0)
    total_trades     = Column(Integer, nullable=False, default=0)
    winning_trades   = Column(Integer, nullable=False, default=0)
    total_pnl_usd    = Column(Float, nullable=False, default=0.0)
    updated_at       = Column(PG_TIMESTAMP(timezone=True), nullable=False, default=datetime.now)
