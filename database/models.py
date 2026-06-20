from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    PrimaryKeyConstraint,
    UniqueConstraint,
    ForeignKeyConstraint
)
from sqlalchemy.orm import DeclarativeBase, Mapped, relationship
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
