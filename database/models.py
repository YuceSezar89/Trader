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

    id = Column(Integer, primary_key=True, autoincrement=True)  # Auto-increment primary key
    symbol = Column(String)  # Artık primary key değil
    timestamp = Column(DateTime)  # Artık primary key değil
    signal_type = Column(String)
    interval = Column(String)
    price = Column(Float, nullable=True)
    pullback_level = Column(Float, nullable=True)
    strength = Column(Integer, nullable=True)
    indicators = Column(String, nullable=True)
    rsi = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    momentum = Column(Float, nullable=True)
    atr = Column(Float, nullable=True)
    adx = Column(Float, nullable=True)
    plus_di = Column(Float, nullable=True)
    minus_di = Column(Float, nullable=True)
    alpha = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    calmar_ratio = Column(Float, nullable=True)
    omega_ratio = Column(Float, nullable=True)
    treynor_ratio = Column(Float, nullable=True)
    information_ratio = Column(Float, nullable=True)
    scaled_avg_normalized = Column(Float, nullable=True)
    normalized_composite = Column(Float, nullable=True)
    normalized_price_change = Column(Float, nullable=True)
    zscore_ratio_percent = Column(Float, nullable=True)

    # V-P-M onay ve skor
    vpms_score = Column(Float, nullable=True)
    vpm_confirmed = Column(Boolean, nullable=True)

    # MTF bonus ve birleşik skor
    mtf_score = Column(Float, nullable=True)
    vpms_mtf_score = Column(Float, nullable=True)

    # Lifecycle yönetimi kolonları
    status = Column(String(20), nullable=True, default='active')
    superseded_by = Column(Integer, nullable=True)
    superseded_at = Column(DateTime, nullable=True)
    performance_period = Column(Integer, nullable=True)
    lifecycle_end_reason = Column(String(50), nullable=True)

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", "signal_type", "interval"),
    )
