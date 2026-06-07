"""
CandleChart — pyqtgraph tabanlı mumlu grafik.

Layout (3 satır):
    Satır 0 (65%): Mumluk + EMA-21
    Satır 1 (15%): Hacim barları
    Satır 2 (20%): RSI-14
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QPainter, QPicture
from PyQt6.QtWidgets import QSizePolicy

from desktop.theme import COLORS

_BULL = COLORS["green"]
_BEAR = COLORS["red"]
_EMA_COLOR = "#58a6ff"
_RSI_COLOR = "#a371f7"
_VOL_BULL = COLORS["green"] + "99"
_VOL_BEAR = COLORS["red"] + "99"

pg.setConfigOption("background", COLORS["bg_primary"])
pg.setConfigOption("foreground", COLORS["text_muted"])
pg.setConfigOption("antialias", True)


# ── Candlestick Item ──────────────────────────────────────────────────────────

class CandlestickItem(pg.GraphicsObject):
    """OHLCV mum grafiği çizen GraphicsObject."""

    def __init__(self):
        super().__init__()
        self._data: list[tuple] = []
        self._picture: Optional[QPicture] = None
        self._bar_w = 0.4

    def set_data(self, data: list[tuple]) -> None:
        """data: [(timestamp_float, open, high, low, close), ...]"""
        self._data = data
        self._bar_w = max(0.3, 0.4 * (600 / max(len(data), 1)) ** 0.3) if data else 0.4
        self._build_picture()
        self.update()
        self.informViewBoundsChanged()

    def _build_picture(self) -> None:
        self._picture = QPicture()
        p = QPainter(self._picture)
        w = self._bar_w

        for t, o, h, l, c in self._data:
            bull = c >= o
            color = _BULL if bull else _BEAR
            pen = pg.mkPen(color, width=1)
            brush = pg.mkBrush(color)

            p.setPen(pen)
            p.setBrush(pg.mkBrush(None))
            p.drawLine(QPointF(t, l), QPointF(t, h))

            p.setBrush(brush)
            body_top = max(o, c)
            body_bot = min(o, c)
            body_h = max(body_top - body_bot, 1e-9)
            p.drawRect(QRectF(t - w, body_bot, w * 2, body_h))

        p.end()

    def paint(self, p: QPainter, *args) -> None:
        if self._picture:
            self._picture.play(p)

    def boundingRect(self) -> QRectF:
        if not self._data:
            return QRectF(0, 0, 1, 1)
        ts = [d[0] for d in self._data]
        highs = [d[2] for d in self._data]
        lows = [d[3] for d in self._data]
        return QRectF(
            min(ts) - 0.5,
            min(lows),
            max(ts) - min(ts) + 1,
            max(highs) - min(lows),
        )


# ── Volume Item ───────────────────────────────────────────────────────────────

class VolumeItem(pg.GraphicsObject):
    """Hacim barlarını çizen GraphicsObject."""

    def __init__(self):
        super().__init__()
        self._data: list[tuple] = []
        self._picture: Optional[QPicture] = None

    def set_data(self, data: list[tuple]) -> None:
        """data: [(timestamp_float, volume, bull: bool), ...]"""
        self._data = data
        self._build_picture()
        self.update()
        self.informViewBoundsChanged()

    def _build_picture(self) -> None:
        self._picture = QPicture()
        p = QPainter(self._picture)
        if not self._data:
            p.end()
            return

        w = max(0.3, 0.4 * (600 / max(len(self._data), 1)) ** 0.3)
        for t, vol, bull in self._data:
            color = _VOL_BULL if bull else _VOL_BEAR
            p.setPen(pg.mkPen(color, width=1))
            p.setBrush(pg.mkBrush(color))
            p.drawRect(QRectF(t - w, 0, w * 2, vol))
        p.end()

    def paint(self, p: QPainter, *args) -> None:
        if self._picture:
            self._picture.play(p)

    def boundingRect(self) -> QRectF:
        if not self._data:
            return QRectF(0, 0, 1, 1)
        ts = [d[0] for d in self._data]
        vols = [d[1] for d in self._data]
        return QRectF(min(ts) - 0.5, 0, max(ts) - min(ts) + 1, max(vols) * 1.05)


# ── CandleChart ───────────────────────────────────────────────────────────────

class CandleChart(pg.GraphicsLayoutWidget):
    """
    Ana grafik widget'i.

    Kullanım:
        chart = CandleChart()
        chart.load_df(df, symbol="BTCUSDT", tf="1h")
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(300)

        self._symbol = ""
        self._tf = ""
        self._df: Optional[pd.DataFrame] = None
        self._store_ts: Optional[np.ndarray] = None
        self._setup_plots()
        self._setup_crosshair()

    # ── Plot kurulumu ─────────────────────────────────────────────────────────

    def _setup_plots(self) -> None:
        self.ci.layout.setSpacing(0)
        self.ci.setContentsMargins(4, 4, 4, 4)

        axis_pen = pg.mkPen(COLORS["border"])

        def _left_axis() -> pg.AxisItem:
            ax = pg.AxisItem("left")
            ax.setPen(axis_pen)
            ax.setTextPen(pg.mkPen(COLORS["text_muted"]))
            ax.setStyle(tickLength=-5)
            return ax

        def _date_axis() -> pg.DateAxisItem:
            ax = pg.DateAxisItem(orientation="bottom")
            ax.setPen(axis_pen)
            ax.setTextPen(pg.mkPen(COLORS["text_muted"]))
            ax.setStyle(tickLength=-5)
            return ax

        # Satır 0: Mumluk
        self._p_candle = self.addPlot(
            row=0, col=0,
            axisItems={"left": _left_axis(), "bottom": _date_axis()},
        )
        self._p_candle.showGrid(x=True, y=True, alpha=0.15)
        self._p_candle.hideAxis("bottom")
        self._p_candle.setMenuEnabled(False)

        # Candle öğesi
        self._candle_item = CandlestickItem()
        self._p_candle.addItem(self._candle_item)

        # EMA çizgisi
        self._ema_line = self._p_candle.plot(
            pen=pg.mkPen(_EMA_COLOR, width=1.5), name="EMA-21"
        )

        # Fiyat etiketi — ignoreBounds: bounding rect hesabını etkilemesin
        self._price_label = pg.TextItem(
            text="", color=COLORS["text_primary"], anchor=(0, 0.5)
        )
        self._p_candle.addItem(self._price_label, ignoreBounds=True)

        # OHLCV tooltip
        self._ohlcv_label = pg.TextItem(
            text="", color=COLORS["text_muted"], anchor=(0, 1)
        )
        self._ohlcv_label.setPos(0, 0)
        self._p_candle.addItem(self._ohlcv_label, ignoreBounds=True)

        # Satır 1: Hacim
        self._p_vol = self.addPlot(
            row=1, col=0,
            axisItems={"left": _left_axis(), "bottom": _date_axis()},
        )
        self._p_vol.showGrid(x=False, y=True, alpha=0.1)
        self._p_vol.hideAxis("bottom")
        self._p_vol.setMenuEnabled(False)
        self._p_vol.setMaximumHeight(80)

        self._vol_item = VolumeItem()
        self._p_vol.addItem(self._vol_item)
        self._p_vol.setXLink(self._p_candle)

        # Satır 2: RSI — sadece bu panelde tarih ekseni görünür
        self._p_rsi = self.addPlot(
            row=2, col=0,
            axisItems={"left": _left_axis(), "bottom": _date_axis()},
        )
        self._p_rsi.showGrid(x=True, y=True, alpha=0.1)
        self._p_rsi.setMenuEnabled(False)
        self._p_rsi.setMaximumHeight(90)
        self._p_rsi.setYRange(0, 100)
        self._p_rsi.setXLink(self._p_candle)

        self._rsi_line = self._p_rsi.plot(
            pen=pg.mkPen(_RSI_COLOR, width=1.5), name="RSI-14"
        )

        for level, color in [(30, COLORS["green"]), (70, COLORS["red"])]:
            self._p_rsi.addItem(
                pg.InfiniteLine(
                    pos=level, angle=0,
                    pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                )
            )

        self._p_rsi.showAxis("bottom")

        # Auto-range kapalı — kullanıcı zoom/pan yapabilsin
        for plot in [self._p_candle, self._p_vol, self._p_rsi]:
            plot.vb.disableAutoRange()

    def _setup_crosshair(self) -> None:
        ch_pen = pg.mkPen(COLORS["border_hover"], width=1, style=Qt.PenStyle.DashLine)

        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=ch_pen)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=ch_pen)
        self._p_candle.addItem(self._vline)
        self._p_candle.addItem(self._hline)

        self._p_candle.scene().sigMouseMoved.connect(self._on_mouse_move)

    # ── Veri yükleme ──────────────────────────────────────────────────────────

    def load_df(self, df: pd.DataFrame, symbol: str = "", tf: str = "", auto_range: bool = False) -> None:
        """DataFrame'i grafiklere yükler. Sütunlar: timestamp, open, high, low, close, volume."""
        if df is None or df.empty:
            self.clear_chart()
            return

        symbol_changed = (symbol != self._symbol or tf != self._tf)
        self._symbol = symbol
        self._tf = tf
        self._df = df.copy()

        ts = df["timestamp"].astype("int64") // 10**9  # Unix saniye
        opens  = df["open"].astype(float)
        highs  = df["high"].astype(float)
        lows   = df["low"].astype(float)
        closes = df["close"].astype(float)
        vols   = df["volume"].astype(float)

        # Mumluk
        candle_data = list(zip(ts, opens, highs, lows, closes))
        self._candle_item.set_data(candle_data)

        # EMA-21
        ema = self._calc_ema(closes, 21)
        self._ema_line.setData(ts.values, ema.values)

        # Hacim
        bulls = (closes >= opens).values
        vol_data = list(zip(ts, vols, bulls))
        self._vol_item.set_data(vol_data)

        # RSI-14
        rsi = self._calc_rsi(closes)
        valid = ~rsi.isna()
        self._rsi_line.setData(ts.values[valid], rsi.values[valid])

        # Son fiyat etiketi
        last_price = closes.iloc[-1]
        last_ts    = ts.iloc[-1]
        self._price_label.setPos(last_ts + 1, last_price)
        self._price_label.setText(f" {last_price:,.4f}")

        # Sadece sembol/TF değişince veya açık talep varsa görünümü sıfırla
        if auto_range or symbol_changed:
            self._p_candle.autoRange()
            for plot in [self._p_candle, self._p_vol, self._p_rsi]:
                plot.vb.disableAutoRange()
        self._store_ts = ts.values

    def clear_chart(self) -> None:
        self._candle_item.set_data([])
        self._vol_item.set_data([])
        self._ema_line.setData([], [])
        self._rsi_line.setData([], [])
        self._price_label.setText("")

    # ── Mouse crosshair ───────────────────────────────────────────────────────

    def _on_mouse_move(self, pos) -> None:
        if not self._p_candle.sceneBoundingRect().contains(pos):
            return
        mouse_pt = self._p_candle.vb.mapSceneToView(pos)
        self._vline.setPos(mouse_pt.x())
        self._hline.setPos(mouse_pt.y())

        df = getattr(self, "_df", None)
        if df is None or df.empty:
            return

        ts_vals = getattr(self, "_store_ts", None)
        if ts_vals is None or len(ts_vals) == 0:
            return

        idx = int(np.searchsorted(ts_vals, mouse_pt.x()))
        idx = min(max(idx, 0), len(df) - 1)
        row = df.iloc[idx]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = COLORS["green"] if c >= o else COLORS["red"]
        self._ohlcv_label.setText(
            f"O:{o:.4f}  H:{h:.4f}  L:{l:.4f}  C:{c:.4f}",
            color=color,
        )
        vr = self._p_candle.viewRect()
        self._ohlcv_label.setPos(vr.left(), vr.top())

    # ── Göstergeler ───────────────────────────────────────────────────────────

    @staticmethod
    def _calc_ema(closes: pd.Series, period: int) -> pd.Series:
        return closes.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _calc_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))
