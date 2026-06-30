"""
ChartPanel — sembol + TF seçimi + CandleChart.

MarketWorker'dan gelen klines_updated sinyaliyle veya watchlist'ten
symbol_selected ile tetiklenir.
"""

from __future__ import annotations

import io
from typing import Optional

import pandas as pd
from PyQt6.QtCore import pyqtSignal, pyqtSlot  # pylint: disable=no-name-in-module
from PyQt6.QtWidgets import (  # pylint: disable=no-name-in-module
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import COLORS
from desktop.widgets.tv_chart import TVChart

_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
_DEFAULT_TF = "1h"
_DEFAULT_SYM = "BTCUSDT"
_LIMIT = 200


class ChartPanel(QWidget):  # pylint: disable=too-many-instance-attributes
    """
    Grafik paneli içeriği (QDockWidget'a yerleştirilen widget).

    Sinyaller:
        symbol_changed(str, str): Kullanıcı sembol/TF değiştirdiğinde (symbol, tf).
    """

    symbol_changed = pyqtSignal(str, str)

    def __init__(self, redis_url: str, db_cfg: dict, parent=None):
        super().__init__(parent)
        self._redis_url = redis_url
        self._db_cfg = db_cfg
        self._symbol = _DEFAULT_SYM
        self._tf = _DEFAULT_TF
        self._tf_buttons: dict[str, QPushButton] = {}

        self._setup_ui()
        self.load_symbol(_DEFAULT_SYM, _DEFAULT_TF)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Başlık + TF butonları
        header = QHBoxLayout()
        self._sym_label = QLabel(_DEFAULT_SYM)
        self._sym_label.setObjectName("section_title")
        self._sym_label.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 14px; font-weight: 700;"
        )
        header.addWidget(self._sym_label)

        self._price_label = QLabel("—")
        self._price_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 13px; margin-left: 8px;"
        )
        header.addWidget(self._price_label)
        header.addStretch()

        for tf in _TIMEFRAMES:
            btn = QPushButton(tf)
            btn.setFixedHeight(24)
            btn.setMinimumWidth(36)
            btn.setCheckable(True)
            btn.setChecked(tf == _DEFAULT_TF)
            btn.clicked.connect(lambda _, t=tf: self._on_tf_clicked(t))
            btn.setStyleSheet(self._tf_btn_style(tf == _DEFAULT_TF))
            header.addWidget(btn)
            self._tf_buttons[tf] = btn

        self._log_btn = QPushButton("Log")
        self._log_btn.setFixedHeight(24)
        self._log_btn.setFixedWidth(36)
        self._log_btn.setCheckable(True)
        self._log_btn.setChecked(False)
        self._log_btn.setStyleSheet(self._tf_btn_style(False))
        self._log_btn.clicked.connect(self._on_log_toggled)
        header.addWidget(self._log_btn)

        self._smc_btn = QPushButton("SMC")
        self._smc_btn.setFixedHeight(24)
        self._smc_btn.setFixedWidth(42)
        self._smc_btn.setCheckable(True)
        self._smc_btn.setChecked(False)
        self._smc_btn.setStyleSheet(self._tf_btn_style(False))
        self._smc_btn.clicked.connect(self._on_smc_toggled)
        header.addWidget(self._smc_btn)

        self._fvg_btn = QPushButton("FVG")
        self._fvg_btn.setFixedHeight(24)
        self._fvg_btn.setFixedWidth(42)
        self._fvg_btn.setCheckable(True)
        self._fvg_btn.setChecked(False)
        self._fvg_btn.setStyleSheet(self._tf_btn_style(False))
        self._fvg_btn.clicked.connect(self._on_fvg_toggled)
        header.addWidget(self._fvg_btn)

        self._phl_btn = QPushButton("PDH/L")
        self._phl_btn.setFixedHeight(24)
        self._phl_btn.setFixedWidth(52)
        self._phl_btn.setCheckable(True)
        self._phl_btn.setChecked(False)
        self._phl_btn.setStyleSheet(self._tf_btn_style(False))
        self._phl_btn.clicked.connect(self._on_phl_toggled)
        header.addWidget(self._phl_btn)

        root.addLayout(header)

        # Grafik
        self._chart = TVChart(self)
        root.addWidget(self._chart)

    @staticmethod
    def _tf_btn_style(active: bool) -> str:
        if active:
            return (
                f"QPushButton {{ background-color: {COLORS['accent']}; "
                f"color: #fff; border: none; border-radius: 3px; "
                f"font-size: 11px; padding: 0 8px; }}"
            )
        return (
            f"QPushButton {{ background-color: {COLORS['bg_tertiary']}; "
            f"color: {COLORS['text_muted']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 3px; font-size: 11px; padding: 0 8px; }}"
            f"QPushButton:hover {{ color: {COLORS['text_primary']}; }}"
        )

    # ── Veri yükleme ──────────────────────────────────────────────────────────

    def _load_and_draw(self, symbol: str, tf: str, auto_range: bool = False) -> None:  # pylint: disable=unused-argument
        df = self._fetch_data(symbol, tf)
        self._chart.load_df(df, symbol, tf)
        if df is not None and not df.empty:
            last = df["close"].iloc[-1]
            self._price_label.setText(f"{last:,.4f}")
        # Sembol değişince marker'ları temizle
        self._chart.clear_signal_marker()

    def _fetch_data(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        """Redis Arrow → JSON fallback → DB fallback."""
        try:
            import redis as _redis  # pylint: disable=import-outside-toplevel
            r = _redis.Redis.from_url(
                self._redis_url,
                decode_responses=False,
                socket_connect_timeout=2,
            )
            raw = r.get(f"live_kline_data:{symbol}:{tf}")
            if raw:
                if raw[:4] == b"ARDF":
                    import pyarrow as pa  # pylint: disable=import-outside-toplevel
                    reader = pa.ipc.open_stream(raw[4:])
                    df = reader.read_pandas()
                else:
                    df = pd.read_json(io.StringIO(raw.decode("utf-8")), orient="split")
                df = self._normalize_df(df)
                if df is not None:
                    return df.tail(_LIMIT)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return self._fetch_from_db(symbol, tf)

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        # open_time → timestamp alias (Binance/Redis formatı)
        if "timestamp" not in df.columns and "open_time" in df.columns:
            df = df.rename(columns={"open_time": "timestamp"})

        needed = {"timestamp", "open", "high", "low", "close", "volume"}
        cols_lower = {c.lower(): c for c in df.columns}
        rename = {}
        for need in needed:
            if need not in df.columns and need in cols_lower:
                rename[cols_lower[need]] = need
        if rename:
            df = df.rename(columns=rename)
        if not needed.issubset(df.columns):
            return None
        if pd.api.types.is_integer_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _fetch_from_db(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        try:
            import psycopg2  # pylint: disable=import-outside-toplevel
            conn = psycopg2.connect(**self._db_cfg)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM price_data
                WHERE symbol = %s AND interval = %s
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (symbol, tf, _LIMIT),
            )
            rows = cur.fetchall()
            conn.close()
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    # ── Slot'lar ──────────────────────────────────────────────────────────────

    @pyqtSlot(str, str, object)
    def on_klines_updated(self, symbol: str, tf: str, df: object) -> None:
        """MarketWorker pub/sub güncellemesini doğrudan çizer."""
        if symbol != self._symbol or tf != self._tf:
            return
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        df_norm = self._normalize_df(df.tail(_LIMIT))
        if df_norm is not None:
            self._chart.update_last_bar(df_norm)
            self._price_label.setText(f"{float(df_norm['close'].iloc[-1]):,.4f}")

    @pyqtSlot(str)
    def load_symbol(self, symbol: str, tf: Optional[str] = None) -> None:
        """Sembolü yükler ve grafiği çizer; TF verilmezse mevcut TF kullanılır."""
        if tf is None:
            tf = self._tf
        self._symbol = symbol
        self._tf = tf
        self._sym_label.setText(symbol)
        self._update_tf_buttons(tf)
        self._load_and_draw(symbol, tf, auto_range=True)
        self.symbol_changed.emit(symbol, tf)

    def _on_log_toggled(self, enabled: bool) -> None:
        self._log_btn.setStyleSheet(self._tf_btn_style(enabled))
        self._chart.set_log_scale(enabled)

    def _on_smc_toggled(self, enabled: bool) -> None:
        self._smc_btn.setStyleSheet(self._tf_btn_style(enabled))
        self._chart.toggle_smc(enabled)

    def _on_fvg_toggled(self, enabled: bool) -> None:
        self._fvg_btn.setStyleSheet(self._tf_btn_style(enabled))
        self._chart.toggle_fvg(enabled)

    def _on_phl_toggled(self, enabled: bool) -> None:
        self._phl_btn.setStyleSheet(self._tf_btn_style(enabled))
        self._chart.toggle_phl(enabled)

    def _on_tf_clicked(self, tf: str) -> None:
        """TF butonuna tıklanınca grafiği günceller."""
        self._tf = tf
        self._update_tf_buttons(tf)
        if self._symbol:
            self._load_and_draw(self._symbol, tf, auto_range=True)
        self.symbol_changed.emit(self._symbol, tf)

    def _update_tf_buttons(self, active_tf: str) -> None:
        """Aktif TF butonunu vurgular."""
        for tf, btn in self._tf_buttons.items():
            btn.setChecked(tf == active_tf)
            btn.setStyleSheet(self._tf_btn_style(tf == active_tf))

    # ── Public API ────────────────────────────────────────────────────────────

    @pyqtSlot(str, str)
    def set_tf(self, tf: str) -> None:
        """Dışarıdan TF seçimi yapar."""
        if tf in self._tf_buttons:
            self._on_tf_clicked(tf)

    def current_symbol(self) -> str:
        """Şu an gösterilen sembolü döner."""
        return self._symbol

    def current_tf(self) -> str:
        """Şu an gösterilen TF'i döner."""
        return self._tf

    @pyqtSlot(dict)
    def set_signal_marker(self, signal_data: dict) -> None:
        """Aktif sinyal panelinden çift tıkla gelen sinyal verisini grafik üzerinde gösterir."""
        self._chart.set_signal_marker(signal_data)
