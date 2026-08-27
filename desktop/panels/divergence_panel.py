"""
DivergencePanel — sinyal coinlerin fiyat Z-score ayrışma tablosu + çizgi grafik.
Z = (close - EMA200) / StdDev200

Üst: tüm sembollerin z-score zaman serisi çizgileri (Pine benzeri)
Alt sol: pozitif ayrışanlar (Z > 0), büyükten küçüğe
Alt sağ: negatif ayrışanlar (Z < 0), küçükten büyüğe
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSlot  # pylint: disable=no-name-in-module
from PyQt6.QtGui import QColor, QFont  # pylint: disable=no-name-in-module
from PyQt6.QtWidgets import (  # pylint: disable=no-name-in-module
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from desktop.theme import COLORS

pg.setConfigOption("background", "#0d0d12")
pg.setConfigOption("foreground", "#555566")

# TF başına dakika — bayatlık eşiği hesaplamak için (bkz. _fill_table).
# divergence_live:{symbol}:{tf} backend'de her bar kapanışında yazılıyor,
# bir sonraki bar kapanışına kadar taze sayılır; eşik TF süresinin 3 katı +
# ağ/işlem gecikmesi payı.
_TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
_STALE_TOLERANCE_BARS = 3
_STALE_BUFFER_SEC = 90

_COLS = ["Sembol", "Z-score", "VPMV", "Rank", "Zaman"]
_COL_SYMBOL = 0
_COL_ZSCORE = 1
_COL_VPMV = 2
_COL_RANK = 3
_COL_TIME = 4

_C_VPMV_HIGH = QColor(80, 200, 80)
_C_VPMV_MID = QColor(200, 180, 60)
_C_VPMV_LOW = QColor(200, 80, 80)

_C_GREEN = QColor(COLORS["green"])
_C_RED = QColor(COLORS["red"])
_C_MUTED = QColor(COLORS["text_muted"])
_C_TRANSPARENT = QColor(0, 0, 0, 0)

_BG_GREEN_STRONG = QColor(0, 120, 40, 150)
_BG_GREEN_SOFT = QColor(0, 80, 20, 80)
_BG_RED_STRONG = QColor(180, 20, 20, 150)
_BG_RED_SOFT = QColor(120, 10, 10, 80)
_BG_STALE = QColor(120, 70, 0, 140)

_PALETTE = [
    (100, 220, 100),
    (220, 100, 100),
    (100, 160, 240),
    (240, 180, 80),
    (180, 100, 240),
    (80, 220, 220),
    (240, 240, 80),
    (240, 140, 180),
    (140, 240, 160),
    (200, 160, 240),
    (240, 140, 80),
    (80, 180, 200),
    (160, 240, 100),
    (240, 100, 140),
    (100, 200, 240),
    (200, 240, 120),
    (240, 200, 80),
    (120, 120, 240),
]

# Grafikte aynı anda çok sayıda aktif sinyal varsa (>18) _PALETTE renkleri
# tekrar etmeye başlayıp farklı sembolleri aynı renkte gösteriyordu, tablo
# ile birlikte okununca "hangi çizgi hangisi" ayırt edilemez hale geliyordu
# (31 Tem 2026). Artık sadece en yüksek skorlu (vpmv×|z|) SPOTLIGHT_N pozitif
# + SPOTLIGHT_N negatif sembol gerçek renk+etiketle çiziliyor, geri kalanı
# bu soluk gri tonla — race grafiğindeki aynı çözüm (bkz. do_open_streak_race_generator).
_SPOTLIGHT_N = 10
_C_LINE_MUTED = (90, 92, 105)


class _NumericItem(QTableWidgetItem):
    def __lt__(self, other: "QTableWidgetItem") -> bool:
        try:
            return float(self.data(Qt.ItemDataRole.UserRole)) < float(
                other.data(Qt.ItemDataRole.UserRole)
            )
        except (TypeError, ValueError):
            return super().__lt__(other)


def _make_table() -> QTableWidget:
    t = QTableWidget(0, len(_COLS))
    t.setHorizontalHeaderLabels(_COLS)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    t.setAlternatingRowColors(False)
    t.setSortingEnabled(False)
    t.setShowGrid(False)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(24)
    hh = t.horizontalHeader()
    # ResizeToContents sürekli modda HER setItem() çağrısında tüm sütunu yeniden
    # ölçüyor (O(satır) maliyet × N setItem = O(satır²)) — 550 sembolle bu, ana
    # thread'i kilitleyip panel kasmasına yol açıyordu. Interactive + tabloyu
    # dolduran fonksiyonun sonunda tek seferlik resizeColumnsToContents() aynı
    # görünümü verir, sürekli yeniden ölçüm olmadan.
    hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    hh.setSectionResizeMode(_COL_SYMBOL, QHeaderView.ResizeMode.Interactive)
    hh.setSectionResizeMode(_COL_VPMV, QHeaderView.ResizeMode.Interactive)
    hh.setSectionResizeMode(_COL_RANK, QHeaderView.ResizeMode.Interactive)
    hh.setSectionResizeMode(_COL_TIME, QHeaderView.ResizeMode.Interactive)
    return t


def _make_search_box(placeholder: str) -> QLineEdit:
    box = QLineEdit()
    box.setPlaceholderText(placeholder)
    box.setFixedHeight(24)
    box.setStyleSheet(
        f"background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']}; "
        f"border: 1px solid {COLORS['border']}; border-radius: 3px; "
        f"padding: 0 4px; font-size: 11px;"
    )
    return box


class DivergencePanel(QWidget):
    """Z-score çizgi grafik + pozitif/negatif ayrışma tabloları."""

    _INDICATOR_FILTERS = [
        ("Tümü", ""),
        ("RSI Cross", "RSI_Cross"),
        ("Supertrend", "Supertrend"),
        ("HA Cross", "HA_Cross"),
        ("MA200 Cross", "MA200_Cross"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_result: Optional[dict] = None
        self._prev_pos_ranks: dict[str, int] = {}
        self._prev_neg_ranks: dict[str, int] = {}
        self._pos_symbol_to_row: dict[str, int] = {}
        self._neg_symbol_to_row: dict[str, int] = {}
        self._pos_resized_once = False
        self._neg_resized_once = False
        self._ranking: dict[str, int] = {}
        self._pos_search = ""
        self._neg_search = ""
        self._indicator_filter = ""
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._labels: dict[str, pg.TextItem] = {}
        self._sym_colors: dict[str, tuple] = {}
        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Kontrol çubuğu
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        ctrl.addWidget(self._muted_label("TF:"))
        self._tf_combo = QComboBox()
        self._tf_combo.addItems(["1m", "5m", "15m", "1h", "4h", "1d"])
        self._tf_combo.setCurrentText("1h")
        self._tf_combo.setFixedWidth(70)
        ctrl.addWidget(self._tf_combo)
        ctrl.addWidget(self._muted_label("İndikatör:"))
        self._ind_combo = QComboBox()
        for label, _ in self._INDICATOR_FILTERS:
            self._ind_combo.addItem(label)
        self._ind_combo.setFixedWidth(110)
        self._ind_combo.currentIndexChanged.connect(self._on_indicator_changed)
        ctrl.addWidget(self._ind_combo)
        ctrl.addStretch()
        self._status_label = QLabel("Sinyal bekleniyor…")
        self._status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        ctrl.addWidget(self._status_label)
        root.addLayout(ctrl)

        # Tab widget
        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        # ── Tab 0: Tablo ──────────────────────────────────────────────────
        tables_widget = QWidget()
        tables_layout = QHBoxLayout(tables_widget)
        tables_layout.setContentsMargins(0, 4, 0, 0)
        tables_layout.setSpacing(8)

        pos_col = QVBoxLayout()
        pos_col.setSpacing(4)
        pos_title = QLabel("▲ POZİTİF AYRIŞMA")
        pos_title.setStyleSheet(
            f"color: {COLORS['green']}; font-size: 11px; font-weight: bold; padding: 0 4px;"
        )
        self._pos_search_box = _make_search_box("Ara…")
        self._pos_search_box.textChanged.connect(self._on_pos_search)
        pos_hdr = QHBoxLayout()
        pos_hdr.addWidget(pos_title)
        pos_hdr.addStretch()
        pos_hdr.addWidget(self._pos_search_box)
        self._pos_table = _make_table()
        pos_col.addLayout(pos_hdr)
        pos_col.addWidget(self._pos_table)

        neg_col = QVBoxLayout()
        neg_col.setSpacing(4)
        neg_title = QLabel("▼ NEGATİF AYRIŞMA")
        neg_title.setStyleSheet(
            f"color: {COLORS['red']}; font-size: 11px; font-weight: bold; padding: 0 4px;"
        )
        self._neg_search_box = _make_search_box("Ara…")
        self._neg_search_box.textChanged.connect(self._on_neg_search)
        neg_hdr = QHBoxLayout()
        neg_hdr.addWidget(neg_title)
        neg_hdr.addStretch()
        neg_hdr.addWidget(self._neg_search_box)
        self._neg_table = _make_table()
        neg_col.addLayout(neg_hdr)
        neg_col.addWidget(self._neg_table)

        tables_layout.addLayout(pos_col)
        tables_layout.addLayout(neg_col)

        tabs.addTab(tables_widget, "Tablo")

        # ── Tab 1: Grafik ─────────────────────────────────────────────────
        self._chart = pg.PlotWidget()
        self._chart.showGrid(x=False, y=True, alpha=0.15)
        self._chart.getAxis("bottom").hide()
        self._chart.getAxis("left").setStyle(tickFont=QFont("Courier New", 9))
        self._chart.setLabel("left", "Z-score", color="#555566", size="9pt")
        self._chart.addLine(y=0, pen=pg.mkPen("#888888", width=1))
        for y_val in [1, -1, 2, -2]:
            self._chart.addLine(
                y=y_val,
                pen=pg.mkPen("#444455", width=1, style=Qt.PenStyle.DashLine),
            )

        tabs.addTab(self._chart, "Grafik")

        root.addWidget(tabs)

    def _muted_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        return lbl

    def tf_combo(self) -> QComboBox:
        return self._tf_combo

    # ── Slot'lar ──────────────────────────────────────────────────────────

    @pyqtSlot(object)
    def on_ranking_updated(self, result: list) -> None:
        self._ranking = {r["symbol"]: r["rank"] for r in result}
        if self._last_result:
            self._populate(self._last_result)

    def _on_indicator_changed(self, index: int) -> None:
        self._indicator_filter = self._INDICATOR_FILTERS[index][1]
        if self._last_result:
            self._populate(self._last_result)

    def _on_pos_search(self, text: str) -> None:
        self._pos_search = text.strip().upper()
        self._apply_filter(self._pos_table, self._pos_search)

    def _on_neg_search(self, text: str) -> None:
        self._neg_search = text.strip().upper()
        self._apply_filter(self._neg_table, self._neg_search)

    @staticmethod
    def _apply_filter(table: QTableWidget, search: str) -> None:
        for row in range(table.rowCount()):
            item = table.item(row, _COL_SYMBOL)
            if item is None:
                continue
            symbol = item.text().split()[0]
            table.setRowHidden(row, bool(search) and search not in symbol)

    @pyqtSlot(object)
    def on_divergence_updated(self, result: dict) -> None:
        self._last_result = result
        n = len(result.get("current", {}))
        self._status_label.setText(
            f"Son güncelleme: {datetime.now().strftime('%H:%M:%S')}  •  {n} sembol"
        )
        self._populate(result)

    @pyqtSlot(str)
    def on_status_updated(self, msg: str) -> None:
        self._status_label.setText(msg)

    # ── Grafik güncelleme ─────────────────────────────────────────────────

    def _update_chart(self, series: dict, spotlight: set) -> None:
        active = set(series.keys())
        stale = set(self._curves.keys()) - active

        # Eski sembolleri kaldır
        for sym in stale:
            self._chart.removeItem(self._curves.pop(sym))
            if sym in self._labels:
                self._chart.removeItem(self._labels.pop(sym))
            self._sym_colors.pop(sym, None)

        # Renk ata (yeni sembollere) — sadece spotlight'a girmişse anlamlı,
        # ama sembol daha sonra tekrar spotlight'a girerse aynı rengi
        # koruması için tüm aktiflere kimlik rengi atanıyor, kullanımı
        # aşağıda spotlight şartına bağlı.
        color_idx = 0
        for sym in sorted(active):
            if sym not in self._sym_colors:
                self._sym_colors[sym] = _PALETTE[color_idx % len(_PALETTE)]
            color_idx += 1

        # Çizgileri güncelle / oluştur
        for sym, z_arr in series.items():
            # Boş (0 noktalı) seri pyqtgraph'ta dataBounds() içinde
            # "'<=' not supported between NoneType" hatasına yol açıyor
            # (ViewBox auto-range boş eğrinin sınırını None döndürüyor,
            # 26 Tem 2026 — sürekli tekrarlayan bu hata masaüstü
            # donma/bellek sızıntısına neden oluyordu).
            if len(z_arr) == 0:
                if sym in self._curves:
                    self._chart.removeItem(self._curves.pop(sym))
                if sym in self._labels:
                    self._chart.removeItem(self._labels.pop(sym))
                continue

            is_spot = sym in spotlight
            color = self._sym_colors[sym] if is_spot else _C_LINE_MUTED
            width = 1.5 if is_spot else 1.0
            x_data = np.arange(len(z_arr), dtype=np.float32)
            y_data = z_arr.astype(np.float32)

            if sym in self._curves:
                self._curves[sym].setData(x_data, y_data)
                self._curves[sym].setPen(pg.mkPen(color=color, width=width))
            else:
                pen = pg.mkPen(color=color, width=width)
                curve = self._chart.plot(x_data, y_data, pen=pen)
                self._curves[sym] = curve

            # Uç etiket — sadece spotlight'takiler için (18 renk kapasitesini
            # aşan sayıda etiket zaten okunaksız oluyordu)
            if not is_spot:
                if sym in self._labels:
                    self._chart.removeItem(self._labels.pop(sym))
                continue
            label_text = sym.replace("USDT", "")
            if sym in self._labels:
                self._labels[sym].setPos(float(x_data[-1]), float(y_data[-1]))
                self._labels[sym].setText(label_text)
                self._labels[sym].setColor(color)
            else:
                txt = pg.TextItem(
                    text=label_text,
                    color=color,
                    anchor=(0, 0.5),
                )
                txt.setFont(QFont("Courier New", 8))
                txt.setPos(float(x_data[-1]), float(y_data[-1]))
                self._chart.addItem(txt)
                self._labels[sym] = txt

    # ── Tablo doldurma ────────────────────────────────────────────────────

    def _populate(self, result: dict) -> None:
        current = result.get("current", {})
        diverge_since = result.get("diverge_since", {})
        indicators_map = result.get("indicators", {})
        series = result.get("series", {})
        vpmv_map = result.get("vpmv", {})
        timestamps = result.get("timestamps", {})

        ind_filter = self._indicator_filter
        if ind_filter:
            current = {
                sym: z
                for sym, z in current.items()
                if ind_filter in (indicators_map.get(sym) or "")
            }
            series = {
                sym: s for sym, s in series.items() if ind_filter in (indicators_map.get(sym) or "")
            }

        def _score(sym: str, z: float) -> float:
            return (vpmv_map.get(sym) or 0.0) * abs(z)

        pos_rows = sorted(
            [(sym, z) for sym, z in current.items() if z > 0],
            key=lambda x: _score(x[0], x[1]),
            reverse=True,
        )
        neg_rows = sorted(
            [(sym, z) for sym, z in current.items() if z <= 0],
            key=lambda x: _score(x[0], x[1]),
            reverse=True,
        )

        spotlight = {sym for sym, _ in pos_rows[:_SPOTLIGHT_N]} | {
            sym for sym, _ in neg_rows[:_SPOTLIGHT_N]
        }
        self._update_chart(series, spotlight)

        pos_deltas = {
            sym: self._prev_pos_ranks[sym] - i
            for i, (sym, _) in enumerate(pos_rows)
            if sym in self._prev_pos_ranks
        }
        neg_deltas = {
            sym: self._prev_neg_ranks[sym] - i
            for i, (sym, _) in enumerate(neg_rows)
            if sym in self._prev_neg_ranks
        }

        self._prev_pos_ranks = {sym: i for i, (sym, _) in enumerate(pos_rows)}
        self._prev_neg_ranks = {sym: i for i, (sym, _) in enumerate(neg_rows)}

        self._fill_table(
            self._pos_table,
            pos_rows,
            diverge_since,
            vpmv_map,
            positive=True,
            rank_deltas=pos_deltas,
            symbol_to_row=self._pos_symbol_to_row,
            timestamps=timestamps,
        )
        self._fill_table(
            self._neg_table,
            neg_rows,
            diverge_since,
            vpmv_map,
            positive=False,
            rank_deltas=neg_deltas,
            symbol_to_row=self._neg_symbol_to_row,
            timestamps=timestamps,
        )
        self._apply_filter(self._pos_table, self._pos_search)
        self._apply_filter(self._neg_table, self._neg_search)

    @staticmethod
    def _get_item(table: QTableWidget, row: int, col: int, item_cls) -> QTableWidgetItem:
        item = table.item(row, col)
        if item is None:
            item = item_cls("")
            table.setItem(row, col, item)
        return item

    @staticmethod
    def _rebuild_symbol_to_row(table: QTableWidget, symbol_to_row: dict[str, int]) -> None:
        symbol_to_row.clear()
        for r in range(table.rowCount()):
            it = table.item(r, _COL_SYMBOL)
            if it is not None:
                symbol_to_row[it.text().split()[0]] = r

    def _fill_table(  # pylint: disable=too-many-locals
        self,
        table: QTableWidget,
        rows: list,
        diverge_since: dict,
        vpmv_map: dict,
        positive: bool,
        rank_deltas: Optional[dict] = None,
        symbol_to_row: Optional[dict[str, int]] = None,
        timestamps: Optional[dict] = None,
    ) -> None:
        # 27 Ağu 2026: setRowCount(len(rows)) + tam yeniden inşa her satır için
        # yeni QTableWidgetItem yaratıp eskisini yok ediyordu — ranking_panel.py
        # ile AYNI kök nedenle panel 87.9GB'a çıkıp kernel tarafından durduruldu
        # (bkz. ranking_panel.py::_render). Artık var olan satır/hücreler
        # YERİNDE güncelleniyor, sadece evrenden çıkan/giren semboller için
        # satır silinip/ekleniyor. Satır SIRASI (skora göre) artık POZİSYONEL
        # olarak garanti edilmiyor — sembol hep aynı satırda kalır, kullanıcı
        # bir sütuna tıklayarak (setSortingEnabled) istediği kritere göre
        # sıralayabilir (ranking_panel'deki davranışla tutarlı).
        table.setSortingEnabled(False)
        if symbol_to_row is None:
            symbol_to_row = {}

        # Kullanıcı iki _fill_table() çağrısı arasında bir sütun başlığına
        # tıklayıp tabloyu yeniden sıralayabilir — harita sadece fonksiyon
        # SONUNDA kurulduğu için bu durumda bayatlar (paper_trade_panel.py'deki
        # "P&L önce 31 sonra -2" bug'ıyla aynı kök neden sınıfı, 27 Ağu 2026).
        # Her çağrı başında haritayı tablonun GERÇEK anlık durumundan yeniden
        # kurup bu riski tamamen ortadan kaldırıyoruz.
        self._rebuild_symbol_to_row(table, symbol_to_row)

        mono = QFont("Courier New", 11)
        bold = QFont("Courier New", 11, QFont.Weight.Bold)
        now = datetime.now()
        z_color = _C_GREEN if positive else _C_RED
        if rank_deltas is None:
            rank_deltas = {}
        if timestamps is None:
            timestamps = {}
        tf_minutes = _TF_MINUTES.get(self._tf_combo.currentText(), 60)
        stale_threshold_sec = tf_minutes * 60 * _STALE_TOLERANCE_BARS + _STALE_BUFFER_SEC
        now_ts = now.timestamp()

        incoming = {symbol for symbol, _ in rows}
        removed = set(symbol_to_row) - incoming
        if removed:
            rows_to_remove = sorted(
                (symbol_to_row[s] for s in removed if s in symbol_to_row), reverse=True
            )
            for r in rows_to_remove:
                table.removeRow(r)
            # removeRow altındaki satırların index'ini kaydırır — haritayı hemen
            # yeniden kurmazsak kalan semboller YANLIŞ (kaymış) satırı işaret
            # eder, bir sonraki sembolün verisi o satıra yazılabilir (27 Ağu
            # 2026, deviso_panel.py'de bulundu, aynı desen burada da geçerli).
            self._rebuild_symbol_to_row(table, symbol_to_row)

        for symbol, z in rows:
            row_idx = symbol_to_row.get(symbol)
            if (
                row_idx is None
                or row_idx >= table.rowCount()
                or table.item(row_idx, _COL_SYMBOL) is None
            ):
                row_idx = table.rowCount()
                table.insertRow(row_idx)

            delta = rank_deltas.get(symbol, 0)
            if delta > 0:
                sym_text = f"{symbol} ↑{delta}"
                sym_color = _C_GREEN
            elif delta < 0:
                sym_text = f"{symbol} ↓{abs(delta)}"
                sym_color = _C_RED
            else:
                sym_text = symbol
                sym_color = z_color
            sym_item = self._get_item(table, row_idx, _COL_SYMBOL, QTableWidgetItem)
            sym_item.setText(sym_text)
            sym_item.setFont(bold)
            sym_item.setForeground(sym_color)

            z_item = self._get_item(table, row_idx, _COL_ZSCORE, _NumericItem)
            z_item.setText(f"{z:+.2f}")
            z_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            z_item.setFont(mono)
            z_item.setData(Qt.ItemDataRole.UserRole, z)
            z_item.setForeground(z_color)
            abs_z = abs(z)
            if abs_z >= 2.0:
                z_item.setBackground(_BG_GREEN_STRONG if positive else _BG_RED_STRONG)
            elif abs_z >= 1.0:
                z_item.setBackground(_BG_GREEN_SOFT if positive else _BG_RED_SOFT)
            else:
                z_item.setBackground(_C_TRANSPARENT)

            vpmv = vpmv_map.get(symbol) or 0.0
            vpmv_item = self._get_item(table, row_idx, _COL_VPMV, _NumericItem)
            vpmv_item.setText(f"{vpmv:.0f}")
            vpmv_item.setData(Qt.ItemDataRole.UserRole, vpmv)
            vpmv_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            vpmv_item.setFont(mono)
            if vpmv >= 60:
                vpmv_item.setForeground(_C_VPMV_HIGH)
            elif vpmv >= 45:
                vpmv_item.setForeground(_C_VPMV_MID)
            else:
                vpmv_item.setForeground(_C_VPMV_LOW)

            rank = self._ranking.get(symbol)
            rank_item = self._get_item(table, row_idx, _COL_RANK, _NumericItem)
            rank_item.setText(str(rank) if rank is not None else "—")
            rank_item.setData(Qt.ItemDataRole.UserRole, rank if rank is not None else 9999)
            rank_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            rank_item.setFont(mono)
            rank_item.setForeground(_C_MUTED)

            ts = diverge_since.get(symbol)
            if ts:
                dt = datetime.fromtimestamp(ts)
                time_str = (
                    dt.strftime("%H:%M") if dt.date() == now.date() else dt.strftime("%m/%d %H:%M")
                )
            else:
                time_str = "—"
            t_item = self._get_item(table, row_idx, _COL_TIME, QTableWidgetItem)
            t_item.setText(time_str)
            t_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            t_item.setFont(mono)
            t_item.setForeground(_C_MUTED)

            # Bayatlık kontrolü: divergence_live'ın son barı (ts_recent) bu TF'nin
            # normal bar aralığının birkaç katından eskiyse, backend'in bu sembol
            # için veri üretmeyi durdurmuş olma ihtimali var (27 Ağu 2026, MOVR
            # olayıyla aynı sınıf risk) — sessizce eski Z-score göstermek yerine
            # satır turuncu işaretlenir.
            ts_arr = timestamps.get(symbol)
            last_ts = float(ts_arr[-1]) if ts_arr is not None and len(ts_arr) > 0 else None
            is_stale = last_ts is None or (now_ts - last_ts) > stale_threshold_sec
            if is_stale:
                sym_item.setText(f"{sym_text} ⚠")
                for stale_item in (sym_item, z_item, vpmv_item, rank_item, t_item):
                    stale_item.setBackground(_BG_STALE)

        self._rebuild_symbol_to_row(table, symbol_to_row)

        table.setSortingEnabled(True)
        # resizeColumnsToContents() satır başına font-shaping (CoreText) çağırıyor
        # — periyodik olarak CPU'yu tıkıyordu (27 Ağu 2026, sample ile ölçüldü).
        # İlk dolduruluşta bir kez yapılması yeterli.
        already_resized = self._pos_resized_once if positive else self._neg_resized_once
        if not already_resized:
            table.resizeColumnsToContents()
            if positive:
                self._pos_resized_once = True
            else:
                self._neg_resized_once = True
