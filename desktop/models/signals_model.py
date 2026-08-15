"""
SignalsModel — aktif sinyaller için QAbstractTableModel.

Sütunlar: Sembol | Tip | TF | VPM | MTF | α | β | Z | P&L% | Süre

── dataChanged KURALI (15 Ağu 2026, proje geneli mimari ilke) ──────────────
`dataChanged.emit(tl, br, ...)` ASLA "kolayca tüm tabloyu kapsasın" diye
koşulsuz `index(0, ilk_sütun)`→`index(rowCount()-1, son_sütun)` ile
çağrılmaz. Sadece GERÇEKTEN değişen satır aralığı (min/max index, tek
satırsa aynı index) VE gerçekten etkilenen sütun(lar) bildirilir — arada
kalan ilgisiz sütunlar için (Qt'nin dataChanged'i dikdörtgen bir aralık
istediğinden) gerekiyorsa AYRI dataChanged çağrıları yapılır.

Neden: `on_prices_updated` (MarketWorker, saniyede bir) ~1758 satırlık
Aktif Sinyaller tablosunda, kaç sembol değişirse değişsin, KOŞULSUZ
tüm-satır × 10-sütunluk bir dataChanged yayınlıyordu — sadece 2 sütun
(current_price/pnl_pct'e bağlı COL_PNL+COL_GUV) gerçekten değişirken
aradaki 8 ilgisiz sütun (RANK/VPM_DELTA/Z_DELTA/TF_ALIGN/VERIM*/AGE) da
dahil ediliyordu. Bu, saniyede binlerce gereksiz "hücre değişti"
bildirimi anlamına geliyordu — masaüstü panelde tekrarlayan, açıklanamamış
bellek/CPU patlamalarının (13-15 Ağustos 2026) güçlü şüphelilerinden
biriydi (Qt'nin native tarafı + varsa macOS erişilebilirlik köprüsü, her
bildirimi işlemek zorunda). Aynı kural: yeni bir worker/panel eklerken,
bir Timer'la periyodik "tüm tabloyu güncelle" YAZMA — sadece gerçekten
değişeni izle ve onu bildir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    pyqtSlot,
)
from PyQt6.QtGui import QColor

from config import Config
from desktop.theme import COLORS

COLUMNS = [
    "Sembol",
    "Tip",
    "TF",
    "İndikatör",
    "VPMV",
    "MTF",
    "α",
    "β",
    "Z",
    "P&L%",
    "Sıra",
    "VPMV Δ",
    "Z Δ",
    "TF Hiz.",
    "Verim",
    "Verim Δ",
    "Verim Δ(önc)",
    "Süre",
    "Güv",
]

COL_SYMBOL = 0
COL_TYPE = 1
COL_TF = 2
COL_INDICATOR = 3
COL_VPM = 4
COL_MTF = 5
COL_ALPHA = 6
COL_BETA = 7
COL_ZSCORE = 8
COL_PNL = 9
COL_RANK = 10
COL_VPM_DELTA = 11
COL_Z_DELTA = 12
COL_TF_ALIGN = 13
COL_VERIM = 14
COL_VERIM_DELTA = 15
COL_VERIM_DELTA_PREV = 16
COL_AGE = 17
COL_GUV = 18

# 28 Tem 2026: yeni sütunların başlıkları kısaltma olduğu için kullanıcı
# geri bildirimiyle (anlaşılır değil) her birine açıklayıcı hover-tooltip
# eklendi — ne olduğu VE yüksek/pozitifin ne anlama geldiği net yazılıyor.
_COLUMN_TOOLTIPS: dict[int, str] = {
    COL_RANK: (
        "Güç Sıralaması — tüm izlenen evrende VPMV skoruna göre canlı\n"
        "percentile (0-100), ~90sn'de bir güncellenir.\n"
        "Yüksek (yeşil): göreceli güçlü.  Düşük: göreceli zayıf."
    ),
    COL_VPM_DELTA: (
        "VPMV Ayrışması — canlı VPMV, sinyal AÇILMADAN ÖNCEki ortalamaya\n"
        "(VPMV sütunu) göre ne kadar değişti.\n"
        "Pozitif (yeşil): momentum güçleniyor.  Negatif (kırmızı): zayıflıyor."
    ),
    COL_Z_DELTA: (
        "Ayrışma Z-score değişimi — canlı Z-score, sinyal açılış anındaki\n"
        "Z'ye (Z sütunu) göre ne kadar değişti.\n"
        "Pozitif: ortalamadan uzaklaşma artıyor.  Negatif: ortalamaya dönüyor."
    ),
    COL_TF_ALIGN: (
        "TF Hizalanma — 4h/6h/8h/12h Heikin-Ashi renginden kaçı sinyalin\n"
        "yönüyle (Long→boğa, Short→ayı) aynı (0-4).\n"
        "4/4 (yeşil): tam hizalı.  Detay için üzerine gelin."
    ),
    COL_VERIM: (
        "Verimlilik (ERSI) — fiyatın birim RSI değişimi başına ne kadar\n"
        "hareket ettiği, kendi son 100 barına göre canlı percentile (0-100).\n"
        "Yüksek (yeşil): verimli/sağlıklı trend.  Düşük: RSI çok yoruluyor."
    ),
    COL_VERIM_DELTA: (
        "Verimlilik değişimi — canlı Verimlilik, sinyal AÇILIŞ ANINDAKİNE\n"
        "göre ne kadar değişti.\n"
        "Pozitif (yeşil): trend sinyalden beri sağlıklılaşıyor."
    ),
    COL_VERIM_DELTA_PREV: (
        "Verimlilik, bir ÖNCEKİ aynı-yön sinyale göre (statik, değişmez).\n"
        "Pozitif (yeşil): bu sinyal öncekinden daha verimli başladı."
    ),
}


def _fmt_score(v: Optional[float]) -> str:
    return f"{v:.1f}" if v is not None else "—"


def _fmt_ratio(v: Optional[float]) -> str:
    return f"{v:.2f}" if v is not None else "—"


def _fmt_pnl(v: Optional[float]) -> str:
    if v is None:
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.2f}%"


def _fmt_delta(v: Optional[float]) -> str:
    if v is None:
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.1f}"


_INTERVAL_MINUTES: dict[str, int] = Config.INTERVAL_MINUTES


def _fmt_age(ts: Optional[datetime], interval: str = "") -> str:
    if ts is None:
        return "—"
    time_str = ts.strftime("%H:%M")
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    now = datetime.now()
    secs = int((now - ts).total_seconds())
    if secs < 0:
        return time_str
    mins = secs // 60
    iv_min = _INTERVAL_MINUTES.get(interval)
    if iv_min and mins > 0:
        candles = mins // iv_min
        return f"{time_str} • {candles}m"
    return time_str


@dataclass
class SignalRow:
    id: int
    symbol: str
    signal_type: str  # "LONG" | "SHORT"
    interval: str
    entry_price: float
    vpm: Optional[float]
    mtf: Optional[float]
    alpha: Optional[float]
    beta: Optional[float]
    zscore: Optional[float]
    timestamp: Optional[datetime]
    indicators: str = ""
    status: str = "active"
    current_price: float = 0.0
    st_confirmed: Optional[bool] = None
    sharpe: Optional[float] = None
    oi_data: Optional[str] = None
    z_score_entry: Optional[float] = None
    is_confluence: bool = False
    sortino: Optional[float] = None
    calmar: Optional[float] = None
    vpmv_pre_avg: Optional[float] = None
    vpmv_slope: Optional[float] = None
    vpmv_ratio: Optional[float] = None
    cvd_slope: Optional[float] = None
    vp_buy_avg: Optional[float] = None
    vp_sell_avg: Optional[float] = None
    vp_score: Optional[float] = None
    pd_zone: Optional[float] = None
    market_structure: str = "-"
    fvg_tfs: str = "-"
    candle_pattern: str = "-"
    atr: Optional[float] = None
    devisso_score: Optional[float] = None  # sinyal açılış anı (statik)
    devisso_delta: Optional[float] = None  # önceki aynı-tip sinyale göre (statik)
    pnl_pct: Optional[float] = field(default=None, init=False)
    # 27 Tem 2026: canlı metrikler — LiveMetricsWorker tarafından periyodik
    # doldurulur, DB'den GELMEZ (init=False, sinyal DB'den yüklenirken boş).
    rank_score_live: Optional[float] = field(default=None, init=False)
    vpmv_live_val: Optional[float] = field(default=None, init=False)
    z_live_val: Optional[float] = field(default=None, init=False)
    verim_live: Optional[float] = field(default=None, init=False)
    ha_4h: Optional[bool] = field(default=None, init=False)
    ha_6h: Optional[bool] = field(default=None, init=False)
    ha_8h: Optional[bool] = field(default=None, init=False)
    ha_12h: Optional[bool] = field(default=None, init=False)

    @property
    def mae_atr_now(self) -> Optional[float]:
        if not self.atr or self.atr <= 0 or not self.entry_price or not self.current_price:
            return None
        if self.signal_type == "LONG":
            return (self.current_price - self.entry_price) / self.atr
        return (self.entry_price - self.current_price) / self.atr

    @property
    def vpmv_delta_live(self) -> Optional[float]:
        """Canlı VPMV − sinyal-öncesi ortalama (vpmv_pre_avg, statik snapshot)."""
        if self.vpmv_live_val is None or self.vpmv_pre_avg is None:
            return None
        return self.vpmv_live_val - self.vpmv_pre_avg

    @property
    def z_delta_live(self) -> Optional[float]:
        """Canlı Z-score − sinyal açılış anındaki Z (z_score_entry, statik)."""
        if self.z_live_val is None or self.z_score_entry is None:
            return None
        return self.z_live_val - self.z_score_entry

    @property
    def verim_delta_live(self) -> Optional[float]:
        """Canlı Verimlilik − sinyal açılış anındaki Verimlilik (devisso_score, statik)."""
        if self.verim_live is None or self.devisso_score is None:
            return None
        return self.verim_live - self.devisso_score

    @property
    def tf_align_count(self) -> Optional[int]:
        """4h/6h/8h/12h'den kaçı sinyalin KENDİ yönüyle hizalı (0-4)."""
        vals = (self.ha_4h, self.ha_6h, self.ha_8h, self.ha_12h)
        if any(v is None for v in vals):
            return None
        want_bull = self.signal_type == "LONG"
        return sum(1 for v in vals if v == want_bull)

    def update_price(self, price: float) -> None:
        self.current_price = price
        if self.entry_price and price:
            if self.signal_type == "LONG":
                self.pnl_pct = (price - self.entry_price) / self.entry_price * 100
            else:
                self.pnl_pct = (self.entry_price - price) / self.entry_price * 100


class SignalsModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[SignalRow] = []
        self._id_index: dict[int, int] = {}  # signal id → satır
        self._sym_rows: dict[str, list[int]] = {}  # symbol → [row indices]
        # (symbol, interval, signal_type) → aktif satır sayısı — 2+ ise çoklu sinyal var
        self._coincident: dict[tuple[str, str, str], int] = {}

    # ── QAbstractTableModel ───────────────────────────────────────────────────

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(COLUMNS)

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return COLUMNS[section]
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignCenter)
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.ToolTipRole:
            return _COLUMN_TOOLTIPS.get(section)
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or index.row() >= len(self._rows):
            return None
        row = self._rows[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            return self._display(row, col)
        if role == Qt.ItemDataRole.ToolTipRole:
            if col == COL_TF_ALIGN:
                return self._tf_align_tooltip(row)
            return self._tooltip(row)
        if role == Qt.ItemDataRole.ForegroundRole:
            return self._foreground(row, col)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col == COL_SYMBOL:
                return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        if role == Qt.ItemDataRole.UserRole:
            return row.symbol
        return None

    def _display(self, row: SignalRow, col: int) -> str:
        match col:
            case _ if col == COL_SYMBOL:
                return f"★ {row.symbol}" if row.is_confluence else row.symbol
            case _ if col == COL_TYPE:
                return row.signal_type
            case _ if col == COL_TF:
                return row.interval
            case _ if col == COL_INDICATOR:
                return row.indicators or "—"
            case _ if col == COL_VPM:
                return _fmt_score(row.vpm)
            case _ if col == COL_MTF:
                return f"{int(row.mtf)}" if row.mtf is not None else "—"
            case _ if col == COL_ALPHA:
                return _fmt_ratio(row.alpha)
            case _ if col == COL_BETA:
                return _fmt_ratio(row.beta)
            case _ if col == COL_ZSCORE:
                return f"{row.zscore:+.2f}" if row.zscore is not None else "—"
            case _ if col == COL_PNL:
                return _fmt_pnl(row.pnl_pct)
            case _ if col == COL_RANK:
                return _fmt_score(row.rank_score_live)
            case _ if col == COL_VPM_DELTA:
                return _fmt_delta(row.vpmv_delta_live)
            case _ if col == COL_Z_DELTA:
                return _fmt_delta(row.z_delta_live)
            case _ if col == COL_TF_ALIGN:
                n = row.tf_align_count
                return f"{n}/4" if n is not None else "—"
            case _ if col == COL_VERIM:
                return _fmt_score(row.verim_live)
            case _ if col == COL_VERIM_DELTA:
                return _fmt_delta(row.verim_delta_live)
            case _ if col == COL_VERIM_DELTA_PREV:
                return _fmt_delta(row.devisso_delta)
            case _ if col == COL_AGE:
                return _fmt_age(row.timestamp, row.interval)
            case _ if col == COL_GUV:
                v = row.mae_atr_now
                if v is None:
                    return "—"
                return f"{v:+.1f}"
        return ""

    @staticmethod
    def _oi_line(row: SignalRow) -> str:
        if not row.oi_data:
            return "  OI: —"
        try:
            oi = json.loads(row.oi_data)
            change = oi.get("change_pct", 0.0)
            sign = "+" if change >= 0 else ""
            if change >= 3:
                yorum = "▲ güçlü trend"
            elif change >= 0:
                yorum = "→ nötr"
            elif change >= -3:
                yorum = "↓ zayıflıyor"
            else:
                yorum = "▼ belirgin çıkış"
            return f"  OI: {sign}{change:.1f}%  {yorum}"
        except Exception:  # pylint: disable=broad-exception-caught
            return "  OI: —"

    def _tooltip(self, row: SignalRow) -> str:
        def _r(v, fmt=".2f"):
            return f"{v:{fmt}}" if v is not None else "—"

        st = (
            ("✓ Onaylı" if row.st_confirmed else "✗ Onaysız")
            if row.st_confirmed is not None
            else "—"
        )
        mtf = f"{int(row.mtf)}" if row.mtf is not None else "—"
        cf_line = f"  ★ KONFLUANS  Z={_r(row.z_score_entry, '+.2f')}\n" if row.is_confluence else ""
        pre = _r(row.vpmv_pre_avg, ".1f")
        slop = _r(row.vpmv_slope, "+.1f")
        rat = _r(row.vpmv_ratio, ".3f")
        return (
            f"{row.symbol}  {row.signal_type}  {row.interval}  |  {row.indicators}\n"
            f"{'─'*52}\n"
            f"{cf_line}"
            f"  Alpha    {_r(row.alpha, '+.4f'):>10}    Beta     {_r(row.beta):>8}\n"
            f"  Sharpe   {_r(row.sharpe):>10}    Sortino  {_r(row.sortino):>8}\n"
            f"  Calmar   {_r(row.calmar):>10}\n"
            f"{'─'*52}\n"
            f"  VPMV: {_r(row.vpm, '.1f')}   MTF: {mtf}   ST: {st}\n"
            f"  pre: {pre}   slope: {slop}   ratio: {rat}\n"
            f"{self._oi_line(row)}"
        )

    @staticmethod
    def _tf_align_tooltip(row: SignalRow) -> str:
        def _renk(v: Optional[bool]) -> str:
            if v is None:
                return "—"
            return "Boğa" if v else "Ayı"

        n = row.tf_align_count
        ozet = f"{n}/4 {row.signal_type} yönüyle hizalı" if n is not None else "veri bekleniyor"
        return (
            f"TF Hizalanma — {ozet}\n"
            f"{'─'*30}\n"
            f"  4h:  {_renk(row.ha_4h)}\n"
            f"  6h:  {_renk(row.ha_6h)}\n"
            f"  8h:  {_renk(row.ha_8h)}\n"
            f"  12h: {_renk(row.ha_12h)}"
        )

    def _foreground(self, row: SignalRow, col: int) -> Optional[QColor]:
        if col == COL_TYPE:
            return QColor(COLORS["green"] if row.signal_type == "LONG" else COLORS["red"])
        if col == COL_INDICATOR:
            key = (row.symbol, row.interval, row.signal_type)
            if self._coincident.get(key, 0) >= 2:
                return QColor(COLORS["yellow"])
        if col == COL_PNL and row.pnl_pct is not None:
            if row.pnl_pct > 0:
                return QColor(COLORS["green"])
            if row.pnl_pct < 0:
                return QColor(COLORS["red"])
        if col == COL_MTF and row.mtf is not None:
            if row.mtf >= 100:
                return QColor(COLORS["green"])
            if row.mtf >= 50:
                return QColor(COLORS["yellow"])
            return QColor(COLORS["red"])
        if col == COL_VPM and row.vpm is not None:
            if row.vpm >= 70:
                return QColor(COLORS["green"])
            if row.vpm >= 50:
                return QColor(COLORS["yellow"])
            return QColor(COLORS["text_muted"])
        if col == COL_ALPHA and row.alpha is not None:
            return QColor(COLORS["green"] if row.alpha > 0 else COLORS["red"])
        if col == COL_ZSCORE and row.zscore is not None:
            if abs(row.zscore) >= 2.0:
                return QColor(COLORS["yellow"])
            return QColor(COLORS["text_muted"])
        if col == COL_SYMBOL and row.is_confluence:
            return QColor("#FFD700")
        if col == COL_RANK and row.rank_score_live is not None:
            if row.rank_score_live >= 70:
                return QColor(COLORS["green"])
            if row.rank_score_live >= 50:
                return QColor(COLORS["yellow"])
            return QColor(COLORS["text_muted"])
        if col in (COL_VPM_DELTA, COL_Z_DELTA, COL_VERIM_DELTA, COL_VERIM_DELTA_PREV):
            v = {
                COL_VPM_DELTA: row.vpmv_delta_live,
                COL_Z_DELTA: row.z_delta_live,
                COL_VERIM_DELTA: row.verim_delta_live,
                COL_VERIM_DELTA_PREV: row.devisso_delta,
            }[col]
            if v is None:
                return None
            if v > 0:
                return QColor(COLORS["green"])
            if v < 0:
                return QColor(COLORS["red"])
            return QColor(COLORS["text_muted"])
        if col == COL_TF_ALIGN:
            n = row.tf_align_count
            if n is None:
                return None
            if n == 4:
                return QColor(COLORS["green"])
            if n == 3:
                return QColor(COLORS["yellow"])
            return QColor(COLORS["text_muted"])
        if col == COL_VERIM and row.verim_live is not None:
            if row.verim_live >= 70:
                return QColor(COLORS["green"])
            if row.verim_live >= 50:
                return QColor(COLORS["yellow"])
            return QColor(COLORS["text_muted"])
        if col == COL_GUV:
            v = row.mae_atr_now
            if v is None:
                return None
            if v >= -0.5:
                return QColor(COLORS["green"])
            if v >= -1.0:
                return QColor(COLORS["yellow"])
            if v >= -1.5:
                return QColor("#FF8C00")
            return QColor(COLORS["red"])
        return None

    # ── Veri yükleme / güncelleme ─────────────────────────────────────────────

    def load_signals(self, signals: list[dict]) -> None:
        self.beginResetModel()
        self._rows.clear()
        self._id_index.clear()
        self._sym_rows.clear()
        self._coincident.clear()
        for s in signals:
            self._append_row(s)
        self.endResetModel()

    def add_or_update(self, signal: dict) -> None:
        sid = signal.get("id")
        if sid in self._id_index:
            idx = self._id_index[sid]
            row = self._rows[idx]
            row.vpm = signal.get("vpms_score", row.vpm)
            row.mtf = signal.get("vpms_mtf_score", row.mtf)
            tl = self.index(idx, 0)
            br = self.index(idx, len(COLUMNS) - 1)
            self.dataChanged.emit(tl, br, [Qt.ItemDataRole.DisplayRole])
        else:
            self.beginInsertRows(QModelIndex(), len(self._rows), len(self._rows))
            self._append_row(signal)
            self.endInsertRows()

    def _append_row(self, s: dict) -> None:
        ts = s.get("opened_at")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = None

        row = SignalRow(
            id=s.get("id", 0),
            symbol=s.get("symbol", ""),
            signal_type=s.get("signal_type", "").upper(),
            interval=s.get("interval", ""),
            entry_price=float(s.get("open_price") or 0),
            vpm=s.get("vpms_score"),
            mtf=s.get("mtf_score"),
            alpha=s.get("alpha"),
            beta=s.get("beta"),
            zscore=s.get("z_score_entry"),
            timestamp=ts,
            indicators=s.get("indicators") or "",
            status=s.get("status", "active"),
            st_confirmed=s.get("st_confirmed"),
            sharpe=s.get("sharpe_ratio"),
            sortino=s.get("sortino_ratio"),
            calmar=s.get("calmar_ratio"),
            oi_data=s.get("oi_data"),
            z_score_entry=s.get("z_score_entry"),
            is_confluence=bool(s.get("is_confluence", False)),
            vpmv_pre_avg=s.get("vpmv_pre_avg"),
            vpmv_slope=s.get("vpmv_slope"),
            vpmv_ratio=s.get("vpmv_ratio"),
            cvd_slope=s.get("cvd_slope"),
            vp_buy_avg=s.get("vp_buy_avg"),
            vp_sell_avg=s.get("vp_sell_avg"),
            vp_score=s.get("vp_score"),
            pd_zone=s.get("pd_zone"),
            market_structure=s.get("market_structure") or "-",
            fvg_tfs=s.get("fvg_tfs") or "-",
            candle_pattern=s.get("candle_pattern") or "-",
            atr=s.get("atr"),
            devisso_score=s.get("devisso_score"),
            devisso_delta=s.get("devisso_delta"),
        )
        idx = len(self._rows)
        self._rows.append(row)
        self._id_index[row.id] = idx
        self._sym_rows.setdefault(row.symbol, []).append(idx)
        key = (row.symbol, row.interval, row.signal_type)
        self._coincident[key] = self._coincident.get(key, 0) + 1

    def remove_signals(self, signal_ids: list[int]) -> None:
        indices = sorted(
            [self._id_index[sid] for sid in signal_ids if sid in self._id_index],
            reverse=True,
        )
        for idx in indices:
            self.beginRemoveRows(QModelIndex(), idx, idx)
            self._rows.pop(idx)
            self.endRemoveRows()
        if indices:
            self._id_index.clear()
            self._sym_rows.clear()
            self._coincident.clear()
            for i, row in enumerate(self._rows):
                self._id_index[row.id] = i
                self._sym_rows.setdefault(row.symbol, []).append(i)
                key = (row.symbol, row.interval, row.signal_type)
                self._coincident[key] = self._coincident.get(key, 0) + 1

    @pyqtSlot(str, float, float)
    def on_prices_updated(self, prices: dict) -> None:
        """15 Ağu 2026 mimari düzeltmesi (bkz. dosya başındaki "dataChanged
        kuralı"): SADECE gerçekten değişen satır aralığı + SADECE
        update_price()'ın etkilediği iki sütun (COL_PNL, COL_GUV — arada
        kalan RANK/VPM_DELTA/Z_DELTA/TF_ALIGN/VERIM*/AGE hiç etkilenmiyor)
        için dataChanged yayınlanır. Eskiden HER saniye (MarketWorker
        _FALLBACK_MS=1000), kaç sembol değişirse değişsin, ~1758 satır ×
        10 sütunluk KOŞULSUZ tam-tablo bildirimi atılıyordu — Qt'nin native
        tarafında (ve varsa erişilebilirlik köprüsünde) sürekli, gereksiz
        bir yeniden-tarama/işleme yüküne yol açıyordu."""
        min_idx: Optional[int] = None
        max_idx: Optional[int] = None
        for sym, indices in self._sym_rows.items():
            p = prices.get(sym)
            if p is None:
                continue
            for idx in indices:
                self._rows[idx].update_price(float(p))
                if min_idx is None or idx < min_idx:
                    min_idx = idx
                if max_idx is None or idx > max_idx:
                    max_idx = idx
        if min_idx is None:
            return
        roles = [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ForegroundRole]
        self.dataChanged.emit(self.index(min_idx, COL_PNL), self.index(max_idx, COL_PNL), roles)
        self.dataChanged.emit(self.index(min_idx, COL_GUV), self.index(max_idx, COL_GUV), roles)

    def on_price_updated(self, symbol: str, price: float, _change_pct: float) -> None:
        indices = self._sym_rows.get(symbol, [])
        roles = [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ForegroundRole]
        for idx in indices:
            self._rows[idx].update_price(price)
            # Aradaki ilgisiz sütunları içermesin diye COL_PNL/COL_GUV AYRI
            # yayınlanıyor (bkz. on_prices_updated'in dosya başı kuralına atıfı).
            self.dataChanged.emit(self.index(idx, COL_PNL), self.index(idx, COL_PNL), roles)
            self.dataChanged.emit(self.index(idx, COL_GUV), self.index(idx, COL_GUV), roles)

    def signal_at(self, row: int) -> Optional[SignalRow]:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def apply_live_metrics(self, payload: dict) -> None:
        """LiveMetricsWorker.metrics_updated'ten gelen toplu güncelleme —
        Güç Sıralaması/TF Hizalanma sembol bazlı, VPMV/Verimlilik sinyal-id
        bazlı, Ayrışma Z-score (symbol, interval) bazlı eşleniyor.

        15 Ağu 2026: dataChanged artık SADECE gerçekten güncellenen satır
        aralığı için yayınlanıyor (bkz. dosya başındaki "dataChanged
        kuralı") — her sembol/sinyal için payload'da veri olmayabilir,
        eskiden yine de TÜM satırlar için bildirim atılıyordu."""
        if not self._rows:
            return
        ranking = payload.get("ranking") or {}
        ha = payload.get("ha") or {}
        devisso = payload.get("devisso") or {}
        vpmv = payload.get("vpmv") or {}
        divergence = payload.get("divergence") or {}

        min_idx: Optional[int] = None
        max_idx: Optional[int] = None
        for idx, row in enumerate(self._rows):
            row_changed = False
            if row.symbol in ranking:
                row.rank_score_live = ranking[row.symbol]
                row_changed = True
            ha_row = ha.get(row.symbol)
            if ha_row:
                row.ha_4h = ha_row.get("ha_4h")
                row.ha_6h = ha_row.get("ha_6h")
                row.ha_8h = ha_row.get("ha_8h")
                row.ha_12h = ha_row.get("ha_12h")
                row_changed = True
            if row.id in devisso:
                row.verim_live = devisso[row.id]
                row_changed = True
            if row.id in vpmv:
                row.vpmv_live_val = vpmv[row.id]
                row_changed = True
            z = divergence.get((row.symbol, row.interval))
            if z is not None:
                row.z_live_val = z
                row_changed = True
            if row_changed:
                if min_idx is None or idx < min_idx:
                    min_idx = idx
                if max_idx is None or idx > max_idx:
                    max_idx = idx

        if min_idx is None:
            return
        tl = self.index(min_idx, COL_RANK)
        br = self.index(max_idx, COL_VERIM_DELTA_PREV)
        self.dataChanged.emit(tl, br, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ForegroundRole])


_RANGE_FIELDS = ("vpm", "mtf", "alpha", "beta", "zscore")


class SignalsProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._type_filter = ""
        self._tf_filter = ""
        self._indicator_filter = ""
        self._st_only = False
        self._cf_only = False
        # VPMV/MTF/α/β/Z için min-max eşik filtreleri — None = o yönde sınır yok.
        self._ranges: dict[str, list[Optional[float]]] = {f: [None, None] for f in _RANGE_FIELDS}
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setFilterKeyColumn(COL_SYMBOL)

    def set_range_filter(
        self, field_name: str, min_val: Optional[float], max_val: Optional[float]
    ) -> None:
        """field_name: 'vpm'|'mtf'|'alpha'|'beta'|'zscore'. None sınır konulmadığı anlamına gelir."""
        self._ranges[field_name] = [min_val, max_val]
        self.invalidateFilter()

    def set_type_filter(self, type_filter: str) -> None:
        self._type_filter = type_filter
        self.invalidateFilter()

    def set_tf_filter(self, tf: str) -> None:
        self._tf_filter = tf
        self.invalidateFilter()

    def set_indicator_filter(self, indicator: str) -> None:
        self._indicator_filter = indicator
        self.invalidateFilter()

    def set_st_filter(self, only_confirmed: bool) -> None:
        self._st_only = only_confirmed
        self.invalidateFilter()

    def set_confluence_filter(self, only_confluence: bool) -> None:
        self._cf_only = only_confluence
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        src = self.sourceModel()
        if source_row >= len(src._rows):  # noqa: SLF001
            return False
        row = src._rows[source_row]  # noqa: SLF001
        if self._type_filter and row.signal_type != self._type_filter:
            return False
        if self._tf_filter and row.interval != self._tf_filter:
            return False
        if self._indicator_filter and not row.indicators.startswith(self._indicator_filter):
            return False
        if self._st_only and row.st_confirmed is False:
            return False
        if self._cf_only and not row.is_confluence:
            return False
        for field_name, (lo, hi) in self._ranges.items():
            if lo is None and hi is None:
                continue
            val = getattr(row, field_name, None)
            if val is None:
                return False  # eşik konulmuşsa ve değer bilinmiyorsa gösterme
            if lo is not None and val < lo:
                return False
            if hi is not None and val > hi:
                return False
        return super().filterAcceptsRow(source_row, source_parent)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        src = self.sourceModel()
        col = left.column()
        l_row = src._rows[left.row()]  # noqa: SLF001
        r_row = src._rows[right.row()]  # noqa: SLF001

        def _cmp(a, b):
            if a is None and b is None:
                return False
            if a is None:
                return True
            if b is None:
                return False
            return a < b

        match col:
            case _ if col == COL_VPM:
                return _cmp(l_row.vpm, r_row.vpm)
            case _ if col == COL_MTF:
                return _cmp(l_row.mtf, r_row.mtf)
            case _ if col == COL_ALPHA:
                return _cmp(l_row.alpha, r_row.alpha)
            case _ if col == COL_BETA:
                return _cmp(l_row.beta, r_row.beta)
            case _ if col == COL_ZSCORE:
                return _cmp(l_row.zscore, r_row.zscore)
            case _ if col == COL_PNL:
                return _cmp(l_row.pnl_pct, r_row.pnl_pct)
            case _ if col == COL_RANK:
                return _cmp(l_row.rank_score_live, r_row.rank_score_live)
            case _ if col == COL_VPM_DELTA:
                return _cmp(l_row.vpmv_delta_live, r_row.vpmv_delta_live)
            case _ if col == COL_Z_DELTA:
                return _cmp(l_row.z_delta_live, r_row.z_delta_live)
            case _ if col == COL_TF_ALIGN:
                return _cmp(l_row.tf_align_count, r_row.tf_align_count)
            case _ if col == COL_VERIM:
                return _cmp(l_row.verim_live, r_row.verim_live)
            case _ if col == COL_VERIM_DELTA:
                return _cmp(l_row.verim_delta_live, r_row.verim_delta_live)
            case _ if col == COL_VERIM_DELTA_PREV:
                return _cmp(l_row.devisso_delta, r_row.devisso_delta)
            case _ if col == COL_AGE:

                def _ts(t):
                    if t is None:
                        return datetime.max
                    if t.tzinfo is not None:
                        return t.replace(tzinfo=None)
                    return t

                return _ts(l_row.timestamp) > _ts(r_row.timestamp)
            case _:
                return super().lessThan(left, right)
