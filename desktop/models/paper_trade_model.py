"""PaperOpenModel / PaperHistModel — sanal portföy açık pozisyon + kapalı
işlem tabloları için QAbstractTableModel (28 Ağu 2026, Model/View göçü
Faz 8 — SON faz, en büyük/riskli panel).

PaperOpenModel'in özel yanı: iki farklı kadanslı güncelleme kaynağı var —
5sn'de bir tam DB fetch (bulk_upsert, her şeyi yeniden hesaplar) VE 2sn'de
bir sadece canlı fiyat/PnL/SL/TP güncellemesi (update_prices). Eskiden
(_refresh_price_cells) bu ikinci yol, "harita bayat olabilir" riskiyle
tabloyu sembole göre TARAYARAK doğru satırı buluyordu (27 Ağu 2026 MOVR
"P&L önce 31 sonra -2" bug'ının kök nedeni) — id→index eşlemesi Model/View'da
zaten her zaman doğru olduğu için bu tarama TAMAMEN GEREKSİZLEŞİYOR, bug
sınıfı mimari olarak imkansız hale geliyor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from PyQt6.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from PyQt6.QtGui import QColor

from desktop.models.base_table_model import IdKeyedTableModel
from desktop.models.common import none_safe_lt
from desktop.theme import COLORS
from signals.paper_trade_manager import LEVERAGE_BY_STRATEGY

# ── Açık pozisyonlar ─────────────────────────────────────────────────────────

OPEN_COLUMNS = [
    "Sembol",
    "Yön",
    "TF",
    "Strateji",
    "Giriş$",
    "Fiyat$",
    "P&L$",
    "P&L%",
    "SL%",
    "TP%",
    "Trail$",
    "VPMV",
    "Kolaylık",
    "Süre",
]
(
    OPEN_COL_SYMBOL,
    OPEN_COL_SIDE,
    OPEN_COL_TF,
    OPEN_COL_STRATEGY,
    OPEN_COL_ENTRY,
    OPEN_COL_PRICE,
    OPEN_COL_PNL_USD,
    OPEN_COL_PNL_PCT,
    OPEN_COL_SL,
    OPEN_COL_TP,
    OPEN_COL_TRAIL,
    OPEN_COL_VPMV,
    OPEN_COL_DEVISSO,
    OPEN_COL_AGE,
) = range(14)

HIST_COLUMNS = [
    "Sembol",
    "Yön",
    "TF",
    "Strateji",
    "Giriş$",
    "Çıkış$",
    "P&L$",
    "P&L%",
    "Neden",
    "Kapatma",
]
(
    HIST_COL_SYMBOL,
    HIST_COL_SIDE,
    HIST_COL_TF,
    HIST_COL_STRATEGY,
    HIST_COL_ENTRY,
    HIST_COL_EXIT,
    HIST_COL_PNL_USD,
    HIST_COL_PNL_PCT,
    HIST_COL_REASON,
    HIST_COL_CLOSED,
) = range(10)

_C_GREEN = QColor(COLORS["green"])
_C_RED = QColor(COLORS["red"])
_C_RED_BRIGHT = QColor("#ff4444")
_C_MUTED = QColor(COLORS["text_muted"])
_C_ACCENT = QColor(COLORS["accent"])
_BG_DANGER = QColor("#3d1515")
_BG_STALE = QColor(120, 70, 0, 140)


def _pnl_color(val: Optional[float]) -> QColor:
    if val is None:
        return _C_MUTED
    return _C_GREEN if val >= 0 else _C_RED


def _age_str(dt: Optional[datetime]) -> str:
    if dt is None:
        return "—"
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    secs = max(0, int((datetime.now() - dt).total_seconds()))
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}dk"
    return f"{secs // 3600}s {(secs % 3600) // 60}dk"


def _compute_live_fields(
    side: str, entry: float, sl: Optional[float], tp: Optional[float], live: float
) -> dict:
    """entry/sl/tp/side sabit (satır açıldığında belirlenir), live her tick
    değişir — bu yüzden hem tam-DB-yenilemede (build/update_row) hem canlı
    fiyat tick'inde (update_prices) AYNI formül kullanılıyor, tek yerde."""
    pnl_pct = (live - entry) / entry * 100 if side == "Long" else (entry - live) / entry * 100
    if sl:
        sl_dist = (
            (float(sl) - live) / live * 100 if side == "Long" else (live - float(sl)) / live * 100
        )
    else:
        sl_dist = None
    if tp:
        tp_dist = (
            (float(tp) - live) / live * 100 if side == "Long" else (live - float(tp)) / live * 100
        )
    else:
        tp_dist = None
    return {
        "live": live,
        "pnl_pct": pnl_pct,
        "sl_dist": sl_dist,
        "tp_dist": tp_dist,
        "sl_danger": sl_dist is not None and abs(sl_dist) < 1.0,
    }


@dataclass
class PaperOpenRow:  # pylint: disable=too-many-instance-attributes
    id: int
    symbol: str
    signal_type: str
    interval: str
    strategy_label: str
    entry: float
    sl: Optional[float]
    tp: Optional[float]
    trail: Optional[float]
    position_usd: float
    leverage: float
    vpms_val: Optional[float]
    devisso_val: Optional[float]
    age_str: str
    live: float
    pnl_usd: float
    pnl_pct: float
    sl_dist: Optional[float]
    tp_dist: Optional[float]
    sl_danger: bool
    stale: bool
    raw: dict


def _build_open_row(item: dict) -> PaperOpenRow:
    side = item["signal_type"]
    entry = float(item["entry_price"])
    sl = item["stop_loss_price"]
    tp = item["take_profit_price"]
    trail = item["trailing_stop_price"]
    position_usd = float(item.get("position_usd") or 100.0)
    leverage = LEVERAGE_BY_STRATEGY.get(item.get("strategy", ""), 1.0)
    live = item.get("live_price", entry)
    lf = _compute_live_fields(side, entry, sl, tp, live)
    pnl_usd = lf["pnl_pct"] / 100 * position_usd * leverage
    strat = item.get("strategy", "")
    strat_label = "✋ " + strat if item.get("source") == "manual" else strat
    vpms = item.get("vpms_score")
    devisso = item.get("devisso_score")
    return PaperOpenRow(
        id=item["id"],
        symbol=item["symbol"],
        signal_type=side,
        interval=item["interval"],
        strategy_label=strat_label,
        entry=entry,
        sl=float(sl) if sl else None,
        tp=float(tp) if tp else None,
        trail=float(trail) if trail else None,
        position_usd=position_usd,
        leverage=leverage,
        vpms_val=float(vpms) if vpms is not None else None,
        devisso_val=float(devisso) if devisso is not None else None,
        age_str=_age_str(item["opened_at"]),
        live=lf["live"],
        pnl_usd=pnl_usd,
        pnl_pct=lf["pnl_pct"],
        sl_dist=lf["sl_dist"],
        tp_dist=lf["tp_dist"],
        sl_danger=lf["sl_danger"],
        stale=item.get("stale", False),
        raw=item,
    )


def _update_open_row(row: PaperOpenRow, item: dict) -> None:
    new = _build_open_row(item)
    row.__dict__.update(new.__dict__)


class PaperOpenModel(IdKeyedTableModel):
    COLUMNS = OPEN_COLUMNS
    build_row = staticmethod(_build_open_row)
    update_row = staticmethod(_update_open_row)

    def _id_of_item(self, item: dict) -> int:
        return item["id"]

    def _id_of_row(self, row: PaperOpenRow) -> int:
        return row.id

    def symbols(self) -> set[str]:
        return {row.symbol for row in self._rows}

    def _display(self, row: PaperOpenRow, col: int) -> str:
        if col == OPEN_COL_SYMBOL:
            return row.symbol
        if col == OPEN_COL_SIDE:
            return row.signal_type
        if col == OPEN_COL_TF:
            return row.interval
        if col == OPEN_COL_STRATEGY:
            return row.strategy_label
        if col == OPEN_COL_ENTRY:
            return f"{row.entry:.5g}"
        if col == OPEN_COL_PRICE:
            return f"{row.live:.5g} ⚠bayat" if row.stale else f"{row.live:.5g}"
        if col == OPEN_COL_PNL_USD:
            return f"{row.pnl_usd:+.2f}$"
        if col == OPEN_COL_PNL_PCT:
            return f"{row.pnl_pct:+.2f}%"
        if col == OPEN_COL_SL:
            return f"{row.sl_dist:+.2f}%" if row.sl_dist is not None else "—"
        if col == OPEN_COL_TP:
            return f"{row.tp_dist:+.2f}%" if row.tp_dist is not None else "—"
        if col == OPEN_COL_TRAIL:
            return f"{row.trail:.5g}" if row.trail else "—"
        if col == OPEN_COL_VPMV:
            return f"{row.vpms_val:.1f}" if row.vpms_val is not None else "—"
        if col == OPEN_COL_DEVISSO:
            return f"{row.devisso_val:.1f}" if row.devisso_val is not None else "—"
        if col == OPEN_COL_AGE:
            return row.age_str
        return ""

    def _foreground(self, row: PaperOpenRow, col: int):
        if col == OPEN_COL_SIDE:
            return _C_GREEN if row.signal_type == "Long" else _C_RED
        if col in (OPEN_COL_PNL_USD, OPEN_COL_PNL_PCT):
            return _pnl_color(row.pnl_usd)
        if col == OPEN_COL_SL:
            return _C_RED_BRIGHT if row.sl_danger else _C_RED
        if col == OPEN_COL_TP:
            return _C_GREEN
        if col == OPEN_COL_TRAIL:
            return _C_ACCENT if row.trail else _C_MUTED
        return None

    def _background(self, row: PaperOpenRow, col: int):
        # 28 Ağu 2026: eskiden tam-DB-yenileme (5sn) satırın TÜM sütunlarını,
        # canlı fiyat tick'i (2sn) SADECE Fiyat$..TP% sütunlarını boyuyordu —
        # ikisi arasındaki ≤3sn'de sl_danger/stale durumu değişirse sütunlar
        # arası GÖRSEL TUTARSIZLIK oluşabiliyordu (veri her zaman doğruydu,
        # sadece boyama gecikiyordu). Model/View'da _background tüm sütunlar
        # için TEK bir yerden, satırın GÜNCEL durumundan okunduğu için bu
        # tutarsızlık sınıfı da kendiliğinden ortadan kalkıyor.
        if row.stale:
            return _BG_STALE
        if row.sl_danger:
            return _BG_DANGER
        return None

    def _text_alignment(self, col: int) -> int:
        if col == OPEN_COL_SYMBOL:
            return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        return int(Qt.AlignmentFlag.AlignCenter)

    # ── Canlı fiyat tick'i (2sn'de bir, Redis) ──────────────────────────────

    def update_prices(self, price_updates: dict[str, tuple[float, bool]]) -> None:
        """price_updates: {symbol: (live_price, stale)}. SADECE bu sembollere
        sahip satırlar için PnL/SL/TP yeniden hesaplanır; id→index eşlemesi
        (_id_index) her zaman doğru olduğu için MOVR'daki gibi bir "yanlış
        satıra yazma" riski yapısal olarak yok."""
        min_idx: Optional[int] = None
        max_idx: Optional[int] = None
        for row in self._rows:
            pv = price_updates.get(row.symbol)
            if pv is None:
                continue
            live, stale = pv
            lf = _compute_live_fields(row.signal_type, row.entry, row.sl, row.tp, live)
            row.live = lf["live"]
            row.pnl_pct = lf["pnl_pct"]
            row.pnl_usd = lf["pnl_pct"] / 100 * row.position_usd * row.leverage
            row.sl_dist = lf["sl_dist"]
            row.tp_dist = lf["tp_dist"]
            row.sl_danger = lf["sl_danger"]
            row.stale = stale
            idx = self._id_index[row.id]
            if min_idx is None or idx < min_idx:
                min_idx = idx
            if max_idx is None or idx > max_idx:
                max_idx = idx
        if min_idx is not None:
            tl = self.index(min_idx, 0)
            br = self.index(max_idx, len(self.COLUMNS) - 1)
            self.dataChanged.emit(tl, br, [])


class PaperOpenProxyModel(QSortFilterProxyModel):
    _NUMERIC_ATTR = {
        OPEN_COL_ENTRY: "entry",
        OPEN_COL_PRICE: "live",
        OPEN_COL_PNL_USD: "pnl_usd",
        OPEN_COL_PNL_PCT: "pnl_pct",
        OPEN_COL_SL: "sl_dist",
        OPEN_COL_TP: "tp_dist",
        OPEN_COL_TRAIL: "trail",
        OPEN_COL_VPMV: "vpms_val",
        OPEN_COL_DEVISSO: "devisso_val",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._side = "Tümü"
        self._tf = "Tümü"
        self._strategy = "Tümü"
        self._search = ""

    def set_side(self, v: str) -> None:
        self._side = v
        self.invalidateFilter()

    def set_tf(self, v: str) -> None:
        self._tf = v
        self.invalidateFilter()

    def set_strategy(self, v: str) -> None:
        self._strategy = v
        self.invalidateFilter()

    def set_search(self, v: str) -> None:
        self._search = v.strip().upper()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        src = self.sourceModel()
        if source_row >= len(src._rows):  # noqa: SLF001
            return False
        row = src._rows[source_row]  # noqa: SLF001
        if self._side != "Tümü" and row.signal_type != self._side:
            return False
        if self._tf != "Tümü" and row.interval != self._tf:
            return False
        if self._strategy != "Tümü" and row.strategy_label != self._strategy:
            return False
        if self._search and self._search not in row.symbol.upper():
            return False
        return True

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        attr = self._NUMERIC_ATTR.get(left.column())
        if attr is not None:
            src = self.sourceModel()
            l_row = src._rows[left.row()]  # noqa: SLF001
            r_row = src._rows[right.row()]  # noqa: SLF001
            return none_safe_lt(getattr(l_row, attr), getattr(r_row, attr))
        return super().lessThan(left, right)


# ── Kapalı işlemler ──────────────────────────────────────────────────────────


@dataclass
class PaperHistRow:
    id: int
    symbol: str
    signal_type: str
    interval: str
    strategy_label: str
    entry: float
    exit: Optional[float]
    pnl_usd: float
    pnl_pct: float
    reason: str
    closed_str: str
    raw: dict


def _build_hist_row(item: dict) -> PaperHistRow:
    strat = item.get("strategy", "")
    strat_label = "✋ " + strat if item.get("source") == "manual" else strat
    closed = item["closed_at"]
    if isinstance(closed, datetime) and closed.tzinfo is not None:
        closed = closed.replace(tzinfo=None)
    return PaperHistRow(
        id=item["id"],
        symbol=item["symbol"],
        signal_type=item["signal_type"],
        interval=item["interval"],
        strategy_label=strat_label,
        entry=float(item["entry_price"]),
        exit=float(item["exit_price"]) if item["exit_price"] else None,
        pnl_usd=float(item["pnl_usd"]) if item["pnl_usd"] else 0.0,
        pnl_pct=float(item["pnl_pct"]) if item["pnl_pct"] else 0.0,
        reason=item["close_reason"] or "—",
        closed_str=closed.strftime("%d/%m %H:%M") if closed else "—",
        raw=item,
    )


def _update_hist_row(row: PaperHistRow, item: dict) -> None:
    new = _build_hist_row(item)
    row.__dict__.update(new.__dict__)


class PaperHistModel(IdKeyedTableModel):
    COLUMNS = HIST_COLUMNS
    build_row = staticmethod(_build_hist_row)
    update_row = staticmethod(_update_hist_row)

    def _id_of_item(self, item: dict) -> int:
        return item["id"]

    def _id_of_row(self, row: PaperHistRow) -> int:
        return row.id

    def _display(self, row: PaperHistRow, col: int) -> str:
        if col == HIST_COL_SYMBOL:
            return row.symbol
        if col == HIST_COL_SIDE:
            return row.signal_type
        if col == HIST_COL_TF:
            return row.interval
        if col == HIST_COL_STRATEGY:
            return row.strategy_label
        if col == HIST_COL_ENTRY:
            return f"{row.entry:.5g}"
        if col == HIST_COL_EXIT:
            return f"{row.exit:.5g}" if row.exit is not None else "—"
        if col == HIST_COL_PNL_USD:
            return f"{row.pnl_usd:+.2f}$"
        if col == HIST_COL_PNL_PCT:
            return f"{row.pnl_pct:+.2f}%"
        if col == HIST_COL_REASON:
            return row.reason
        if col == HIST_COL_CLOSED:
            return row.closed_str
        return ""

    def _foreground(self, row: PaperHistRow, col: int):
        if col == HIST_COL_SIDE:
            return _C_GREEN if row.signal_type == "Long" else _C_RED
        if col in (HIST_COL_PNL_USD, HIST_COL_PNL_PCT):
            return _pnl_color(row.pnl_usd)
        return None

    def _text_alignment(self, col: int) -> int:
        if col == HIST_COL_SYMBOL:
            return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        return int(Qt.AlignmentFlag.AlignCenter)


class PaperHistProxyModel(QSortFilterProxyModel):
    _NUMERIC_ATTR = {
        HIST_COL_ENTRY: "entry",
        HIST_COL_EXIT: "exit",
        HIST_COL_PNL_USD: "pnl_usd",
        HIST_COL_PNL_PCT: "pnl_pct",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._side = "Tümü"
        self._tf = "Tümü"
        self._reason = "Tümü"
        self._strategy = "Tümü"
        self._search = ""

    def set_side(self, v: str) -> None:
        self._side = v
        self.invalidateFilter()

    def set_tf(self, v: str) -> None:
        self._tf = v
        self.invalidateFilter()

    def set_reason(self, v: str) -> None:
        self._reason = v
        self.invalidateFilter()

    def set_strategy(self, v: str) -> None:
        self._strategy = v
        self.invalidateFilter()

    def set_search(self, v: str) -> None:
        self._search = v.strip().upper()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        src = self.sourceModel()
        if source_row >= len(src._rows):  # noqa: SLF001
            return False
        row = src._rows[source_row]  # noqa: SLF001
        if self._side != "Tümü" and row.signal_type != self._side:
            return False
        if self._tf != "Tümü" and row.interval != self._tf:
            return False
        if self._reason != "Tümü" and row.reason != self._reason:
            return False
        if self._strategy != "Tümü" and row.strategy_label != self._strategy:
            return False
        if self._search and self._search not in row.symbol.upper():
            return False
        return True

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        attr = self._NUMERIC_ATTR.get(left.column())
        if attr is not None:
            src = self.sourceModel()
            l_row = src._rows[left.row()]  # noqa: SLF001
            r_row = src._rows[right.row()]  # noqa: SLF001
            return none_safe_lt(getattr(l_row, attr), getattr(r_row, attr))
        return super().lessThan(left, right)
