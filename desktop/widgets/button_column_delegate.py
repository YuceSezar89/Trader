"""ButtonColumnDelegate — sıralanabilir/filtrelenebilir bir QTableView
sütununda tıklanabilir buton hücresi.

Neden setCellWidget/setIndexWidget DEĞİL: bir QSortFilterProxyModel altında
görünür-satır↔widget eşlemesi her sıralama/filtrelemede geçersiz olur (Qt'nin
bilinen kısıtı — index widget'lar proxy re-map'lerini takip etmez). Ayrıca
tf_alignment_panel.py'de 26 Tem 2026'da tespit edilen kök neden: her
render()'da yeni bir QPushButton yaratmak, setVisible(True) sırasında
ensurePolished() → TÜM QSS stylesheet'in yeniden ayrıştırılmasını tetikliyor
(20sn'de bir ~225 satır = sürekli CPU/bellek maliyeti, masaüstü donmasının
kaynağıydı). Bu yüzden buton burada gerçek bir widget olarak hiç
YARATILMAZ — sadece çizilir (paint), tıklama editorEvent'te yakalanıp bir
Qt sinyaliyle bildirilir.

Kullanım:
    delegate = ButtonColumnDelegate("Aç")
    delegate.clicked.connect(self._on_open_clicked)
    view.setItemDelegateForColumn(COL_ACTION, delegate)
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QModelIndex, QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionButton,
)


class ButtonColumnDelegate(QStyledItemDelegate):
    clicked = pyqtSignal(QModelIndex)

    def __init__(self, text: str = "Aç", parent: QObject = None):
        super().__init__(parent)
        self._text = text

    def paint(self, painter, option, index) -> None:  # noqa: N802
        btn = QStyleOptionButton()
        btn.rect = option.rect.adjusted(3, 2, -3, -2)
        btn.text = self._text
        btn.state = QStyle.StateFlag.State_Enabled
        if option.state & QStyle.StateFlag.State_MouseOver:
            btn.state |= QStyle.StateFlag.State_MouseOver
        style = option.widget.style() if option.widget is not None else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_PushButton, btn, painter, option.widget)

    def editorEvent(self, event, model, option, index) -> bool:  # noqa: N802
        if (
            event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
            and option.rect.contains(event.pos())
        ):
            self.clicked.emit(index)
            return True
        return False

    def createEditor(self, parent, option, index):  # noqa: N802
        return None  # düzenleme yok — sadece tıklama yakalanır
