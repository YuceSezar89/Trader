"""Dark theme stylesheet for TRader Desktop Terminal."""

DARK_QSS = """
QMainWindow {
    background-color: #0d1117;
}

QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-size: 13px;
}

/* ── Menü Bar ─────────────────────────────────────────────── */
QMenuBar {
    background-color: #161b22;
    color: #e6edf3;
    border-bottom: 1px solid #30363d;
    padding: 2px 0;
}
QMenuBar::item {
    background: transparent;
    padding: 4px 10px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #21262d;
}
QMenu {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #21262d;
}
QMenu::separator {
    height: 1px;
    background-color: #30363d;
    margin: 4px 8px;
}

/* ── Toolbar ──────────────────────────────────────────────── */
QToolBar {
    background-color: #161b22;
    border-bottom: 1px solid #30363d;
    spacing: 4px;
    padding: 4px 8px;
}
QToolBar::separator {
    background-color: #30363d;
    width: 1px;
    margin: 4px 4px;
}
QToolButton {
    background-color: transparent;
    color: #e6edf3;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}
QToolButton:hover {
    background-color: #21262d;
}
QToolButton:pressed {
    background-color: #30363d;
}
QToolButton:checked {
    background-color: #1f6feb;
    color: #ffffff;
}

/* ── Status Bar ───────────────────────────────────────────── */
QStatusBar {
    background-color: #161b22;
    color: #8b949e;
    border-top: 1px solid #30363d;
    font-size: 11px;
    padding: 0 8px;
}
QStatusBar::item {
    border: none;
}
QLabel#status_ok {
    color: #3fb950;
}
QLabel#status_err {
    color: #f85149;
}
QLabel#status_warn {
    color: #d29922;
}

/* ── Dock Widget ──────────────────────────────────────────── */
QDockWidget {
    color: #e6edf3;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}
QDockWidget::title {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-bottom: none;
    padding: 4px 8px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #8b949e;
}
QDockWidget::close-button, QDockWidget::float-button {
    background: transparent;
    border: none;
    border-radius: 3px;
    padding: 2px;
}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {
    background-color: #21262d;
}

/* ── Tablo ────────────────────────────────────────────────── */
QTableView, QTableWidget {
    background-color: #0d1117;
    alternate-background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    gridline-color: #21262d;
    selection-background-color: #1f6feb33;
    selection-color: #e6edf3;
}
QTableView::item, QTableWidget::item {
    padding: 4px 8px;
    border: none;
}
QHeaderView::section {
    background-color: #161b22;
    color: #8b949e;
    border: none;
    border-right: 1px solid #30363d;
    border-bottom: 1px solid #30363d;
    padding: 6px 8px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
}
QHeaderView::section:hover {
    background-color: #21262d;
    color: #e6edf3;
}

/* ── Scrollbar ────────────────────────────────────────────── */
QScrollBar:vertical {
    background-color: #0d1117;
    width: 8px;
    border: none;
}
QScrollBar::handle:vertical {
    background-color: #30363d;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background-color: #484f58;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background-color: #0d1117;
    height: 8px;
    border: none;
}
QScrollBar::handle:horizontal {
    background-color: #30363d;
    border-radius: 4px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #484f58;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ── ComboBox ─────────────────────────────────────────────── */
QComboBox {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 4px 8px;
    color: #e6edf3;
    min-width: 80px;
}
QComboBox:hover {
    border-color: #484f58;
}
QComboBox:focus {
    border-color: #1f6feb;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    selection-background-color: #21262d;
    outline: none;
}

/* ── LineEdit ─────────────────────────────────────────────── */
QLineEdit {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 4px 8px;
    color: #e6edf3;
}
QLineEdit:focus {
    border-color: #1f6feb;
}
QLineEdit::placeholder {
    color: #484f58;
}

/* ── PushButton ───────────────────────────────────────────── */
QPushButton {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 6px 14px;
    color: #e6edf3;
}
QPushButton:hover {
    background-color: #30363d;
    border-color: #484f58;
}
QPushButton:pressed {
    background-color: #161b22;
}
QPushButton#primary {
    background-color: #1f6feb;
    border-color: #1f6feb;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primary:hover {
    background-color: #388bfd;
    border-color: #388bfd;
}

/* ── Splitter ─────────────────────────────────────────────── */
QSplitter::handle {
    background-color: #30363d;
}
QSplitter::handle:horizontal {
    width: 2px;
}
QSplitter::handle:vertical {
    height: 2px;
}

/* ── Tab Widget ───────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #30363d;
    background-color: #0d1117;
}
QTabBar::tab {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-bottom: none;
    padding: 6px 14px;
    color: #8b949e;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #0d1117;
    color: #e6edf3;
    border-bottom: 2px solid #1f6feb;
}
QTabBar::tab:hover:!selected {
    background-color: #21262d;
    color: #e6edf3;
}

/* ── Label ────────────────────────────────────────────────── */
QLabel {
    background: transparent;
    color: #e6edf3;
}
QLabel#section_title {
    font-size: 11px;
    font-weight: 600;
    color: #8b949e;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
QLabel#price_up {
    color: #3fb950;
    font-weight: 600;
}
QLabel#price_down {
    color: #f85149;
    font-weight: 600;
}
QLabel#neutral {
    color: #8b949e;
}

/* ── GroupBox ─────────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #30363d;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 8px;
    font-size: 11px;
    color: #8b949e;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    left: 8px;
}
"""

# Renk paleti — Python tarafında da kullanılabilir
COLORS = {
    "bg_primary":    "#0d1117",
    "bg_secondary":  "#161b22",
    "bg_tertiary":   "#21262d",
    "border":        "#30363d",
    "border_hover":  "#484f58",
    "text_primary":  "#e6edf3",
    "text_muted":    "#8b949e",
    "accent":        "#1f6feb",
    "accent_hover":  "#388bfd",
    "green":         "#3fb950",
    "red":           "#f85149",
    "yellow":        "#d29922",
    "orange":        "#f0883e",
    "purple":        "#a371f7",
}
