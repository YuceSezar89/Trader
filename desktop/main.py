"""
TRader Desktop Terminal — giriş noktası.

Kullanım:
    python -m desktop.main
    # veya
    .venv/bin/python -m desktop.main
"""

import os
import sys

os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from desktop.main_window import MainWindow
from desktop.theme import DARK_QSS


def _load_config() -> dict:
    """Config'i merkezi config.py'den yükler, bulunamazsa varsayılanları kullanır."""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from config import Config
        return {
            "redis_url":   Config.REDIS_URL,
            "db_host":     Config.DB_HOST,
            "db_port":     Config.DB_PORT,
            "db_name":     Config.DB_NAME,
            "db_user":     Config.DB_USER,
            "db_password": Config.DB_PASSWORD,
        }
    except Exception:
        return {
            "redis_url":   "redis://localhost:6379/0",
            "db_host":     "localhost",
            "db_port":     5432,
            "db_name":     "trader_panel",
            "db_user":     "yusuf",
            "db_password": "",
        }


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("TRader Terminal")
    app.setOrganizationName("TRader")
    app.setStyleSheet(DARK_QSS)

    font = QFont()
    font.setFamily("Helvetica Neue")
    font.setPointSize(13)
    app.setFont(font)

    config = _load_config()
    window = MainWindow(config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
