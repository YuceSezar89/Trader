"""
TRader Desktop Terminal — giriş noktası.

Kullanım:
    python -m desktop.main
    # veya
    .venv/bin/python -m desktop.main
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from desktop.main_window import MainWindow
from desktop.theme import DARK_QSS

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _backend_already_running() -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_services.py"],
            capture_output=True, text=True
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def _start_backend() -> "subprocess.Popen | None":
    services_script = os.path.join(_PROJECT_ROOT, "run_services.py")
    if not os.path.exists(services_script):
        return None

    if _backend_already_running():
        return None

    log_dir = os.path.join(_PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "services.log")

    try:
        log_file = open(log_path, "a")  # pylint: disable=consider-using-with
        return subprocess.Popen(
            [sys.executable, "run_services.py"],
            cwd=_PROJECT_ROOT,
            stdout=log_file,
            stderr=log_file,
        )
    except Exception as exc:
        print(f"Backend başlatılamadı: {exc}", file=sys.stderr)
        return None


def _load_config() -> dict:
    try:
        sys.path.insert(0, _PROJECT_ROOT)
        from config import Config  # pylint: disable=import-outside-toplevel
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
    backend = _start_backend()

    app = QApplication(sys.argv)
    app.setApplicationName("TRader Terminal")
    app.setOrganizationName("TRader")
    app.setStyleSheet(DARK_QSS)

    font = QFont()
    font.setFamily("Helvetica Neue")
    font.setPointSize(13)
    app.setFont(font)

    if backend is not None:
        app.aboutToQuit.connect(backend.terminate)

    config = _load_config()
    window = MainWindow(config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
