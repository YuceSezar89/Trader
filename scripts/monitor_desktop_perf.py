"""
Sistem sağlığı izleyicisi — desktop.main / run_services.py / signal_service.py
süreçlerinin bellek/CPU büyüme trendini sürekli takip eder. 13 Tem 2026'daki
74GB masaüstü sızıntısı (kapatılmayan Redis/psycopg2 bağlantıları) ve 14 Tem
2026'daki 96GB tekrarı (PC kilitlenmesine yol açtı) sonrası eklendi — artık
sadece pasif CSV loglamıyor, eşik aşılınca Telegram'a da uyarı atıyor.

Kullanım:
    .venv/bin/python scripts/monitor_desktop_perf.py &

Çıktı: logs/desktop_perf.csv (timestamp, process, elapsed_min, rss_mb,
       footprint_mb, cpu_pct, num_threads, num_fds)
"""

import asyncio
import csv
import os
import re
import subprocess
import time
from datetime import datetime

import psutil

from utils.telegram_notify import send_telegram_message

_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "desktop_perf.csv"
)
_INTERVAL_SEC = 30

# Süreç başına eşik (MB) — desktop.main normalde <1GB, backend'ler (657 sembollük
# MTF buffer'lar nedeniyle) birkaç GB'ta seyrediyor; bu yüzden eşikler süreç bazlı.
_WATCHED: dict[str, dict[str, float]] = {
    "desktop.main": {"warn_mb": 2000, "critical_mb": 4000},
    "run_services.py": {"warn_mb": 6000, "critical_mb": 10000},
    "signal_service.py": {"warn_mb": 3000, "critical_mb": 6000},
}

# 14 Ağu 2026 bugfix: desktop.main gerçek bir olayda (~42-66GB, PC kilitlendi)
# RSS'i psutil/ps'te 54+ dakika boyunca 37-40MB'DE SABİT gösterdi — macOS
# bellek baskısı altında (WebEngine/QtWebEngine sürecinin) sayfalarını
# SIKIŞTIRIYOR/takas ediyor, RSS bunu SAYMIYOR (Activity Monitor'ün "Bellek"
# sütunu farklı, gerçek ayak izini gösteriyor). 13 Tem 2026'daki 74GB olayında
# da aynı ders çıkmış, `footprint` komutuyla (macOS yerleşik, sudo gerekmiyor)
# doğrulanmıştı ama bu script'e o zaman eklenmemişti. Sadece desktop.main
# için (backend'lerde bu sorun hiç gözlenmedi, gereksiz subprocess maliyeti).
_FOOTPRINT_PROCS = {"desktop.main"}
_FOOTPRINT_PATTERN = re.compile(r"Footprint:\s*([\d.]+)\s*MB")


def _footprint_mb(pid: int) -> float | None:
    try:
        out = subprocess.run(
            ["footprint", str(pid)], capture_output=True, text=True, timeout=5, check=False
        ).stdout
        match = _FOOTPRINT_PATTERN.search(out)
        return float(match.group(1)) if match else None
    except Exception:  # pylint: disable=broad-exception-caught
        return None


# fd limiti (setrlimit ile 4096/8192'ye çıkarıldı, 17 Tem 2026) doluşu sessizce
# ilerleyip "DNS'e ulaşılamıyor" gibi kafa karıştırıcı hatalara yol açabiliyor —
# eşiğe yaklaşınca erken uyar (bkz. signal_service.py'deki setrlimit yorumu).
_FD_WARN = 3000
_FD_CRITICAL = 3800


def _find_pids() -> dict[str, int]:
    found: dict[str, int] = {}
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmdline = proc.info.get("cmdline") or []
        for name in _WATCHED:
            if name not in found and any(name in part for part in cmdline):
                found[name] = proc.info["pid"]
    return found


def _alert(text: str) -> None:
    print(f"[UYARI] {text}")
    try:
        asyncio.run(send_telegram_message(f"⚠️ Bellek uyarısı\n{text}"))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[UYARI] Telegram gönderilemedi: {exc}")


def main() -> None:
    procs: dict[str, psutil.Process] = {}
    starts: dict[str, float] = {}
    # Bir eşik için tekrar tekrar uyarı basmamak için (sadece durum değişince alarm)
    alerted: dict[str, set[str]] = {name: set() for name in _WATCHED}

    is_new = not os.path.exists(_LOG_PATH)
    with open(_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                [
                    "timestamp",
                    "process",
                    "elapsed_min",
                    "rss_mb",
                    "footprint_mb",
                    "cpu_pct",
                    "num_threads",
                    "num_fds",
                ]
            )

        print(f"İzleniyor: {list(_WATCHED)}, her {_INTERVAL_SEC}s'de bir {_LOG_PATH}'e yazılıyor")

        while True:
            # Yeni/kaybolan süreçleri periyodik olarak yeniden keşfet
            pids = _find_pids()
            for name, pid in pids.items():
                if name not in procs or procs[name].pid != pid:
                    try:
                        procs[name] = psutil.Process(pid)
                        procs[name].cpu_percent()  # ilk çağrı 0.0 döner, ısıtma
                        starts[name] = time.time()
                        alerted[name] = set()
                        print(f"[{name}] izleniyor: PID={pid}")
                    except psutil.NoSuchProcess:
                        continue

            time.sleep(_INTERVAL_SEC)

            for name in list(procs):
                proc = procs[name]
                try:
                    rss_mb = proc.memory_info().rss / (1024 * 1024)
                    footprint_mb = _footprint_mb(proc.pid) if name in _FOOTPRINT_PROCS else None
                    # Sıkıştırma altında RSS küçük kalabilir — eşik karşılaştırması
                    # ikisinin büyüğüyle yapılır (bkz. yukarıdaki 14 Ağu notu).
                    effective_mb = max(rss_mb, footprint_mb or 0.0)
                    cpu_pct = proc.cpu_percent()
                    num_threads = proc.num_threads()
                    try:
                        num_fds = proc.num_fds()
                    except Exception:  # pylint: disable=broad-exception-caught
                        num_fds = -1
                    elapsed_min = (time.time() - starts[name]) / 60

                    writer.writerow(
                        [
                            datetime.now().isoformat(timespec="seconds"),
                            name,
                            round(elapsed_min, 1),
                            round(rss_mb, 1),
                            round(footprint_mb, 1) if footprint_mb is not None else "",
                            round(cpu_pct, 1),
                            num_threads,
                            num_fds,
                        ]
                    )
                    f.flush()

                    thresholds = _WATCHED[name]
                    if (
                        effective_mb >= thresholds["critical_mb"]
                        and "critical" not in alerted[name]
                    ):
                        _alert(
                            f"{name} (PID {proc.pid}) bellek {effective_mb:.0f}MB "
                            f"(RSS={rss_mb:.0f}MB, footprint={footprint_mb or 0:.0f}MB) — "
                            f"KRİTİK eşik ({thresholds['critical_mb']:.0f}MB) aşıldı, "
                            f"{elapsed_min:.0f} dk çalışıyor."
                        )
                        alerted[name].add("critical")
                    elif effective_mb >= thresholds["warn_mb"] and "warn" not in alerted[name]:
                        _alert(
                            f"{name} (PID {proc.pid}) bellek {effective_mb:.0f}MB "
                            f"(RSS={rss_mb:.0f}MB, footprint={footprint_mb or 0:.0f}MB) — "
                            f"uyarı eşiği ({thresholds['warn_mb']:.0f}MB) aşıldı, "
                            f"{elapsed_min:.0f} dk çalışıyor."
                        )
                        alerted[name].add("warn")
                    elif effective_mb < thresholds["warn_mb"] and alerted[name] - {
                        "fd_warn",
                        "fd_critical",
                    }:
                        alerted[name] -= {"warn", "critical"}  # düştü, tekrar uyarabiliriz

                    if num_fds >= 0:
                        if num_fds >= _FD_CRITICAL and "fd_critical" not in alerted[name]:
                            _alert(
                                f"{name} (PID {proc.pid}) açık dosya tanıtıcısı {num_fds} — "
                                f"KRİTİK eşik ({_FD_CRITICAL}) aşıldı, fd limiti dolabilir."
                            )
                            alerted[name].add("fd_critical")
                        elif num_fds >= _FD_WARN and "fd_warn" not in alerted[name]:
                            _alert(
                                f"{name} (PID {proc.pid}) açık dosya tanıtıcısı {num_fds} — "
                                f"uyarı eşiği ({_FD_WARN}) aşıldı."
                            )
                            alerted[name].add("fd_warn")
                        elif num_fds < _FD_WARN:
                            alerted[name] -= {"fd_warn", "fd_critical"}

                except psutil.NoSuchProcess:
                    print(f"[{name}] artık çalışmıyor, izlemeden çıkarılıyor")
                    del procs[name]


if __name__ == "__main__":
    main()
