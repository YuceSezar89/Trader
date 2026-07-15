"""
Sistem sağlığı izleyicisi — desktop.main / run_services.py / signal_service.py
süreçlerinin bellek/CPU büyüme trendini sürekli takip eder. 13 Tem 2026'daki
74GB masaüstü sızıntısı (kapatılmayan Redis/psycopg2 bağlantıları) ve 14 Tem
2026'daki 96GB tekrarı (PC kilitlenmesine yol açtı) sonrası eklendi — artık
sadece pasif CSV loglamıyor, eşik aşılınca Telegram'a da uyarı atıyor.

Kullanım:
    .venv/bin/python scripts/monitor_desktop_perf.py &

Çıktı: logs/desktop_perf.csv (timestamp, process, elapsed_min, rss_mb, cpu_pct,
       num_threads, num_fds)
"""

import asyncio
import csv
import os
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
                            round(cpu_pct, 1),
                            num_threads,
                            num_fds,
                        ]
                    )
                    f.flush()

                    thresholds = _WATCHED[name]
                    if rss_mb >= thresholds["critical_mb"] and "critical" not in alerted[name]:
                        _alert(
                            f"{name} (PID {proc.pid}) bellek {rss_mb:.0f}MB — KRİTİK eşik "
                            f"({thresholds['critical_mb']:.0f}MB) aşıldı, {elapsed_min:.0f} dk çalışıyor."
                        )
                        alerted[name].add("critical")
                    elif rss_mb >= thresholds["warn_mb"] and "warn" not in alerted[name]:
                        _alert(
                            f"{name} (PID {proc.pid}) bellek {rss_mb:.0f}MB — uyarı eşiği "
                            f"({thresholds['warn_mb']:.0f}MB) aşıldı, {elapsed_min:.0f} dk çalışıyor."
                        )
                        alerted[name].add("warn")
                    elif rss_mb < thresholds["warn_mb"] and alerted[name]:
                        alerted[name].clear()  # düştü, bir sonraki aşımda tekrar uyarabiliriz

                except psutil.NoSuchProcess:
                    print(f"[{name}] artık çalışmıyor, izlemeden çıkarılıyor")
                    del procs[name]


if __name__ == "__main__":
    main()
