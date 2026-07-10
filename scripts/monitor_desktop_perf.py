"""
Masaüstü uygulamasının (desktop.main) uzun süreli çalışırken bellek/CPU
büyüme trendini takip eder — "uzun süreli kullanımda panel yavaşlıyor"
şikayetini araştırmak için. RSS sürekli artıyorsa gerçek bir sızıntıya
işaret eder; artmıyorsa yavaşlık başka bir yerden (ör. periyodik pahalı
UI işlemleri) geliyor demektir.

Kullanım:
    .venv/bin/python scripts/monitor_desktop_perf.py &

Çıktı: logs/desktop_perf.csv (timestamp, rss_mb, cpu_pct, num_threads, num_fds)
"""

import csv
import os
import time
from datetime import datetime

import psutil

_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "desktop_perf.csv")
_INTERVAL_SEC = 30
_PROCESS_MATCH = "desktop.main"


def _find_pid() -> int:
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmdline = proc.info.get("cmdline") or []
        if any(_PROCESS_MATCH in part for part in cmdline):
            return proc.info["pid"]
    raise RuntimeError(f"'{_PROCESS_MATCH}' çalışan bir process bulunamadı")


def main() -> None:
    pid = _find_pid()
    proc = psutil.Process(pid)
    print(f"İzleniyor: PID={pid}, her {_INTERVAL_SEC}s'de bir {_LOG_PATH}'e yazılıyor")

    is_new = not os.path.exists(_LOG_PATH)
    with open(_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "elapsed_min", "rss_mb", "cpu_pct", "num_threads", "num_fds"])

        proc.cpu_percent()  # ilk çağrı her zaman 0.0 döner, ısıtma
        start = time.time()

        while True:
            time.sleep(_INTERVAL_SEC)
            try:
                rss_mb = proc.memory_info().rss / (1024 * 1024)
                cpu_pct = proc.cpu_percent()
                num_threads = proc.num_threads()
                try:
                    num_fds = proc.num_fds()
                except Exception:  # pylint: disable=broad-exception-caught
                    num_fds = -1
                elapsed_min = (time.time() - start) / 60

                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    round(elapsed_min, 1),
                    round(rss_mb, 1),
                    round(cpu_pct, 1),
                    num_threads,
                    num_fds,
                ])
                f.flush()
            except psutil.NoSuchProcess:
                print("Process artık çalışmıyor, izleme durduruluyor")
                return


if __name__ == "__main__":
    main()
