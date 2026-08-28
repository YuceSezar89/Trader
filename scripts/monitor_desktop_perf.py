"""
Sistem sağlığı izleyicisi — desktop.main / run_services.py / signal_service.py
süreçlerinin bellek/CPU büyüme trendini sürekli takip eder. 13 Tem 2026'daki
74GB masaüstü sızıntısı (kapatılmayan Redis/psycopg2 bağlantıları) ve 14 Tem
2026'daki 96GB tekrarı (PC kilitlenmesine yol açtı) sonrası eklendi — artık
sadece pasif CSV loglamıyor, eşik aşılınca Telegram'a da uyarı atıyor.

27 Ağu 2026: script bugüne kadar launchd'ye hiç kaydedilmemişti (elle
başlatılması gerekiyordu) — bu yüzden 87.9GB'a çıkıp kernel'in "swap
exhaustion" nedeniyle durdurduğu masaüstü panel olayı hiç Telegram'a
düşmedi, günlerce fark edilmedi. Artık com.trader.desktop_perf_monitor
LaunchAgent'ı ile sürekli çalışıyor. Ayrıca: desktop.main (SADECE bu —
backend'ler ASLA) kritik eşiği aşınca artık sadece uyarmıyor, GERÇEKTEN
sonlandırıyor — bir UI bug'ının tüm sistemi (swap'ı tüketip Postgres'in
arka plan job'larını etkileyerek) felç etmesini önlemek için (bulkhead
izolasyonu, bkz. proje CLAUDE.md "Backend Hesaplar, Arayüz Okur" ilkesi).

28 Ağu 2026: event-loop donması tespiti de eklendi — desktop.main'in
GUI thread'i her 15sn'de bir heartbeat:desktop_panel'i Redis'e yazıyor
(desktop/main_window.py), donarsa bu da durur. Bu script zaten PID'yi
izlediği için hem "süreç çalışıyor" hem "heartbeat bayat" korelasyonunu
tek yerde yapabiliyor (run_services.py'deki genel heartbeat_watchdog_loop
bunu yapamıyordu — process çöktüğünde/kapandığında da key bayat kalıyor,
"donmuş" ile "kapanmış"ı ayıramıyordu, bkz. proje hafızası 28 Ağu).

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
import redis

from config import Config
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

# 27 Ağu 2026: SADECE desktop.main (bir UI süreci, kapanması güvenli/geri
# dönüşü kolay — kullanıcı tekrar açabilir) kritik eşikte GERÇEKTEN
# sonlandırılıyor. run_services.py/signal_service.py KESİNLİKLE bu kümede
# OLMAMALI — bunlar kritik-iş backend'leri, otomatik öldürme paper trade
# pozisyon yönetimini/sinyal üretimini sessizce durdurup gerçek karar
# kaybına yol açar.
_KILL_ON_CRITICAL = {"desktop.main"}
_KILL_GRACE_SEC = 5  # terminate() sonrası kill()'e geçmeden önce bekleme

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

# 28 Ağu 2026: event-loop donması — desktop.main'in GUI thread'i her 15sn'de
# heartbeat:desktop_panel'i yazıyor (ex=90 TTL'li). Süreç çalışıyor AMA key
# max_age_sec'ten uzun süredir güncellenmemişse GUI thread tıkanmış demektir.
_HEARTBEAT_WATCHED: dict[str, dict[str, object]] = {
    "desktop.main": {"key": "heartbeat:desktop_panel", "max_age_sec": 90},
}

_redis_client: "redis.Redis | None" = None


def _get_redis_client() -> "redis.Redis | None":
    global _redis_client  # pylint: disable=global-statement
    if _redis_client is None:
        try:
            _redis_client = redis.Redis.from_url(
                Config.REDIS_URL,
                decode_responses=True,
                socket_timeout=2,
                socket_connect_timeout=2,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return None
    return _redis_client


def _heartbeat_age_sec(key: str) -> float | None:
    client = _get_redis_client()
    if client is None:
        return None
    try:
        raw = client.get(key)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    if not raw:
        return None
    try:
        last = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return (datetime.now() - last).total_seconds()


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


def _kill_process(proc: psutil.Process, name: str, reason: str) -> None:
    """SADECE _KILL_ON_CRITICAL kümesindeki süreçler için çağrılır — önce
    nazikçe (terminate/SIGTERM) kapatmayı dener, _KILL_GRACE_SEC içinde
    kapanmazsa zorla (kill/SIGKILL) sonlandırır. `reason` sonlandırma
    nedenini açıklar (ör. "bellek 4200MB" veya "event-loop donması").
    Amaç: 27 Ağu 2026'daki 87.9GB swap-exhaustion olayının tekrarında,
    bu sürecin sistemin geri kalanını (Postgres arka plan job'ları dahil)
    etkilemesine izin vermeden, kendi kendine sessizce sonlanmasını
    sağlamak."""
    try:
        proc.terminate()
        proc.wait(timeout=_KILL_GRACE_SEC)
        _alert(
            f"{name} (PID {proc.pid}) {reason} nedeniyle "
            f"OTOMATİK KAPATILDI (terminate) — sistemin geri kalanını korumak için."
        )
    except psutil.TimeoutExpired:
        try:
            proc.kill()
            _alert(
                f"{name} (PID {proc.pid}) {reason} nedeniyle "
                f"ZORLA SONLANDIRILDI (kill, terminate yanıt vermedi)."
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _alert(f"{name} (PID {proc.pid}) sonlandırılamadı: {exc}")
    except psutil.NoSuchProcess:
        pass  # zaten kapanmış
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _alert(f"{name} (PID {proc.pid}) sonlandırma denemesi başarısız: {exc}")


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
                        if name in _KILL_ON_CRITICAL:
                            _kill_process(proc, name, f"bellek {effective_mb:.0f}MB")
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

                    if name in _HEARTBEAT_WATCHED:
                        hb = _HEARTBEAT_WATCHED[name]
                        age = _heartbeat_age_sec(hb["key"])
                        max_age = hb["max_age_sec"]
                        if age is not None and age >= max_age and "freeze" not in alerted[name]:
                            alerted[name].add("freeze")
                            _alert(
                                f"{name} (PID {proc.pid}) event-loop {age:.0f}s'dir "
                                f"heartbeat göndermiyor (limit {max_age}s) — donmuş olabilir."
                            )
                            if name in _KILL_ON_CRITICAL:
                                _kill_process(
                                    proc, name, f"event-loop donması (heartbeat {age:.0f}s bayat)"
                                )
                        elif age is not None and age < max_age:
                            alerted[name].discard("freeze")

                except psutil.NoSuchProcess:
                    print(f"[{name}] artık çalışmıyor, izlemeden çıkarılıyor")
                    del procs[name]


if __name__ == "__main__":
    main()
