"""PgBouncer log rotasyonu.

homebrew.mxcl.pgbouncer için hiç rotasyon yoktu (Ekim 2025'ten beri tek
dosya, 16 Ağu 2026'da disk dolmasına katkıda bulundu). com.trader.pgbouncer_logrotate
LaunchAgent'ı bu script'i günlük çalıştırır.
"""

import gzip
import os
import shutil
import signal
import subprocess

LOG = "/opt/homebrew/var/log/pgbouncer.log"
KEEP = 7


def main() -> None:
    if not os.path.exists(LOG) or os.path.getsize(LOG) == 0:
        return

    for i in range(KEEP - 1, 0, -1):
        src = f"{LOG}.{i}.gz"
        dst = f"{LOG}.{i + 1}.gz"
        if os.path.exists(src):
            os.replace(src, dst)

    oldest = f"{LOG}.{KEEP}.gz"
    if os.path.exists(oldest):
        os.remove(oldest)

    rotated = f"{LOG}.1"
    os.replace(LOG, rotated)
    with open(rotated, "rb") as f_in, gzip.open(f"{rotated}.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(rotated)

    out = subprocess.run(
        ["pgrep", "-f", "pgbouncer -q /opt/homebrew/etc/pgbouncer.ini"],
        capture_output=True,
        text=True,
        check=False,
    )
    pid = out.stdout.strip().splitlines()
    if pid:
        os.kill(int(pid[0]), signal.SIGHUP)


if __name__ == "__main__":
    main()
