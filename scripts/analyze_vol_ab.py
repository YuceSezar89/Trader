"""
VOL_AB karşılaştırma scripti.
Logdan yönsüz vs yönlü hacim skorlarını okur, istatistik verir.

Kullanım:
    python scripts/analyze_vol_ab.py [log_dosyası]

Varsayılan log: /tmp/trader_service.log
"""

import sys
import re
import statistics

log_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/trader_service.log"

pattern = re.compile(
    r"VOL_AB \| (\w+) \| [\d.]+ \| yonsuz=([\d.]+) yonlu=([\d.]+) fark=(-?[\d.]+)"
)

long_rows, short_rows = [], []

with open(log_file) as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue
        side, yonsuz, yonlu, fark = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
        row = {"yonsuz": yonsuz, "yonlu": yonlu, "fark": fark}
        (long_rows if side == "Long" else short_rows).append(row)

def rapor(rows, baslik):
    if not rows:
        print(f"{baslik}: veri yok")
        return
    farklar = [r["fark"] for r in rows]
    yonsuz  = [r["yonsuz"] for r in rows]
    yonlu   = [r["yonlu"] for r in rows]
    ayni_yon = sum(1 for r in rows if (r["yonlu"] - 50) * (r["yonsuz"] - 50) > 0)

    print(f"\n{'='*40}")
    print(f"{baslik}  ({len(rows)} sinyal)")
    print(f"{'='*40}")
    print(f"Yönsüz  ort={statistics.mean(yonsuz):.1f}  std={statistics.stdev(yonsuz) if len(yonsuz)>1 else 0:.1f}")
    print(f"Yönlü   ort={statistics.mean(yonlu):.1f}  std={statistics.stdev(yonlu) if len(yonlu)>1 else 0:.1f}")
    print(f"Fark    ort={statistics.mean(farklar):.1f}  min={min(farklar):.1f}  max={max(farklar):.1f}")
    print(f"Aynı yön: {ayni_yon}/{len(rows)} ({ayni_yon/len(rows)*100:.0f}%)")
    print(f"Yönlü > Yönsüz: {sum(1 for r in rows if r['fark']>0)}/{len(rows)}")
    print(f"Büyük fark (>20): {sum(1 for r in rows if abs(r['fark'])>20)}/{len(rows)}")

rapor(long_rows,  "LONG  sinyalleri")
rapor(short_rows, "SHORT sinyalleri")

total = len(long_rows) + len(short_rows)
print(f"\nToplam: {total} sinyal")
