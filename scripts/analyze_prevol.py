"""
PREVOL analiz scripti.
Sinyal öncesi 5 barın buy_vol oranını DB'deki PnL ile karşılaştırır.

Kullanım:
    python scripts/analyze_prevol.py [log_dosyası]
"""

import re
import sys
import statistics
from datetime import datetime, timedelta

import psycopg2
import psycopg2.extras

DB_DSN = "dbname=trader_panel user=yusuf host=localhost port=5432"
log_file = sys.argv[1] if len(sys.argv) > 1 else "logs/services.log"

pattern = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*PREVOL \| (\w+) \| (\w+) \| (\w+) \| buy_pct=([\d.]+)"
)

entries = []
with open(log_file) as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue
        ts_str, symbol, sig_type, interval, buy_pct = m.groups()
        entries.append({
            "ts":       datetime.fromisoformat(ts_str),
            "symbol":   symbol,
            "sig_type": sig_type,
            "interval": interval,
            "buy_pct":  float(buy_pct),
        })

print(f"Log'da {len(entries)} PREVOL girişi bulundu.")
if not entries:
    sys.exit(0)

# Tek sorguda tüm sinyalleri çek, sonra Python'da eşleştir
conn = psycopg2.connect(DB_DSN)
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
cur.execute("""
    SELECT symbol, signal_type, interval, opened_at, realized_pnl
    FROM signals
    WHERE status = 'closed'
      AND realized_pnl IS NOT NULL
    ORDER BY opened_at
""")
signals = cur.fetchall()
cur.close()
conn.close()

# Python'da eşleştir
sig_index = {}
for s in signals:
    key = (s["symbol"], s["signal_type"], s["interval"])
    sig_index.setdefault(key, []).append(s)

matched = []
for e in entries:
    key = (e["symbol"], e["sig_type"], e["interval"])
    candidates = sig_index.get(key, [])
    lo = e["ts"] - timedelta(minutes=2)
    hi = e["ts"] + timedelta(minutes=2)
    for s in candidates:
        if lo <= s["opened_at"] <= hi:
            matched.append({**e, "pnl_pct": float(s["realized_pnl"])})
            break

print(f"DB ile eşleşen: {len(matched)} sinyal\n")

if len(matched) < 10:
    print("Yeterli veri yok.")
    sys.exit(0)


def rapor(rows, baslik):
    if not rows:
        print(f"{baslik}: veri yok")
        return
    pnls = [r["pnl_pct"] for r in rows]
    pos  = sum(1 for p in pnls if p > 0)
    print(f"\n{'='*45}")
    print(f"{baslik}  ({len(rows)} sinyal)")
    print(f"{'='*45}")
    print(f"Win rate : {pos}/{len(rows)} ({pos/len(rows)*100:.0f}%)")
    print(f"Ort PnL  : {statistics.mean(pnls):+.3f}%")
    print(f"Medyan   : {statistics.median(pnls):+.3f}%")


for label, sig_type, good_thr, bad_thr in [
    ("LONG",  "Long",  60, 40),
    ("SHORT", "Short", 40, 60),
]:
    sigs = [r for r in matched if r["sig_type"] == sig_type]
    if not sigs:
        continue
    good = [r for r in sigs if (sig_type == "Long"  and r["buy_pct"] >= good_thr) or
                                (sig_type == "Short" and r["buy_pct"] <= good_thr)]
    bad  = [r for r in sigs if (sig_type == "Long"  and r["buy_pct"] <= bad_thr) or
                                (sig_type == "Short" and r["buy_pct"] >= bad_thr)]
    neut = [r for r in sigs if r not in good and r not in bad]

    rapor(good, f"{label} — Yön uyumlu hacim")
    rapor(bad,  f"{label} — Ters yönde hacim")
    rapor(neut, f"{label} — Nötr hacim")

print(f"\nToplam eşleşen: {len(matched)}")
