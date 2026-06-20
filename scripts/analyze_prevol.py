"""
PREVOL analiz scripti.
Sinyal öncesi 5 barın buy_vol oranını DB'deki PnL ile karşılaştırır.

Kullanım:
    python scripts/analyze_prevol.py [log_dosyası]

Varsayılan log: /tmp/trader_service.log
"""

import re
import sys
import statistics
import asyncio
from datetime import datetime, timedelta

import asyncpg

DB_DSN = "postgresql://yusuf@localhost/trader_panel"
log_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/trader_service.log"

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


async def fetch_pnl(entries):
    conn = await asyncpg.connect(DB_DSN)
    results = []
    for e in entries:
        row = await conn.fetchrow("""
            SELECT pnl_pct, closed_at
            FROM signals
            WHERE symbol      = $1
              AND signal_type = $2
              AND interval    = $3
              AND opened_at BETWEEN $4 AND $5
              AND status = 'closed'
              AND pnl_pct IS NOT NULL
            ORDER BY opened_at DESC
            LIMIT 1
        """, e["symbol"], e["sig_type"], e["interval"],
            e["ts"] - timedelta(minutes=2),
            e["ts"] + timedelta(minutes=2))
        if row:
            results.append({**e, "pnl_pct": row["pnl_pct"]})
    await conn.close()
    return results


matched = asyncio.run(fetch_pnl(entries))
print(f"DB ile eşleşen: {len(matched)} sinyal\n")

if len(matched) < 10:
    print("Yeterli veri yok, daha fazla sinyal birikmesi lazım.")
    sys.exit(0)

# buy_pct > 60 → alıcı baskın, < 40 → satıcı baskın
def rapor(rows, baslik):
    if not rows:
        print(f"{baslik}: veri yok")
        return
    pnls = [r["pnl_pct"] for r in rows]
    pos  = sum(1 for p in pnls if p > 0)
    print(f"\n{'='*40}")
    print(f"{baslik}  ({len(rows)} sinyal)")
    print(f"{'='*40}")
    print(f"PnL  ort={statistics.mean(pnls):.2f}%  medyan={statistics.median(pnls):.2f}%")
    print(f"Pozitif: {pos}/{len(rows)} ({pos/len(rows)*100:.0f}%)")

long_signals  = [r for r in matched if r["sig_type"] == "Long"]
short_signals = [r for r in matched if r["sig_type"] == "Short"]

for label, signals, good_threshold, bad_threshold in [
    ("LONG",  long_signals,  60, 40),
    ("SHORT", short_signals, 40, 60),
]:
    good = [r for r in signals if r["buy_pct"] >= good_threshold]
    bad  = [r for r in signals if r["buy_pct"] <= bad_threshold]
    neut = [r for r in signals if bad_threshold < r["buy_pct"] < good_threshold]

    rapor(good, f"{label} — Sinyal yönünde hacim (buy_pct {'≥' if good_threshold==60 else '≤'}{good_threshold})")
    rapor(bad,  f"{label} — Ters yönde hacim     (buy_pct {'≤' if bad_threshold==40 else '≥'}{bad_threshold})")
    rapor(neut, f"{label} — Nötr hacim")

print(f"\nToplam eşleşen: {len(matched)}")
