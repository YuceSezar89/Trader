"""
Ekonomik etki ölçümü — do_break_gauss_bt.py bulgusunun (DO kırılımı + 3 ardışık
yeşil + Gauss büyüklük üst tercili) GERÇEK canlı kullanımda ne kazandıracağını
tahmin eder.

Önceki testlerden (do_break_gauss_bt.py, do_break_gauss_split_check.py) İKİ
FARK var, ikisi de bu testi daha gerçekçi/dürüst kılmak için:

1. **Look-ahead düzeltmesi:** Önceki testlerde Gauss tercil eşiği (q1/q2) 45
   günün TAMAMINDAN hesaplanmıştı — canlıda henüz görülmemiş geleceği kullanmak
   demek. Burada eşik SADECE ilk yarıdan (in-sample) türetilip, SABİT olarak
   ikinci yarıya (out-of-sample) uygulanıyor — "geçmiş veriyle kalibre edip
   bugünden itibaren canlıya alsaydık" senaryosunu simüle ediyor.
2. **$ dönüşümü:** signals/paper_trade_manager.py ile AYNI konvansiyon
   (POSITION_USD=100, FEE_RATE=0.0005/taraf, round-trip=%0.1) kullanılarak
   % getiri → $ P&L'e çevriliyor, aylık/yıllık projeksiyon veriliyor.

Sınır: Bu 24h sabit-tutma (fixed-hold) getirisi — do_kirilimi'nin gerçek
SL/TP/ATR bazlı çıkış mantığından FARKLI, basitleştirilmiş bir ekonomik
tahmin. Gerçek strateji entegrasyonu ayrı bir adım.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _fwd_returns  # pylint: disable=wrong-import-position
from research.pattern_lab.do_open_streak_bt import (  # pylint: disable=wrong-import-position
    DAYS, HORIZON_BARS, MIN_BARS, _fetch, _do_break_gate,
)
from research.pattern_lab.do_open_touch_gauss_bt import (  # pylint: disable=wrong-import-position
    GAUSS_STREAK_THRESHOLD, _gauss_sum, _streak_state, _threshold_events,
)

POSITION_USD = 100.0
FEE_RATE = 0.0005  # signals/paper_trade_manager.py ile aynı
ROUND_TRIP_FEE = POSITION_USD * FEE_RATE * 2


def _dollar_stats(rets: np.ndarray, days_span: float) -> dict:
    if len(rets) == 0:
        return {"n": 0}
    pnl = rets * POSITION_USD - ROUND_TRIP_FEE
    total = float(pnl.sum())
    per_month = total / days_span * 30 if days_span > 0 else 0.0
    return {
        "n": len(rets),
        "wr": round(float((pnl > 0).mean() * 100), 1),
        "avg_usd": round(float(pnl.mean()), 3),
        "total_usd": round(total, 1),
        "usd_per_month": round(per_month, 1),
    }


def run():
    df = _fetch()
    t_min, t_max = df["ts"].min(), df["ts"].max()
    mid = t_min + (t_max - t_min) / 2
    oos_days = (t_max - mid).total_seconds() / 86400
    print(f"dönem: {t_min} .. {t_max} ({DAYS} gün)")
    print(f"kalibrasyon (in-sample): {t_min} .. {mid}")
    print(f"test (out-of-sample):    {mid} .. {t_max}  ({oos_days:.1f} gün)\n")

    # 1. geçiş: in-sample Gauss değerlerini topla → sabit eşiği türet
    is_gauss_vals = []
    per_symbol = []
    n_syms = 0

    for _sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1

        ts = g["ts"]
        ts_np = ts.to_numpy()
        o = g["open"].to_numpy(float)
        h = g["high"].to_numpy(float)
        l = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)

        daily_open, _ = _daily_open(ts, o)
        gate = _do_break_gate(o, c, daily_open)
        count_long, long_perc = _streak_state(o, h, l, c)
        ev3 = _threshold_events(count_long, gate, GAUSS_STREAK_THRESHOLD)
        gauss_perc = _gauss_sum(np.round(long_perc[ev3], 2))
        valid = [(i, gv) for i, gv in zip(ev3, gauss_perc)
                 if np.isfinite(gv) and i < len(c) - HORIZON_BARS]

        is_gauss_vals.extend(gv for i, gv in valid if pd.Timestamp(ts_np[i]) < mid)
        per_symbol.append((ts_np, c, valid))

    oos_threshold = float(np.percentile(is_gauss_vals, 66.7))
    print(f"analiz edilen sembol: {n_syms} | in-sample olay: {len(is_gauss_vals)}")
    print(f"SABİT (out-of-sample'a uygulanan) Gauss eşiği: {oos_threshold:.2f}\n")

    # 2. geçiş: OOS penceresinde sabit eşikle grupla
    oos_high_rets, oos_low_rets, oos_baseline_rets = [], [], []

    for ts_np, c, valid in per_symbol:
        all_idx = list(range(200, len(c) - HORIZON_BARS, 4))
        oos_all_idx = [i for i in all_idx if pd.Timestamp(ts_np[i]) >= mid]
        oos_baseline_rets.append(_fwd_returns(c, oos_all_idx, HORIZON_BARS))

        oos_events = [(i, gv) for i, gv in valid if pd.Timestamp(ts_np[i]) >= mid]
        high_idx = [i for i, gv in oos_events if gv >= oos_threshold]
        low_idx = [i for i, gv in oos_events if gv < oos_threshold]
        oos_high_rets.append(_fwd_returns(c, high_idx, HORIZON_BARS))
        oos_low_rets.append(_fwd_returns(c, low_idx, HORIZON_BARS))

    baseline_rets = np.concatenate(oos_baseline_rets) if oos_baseline_rets else np.array([])
    high_rets = np.concatenate(oos_high_rets) if oos_high_rets else np.array([])
    low_rets = np.concatenate(oos_low_rets) if oos_low_rets else np.array([])

    print(f"── Out-of-sample ekonomik etki (${POSITION_USD:.0f} pozisyon, "
          f"round-trip fee ${ROUND_TRIP_FEE:.2f}) ──")
    print(f"{'grup':32} {'n':>6} {'WR%':>6} {'ort $/işlem':>12} {'toplam $':>10} {'$/ay':>10}")
    for name, rets in (
        ("baseline (tüm barlar)", baseline_rets),
        (f"DO+3yeşil+Gauss YÜKSEK (eşik>={oos_threshold:.1f})", high_rets),
        ("DO+3yeşil+Gauss düşük/orta", low_rets),
    ):
        s = _dollar_stats(rets, oos_days)
        if s.get("n", 0) == 0:
            print(f"{name:32} {'0':>6}")
            continue
        print(f"{name:32} {s['n']:>6} {s['wr']:>6} {s['avg_usd']:>12} "
              f"{s['total_usd']:>10} {s['usd_per_month']:>10}")

    print(f"\nNot: 'ort $/işlem' ve '$/ay' TEK bir ${POSITION_USD:.0f}'lık pozisyonun art arda\n"
          f"aynı sırayla açılıp 24h sonra kapandığı varsayımıyla (kartez/eşzamanlı değil,\n"
          f"basit toplama). Gerçek portföy simülasyonu (eşzamanlı pozisyon limiti, compounding)\n"
          f"kapsam dışı — bu sadece 'ortalama bir işlem ne kazandırır' tahmini.")


if __name__ == "__main__":
    run()
