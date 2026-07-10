"""
D-open kırılımı + ardışık yeşil mum → büyük hareket (24h ufkunda) replay backtest.

Kullanıcı gözlemi (9 Tem 2026): gün içi en çok yükselen coinlerde, günlük açılıştan
(D-open) tepki alıp art arda yeşil mum yakma örüntüsü sonrasında büyük hareket
geliyor. Bu, project_pattern_lab.md hafızasındaki 4 Tem'den kalan, hiç test
edilmemiş İKİ ayrı hipotezin birleşimi: "DO temas anı + kırılım" ve "ardışık
yükselen mum sayısı → büyük hareket". Kullanıcının arşivinden bulduğu bir Pine
script (Ardışık Sistemler.txt) ikinci hipotez için ölçülebilir tanımı veriyor:
ardışık yeşil mum sayısı + ilk mumun low'undan şu anki high'a % yükseliş.

D-open ve kırılım tanımı signals/do_kirilimi.py::_daily_open / do_break ile
BİREBİR AYNI (DO_HOUR=3, lokal-naive gün sınırı) — kod tekrarı yerine doğrudan
import edildi (do_kirilimi.py'nin top-level importları hafif, DB'ye dokunmuyor).

Ablasyon: D-open şartı olmadan SADECE ardışık yeşil mum eşiği de ayrıca ölçülüyor
— D-open koşulunun gerçekten katma değeri var mı, yoksa ardışık yeşil mumun
kendisi mi taşıyor, ayrıştırmak için (placebo/ablasyon disiplini, bkz. Pattern
Lab metodoloji dersleri).
"""
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import psycopg2  # pylint: disable=wrong-import-position

from config import Config  # pylint: disable=wrong-import-position
from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _fwd_returns, _stats  # pylint: disable=wrong-import-position

DAYS = 45
HORIZON_BARS = 96  # 24h @ 15m
MIN_BARS = 700
STREAK_THRESHOLDS = [2, 3, 4, 5]


def _fetch() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, open, high, low, close
        FROM cagg_15m
        WHERE bucket > NOW() - INTERVAL '{DAYS} days'
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def _streak_events(o: np.ndarray, c: np.ndarray, gate: Optional[np.ndarray] = None) -> dict[int, list[int]]:
    """Ardışık yeşil mum sayısı her eşiğe İLK ULAŞTIĞI bar indekslerini döner
    (Pine'ın count_long mantığı — kırmızı mumda sayaç sıfırlanır). gate
    verilirse (D-open kırılımından itibaren yeşil mum bozulana kadar True
    kalan maske), sadece gate=True barlarda eşiğe ulaşma olay sayılır."""
    n = len(c)
    is_long = c > o
    count_long = 0
    fired = {th: False for th in STREAK_THRESHOLDS}
    events: dict[int, list[int]] = {th: [] for th in STREAK_THRESHOLDS}
    for i in range(n):
        if is_long[i]:
            count_long += 1
        else:
            count_long = 0
            fired = {th: False for th in STREAK_THRESHOLDS}
            continue
        if gate is not None and not gate[i]:
            continue
        for th in STREAK_THRESHOLDS:
            if count_long == th and not fired[th]:
                events[th].append(i)
                fired[th] = True
    return events


def _do_break_gate(o: np.ndarray, c: np.ndarray, daily_open: np.ndarray) -> np.ndarray:
    """D-open kırılım anından itibaren, ardışık yeşil mum bozulana (kırmızı mum
    gelene) kadar True kalan maske — 'tepki alıp SONRA art arda yeşil mum'un
    'sonra'sını (kesintisiz devam şartını) kodluyor."""
    n = len(c)
    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan
    do_break = (c > daily_open) & (prev_c <= daily_open) & np.isfinite(daily_open)
    is_long = c > o
    gate = np.zeros(n, dtype=bool)
    active = False
    for i in range(n):
        if do_break[i]:
            active = True
        elif not is_long[i]:
            active = False
        gate[i] = active
    return gate


def run():
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    baseline_fwd = []
    do_streak_fwd = {th: [] for th in STREAK_THRESHOLDS}
    plain_streak_fwd = {th: [] for th in STREAK_THRESHOLDS}
    n_syms = 0
    n_do_events = {th: 0 for th in STREAK_THRESHOLDS}
    n_plain_events = {th: 0 for th in STREAK_THRESHOLDS}

    for _sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1

        ts = g["ts"]
        o = g["open"].to_numpy(float)
        c = g["close"].to_numpy(float)

        daily_open, _ = _daily_open(ts, o)
        gate = _do_break_gate(o, c, daily_open)

        do_events = _streak_events(o, c, gate=gate)
        plain_events = _streak_events(o, c, gate=None)

        all_idx = list(range(200, len(c) - HORIZON_BARS, 4))
        baseline_fwd.append(_fwd_returns(c, all_idx, HORIZON_BARS))
        for th in STREAK_THRESHOLDS:
            n_do_events[th] += len(do_events[th])
            n_plain_events[th] += len(plain_events[th])
            do_streak_fwd[th].append(_fwd_returns(c, do_events[th], HORIZON_BARS))
            plain_streak_fwd[th].append(_fwd_returns(c, plain_events[th], HORIZON_BARS))

    baseline_rets = np.concatenate(baseline_fwd) if baseline_fwd else np.array([])
    print(f"analize giren sembol: {n_syms}\n")
    print(f"{'grup':38} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")

    s = _stats(baseline_rets)
    print(f"{'baseline (tüm barlar)':38} {s.get('n',0):>7} {s.get('wr',0):>6} "
          f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}")
    print()

    for th in STREAK_THRESHOLDS:
        s = _stats(np.concatenate(do_streak_fwd[th]) if do_streak_fwd[th] else np.array([]))
        label = f"D-open kırılım + {th} ardışık yeşil"
        print(f"{label:38} {s.get('n',0):>7} {s.get('wr',0):>6} "
              f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}  (olay={n_do_events[th]})")
    print()

    for th in STREAK_THRESHOLDS:
        s = _stats(np.concatenate(plain_streak_fwd[th]) if plain_streak_fwd[th] else np.array([]))
        label = f"[ablasyon] sadece {th} ardışık yeşil"
        print(f"{label:38} {s.get('n',0):>7} {s.get('wr',0):>6} "
              f"{s.get('ort_%',0):>8} {s.get('pf',0):>7}  (olay={n_plain_events[th]})")


if __name__ == "__main__":
    run()
