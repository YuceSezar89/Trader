"""
do_open_streak'e "3 ardışık yeşil mumun hacmi de ardışık artıyor mu" şartı
eklense ne olurdu? (21 Tem 2026, kullanıcı sorusu üzerine)

do_open_streak_bt.py'nin BİREBİR AYNI D-open kırılım + ardışık-yeşil-mum
tespit mantığını (do_kirilimi.py::_daily_open ile uyumlu) kullanır — sadece
headline eşik (3 ardışık yeşil, canlı sistemle aynı STREAK_THRESHOLD) için,
streak'i oluşturan 3 barın hacminin de KESİNTİSİZ artıp artmadığına göre
ayrıştırır: vol[i-2] < vol[i-1] < vol[i].

Hedef: sabit 96-bar (24h @ 15m) ileri getiri — do_open_streak'in canlı
MAX_HOLD_HOURS=24 ile aynı ufuk, çıkış mekanizmasından (SL/timeout) bağımsız,
bugünün "temiz yöntem" standardına uygun.

Kullanım: python -m research.pattern_lab.do_open_streak_volume_rising_bt
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import psycopg2  # pylint: disable=wrong-import-position

from config import Config  # pylint: disable=wrong-import-position
from research.pattern_lab.vol_exhaustion_bt import _fwd_returns, _stats  # pylint: disable=wrong-import-position
from signals.do_kirilimi import _daily_open  # pylint: disable=wrong-import-position

DAYS = 45
HORIZON_BARS = 96  # 24h @ 15m — canlı do_open_streak'in MAX_HOLD_HOURS ile aynı
MIN_BARS = 700
STREAK_TH = 3  # canlı STREAK_THRESHOLD ile aynı
_PLACEBO_ITER = 300


def _fetch() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, bucket AS ts, open, high, low, close, volume
        FROM cagg_15m
        WHERE bucket > NOW() - INTERVAL '{DAYS} days'
        ORDER BY symbol, bucket
    """
    df = pd.read_sql(q, conn)
    conn.close()
    return df


def _do_break_gate(o: np.ndarray, c: np.ndarray, daily_open: np.ndarray) -> np.ndarray:
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


def _streak_events_vol_split(
    o: np.ndarray, c: np.ndarray, vol: np.ndarray, gate: np.ndarray
) -> tuple[list[int], list[int]]:
    """STREAK_TH'a İLK ULAŞTIĞI bar indekslerini, streak'i oluşturan barların
    hacmi kesintisiz artıyor mu (vol[i-2]<vol[i-1]<vol[i]) diye ikiye ayırır."""
    n = len(c)
    is_long = c > o
    count_long = 0
    fired = False
    vol_rising_idx: list[int] = []
    vol_not_rising_idx: list[int] = []
    for i in range(n):
        if is_long[i]:
            count_long += 1
        else:
            count_long = 0
            fired = False
            continue
        if not gate[i]:
            continue
        if count_long == STREAK_TH and not fired:
            fired = True
            streak_vols = vol[i - STREAK_TH + 1 : i + 1]
            if np.all(np.diff(streak_vols) > 0):
                vol_rising_idx.append(i)
            else:
                vol_not_rising_idx.append(i)
    return vol_rising_idx, vol_not_rising_idx


def run() -> None:
    df = _fetch()
    print(f"{df['symbol'].nunique()} sembol, {len(df):,} 15m bar ({DAYS} gün)\n")

    rising_fwd, not_rising_fwd = [], []
    rising_ts, not_rising_ts = [], []
    n_syms = 0

    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < MIN_BARS:
            continue
        n_syms += 1

        ts = g["ts"]
        o = g["open"].to_numpy(float)
        c = g["close"].to_numpy(float)
        vol = g["volume"].to_numpy(float)

        daily_open, _ = _daily_open(ts, o)
        gate = _do_break_gate(o, c, daily_open)
        rising_idx, not_rising_idx = _streak_events_vol_split(o, c, vol, gate)

        rising_idx = [i for i in rising_idx if i < len(c) - HORIZON_BARS]
        not_rising_idx = [i for i in not_rising_idx if i < len(c) - HORIZON_BARS]

        rising_fwd.append(_fwd_returns(c, rising_idx, HORIZON_BARS))
        not_rising_fwd.append(_fwd_returns(c, not_rising_idx, HORIZON_BARS))
        rising_ts.extend(ts.iloc[i] for i in rising_idx)
        not_rising_ts.extend(ts.iloc[i] for i in not_rising_idx)

    print(f"analize giren sembol: {n_syms}\n")

    rising_rets = np.concatenate(rising_fwd) if rising_fwd else np.array([])
    not_rising_rets = np.concatenate(not_rising_fwd) if not_rising_fwd else np.array([])

    s_r = _stats(rising_rets)
    s_nr = _stats(not_rising_rets)
    print(f"{'grup':45} {'n':>7} {'WR%':>6} {'ort%':>8} {'PF':>7}")
    print(f"{'hacim de ardışık arttı (vol_rising)':45} {s_r.get('n',0):>7} {s_r.get('wr',0):>6} "
          f"{s_r.get('ort_%',0):>8} {s_r.get('pf',0):>7}")
    print(f"{'hacim ardışık artmadı':45} {s_nr.get('n',0):>7} {s_nr.get('wr',0):>6} "
          f"{s_nr.get('ort_%',0):>8} {s_nr.get('pf',0):>7}")

    pf_r = s_r.get("pf", 0) or 0
    pf_nr = s_nr.get("pf", 0) or 0
    real_gap = pf_r - pf_nr
    print(f"\nPF farkı (vol_rising - not_rising): {real_gap:+.3f}")

    all_rets = np.concatenate([rising_rets, not_rising_rets])
    labels = np.array([True] * len(rising_rets) + [False] * len(not_rising_rets))
    rng = np.random.default_rng(42)
    n_true = len(rising_rets)
    count_ge = 0
    for _ in range(_PLACEBO_ITER):
        perm = rng.permutation(len(all_rets))
        fake = np.zeros(len(all_rets), dtype=bool)
        fake[perm[:n_true]] = True
        pf_t = _stats(all_rets[fake]).get("pf", 0) or 0
        pf_f = _stats(all_rets[~fake]).get("pf", 0) or 0
        if abs(pf_t - pf_f) >= abs(real_gap):
            count_ge += 1
    print(f"placebo: rastgele etiketlemede aynı/daha büyük fark sıklığı: %{count_ge/_PLACEBO_ITER*100:.1f}")

    if len(rising_rets) >= 40:
        rdf = pd.DataFrame({"ts": rising_ts, "ret": rising_rets}).sort_values("ts")
        mid = rdf["ts"].iloc[len(rdf)//2]
        fh = _stats(rdf[rdf["ts"] < mid]["ret"].to_numpy())
        sh = _stats(rdf[rdf["ts"] >= mid]["ret"].to_numpy())
        print(f"\nvol_rising split-period: ilk yarı PF={fh.get('pf','-')} (n={fh.get('n',0)})  "
              f"ikinci yarı PF={sh.get('pf','-')} (n={sh.get('n',0)})")
    else:
        print("\nvol_rising split-period: örneklem yetersiz")


if __name__ == "__main__":
    run()
