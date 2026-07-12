"""
HA_Cross sinyallerinde, sinyalin AÇILIŞ barında VPMV'nin Hacim (V) VE Momentum
(M) bileşenlerinin İKİSİ BİRDEN aşırı yüksek olduğu anları (yaklaşık "V=100,
M=100") rejim/filtre olarak test eder — utils/vpmv.py::compute_series ile
BİREBİR AYNI üretim formülü (train/serve skew'den kaçınmak için doğrudan bu
modülün bileşenlerini kullanıyor, kendi kopyasını yazmıyor):

  V (yön-özel): buy_volume (Long) / sell_volume (Short), normalize_volume_0_100
                (rolling-50 log min-max — SERT 0-100 kenetlenmesi, V=100 sık).
  M (yön-özel): RSI(14).diff() × side, normalize_momentum_0_100 (rolling-100
                z-score + SIGMOID — asla tam 100'e değmez, ampirik olarak
                BTCUSDT'de max~99.7, M>=95 sadece barların %0.7'si).

Bu asimetri yüzünden "V=100 VE M=100" tam eşitlikle sorulmaz: V>=99.99 (sert
eşik) VE M>=90 (M'nin kendi dağılımının üst ~%3-5'i — ampirik olarak seçildi,
"tam 100" sigmoid'de pratikte hiç olmuyor).

Veri: price_data (1m, buy_volume/sell_volume — [[project_directional_volume]])
15m'ye toplanmış, sadece temiz kapsamlı dönem (22 Haziran+, DAYS=20).

Look-ahead yok: V/M SADECE geçmiş barlarla (rolling 50/100) hesaplanıyor;
merge_regime'in (opened_at-15dk) kesme mantığıyla bar-kapanış güvenliği korunuyor.
"""
import os
import sys

import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config  # pylint: disable=wrong-import-position
from indicators.core import calculate_rsi  # pylint: disable=wrong-import-position
from research.pattern_lab.rsi_cross_volbreakout_regime_bt import (  # pylint: disable=wrong-import-position
    _fetch_signals, _merge_regime,
)
from research.pattern_lab.threshold_optimizer import _run_single_var_on_df  # pylint: disable=wrong-import-position
from utils.preprocessing import normalize_momentum_0_100, normalize_volume_0_100  # pylint: disable=wrong-import-position

INDICATOR = "HA_Cross"
DAYS = 20  # buy_volume temiz kapsamı 22 Haziran'dan itibaren ([[project_directional_volume]])
BAR_DURATION = pd.Timedelta(minutes=15)
M_THRESHOLD = 90  # ampirik: BTCUSDT'de M>=95 barların sadece %0.7'si, sigmoid pratikte hiç 100'e değmiyor
LOOKBACK_BARS = 8  # ~2 saat @ 15m — V100_kirmizi'de kullanılan pencereyle aynı


def _fetch_15m_with_dir_volume(symbols: list) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    q = f"""
        SELECT symbol, time_bucket('15 minutes', timestamp) AS ts,
               first(open, timestamp) AS open, max(high) AS high, min(low) AS low,
               last(close, timestamp) AS close,
               sum(buy_volume) AS buy_volume, sum(sell_volume) AS sell_volume
        FROM price_data
        WHERE interval = '1m' AND timestamp > NOW() - INTERVAL '{DAYS} days'
          AND symbol = ANY(%s) AND buy_volume IS NOT NULL
        GROUP BY symbol, time_bucket('15 minutes', timestamp)
        ORDER BY symbol, ts
    """
    df = pd.read_sql(q, conn, params=(symbols,))
    conn.close()
    return df


def _vm_joint_series(g: pd.DataFrame, side: float) -> pd.Series:
    """utils/vpmv.py::compute_series ile AYNI V/M formülü — side'a göre yön-özel.
    AYNI bar şartı yerine LOOKBACK_BARS penceresinde her ikisinin de (V ve M,
    farklı barlarda olabilir) en az bir kez aşırı uca değmiş olması aranıyor —
    tam eşleşme çok nadirdi (n<30), pencere gevşetmesiyle örneklem büyütülüyor."""
    dir_vol = g["buy_volume"] if side > 0 else g["sell_volume"]
    vol_score = normalize_volume_0_100(dir_vol)

    rsi = calculate_rsi(g, period=14)
    mom_score = normalize_momentum_0_100(rsi.diff().fillna(0.0) * side)

    v_recent = (vol_score >= 99.99).rolling(LOOKBACK_BARS, min_periods=1).max()
    m_recent = (mom_score >= M_THRESHOLD).rolling(LOOKBACK_BARS, min_periods=1).max()
    joint = (v_recent >= 1) & (m_recent >= 1)
    return joint.astype(float)


def _fetch_regime_vm(symbols: list, side: float, col_name: str) -> pd.DataFrame:
    df = _fetch_15m_with_dir_volume(symbols)
    out = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < 120:
            continue
        state = _vm_joint_series(g, side)
        out.append(pd.DataFrame({"symbol": sym, "ts": g["ts"], col_name: state}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["symbol", "ts", col_name])


def run() -> None:
    for direction, side in (("Long", 1.0), ("Short", -1.0)):
        sig_df = _fetch_signals(INDICATOR, direction)
        if len(sig_df) < 50:
            print(f"{INDICATOR} — {direction}: yetersiz sinyal ({len(sig_df)}), atlanıyor")
            continue

        col_name = "vm_joint_extreme"
        regime_df = _fetch_regime_vm(sig_df["symbol"].unique().tolist(), side, col_name)
        merged = _merge_regime(sig_df, regime_df, col_name, BAR_DURATION)
        print(f"{INDICATOR} — {direction}: {len(sig_df):,} sinyal, {len(merged):,} durumla eşleşti "
              f"(V>=99.99 VE M>={M_THRESHOLD} oranı={merged[col_name].mean():.2%})")

        if merged[col_name].sum() < 30:
            print(f"  Olay sayısı çok az ({int(merged[col_name].sum())}), güvenilir test için yetersiz.\n")
            continue

        label = f"{INDICATOR} — {direction} — V+M birlikte aşırı (V>=99.99, M>={M_THRESHOLD})"
        _run_single_var_on_df(label, merged, col_name)


if __name__ == "__main__":
    run()
