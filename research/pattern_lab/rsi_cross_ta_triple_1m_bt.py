"""
Long ÜÇLÜ (all_up + TA-base + kovalama + HA-hizalı) — 1 DAKİKALIK sinyallerde
test. (24 Tem 2026, kullanıcı isteği.)

1m sinyallerinde `signals.all_up`/`vol_score`/`mom_score`/`volat_score`/
`price_score` DB'de tamamen boş — canlı üretim hattı bu zenginleştirmeyi 1m
için hiç hesaplamıyor. Bu script `all_up`'ı utils/vpmv.py'nin AYNI formülüyle
(vol/mom/vlt/price skorlarının ART ARDA 3 sinyal boyunca — hayır, TEK ADIM,
`rsi_cross_allup_candleshape_clean_bt.py::_add_all_up`'ın basit `all_up`
sütunu — artması) ham 1m price_data'dan kendi hesaplıyor.

1m sinyalleri sadece 18-21 Haziran 2026 arası üretilmiş (292 sembol, n=5055)
— dar bir pencere, sonuçlar buna göre yorumlanmalı.

Kullanım: python -m research.pattern_lab.rsi_cross_ta_triple_1m_bt
"""

import numpy as np
import pandas as pd
import psycopg2

from config import Config
from indicators.core import calculate_atr, calculate_rsi
from research.pattern_lab.do_open_streak_full_clean_bt import _conn
from research.pattern_lab.rsi_cross_combo_realistic_replay_bt import _pnl_usd, _simulate, _stats
from signals.ta_kovalama_gate import _net_ta_series, _percentile_now, _slope_now
from signals.tf_alignment_gate import _heikin_ashi_bull
from utils.preprocessing import (
    normalize_momentum_0_100,
    normalize_price_0_100,
    normalize_volatility_0_100,
    normalize_volume_0_100,
)
from utils.vpmv import directional_volume

_LIVE_KOVALAMA_TH_LONG = 90
_BASE_TH_LONG = 55.0
_MAX_HOLD_BARS_1M = 1440  # 24 saat güvenlik ufku


def _fetch_signals(cur) -> pd.DataFrame:
    cur.execute(
        """
        SELECT symbol, opened_at, open_price
        FROM signals
        WHERE indicators IN ('RSI_Cross(9,24)','HA_Cross') AND interval='1m' AND signal_type='Long'
          AND open_price IS NOT NULL
        ORDER BY symbol, opened_at
        """
    )
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["symbol", "opened_at", "open_price"])


def _fetch_1m(cur, symbol: str) -> pd.DataFrame:
    cur.execute(
        """
        SELECT timestamp AS bucket, open, high, low, close, volume, buy_volume, sell_volume
        FROM price_data
        WHERE symbol=%s AND interval='1m'
          AND timestamp BETWEEN '2026-06-17 00:00:00' AND '2026-06-23 00:00:00'
        ORDER BY timestamp ASC
        """,
        (symbol,),
    )
    rows = cur.fetchall()
    df = pd.DataFrame(
        rows, columns=["bucket", "open", "high", "low", "close", "volume", "buy_volume", "sell_volume"]
    )
    for c in ("open", "high", "low", "close", "volume", "buy_volume", "sell_volume"):
        df[c] = df[c].astype(float)
    return df


def _vpmv_components(df: pd.DataFrame, side: float = 1.0):
    vol = normalize_volume_0_100(directional_volume(df, side))
    rsi = calculate_rsi(df, period=14)
    mom = normalize_momentum_0_100(rsi.diff().fillna(0.0) * side)
    atr = calculate_atr(df, period=Config.ATR_PERIOD)
    vlt = normalize_volatility_0_100(atr)
    prc = normalize_price_0_100(df["close"].pct_change().fillna(0.0) * 100.0 * side)
    return vol.to_numpy(), mom.to_numpy(), vlt.to_numpy(), prc.to_numpy(), atr.to_numpy()


def _ta_ha_at(pg_conn, symbol: str, opened_at) -> "dict | None":
    """1h/4h percentile+slope+HA — cagg_1h/cagg_4h'den, sinyal ANINA kadar
    (gelecek veri sızıntısı yok)."""
    cur = pg_conn.cursor()
    out = {}
    for tf, view, bars_needed in (("1h", "cagg_1h", 220), ("4h", "cagg_4h", 220)):
        cur.execute(
            f"SELECT bucket, open, high, low, close FROM {view} "
            f"WHERE symbol=%s AND bucket <= %s ORDER BY bucket DESC LIMIT %s",
            (symbol, opened_at, bars_needed),
        )
        rows = sorted(cur.fetchall(), key=lambda r: r[0])
        if len(rows) < 53:
            return None
        d = pd.DataFrame(rows, columns=["bucket", "open", "high", "low", "close"])
        for c in ("open", "high", "low", "close"):
            d[c] = d[c].astype(float)
        d["open_time"] = [int(b.timestamp() * 1000) for b in d["bucket"]]
        net = _net_ta_series(d)
        pct = _percentile_now(net)
        slope = _slope_now(net)
        if pct != pct or slope != slope:
            return None
        bull = _heikin_ashi_bull(d)
        out[tf] = (pct, slope, bull)
    return out


def main() -> None:
    conn = _conn()
    cur = conn.cursor()
    sig_df = _fetch_signals(cur)
    print(f"ham 1m Long sinyal: {len(sig_df)} ({sig_df['symbol'].nunique()} sembol)")

    all_up_records = []
    for sym, g in sig_df.groupby("symbol"):
        df1m = _fetch_1m(cur, sym)
        if len(df1m) < 100:
            continue
        vol, mom, vlt, prc, atr = _vpmv_components(df1m, side=1.0)
        bucket = df1m["bucket"].to_numpy()

        g = g.sort_values("opened_at")
        comp_rows = []
        for _, row in g.iterrows():
            idx = np.searchsorted(bucket, np.datetime64(row["opened_at"]), side="right") - 1
            if idx < 60 or idx + 1 >= len(df1m) or np.isnan(atr[idx]) or atr[idx] <= 0:
                continue
            comp_rows.append(
                {"idx": idx, "opened_at": row["opened_at"], "vol": vol[idx], "mom": mom[idx],
                 "vlt": vlt[idx], "prc": prc[idx], "atr": atr[idx]}
            )
        if len(comp_rows) < 2:
            continue
        cdf = pd.DataFrame(comp_rows)
        cdf["d_vol"] = cdf["vol"].diff()
        cdf["d_mom"] = cdf["mom"].diff()
        cdf["d_vlt"] = cdf["vlt"].diff()
        cdf["d_prc"] = cdf["prc"].diff()
        cdf["all_up"] = (cdf["d_vol"] > 0) & (cdf["d_mom"] > 0) & (cdf["d_vlt"] > 0) & (cdf["d_prc"] > 0)
        cdf = cdf.dropna(subset=["d_vol", "d_mom", "d_vlt", "d_prc"])
        for _, r in cdf[cdf["all_up"]].iterrows():
            all_up_records.append(
                {"symbol": sym, "idx": int(r["idx"]), "opened_at": r["opened_at"], "atr": r["atr"]}
            )

    print(f"all_up=True (kendi hesabımızla) 1m Long sinyal: {len(all_up_records)}")
    if not all_up_records:
        print("Yeterli all_up sinyali yok, çıkılıyor.")
        return

    kovalama_records = []
    for rec in all_up_records:
        try:
            ta = _ta_ha_at(conn, rec["symbol"], rec["opened_at"])
        except Exception:  # pylint: disable=broad-exception-caught
            ta = None
        if ta is None:
            continue
        pct_1h, slope_1h, bull_1h = ta["1h"]
        pct_4h, slope_4h, bull_4h = ta["4h"]
        ta_base = pct_1h >= _BASE_TH_LONG and pct_4h >= _BASE_TH_LONG
        kovalama = (pct_1h >= _LIVE_KOVALAMA_TH_LONG and slope_1h > 0) or (
            pct_4h >= _LIVE_KOVALAMA_TH_LONG and slope_4h > 0
        )
        ha_ok = bool(bull_1h) and bool(bull_4h)
        if ta_base and kovalama and ha_ok:
            kovalama_records.append(rec)

    print(f"kovalama üçlüsünü geçen 1m Long sinyal: {len(kovalama_records)}")
    if not kovalama_records:
        print("Yeterli kovalama sinyali yok, çıkılıyor.")
        return

    results = []
    by_symbol: dict[str, list] = {}
    for rec in kovalama_records:
        by_symbol.setdefault(rec["symbol"], []).append(rec)

    for sym, recs in by_symbol.items():
        df1m = _fetch_1m(cur, sym)
        h = df1m["high"].to_numpy()
        l = df1m["low"].to_numpy()
        c = df1m["close"].to_numpy()
        for rec in recs:
            idx = rec["idx"]
            if idx + 1 >= len(c):
                continue
            entry_price = c[idx]
            atr_entry = float(rec["atr"])
            exit_price, reason, bars_held = _simulate(h, l, c, idx + 1, "Long", entry_price, atr_entry)
            # _simulate MAX_HOLD_BARS 500 sabit; 1m'de bu ~8.3 saat, kısa
            # kalabilir ama şimdilik aynı fonksiyonu (tutarlılık) kullanıyoruz
            pnl = _pnl_usd("Long", entry_price, exit_price)
            results.append({"symbol": sym, "opened_at": rec["opened_at"], "pnl_usd": pnl, "reason": reason})

    conn.close()
    df = pd.DataFrame(results)
    print(f"\nGerçekçi replay tamamlanan işlem: {len(df)}")
    print(_stats(df["pnl_usd"].to_numpy()))
    print(f"kapanış nedeni: {df['reason'].value_counts().to_dict()}")

    if len(df) >= 20:
        d = df.sort_values("opened_at")
        mid = d["opened_at"].iloc[len(d) // 2]
        fh = _stats(d[d["opened_at"] < mid]["pnl_usd"].to_numpy())
        sh = _stats(d[d["opened_at"] >= mid]["pnl_usd"].to_numpy())
        print(f"\nsplit-period: ilk yarı {fh} | ikinci yarı {sh}")


if __name__ == "__main__":
    main()
