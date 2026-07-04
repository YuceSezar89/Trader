"""
Adım 3 — Özellik çıkarıcı.

SÖZLEŞME: extract_features içinde ilk iş df = df[df.ts < t0].
Hiçbir özellik t0 veya sonrası barı göremez (test_lookahead bunu doğrular).

Pencereler: SHORT = son FEAT_SHORT_H saat (ateşleme),
            LONG  = ondan önceki FEAT_LONG_H − FEAT_SHORT_H saat (bağlam).
Kıyas hep SHORT / LONG — "son saatlerde ne değişti?"
"""
import numpy as np
import pandas as pd

from research.pattern_lab import config as C

SHORT_BARS = C.FEAT_SHORT_H * 12
LONG_BARS = (C.FEAT_LONG_H - C.FEAT_SHORT_H) * 12
DO_HOUR = 3


def _vol_percentile_rank(volume: pd.Series, length: int = 100, smoothing: int = 3) -> pd.Series:
    """Volume Heatmap Oscillator formülü (Pine v6 portu): SMA(3) hacim,
    son `length` bara göre percentile rank (0-100, mavi=soğuk kırmızı=sıcak)."""
    smooth = volume.rolling(smoothing).mean()
    return smooth.rolling(length).apply(lambda w: (w < w[-1]).mean() * 100, raw=True)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(alpha=1 / n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _slope_norm(s: pd.Series) -> float:
    """Basit normalize eğim: (son − ilk) / ortalama."""
    s = s.dropna()
    if len(s) < 2 or s.mean() == 0:
        return np.nan
    return float((s.iloc[-1] - s.iloc[0]) / abs(s.mean()))


def _st_equity(df: pd.DataFrame, length: int = 10, mult: float = 3.0) -> dict:
    """Sanal ST(10,3) flip stratejisi — 'para kazandıran bölge' testi.
    Long-only flip: yön +1 olduğunda al, -1'e dönünce sat."""
    atr = _atr(df, length)
    mid = (df["high"] + df["low"]) / 2
    ub, lb = mid + mult * atr, mid - mult * atr
    close = df["close"].to_numpy()
    f_ub, f_lb = ub.to_numpy().copy(), lb.to_numpy().copy()
    direction = np.ones(len(df), dtype=int)
    for i in range(1, len(df)):
        f_ub[i] = min(ub.iloc[i], f_ub[i - 1]) if close[i - 1] <= f_ub[i - 1] else ub.iloc[i]
        f_lb[i] = max(lb.iloc[i], f_lb[i - 1]) if close[i - 1] >= f_lb[i - 1] else lb.iloc[i]
        if close[i] > f_ub[i - 1]:
            direction[i] = 1
        elif close[i] < f_lb[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    trades = []
    entry = None
    for i in range(1, len(df)):
        if direction[i] == 1 and direction[i - 1] == -1:
            entry = close[i]
        elif direction[i] == -1 and direction[i - 1] == 1 and entry:
            trades.append(close[i] / entry - 1)
            entry = None
    if not trades:
        return {"st_trade_sayisi": 0, "st_son10_pf": np.nan, "st_equity_egim": np.nan}
    eq = pd.Series(trades).add(1).cumprod()
    last = pd.Series(trades[-10:])
    g, b = last[last > 0].sum(), -last[last < 0].sum()
    return {
        "st_trade_sayisi": len(trades),
        "st_son10_pf": float(g / b) if b > 0 else np.inf,
        "st_equity_egim": _slope_norm(eq.tail(10)),
    }


def _htf_ha_count(df: pd.DataFrame) -> int:
    """Hoca merdiveni: 90/180/360/720 dk HTF'lerde HA yeşil sayısı (0-4)."""
    n = 0
    x = df.set_index("ts")
    for minutes in (90, 180, 360, 720):
        agg = x.resample(f"{minutes}min", origin=pd.Timestamp("2026-01-01 03:00:00")).agg(
            o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last")
        ).dropna()
        if len(agg) < 3:
            continue
        ha_c = (agg["o"] + agg["h"] + agg["l"] + agg["c"]) / 4
        ha_o = ha_c.copy()
        ha_o.iloc[0] = (agg["o"].iloc[0] + agg["c"].iloc[0]) / 2
        for i in range(1, len(agg)):
            ha_o.iloc[i] = (ha_o.iloc[i - 1] + ha_c.iloc[i - 1]) / 2
        if ha_c.iloc[-1] > ha_o.iloc[-1]:
            n += 1
    return n


def extract_features(sym_df: pd.DataFrame, btc_close: pd.Series, t0) -> dict:
    """Tek gözlem: (sembolün 5m barları, BTC close serisi, t0) → özellikler.
    SADECE t0 öncesi veriyle çalışır."""
    t0 = pd.Timestamp(t0)
    df = sym_df[sym_df["ts"] < t0].sort_values("ts").reset_index(drop=True)
    btc = btc_close[btc_close.index < t0]
    if len(df) < LONG_BARS + SHORT_BARS + 50:
        return {}

    short, long_ = df.tail(SHORT_BARS), df.iloc[-(SHORT_BARS + LONG_BARS):-SHORT_BARS]
    close, out = df["close"], {}

    # 1 — Δhacim
    v_s, v_l = short["volume"].mean(), long_["volume"].mean()
    out["dvol_oran"] = float(v_s / v_l) if v_l > 0 else np.nan
    out["vol_egim_48h"] = _slope_norm(df["volume"].tail(SHORT_BARS + LONG_BARS).rolling(36).mean())

    # 2 — Sıkışma (yay)
    atr = _atr(df)
    out["atr_daralma"] = float(atr.tail(SHORT_BARS).mean() / atr.iloc[-(SHORT_BARS + LONG_BARS):-SHORT_BARS].mean())
    bw = close.rolling(60).std() / close.rolling(60).mean()
    out["bant_daralma"] = float(bw.tail(SHORT_BARS).mean() / bw.iloc[-(SHORT_BARS + LONG_BARS):-SHORT_BARS].mean())

    # 3 — Momentum
    rsi = _rsi(close)
    out["rsi"] = float(rsi.iloc[-1])
    out["rsi_delta_6h"] = float(rsi.iloc[-1] - rsi.iloc[-SHORT_BARS])
    out["roc_24h"] = float(close.iloc[-1] / close.iloc[-288] - 1) * 100 if len(close) > 288 else np.nan

    # 4 — DO / WO konumu
    day_key = (df["ts"] - pd.Timedelta(hours=DO_HOUR)).dt.date
    do = df.loc[day_key == day_key.iloc[-1], "open"].iloc[0]
    week_key = (df["ts"] - pd.Timedelta(hours=DO_HOUR)).dt.isocalendar().week
    wo = df.loc[week_key == week_key.iloc[-1], "open"].iloc[0]
    out["do_ustu"] = int(close.iloc[-1] > do)
    out["do_mesafe_pct"] = float(close.iloc[-1] / do - 1) * 100
    out["wo_ustu"] = int(close.iloc[-1] > wo)
    out["wo_mesafe_pct"] = float(close.iloc[-1] / wo - 1) * 100

    # 5 — BTC'ye ayrışma (1h/4h/24h)
    for name, bars in (("1h", 12), ("4h", 48), ("24h", 288)):
        if len(close) > bars and len(btc) > bars:
            coin_r = close.iloc[-1] / close.iloc[-bars] - 1
            btc_r = btc.iloc[-1] / btc.iloc[-bars] - 1
            out[f"ayrisma_{name}"] = float(coin_r - btc_r) * 100
        else:
            out[f"ayrisma_{name}"] = np.nan

    # 6 — Verimlilik (devisso/ERSI tarzı, percentile rank)
    price_pct = close.pct_change() * 100
    raw = (price_pct / rsi.diff().replace(0, np.nan)).ewm(span=7, adjust=False).mean().dropna()
    if len(raw) >= 100:
        rec = raw.iloc[-100:]
        out["verimlilik"] = float((rec < raw.iloc[-1]).mean() * 100)
    else:
        out["verimlilik"] = np.nan

    # 7 — ST-equity ("para kazandıran bölge")
    out.update(_st_equity(df))

    # 8 — HTF HA merdiveni
    out["htf_ha_yesil"] = _htf_ha_count(df)

    # 9 — Hacim sönmesi ("satıcı kalmayınca ne olur" — toplam hacim percentile rank,
    # yönlü ayrım YOK, sadece büyüklük; Pine'ın da erişebildiği veri)
    vrank = _vol_percentile_rank(df["volume"])
    win48 = vrank.tail(576)  # son 48h (5m barlarda)
    if win48.notna().sum() >= 50:
        peak = float(win48.max())
        at_t0 = float(vrank.iloc[-1])
        out["hacim_tepe_48h"] = peak
        out["hacim_simdi"] = at_t0
        out["hacim_sonme_derinligi"] = peak - at_t0
        out["hacim_tam_sonme"] = int(peak >= 75 and at_t0 <= 20)
    else:
        out["hacim_tepe_48h"] = out["hacim_simdi"] = out["hacim_sonme_derinligi"] = np.nan
        out["hacim_tam_sonme"] = np.nan

    # 10 — Yönlü akış: gerçek + vekil + sapma (emilim)
    if "buy_volume" in df.columns and df["buy_volume"].notna().any():
        for name, w in (("6h", short), ("48h", df.tail(SHORT_BARS + LONG_BARS))):
            tv = w["volume"].sum()
            out[f"akis_gercek_{name}"] = float((w["buy_volume"] - w["sell_volume"]).sum() / tv) if tv > 0 else np.nan
            hl = (w["high"] - w["low"]).clip(lower=1e-12)
            proxy = (w["volume"] * (2 * w["close"] - w["high"] - w["low"]) / hl).sum()
            out[f"akis_vekil_{name}"] = float(proxy / tv) if tv > 0 else np.nan
        out["akis_sapma_6h"] = out["akis_vekil_6h"] - out["akis_gercek_6h"]
    return out
