"""
"Güçlü DO Kırılımı + FVG + Marubozu" stratejisinin tarihsel backtest'i.

Pine v6 indikatörünün 6 kapılı state machine'i birebir kopyalanır:
  T (HTF HA onayı, Markov) · M (SuperTrend SDF güçlü flip) · D (Daily Open temas)
  C (C10/C20 log-RSI delta cross) · F (3 yeşil mum FVG, DO üstü) · B (Marubozu/Engulfing)
  + ADX(14) >= 25 kapısı. Entry: setup ile aynı bar (Pine default).

Pine'ın request.security(lookahead_on) davranışı canlıda "oluşmakta olan HTF mumu"
demektir; burada forming HTF HA barı yalnızca o ana kadarki barlardan kurulur
(geleceğe bakış yok).

Kullanım:
    python scripts/backtest_do_kirilimi.py [--tf 15m] [--symbols N] [--days N]
"""

import argparse
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import pandas_ta_classic as pta

from config import Config

# ── Strateji parametreleri (Pine default'ları) ───────────────────────────────
DO_HOUR          = 3          # Daily Open saati (İstanbul)
SESSION_HOURS    = 24
HTF_MINUTES      = [90, 180, 360, 540, 720]
MIN_ONAY         = 3          # HTF HA bullish eşiği
MARKOV_KEEP      = 0.7
MARKOV_ADD       = 0.3
ST_LEN, ST_MULT  = 10, 3.0
ADX_LEN, ADX_MIN = 14, 25.0
MARU_BODY        = 0.70
MARU_WICK        = 0.20
FVG_INVALIDATE   = True
SL_MULT, TP_MULT = 1.5, 3.0   # canlı risk kuralları
SIM_MAX_BARS     = 1000
HORIZON_BARS     = 50

FUNNEL_KEYS = [
    "session_bar", "t_ok_bar", "sdf_long_bar",
    "c10c20_do_event", "armed_event", "fvg_event",
    "setup_candle", "after_st_gate", "entry",
]


def load_data(tf: str, min_bars: int, max_symbols: int, days: int) -> dict[str, pd.DataFrame]:
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT, dbname=Config.DB_NAME,
        user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    where_days = f"WHERE bucket > NOW() - INTERVAL '{days} days'" if days else ""
    q = f"""
        SELECT symbol, bucket, open, high, low, close, volume
        FROM cagg_{tf} {where_days}
        ORDER BY symbol, bucket
    """
    print(f"[veri] cagg_{tf} yükleniyor...", flush=True)
    df = pd.read_sql(q, conn)
    conn.close()
    print(f"[veri] {len(df):,} satır, {df['symbol'].nunique()} sembol", flush=True)

    out: dict[str, pd.DataFrame] = {}
    for sym, g in df.groupby("symbol"):
        if len(g) < min_bars:
            continue
        g = g.drop(columns="symbol").reset_index(drop=True)
        g[["open", "high", "low", "close", "volume"]] = g[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)
        out[sym] = g
        if max_symbols and len(out) >= max_symbols:
            break
    print(f"[veri] {len(out)} sembol yeterli geçmişe sahip (min {min_bars} bar)", flush=True)
    return out


def heikin_ashi(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray):
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty_like(ha_c)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_c)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    return ha_o, ha_c


def htf_forming_bull(df: pd.DataFrame, tf_min: int) -> np.ndarray:
    """Her taban barında, oluşmakta olan HTF HA mumunun bullish olup olmadığı.
    HTF pencereleri UTC gece yarısına (03:00 lokal) hizalanır — Pine anchor'ı."""
    ts = df["bucket"]
    bucket = (ts - pd.Timedelta(hours=3)).dt.floor(f"{tf_min}min") + pd.Timedelta(hours=3)

    g = df.groupby(bucket)
    o_w = g["open"].transform("first").to_numpy()
    h_w = g["high"].cummax().to_numpy()
    l_w = g["low"].cummin().to_numpy()
    c_w = df["close"].to_numpy()
    ha_close_f = (o_w + h_w + l_w + c_w) / 4.0

    # Tamamlanmış HTF barlarının HA open/close serisi (recursive)
    agg = g.agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"))
    ha_o_full, ha_c_full = heikin_ashi(
        agg["o"].to_numpy(), agg["h"].to_numpy(), agg["l"].to_numpy(), agg["c"].to_numpy()
    )
    # Oluşan barın ha_open'ı = önceki TAMAMLANMIŞ barın (ha_open+ha_close)/2
    prev_ha_o = pd.Series(np.roll(ha_o_full, 1), index=agg.index)
    prev_ha_c = pd.Series(np.roll(ha_c_full, 1), index=agg.index)
    prev_ha_o.iloc[0] = np.nan
    prev_ha_c.iloc[0] = np.nan
    ha_open_f = ((prev_ha_o + prev_ha_c) / 2.0).reindex(bucket).to_numpy()

    return (ha_close_f > ha_open_f) & ~np.isnan(ha_open_f)


def crossover(s: pd.Series, level: float) -> np.ndarray:
    return ((s > level) & (s.shift(1) <= level)).fillna(False).to_numpy()


def prepare(df: pd.DataFrame) -> dict:
    """Vektörize edilebilen her şey: indikatörler, bayraklar, DO/session."""
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy();  c = df["close"].to_numpy()
    n = len(df)

    st = pta.supertrend(df["high"], df["low"], df["close"], length=ST_LEN, multiplier=ST_MULT)
    st_dir = st[f"SUPERTd_{ST_LEN}_{ST_MULT}"].to_numpy()  # 1=long, -1=short

    adx = pta.adx(df["high"], df["low"], df["close"], length=ADX_LEN)[f"ADX_{ADX_LEN}"].to_numpy()
    adx_ok = adx >= ADX_MIN

    atr = pta.atr(df["high"], df["low"], df["close"], length=14).to_numpy()

    rsi = pta.rsi(np.log(df["close"]), length=14)
    d_rsi = rsi.diff()
    c10 = crossover(d_rsi, 10.0)
    c20 = crossover(d_rsi, 20.0)

    # Daily Open (03:00) + session
    ts = df["bucket"]
    day_key = (ts - pd.Timedelta(hours=DO_HOUR)).dt.date
    new_day = (day_key != day_key.shift(1)).to_numpy()
    new_day[0] = True
    do_raw = np.where(new_day, o, np.nan)
    dailyOpen = pd.Series(do_raw).ffill().to_numpy()

    mins_local = ts.dt.hour * 60 + ts.dt.minute
    mins_from_do = ((mins_local - DO_HOUR * 60) % 1440).to_numpy()
    session_ok = mins_from_do < SESSION_HOURS * 60

    # DO koşulları
    do_touch = (l <= dailyOpen) & (h >= dailyOpen)
    prev_c = np.roll(c, 1); prev_c[0] = np.nan
    do_break = (c > dailyOpen) & (prev_c <= dailyOpen)
    do_lift = (l <= dailyOpen) & (c > dailyOpen)
    do_any = do_touch | do_break | do_lift
    c10c20_do = (c10 | c20) & do_any & ~np.isnan(dailyOpen)

    # Mum kalıpları
    rng = h - l
    body = np.abs(c - o)
    up_wick = h - c
    lo_wick = o - l
    with np.errstate(divide="ignore", invalid="ignore"):
        marubozu = (c > o) & (rng > 0) & (body / rng >= MARU_BODY) & \
                   (up_wick / rng <= MARU_WICK) & (lo_wick / rng <= MARU_WICK)
    prev_o = np.roll(o, 1); prev_o[0] = np.nan
    prev_h = np.roll(h, 1); prev_l = np.roll(l, 1)
    engulf = (c > o) & (prev_c < prev_o) & (l < prev_l) & (h > prev_h)

    # 3 yeşil mum FVG
    bull = c > o
    bull1 = np.roll(bull, 2); bull2 = np.roll(bull, 1)
    bull1[:2] = False; bull2[:1] = False
    h2 = np.roll(h, 2); h2[:2] = np.nan
    fvg3 = bull1 & bull2 & bull & (l > h2)
    fvg_above_do = (h2 > dailyOpen) & (l > dailyOpen)
    setup_above_do = (o > dailyOpen) & (c > dailyOpen)

    # T onayı: 5 HTF forming HA bull sayısı + Markov
    bull_count = np.zeros(n, dtype=np.int8)
    for tfm in HTF_MINUTES:
        bull_count += htf_forming_bull(df, tfm).astype(np.int8)
    t_state = np.empty(n)
    s = 0.5
    add_flags = bull_count >= MIN_ONAY
    for i in range(n):
        s = s * MARKOV_KEEP + (MARKOV_ADD if add_flags[i] else 0.0)
        t_state[i] = s
    t_ok = t_state > 0.5

    return dict(
        o=o, h=h, l=l, c=c, n=n, ts=ts.to_numpy(),
        st_dir=st_dir, adx_ok=adx_ok, atr=atr,
        c10c20_do=c10c20_do, session_ok=session_ok, new_day=new_day,
        dailyOpen=dailyOpen, marubozu=marubozu, engulf=engulf,
        fvg3=fvg3, fvg_above_do=fvg_above_do, setup_above_do=setup_above_do,
        t_ok=t_ok, h2=h2,
    )


def run_state_machine(p: dict, funnel: dict) -> list[dict]:
    """Pine bölüm 16-21'in birebir sırası. Entry listesi döner."""
    n = p["n"]
    st_dir = p["st_dir"]

    sdf_last_short_high = np.nan
    sdf_last_long_low = np.nan
    sdf_dir = 0

    do_setup = False
    fvg_mem = False
    fvg_lower = np.nan
    fvg_bar = -1

    entries: list[dict] = []

    for i in range(1, n):
        # ── SDF state (Pine §16: ok kontrolleri, referans güncellemeden ÖNCE) ──
        long_flip = st_dir[i] == 1 and st_dir[i - 1] == -1
        short_flip = st_dir[i] == -1 and st_dir[i - 1] == 1
        if long_flip and not np.isnan(sdf_last_short_high) and p["l"][i] > sdf_last_short_high:
            sdf_dir = 1
        if short_flip and not np.isnan(sdf_last_long_low) and p["h"][i] < sdf_last_long_low:
            sdf_dir = -1
        if short_flip:
            sdf_last_short_high = p["h"][i]
        if long_flip:
            sdf_last_long_low = p["l"][i]
        sdf_long = sdf_dir == 1

        # ── Yeni gün reset (Pine §19 başı) ──
        if p["new_day"][i]:
            do_setup = False
            fvg_mem = False
            fvg_lower = np.nan
            fvg_bar = -1

        sess = p["session_ok"][i]
        t_ok = p["t_ok"][i]

        if sess:
            funnel["session_bar"] += 1
            if t_ok:
                funnel["t_ok_bar"] += 1
                if sdf_long:
                    funnel["sdf_long_bar"] += 1

        # ── Kaynak setup (C10/C20 + DO) ──
        source_ok = sess and sdf_long and t_ok and p["c10c20_do"][i]
        if p["c10c20_do"][i] and sess:
            funnel["c10c20_do_event"] += 1
        if source_ok:
            funnel["armed_event"] += 1
            do_setup = True
            fvg_mem = False
            fvg_lower = np.nan
            fvg_bar = -1

        # ── FVG oluşumu ──
        fvg_core = (do_setup and sess and sdf_long and t_ok
                    and p["fvg3"][i] and p["fvg_above_do"][i])
        if fvg_core:
            funnel["fvg_event"] += 1
            fvg_mem = True
            fvg_lower = p["h2"][i]
            fvg_bar = i

        # ── FVG iptali (alt bant kırılırsa; aynı bar hariç) ──
        if (FVG_INVALIDATE and fvg_mem and not np.isnan(fvg_lower)
                and i > fvg_bar and p["l"][i] < fvg_lower):
            fvg_mem = False
            fvg_lower = np.nan
            fvg_bar = -1

        # ── Setup mumu (Marubozu/Engulfing, DO üstü) ──
        setup_candle = (fvg_mem and sess and sdf_long and t_ok
                        and p["setup_above_do"][i]
                        and (p["marubozu"][i] or p["engulf"][i]))
        if setup_candle:
            funnel["setup_candle"] += 1
            do_setup = False
            fvg_mem = False
            fvg_lower = np.nan
            fvg_bar = -1

            # ── Kapılar: normal ST + ADX (entry aynı bar) ──
            if st_dir[i] == 1:
                funnel["after_st_gate"] += 1
                if p["adx_ok"][i]:
                    funnel["entry"] += 1
                    entries.append(dict(
                        bar=i, ts=p["ts"][i], price=p["c"][i],
                        atr=p["atr"][i], do=p["dailyOpen"][i],
                        pattern="MARU" if p["marubozu"][i] else "ENGULF",
                    ))
    return entries


def simulate_trades(p: dict, entries: list[dict]) -> list[dict]:
    """Canlı risk kuralları: SL=1.5×ATR, TP=3×ATR → trailing. + sabit ufuklar."""
    results = []
    h, l, c = p["h"], p["l"], p["c"]
    n = p["n"]
    for e in entries:
        i0 = e["bar"]
        entry = e["price"]
        atr = e["atr"]
        if not np.isfinite(atr) or atr <= 0:
            continue
        sl = entry - SL_MULT * atr
        tp = entry + TP_MULT * atr
        dist = SL_MULT * atr
        trail = np.nan
        exit_px, exit_reason, exit_bar = None, None, None

        for j in range(i0 + 1, min(i0 + SIM_MAX_BARS, n)):
            if np.isnan(trail):
                if l[j] <= sl:                      # SL önce (muhafazakâr)
                    exit_px, exit_reason, exit_bar = sl, "stop_loss", j
                    break
                if h[j] >= tp:                      # TP → trailing aktive
                    trail = tp - dist
            else:
                new_trail = h[j] - dist
                if new_trail > trail:
                    trail = new_trail
                if l[j] <= trail:
                    exit_px, exit_reason, exit_bar = trail, "trailing_stop", j
                    break
        if exit_px is None:
            j = min(i0 + SIM_MAX_BARS, n) - 1
            exit_px, exit_reason, exit_bar = c[j], "expired", j

        pnl_pct = (exit_px - entry) / entry * 100

        # Sabit ufuklar + MFE/MAE (ATR birimi)
        hor = {}
        for k in (3, 5, 10):
            jj = i0 + k
            hor[f"t{k}_atr"] = (c[jj] - entry) / atr if jj < n else np.nan
        win_end = min(i0 + HORIZON_BARS + 1, n)
        if win_end > i0 + 1:
            mfe = (h[i0 + 1:win_end].max() - entry) / atr
            mae = (l[i0 + 1:win_end].min() - entry) / atr
        else:
            mfe = mae = np.nan

        results.append(dict(
            ts=e["ts"], price=entry, pattern=e["pattern"],
            pnl_pct=pnl_pct, exit_reason=exit_reason,
            bars_held=exit_bar - i0, mfe_atr=mfe, mae_atr=mae, **hor,
        ))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="15m", choices=["5m", "15m"])
    ap.add_argument("--symbols", type=int, default=0, help="0 = hepsi")
    ap.add_argument("--days", type=int, default=0, help="0 = tüm geçmiş")
    args = ap.parse_args()

    tf_min = 5 if args.tf == "5m" else 15
    min_bars = 3000 if args.tf == "5m" else 1000

    data = load_data(args.tf, min_bars, args.symbols, args.days)

    funnel = {k: 0 for k in FUNNEL_KEYS}
    all_trades: list[dict] = []
    sym_entry_counts: dict[str, int] = {}

    for idx, (sym, df) in enumerate(data.items(), 1):
        try:
            p = prepare(df)
            entries = run_state_machine(p, funnel)
            if entries:
                trades = simulate_trades(p, entries)
                for t in trades:
                    t["symbol"] = sym
                all_trades.extend(trades)
                sym_entry_counts[sym] = len(entries)
        except Exception as exc:
            print(f"  [hata] {sym}: {exc}", flush=True)
        if idx % 50 == 0:
            print(f"  {idx}/{len(data)} sembol işlendi, {len(all_trades)} entry", flush=True)

    print("\n" + "=" * 70)
    print(f"SONUÇLAR — {args.tf} | {len(data)} sembol")
    print("=" * 70)

    print("\n── Kapı Hunisi ──")
    for k in FUNNEL_KEYS:
        print(f"  {k:<18} {funnel[k]:>10,}")

    if not all_trades:
        print("\nHiç entry üretilmedi.")
        return

    tr = pd.DataFrame(all_trades)
    out_csv = f"logs/backtest_do_kirilimi_{args.tf}.csv"
    tr.to_csv(out_csv, index=False)

    days_span = (pd.Timestamp(tr["ts"].max()) - pd.Timestamp(tr["ts"].min())).days or 1
    wins = (tr["pnl_pct"] > 0).sum()
    print(f"\n── Trade Özeti ({len(tr)} entry, {days_span} gün, günde {len(tr)/days_span:.1f}) ──")
    print(f"  Win rate           : {wins / len(tr) * 100:.1f}%")
    print(f"  Ortalama PnL       : {tr['pnl_pct'].mean():+.3f}%")
    print(f"  Medyan PnL         : {tr['pnl_pct'].median():+.3f}%")
    g = tr.loc[tr["pnl_pct"] > 0, "pnl_pct"].sum()
    b = -tr.loc[tr["pnl_pct"] < 0, "pnl_pct"].sum()
    print(f"  Profit factor      : {g / b:.2f}" if b > 0 else "  Profit factor      : ∞")
    print(f"  Ort. MFE / MAE     : {tr['mfe_atr'].mean():+.2f} / {tr['mae_atr'].mean():+.2f} ATR")
    print(f"  Ort. tutma süresi  : {tr['bars_held'].mean():.0f} bar")
    print(f"  T+3 / T+5 / T+10   : {tr['t3_atr'].mean():+.3f} / {tr['t5_atr'].mean():+.3f} / {tr['t10_atr'].mean():+.3f} ATR")

    print("\n── Çıkış nedenleri ──")
    for reason, gdf in tr.groupby("exit_reason"):
        print(f"  {reason:<14} n={len(gdf):>5}  ort={gdf['pnl_pct'].mean():+.3f}%  wr={(gdf['pnl_pct']>0).mean()*100:.0f}%")

    print("\n── Pattern bazında ──")
    for pat, gdf in tr.groupby("pattern"):
        print(f"  {pat:<8} n={len(gdf):>5}  ort={gdf['pnl_pct'].mean():+.3f}%  wr={(gdf['pnl_pct']>0).mean()*100:.0f}%")

    top = sorted(sym_entry_counts.items(), key=lambda x: -x[1])[:10]
    print("\n── En çok entry üreten semboller ──")
    for sym, cnt in top:
        print(f"  {sym:<14} {cnt}")

    print(f"\nDetay CSV: {out_csv}")


if __name__ == "__main__":
    main()
