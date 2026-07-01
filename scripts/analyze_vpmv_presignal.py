"""
Pre-signal VPMV analizi.

Kullanım:
    python scripts/analyze_vpmv_presignal.py BTCUSDT ETHUSDT SOLUSDT ...

Her sembol için son 24 saatteki sinyalleri alır ve şunları hesaplar:
  - vpmv_signal   : sinyal barındaki VPMV
  - vpmv_pre_avg  : sinyal öncesi 5 bar ortalaması
  - vpmv_slope    : önceki 5 bar'daki eğim (son - ilk)
  - outcome_pct   : sinyal sonrası 4 bar (sinyal TF'ine göre) fiyat değişimi
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, ".")

from binance_client import BinanceClientManager
from database.engine import get_session
from indicators.core import calculate_atr
from utils.preprocessing import (
    normalize_momentum_0_100,
    normalize_price_0_100,
    normalize_volatility_0_100,
    normalize_volume_0_100,
)
from config import Config
from indicators.core import calculate_rsi
from signals.vpm_calculator import VPMCalculator
from sqlalchemy import text

PRE_BARS   = 5   # sinyal öncesi kaç bar
POST_BARS  = 4   # sonuç için kaç bar sonrası


def _compute_vpmv_series(df: pd.DataFrame, signal_type: str) -> pd.Series:
    """Tüm df için bar bazlı VPMV serisi üretir."""
    side = 1.0 if signal_type == "Long" else -1.0

    vol_series  = normalize_volume_0_100(df["volume"])
    rsi_s       = calculate_rsi(df, period=14)
    mom_series  = normalize_momentum_0_100(rsi_s.diff().fillna(0.0) * side)
    atr_s       = calculate_atr(df, period=Config.ATR_PERIOD)
    vlt_series  = normalize_volatility_0_100(atr_s)
    prc_pct     = df["close"].pct_change().fillna(0.0) * 100.0 * side
    prc_series  = normalize_price_0_100(prc_pct)

    scores = (
        0.35 * vol_series +
        0.35 * mom_series +
        0.20 * vlt_series +
        0.10 * prc_series
    ).clip(0, 100)
    return scores


async def fetch_signals(symbols: list[str]) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(hours=72)).replace(tzinfo=None)
    async with get_session() as s:
        rows = await s.execute(text("""
            SELECT symbol, signal_type, interval, opened_at, open_price, vpms_score, indicators
            FROM signals
            WHERE symbol = ANY(:syms)
              AND opened_at >= :since
              AND status IN ('active', 'closed')
            ORDER BY opened_at DESC
        """), {"syms": symbols, "since": since})
        return [dict(r._mapping) for r in rows.all()]


async def analyze(symbols: list[str]) -> None:
    signals = await fetch_signals(symbols)
    if not signals:
        print("Sinyal bulunamadı.")
        return

    tf_limit = {"1m": 500, "5m": 300, "15m": 200, "1h": 150, "4h": 100}

    results = []
    for sig in signals:
        symbol     = sig["symbol"]
        sig_type   = sig["signal_type"]
        interval   = sig["interval"]
        opened_at  = sig["opened_at"]
        open_price = float(sig["open_price"] or 0)
        vpmv_db    = sig["vpms_score"]

        if interval not in tf_limit:
            continue

        limit = tf_limit[interval] + POST_BARS + 10
        try:
            df = await BinanceClientManager.fetch_klines(symbol, interval, limit=limit)
        except Exception as e:
            print(f"  {symbol} kline hatası: {e}")
            continue

        df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

        # Sinyal barını bul
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)

        # opened_at = bar kapanış zamanı → bar open_time = opened_at - interval
        interval_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
        delta = timedelta(minutes=interval_map[interval])
        bar_open = opened_at - delta

        # En yakın bar'ı bul
        df["diff"] = (df["dt"] - bar_open).abs()
        signal_idx = df["diff"].idxmin()
        pos = df.index.get_loc(signal_idx)

        if pos < PRE_BARS + 14:
            continue

        # Tüm df için VPMV serisi
        vpmv_series = _compute_vpmv_series(df, sig_type)
        vals = vpmv_series.values

        # Sinyal barı VPMV
        vpmv_signal = float(vals[pos])

        # Pre-signal: pos-PRE_BARS .. pos-1
        pre_scores = [float(vals[pos - i]) for i in range(PRE_BARS, 0, -1)]
        vpmv_pre_avg = sum(pre_scores) / len(pre_scores)
        vpmv_slope   = pre_scores[-1] - pre_scores[0]

        # Post-signal
        post_pos = pos + POST_BARS
        outcome_pct = None
        post_avg = None
        post_delta = None
        if post_pos < len(df):
            post_price = float(df.iloc[post_pos]["close"])
            outcome_pct = (post_price - open_price) / open_price * 100 if sig_type == "Long" else (open_price - post_price) / open_price * 100
            post_scores = [float(vals[pos + i]) for i in range(1, POST_BARS + 1) if pos + i < len(vals)]
            if post_scores:
                post_avg = sum(post_scores) / len(post_scores)
                post_delta = post_avg - vpmv_signal

        results.append({
            "Sembol":      symbol,
            "Tip":         sig_type,
            "TF":          interval,
            "Saat":        opened_at.strftime("%H:%M"),
            "İnd":         (sig["indicators"] or "").replace("RSI_Cross(9,24)", "RSI").replace("Supertrend(10,3.0)", "ST").replace("MA200_Cross", "MA200"),
            "pre_avg":     round(vpmv_pre_avg, 1),
            "slope":       round(vpmv_slope, 1),
            "→signal":     round(vpmv_signal, 1),
            "post_avg":    round(post_avg, 1) if post_avg is not None else None,
            "post_delta":  round(post_delta, 1) if post_delta is not None else None,
            "sonuç%":      round(outcome_pct, 2) if outcome_pct is not None else None,
            "✓":           ("✅" if outcome_pct and outcome_pct > 0 else "❌") if outcome_pct is not None else "…",
        })

    if not results:
        print("Analiz edilecek veri yok.")
        return

    df_out = pd.DataFrame(results).sort_values(["Sembol", "Saat"])

    pd.set_option("display.max_rows", 300)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.1f}".format)
    print(df_out.to_string(index=False))

    print("\n--- Özet ---")
    ev = df_out.dropna(subset=["sonuç%"])
    for grp, sub in ev.groupby("Tip"):
        wr = (sub["sonuç%"] > 0).mean() * 100
        print(f"  {grp}: {len(sub)} sinyal | pre_avg={sub['pre_avg'].mean():.1f} | signal={sub['→signal'].mean():.1f} | post_avg={sub['post_avg'].mean():.1f} | post_delta={sub['post_delta'].mean():+.1f} | win=%{wr:.0f}")


async def filter_test(symbols: list[str]) -> None:
    signals = await fetch_signals(symbols)
    tf_limit = {"1m": 500, "5m": 300, "15m": 200, "1h": 150, "4h": 100}
    rows = []
    for sig in signals:
        symbol, sig_type, interval = sig["symbol"], sig["signal_type"], sig["interval"]
        opened_at = sig["opened_at"]
        open_price = float(sig["open_price"] or 0)
        if interval not in tf_limit:
            continue
        try:
            df = await BinanceClientManager.fetch_klines(symbol, interval, limit=tf_limit[interval] + POST_BARS + 10)
        except Exception:
            continue
        df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        interval_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
        bar_open = opened_at - timedelta(minutes=interval_map[interval])
        df["diff"] = (df["dt"] - bar_open).abs()
        pos = df.index.get_loc(df["diff"].idxmin())
        if pos < PRE_BARS + 14:
            continue
        vals = _compute_vpmv_series(df, sig_type).values
        vpmv_signal = float(vals[pos])
        pre_scores = [float(vals[pos - i]) for i in range(PRE_BARS, 0, -1)]
        pre_avg = sum(pre_scores) / len(pre_scores)
        slope = pre_scores[-1] - pre_scores[0]
        post_pos = pos + POST_BARS
        outcome = None
        post_avg = None
        if post_pos < len(df):
            post_price = float(df.iloc[post_pos]["close"])
            outcome = (post_price - open_price) / open_price * 100 if sig_type == "Long" else (open_price - post_price) / open_price * 100
            post_scores = [float(vals[pos + i]) for i in range(1, POST_BARS + 1) if pos + i < len(vals)]
            if post_scores:
                post_avg = sum(post_scores) / len(post_scores)
        rows.append({"tip": sig_type, "pre_avg": pre_avg, "vpmv_signal": vpmv_signal, "slope": slope, "post_avg": post_avg, "post_delta": (post_avg - vpmv_signal) if post_avg is not None else None, "outcome": outcome})

    df_all = pd.DataFrame(rows)
    df_ev = df_all.dropna(subset=["outcome"])

    def stats(sub, label):
        if len(sub) == 0:
            return
        wr = (sub["outcome"] > 0).mean() * 100
        avg = sub["outcome"].mean()
        print(f"  {label:40s} {len(sub):3d} sinyal | win=%{wr:4.0f} | ort={avg:+.2f}%")

    print("\n=== FİLTRESİZ ===")
    stats(df_ev[df_ev["tip"] == "Long"],  "Long")
    stats(df_ev[df_ev["tip"] == "Short"], "Short")

    # Post-signal VPMV delta analizi
    df_post = df_ev.dropna(subset=["post_delta"])
    print("\n=== LONG: vpmv_signal SEVİYESİNE GÖRE POST-DELTA ETKİSİ ===")
    print(f"  {'Grup':30s}  {'N':>4}  {'WinRate':>7}  {'Ort%':>7}  {'post_delta ort':>14}")
    longs = df_post[df_post["tip"] == "Long"]
    for lo, hi in [(50, 63), (63, 72), (72, 100)]:
        grp = longs[(longs["vpmv_signal"] >= lo) & (longs["vpmv_signal"] < hi)]
        if len(grp) == 0:
            continue
        wr  = (grp["outcome"] > 0).mean() * 100
        avg = grp["outcome"].mean()
        pd_avg = grp["post_delta"].mean()
        print(f"  signal {lo}-{hi}:  {len(grp):4d} sinyal | win=%{wr:.0f} | ort={avg:+.2f}% | post_delta={pd_avg:+.1f}")
        rising  = grp[grp["post_delta"] > 0]
        falling = grp[grp["post_delta"] <= 0]
        if len(rising):
            print(f"    ↑ post_delta>0 : {len(rising):3d} | win=%{(rising['outcome']>0).mean()*100:.0f} | ort={rising['outcome'].mean():+.2f}%")
        if len(falling):
            print(f"    ↓ post_delta≤0 : {len(falling):3d} | win=%{(falling['outcome']>0).mean()*100:.0f} | ort={falling['outcome'].mean():+.2f}%")

    print(f"\n=== EŞİK KARŞILAŞTIRMA ===")
    print(f"  {'Filtre':40s} {'Adet':>5}  {'WinRate':>8}  {'Ort%':>7}  {'Elenme%':>8}")
    total = len(df_ev)
    for tip in ["Long", "Short"]:
        base = df_ev[df_ev["tip"] == tip]
        for pre_t, sig_t, oran_t in [(0,0,9),(50,45,2.0),(55,50,1.8),(55,55,1.5),(60,55,1.5),(55,55,1.3)]:
            if pre_t == 0:
                m = pd.Series([True]*len(base), index=base.index)
                label = f"{tip} [baseline]"
            else:
                m = (base["pre_avg"]>pre_t) & (base["vpmv_signal"]>sig_t) & (base["vpmv_signal"]/base["pre_avg"]<oran_t)
                label = f"{tip} pre>{pre_t} sig>{sig_t} oran<{oran_t}"
            sub = base[m]
            if len(sub) == 0:
                continue
            wr = (sub["outcome"] > 0).mean() * 100
            avg = sub["outcome"].mean()
            elenme = (1 - len(sub)/len(base)) * 100
            print(f"  {label:40s} {len(sub):5d}  %{wr:6.0f}    {avg:+.2f}%  %{elenme:5.0f} elendi")
        print()


async def timeline(symbols: list[str]) -> None:
    signals = await fetch_signals(symbols)
    tf_limit = {"1m": 500, "5m": 300, "15m": 200, "1h": 150, "4h": 100}
    interval_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}

    # Başlık
    pre_cols  = [f"p{i}" for i in range(PRE_BARS, 0, -1)]
    post_cols = [f"+{i}" for i in range(1, POST_BARS + 1)]
    header = (
        f"{'Sembol':>12} {'Tip':>5} {'TF':>3} {'Saat':>5} {'İnd':>8}  "
        + "  ".join(f"{c:>4}" for c in pre_cols)
        + "  [SİN]"
        + "  ".join(f"{c:>4}" for c in post_cols)
        + "  sonuç%  ✓"
    )
    print(header)
    print("-" * len(header))

    for sig in sorted(signals, key=lambda s: (s["symbol"], s["opened_at"])):
        symbol, sig_type, interval = sig["symbol"], sig["signal_type"], sig["interval"]
        opened_at = sig["opened_at"]
        open_price = float(sig["open_price"] or 0)
        if interval not in tf_limit:
            continue
        try:
            df = await BinanceClientManager.fetch_klines(symbol, interval, limit=tf_limit[interval] + POST_BARS + 10)
        except Exception:
            continue
        df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)
        bar_open = opened_at - timedelta(minutes=interval_map[interval])
        df["diff"] = (df["dt"] - bar_open).abs()
        pos = df.index.get_loc(df["diff"].idxmin())
        if pos < PRE_BARS + 14:
            continue

        vals = _compute_vpmv_series(df, sig_type).values
        pre_vals  = [int(vals[pos - i]) for i in range(PRE_BARS, 0, -1)]
        sig_val   = int(vals[pos])
        post_vals = [int(vals[pos + i]) if pos + i < len(vals) else None for i in range(1, POST_BARS + 1)]

        outcome = None
        post_pos = pos + POST_BARS
        if post_pos < len(df):
            post_price = float(df.iloc[post_pos]["close"])
            outcome = (post_price - open_price) / open_price * 100 if sig_type == "Long" else (open_price - post_price) / open_price * 100

        ind = (sig["indicators"] or "").replace("RSI_Cross(9,24)", "RSI").replace("Supertrend(10,3.0)", "ST").replace("MA200_Cross", "MA200").replace("HA_Cross", "HA")
        pre_str  = "  ".join(f"{v:>4}" for v in pre_vals)
        post_str = "  ".join(f"{v:>4}" if v is not None else "   …" for v in post_vals)
        sonuc    = f"{outcome:+6.1f}%" if outcome is not None else "     …"
        ok       = ("✅" if outcome and outcome > 0 else "❌") if outcome is not None else "…"

        print(
            f"{symbol:>12} {sig_type:>5} {interval:>3} {opened_at.strftime('%H:%M'):>5} {ind:>8}  "
            f"{pre_str}  [{sig_val:>3}]  {post_str}  {sonuc}  {ok}"
        )


if __name__ == "__main__":
    import sys as _sys
    args = _sys.argv[1:]
    flags = {a for a in args if a.startswith("--")}
    symbols = [a.upper() for a in args if not a.startswith("--")]
    if not symbols:
        print("Kullanım: python scripts/analyze_vpmv_presignal.py [--filter|--timeline] BTCUSDT ...")
        _sys.exit(1)
    if "--filter" in flags:
        asyncio.run(filter_test(symbols))
    elif "--timeline" in flags:
        asyncio.run(timeline(symbols))
    else:
        asyncio.run(analyze(symbols))
