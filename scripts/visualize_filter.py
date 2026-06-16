"""
SignalFilter görselleştirme — mum grafiği + Supertrend + filtre referans çizgileri.

Kullanım:
    python scripts/visualize_filter.py --symbol BTCUSDT --interval 1m \
        --from 2026-06-07 --to 2026-06-09
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from signals.signal_filter import SignalFilter


# ── Supertrend ────────────────────────────────────────────────────────────────

def _rma(series: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / length
    result = series.copy().astype(float)
    result.iloc[:length] = np.nan
    result.iloc[length - 1] = series.iloc[:length].mean()
    for i in range(length, len(series)):
        result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i - 1]
    return result


def supertrend(df: pd.DataFrame, atr_length: int = 10, factor: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = _rma(tr, atr_length)
    hl2 = (high + low) / 2
    upper_b = hl2 + factor * atr
    lower_b = hl2 - factor * atr

    upper     = pd.Series(np.nan, index=df.index, dtype=float)
    lower     = pd.Series(np.nan, index=df.index, dtype=float)
    direction = pd.Series(np.nan, index=df.index, dtype=float)

    s = atr_length - 1
    upper.iloc[s]     = upper_b.iloc[s]
    lower.iloc[s]     = lower_b.iloc[s]
    direction.iloc[s] = 1.0

    for i in range(s + 1, len(df)):
        ub, lb = upper_b.iloc[i], lower_b.iloc[i]
        pu, pl = upper.iloc[i - 1], lower.iloc[i - 1]
        pc     = close.iloc[i - 1]

        upper.iloc[i] = ub if ub < pu or pc > pu else pu
        lower.iloc[i] = lb if lb > pl or pc < pl else pl

        pd_ = direction.iloc[i - 1]
        if pd_ == -1:
            direction.iloc[i] = 1.0 if close.iloc[i] < lower.iloc[i] else -1.0
        else:
            direction.iloc[i] = -1.0 if close.iloc[i] > upper.iloc[i] else 1.0

    df["st_line"]     = np.where(direction == -1, lower, upper)
    df["direction"]   = direction
    df["long_signal"] = (direction == -1) & (direction.shift(1) != -1)
    df["short_signal"]= (direction ==  1) & (direction.shift(1) !=  1)
    return df


# ── Veri çekme ────────────────────────────────────────────────────────────────

_CAGG_MAP = {"5m": "cagg_5m", "15m": "cagg_15m", "1h": "cagg_1h", "4h": "cagg_4h"}


def fetch(symbol, interval, from_date, to_date, limit=1000):
    import psycopg2
    conn = psycopg2.connect(
        host=Config.DB_HOST, port=Config.DB_PORT,
        dbname=Config.DB_NAME, user=Config.DB_USER, password=Config.DB_PASSWORD,
    )
    cur = conn.cursor()
    if interval in _CAGG_MAP:
        view = _CAGG_MAP[interval]
        if from_date and to_date:
            cur.execute(
                f"SELECT bucket,open,high,low,close FROM {view} "
                "WHERE symbol=%s AND bucket BETWEEN %s AND %s ORDER BY bucket",
                (symbol, from_date, to_date),
            )
        else:
            cur.execute(
                f"SELECT bucket,open,high,low,close FROM {view} "
                "WHERE symbol=%s ORDER BY bucket DESC LIMIT %s",
                (symbol, limit),
            )
    elif from_date and to_date:
        cur.execute(
            "SELECT timestamp,open,high,low,close FROM price_data "
            "WHERE symbol=%s AND interval=%s AND timestamp BETWEEN %s AND %s ORDER BY timestamp",
            (symbol, interval, from_date, to_date),
        )
    else:
        cur.execute(
            "SELECT timestamp,open,high,low,close FROM price_data "
            "WHERE symbol=%s AND interval=%s ORDER BY timestamp DESC LIMIT %s",
            (symbol, interval, limit),
        )
    rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Istanbul")
    return df.sort_values("timestamp").reset_index(drop=True)


# ── Grafik ────────────────────────────────────────────────────────────────────

def build_chart(df: pd.DataFrame, symbol: str, interval: str,
                atr_length: int, factor: float) -> go.Figure:
    filt = SignalFilter()
    ind  = f"Supertrend({atr_length},{factor})"

    signals = []
    for _, row in df.iterrows():
        if not row["long_signal"] and not row["short_signal"]:
            continue
        sig   = "Long" if row["long_signal"] else "Short"
        state = filt._state.get((symbol, interval, ind))
        ref   = state.last_short_high if sig == "Long" and state else (
                state.last_long_low   if sig == "Short" and state else None)
        valid = filt.check(sig, float(row["high"]), float(row["low"]),
                           symbol, interval, ind)
        signals.append({
            "ts": row["timestamp"], "sig": sig,
            "high": row["high"], "low": row["low"],
            "close": row["close"],
            "ref": ref, "valid": valid,
        })

    fig = make_subplots(rows=1, cols=1)

    # Mumlar
    fig.add_trace(go.Candlestick(
        x=df["timestamp"],
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        name="Fiyat", showlegend=False,
    ))

    # Supertrend çizgisi (bullish yeşil, bearish kırmızı)
    bull = df[df["direction"] == -1]
    bear = df[df["direction"] ==  1]
    fig.add_trace(go.Scatter(
        x=bull["timestamp"], y=bull["st_line"],
        mode="lines", line=dict(color="#26a69a", width=1.5),
        name="ST Bullish", connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=bear["timestamp"], y=bear["st_line"],
        mode="lines", line=dict(color="#ef5350", width=1.5),
        name="ST Bearish", connectgaps=False,
    ))

    # Sinyaller + referans çizgileri
    for s in signals:
        color  = "#26a69a" if s["sig"] == "Long" else "#ef5350"
        opacity= 1.0       if s["valid"]         else 0.3
        symbol_shape = "triangle-up" if s["sig"] == "Long" else "triangle-down"
        y_pos  = s["low"]  * 0.9995 if s["sig"] == "Long" else s["high"] * 1.0005

        label  = ("✓" if s["valid"] else "✗") + " " + s["sig"]

        fig.add_trace(go.Scatter(
            x=[s["ts"]], y=[y_pos],
            mode="markers+text",
            marker=dict(symbol=symbol_shape, size=12, color=color, opacity=opacity),
            text=[label], textposition="bottom center" if s["sig"] == "Long" else "top center",
            textfont=dict(size=9, color=color),
            name=label, showlegend=False,
            hovertemplate=(
                "<b>{} ({})</b><br>H: {:.1f}  L: {:.1f}<br>Ref: {}<extra></extra>".format(
                    s["sig"], "Geçerli" if s["valid"] else "Geçersiz",
                    s["high"], s["low"],
                    f"{s['ref']:.1f}" if s["ref"] is not None else "—",
                )
            ),
        ))

        # Referans yatay çizgi (sinyal anından 20 bar ileriye)
        if s["ref"] is not None:
            idx = df[df["timestamp"] == s["ts"]].index
            if len(idx):
                i     = idx[0]
                end_i = min(i + 20, len(df) - 1)
                x0, x1 = s["ts"], df["timestamp"].iloc[end_i]
                ref_color = "#ef5350" if s["sig"] == "Long" else "#26a69a"
                fig.add_shape(type="line",
                    x0=x0, x1=x1, y0=s["ref"], y1=s["ref"],
                    line=dict(color=ref_color, width=1, dash="dot"),
                )
                fig.add_annotation(
                    x=x1, y=s["ref"],
                    text=f"{s['ref']:.0f}",
                    showarrow=False, font=dict(size=8, color=ref_color),
                    xanchor="left",
                )

    fig.update_layout(
        title=f"{symbol} {interval} — Supertrend({atr_length},{factor}) + SignalFilter",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#1a1a2e", plot_bgcolor="#0f0f1a",
        font=dict(color="#c9d1d9"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        height=700,
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",     default="BTCUSDT")
    parser.add_argument("--interval",   default="1m")
    parser.add_argument("--from",       default=None, dest="from_date")
    parser.add_argument("--to",         default=None, dest="to_date")
    parser.add_argument("--limit",      type=int, default=500)
    parser.add_argument("--atr-length", type=int, default=10)
    parser.add_argument("--factor",     type=float, default=3.0)
    parser.add_argument("--out",        default="/tmp/filter_chart.html")
    args = parser.parse_args()

    print(f"Veri çekiliyor: {args.symbol} {args.interval}")
    df = fetch(args.symbol, args.interval, args.from_date, args.to_date, args.limit)
    df = supertrend(df, args.atr_length, args.factor)
    print(f"  {len(df)} mum, Supertrend hesaplandı.")

    fig = build_chart(df, args.symbol, args.interval, args.atr_length, args.factor)
    fig.write_html(args.out)
    print(f"Grafik kaydedildi: {args.out}")
    import webbrowser
    webbrowser.open(f"file://{args.out}")
