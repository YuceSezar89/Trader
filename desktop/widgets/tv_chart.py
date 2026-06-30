"""
TVChart — TradingView lightweight-charts tabanlı grafik widget'ı.

QWebEngineView içinde lightweight-charts v4 çalıştırır.
Yerel JS dosyası kullanılır: desktop/assets/lightweight-charts.js
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import QTimer, QUrl, pyqtSlot
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QSizePolicy

from desktop.theme import COLORS

_ASSETS_DIR = Path(__file__).parent.parent / "assets"
_JS_PATH = _ASSETS_DIR / "lightweight-charts.js"

_EMA_SPAN = 21
_EMA_ALPHA = 2 / (_EMA_SPAN + 1)   # ≈ 0.0909
_RSI_PERIOD = 14
_RSI_ALPHA = 1 / _RSI_PERIOD        # ≈ 0.0714
_ST_PERIOD = 10
_ST_MULT = 3.0

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: 100%; height: 100%; background: {bg}; overflow: hidden; }}
  #chart {{ width: 100%; height: 100%; min-width: 400px; min-height: 300px; }}
</style>
</head>
<body>
<div id="chart"></div>
{js_url}
<script>
const COLORS = {{
  bg:     '{bg}',
  bg2:    '{bg2}',
  border: '{border}',
  muted:  '{muted}',
  green:  '{green}',
  red:    '{red}',
  ema:    '#58a6ff',
  rsi:    '{purple}',
}};

const container = document.getElementById('chart');
let chart, candleSeries, emaSeries, volSeries, rsiSeries, rsi70, rsi30;
let stUpSeries, stDownSeries;
let _priceLines = [];
let _fvgLines = [];
let _smcData = null;
let _smcEnabled = false;
let _smcLines = [];
let _signalMarkers = [];

function initChart() {{
  const w = container.clientWidth;
  const h = container.clientHeight;
  if (w < 10 || h < 10) {{ requestAnimationFrame(initChart); return; }}
  try {{

  chart = LightweightCharts.createChart(container, {{
  width: w,
  height: h,
  layout: {{
    background: {{ type: 'solid', color: COLORS.bg }},
    textColor: COLORS.muted,
    fontSize: 11,
  }},
  grid: {{
    vertLines: {{ color: COLORS.border, style: 1 }},
    horzLines: {{ color: COLORS.border, style: 1 }},
  }},
  crosshair: {{
    vertLine: {{ color: COLORS.muted, labelBackgroundColor: COLORS.bg2 }},
    horzLine: {{ color: COLORS.muted, labelBackgroundColor: COLORS.bg2 }},
  }},
  rightPriceScale: {{ borderColor: COLORS.border }},
  timeScale: {{
    borderColor: COLORS.border,
    timeVisible: true,
    secondsVisible: false,
  }},
  handleScroll: true,
  handleScale: true,
}});

  function safeApply(series, opts) {{
    try {{ const ps = series.priceScale(); if (ps) ps.applyOptions(opts); }}
    catch(e) {{ console.log('priceScale err:', e.message); }}
  }}

  // Candlestick serisi
  candleSeries = chart.addCandlestickSeries({{
    upColor:         COLORS.green,
    downColor:       COLORS.red,
    borderUpColor:   COLORS.green,
    borderDownColor: COLORS.red,
    wickUpColor:     COLORS.green,
    wickDownColor:   COLORS.red,
  }});
  safeApply(candleSeries, {{ scaleMargins: {{ top: 0.05, bottom: 0.35 }} }});

  // EMA çizgisi
  emaSeries = chart.addLineSeries({{
    color: COLORS.ema,
    lineWidth: 1.5,
    priceLineVisible: false,
    lastValueVisible: true,
    crosshairMarkerVisible: false,
  }});

  // Hacim histogramı
  volSeries = chart.addHistogramSeries({{
    priceFormat: {{ type: 'volume' }},
    priceScaleId: 'vol',
  }});
  safeApply(volSeries, {{ scaleMargins: {{ top: 0.80, bottom: 0.18 }}, borderVisible: false }});

  // RSI çizgisi
  rsiSeries = chart.addLineSeries({{
    color: COLORS.rsi,
    lineWidth: 1.5,
    priceLineVisible: false,
    lastValueVisible: true,
    crosshairMarkerVisible: false,
    priceScaleId: 'rsi',
  }});
  safeApply(rsiSeries, {{ scaleMargins: {{ top: 0.82, bottom: 0 }}, borderVisible: false }});

  rsi70 = chart.addLineSeries({{
    color: COLORS.red + '55',
    lineWidth: 1, lineStyle: 2,
    priceLineVisible: false, lastValueVisible: false,
    crosshairMarkerVisible: false, priceScaleId: 'rsi',
  }});
  rsi30 = chart.addLineSeries({{
    color: COLORS.green + '55',
    lineWidth: 1, lineStyle: 2,
    priceLineVisible: false, lastValueVisible: false,
    crosshairMarkerVisible: false, priceScaleId: 'rsi',
  }});

  // Supertrend — uptrend (yeşil) ve downtrend (kırmızı) serileri
  stUpSeries = chart.addLineSeries({{
    color: COLORS.green,
    lineWidth: 2,
    priceLineVisible: false,
    lastValueVisible: false,
    crosshairMarkerVisible: false,
  }});
  safeApply(stUpSeries, {{ scaleMargins: {{ top: 0.05, bottom: 0.35 }} }});

  stDownSeries = chart.addLineSeries({{
    color: COLORS.red,
    lineWidth: 2,
    priceLineVisible: false,
    lastValueVisible: false,
    crosshairMarkerVisible: false,
  }});
  safeApply(stDownSeries, {{ scaleMargins: {{ top: 0.05, bottom: 0.35 }} }});

  window._chartReady = true;
  }} catch(e) {{ console.log('initChart HATA:', e.message, e.stack); }}
}}  // initChart sonu

// Python resizeEvent'i tetikler
function resizeChart(w, h) {{
  try {{ if (chart && w > 10 && h > 10) chart.resize(w, h); }}
  catch(e) {{ console.log('resize err:', e.message); }}
}}

// Container boyut kazanana kadar bekle, sonra başlat
(function tryInit() {{
  const el = document.getElementById('chart');
  if (el.clientWidth > 10 && el.clientHeight > 10) {{ initChart(); }}
  else {{ setTimeout(tryInit, 100); }}
}})();

// Python köprüsü — chart hazır değilse bekle
function loadData(candlesJson, emaJson, volJson, rsiJson, attempt) {{
  attempt = attempt || 0;
  if (!window._chartReady || !candleSeries) {{
    if (attempt < 20) setTimeout(() => loadData(candlesJson, emaJson, volJson, rsiJson, attempt+1), 150);
    return;
  }}
  try {{
    const candles = JSON.parse(candlesJson);
    const ema     = JSON.parse(emaJson);
    const vol     = JSON.parse(volJson);
    const rsi     = JSON.parse(rsiJson);

    candleSeries.setData(candles);
    emaSeries.setData(ema);
    volSeries.setData(vol);
    rsiSeries.setData(rsi);

    if (candles.length > 1) {{
      const t0 = candles[0].time;
      const t1 = candles[candles.length - 1].time;
      rsi70.setData([{{time: t0, value: 70}}, {{time: t1, value: 70}}]);
      rsi30.setData([{{time: t0, value: 30}}, {{time: t1, value: 30}}]);
    }}

    chart.timeScale().fitContent();
    try {{ chart.priceScale('right').applyOptions({{ autoScale: true }}); }} catch(e) {{}}
  }} catch(e) {{
    console.log('loadData error:', e.message);
  }}
}}

function loadSupertrend(stUpJson, stDownJson) {{
  if (!stUpSeries || !stDownSeries) return;
  try {{
    stUpSeries.setData(JSON.parse(stUpJson));
    stDownSeries.setData(JSON.parse(stDownJson));
  }} catch(e) {{ console.log('loadSupertrend error:', e.message); }}
}}

function setSignalMarkers(markersJson, priceLineJson) {{
  _priceLines.forEach(pl => {{ try {{ candleSeries.removePriceLine(pl); }} catch(e) {{}} }});
  _priceLines = [];
  try {{ _signalMarkers = JSON.parse(markersJson); }} catch(e) {{ _signalMarkers = []; }}
  try {{
    JSON.parse(priceLineJson).forEach(l => {{
      const pl = candleSeries.createPriceLine({{
        price: l.price, color: l.color, lineWidth: 1,
        lineStyle: 2, axisLabelVisible: true, title: l.title,
      }});
      _priceLines.push(pl);
    }});
  }} catch(e) {{ console.log('setPriceLines error:', e.message); }}
  _refreshAllMarkers();
}}

function loadFVG(fvgJson) {{
  _fvgLines.forEach(pl => {{ try {{ candleSeries.removePriceLine(pl); }} catch(e) {{}} }});
  _fvgLines = [];
  try {{
    JSON.parse(fvgJson).forEach(z => {{
      const top = candleSeries.createPriceLine({{
        price: z.top, color: z.color, lineWidth: 1,
        lineStyle: 3, axisLabelVisible: false, title: z.label,
      }});
      const bot = candleSeries.createPriceLine({{
        price: z.bot, color: z.color, lineWidth: 1,
        lineStyle: 3, axisLabelVisible: false, title: '',
      }});
      _fvgLines.push(top, bot);
    }});
  }} catch(e) {{ console.log('loadFVG error:', e.message); }}
}}

function clearSignalMarkers() {{
  _signalMarkers = [];
  _priceLines.forEach(pl => {{ try {{ candleSeries.removePriceLine(pl); }} catch(e) {{}} }});
  _priceLines = [];
  _refreshAllMarkers();
}}

function _refreshAllMarkers() {{
  const smcM = (_smcEnabled && _smcData) ? (_smcData.markers || []) : [];
  const all = [..._signalMarkers, ...smcM].sort((a, b) => a.time - b.time);
  try {{ candleSeries.setMarkers(all); }} catch(e) {{}}
}}

function _renderSMC() {{
  _smcLines.forEach(pl => {{ try {{ candleSeries.removePriceLine(pl); }} catch(e) {{}} }});
  _smcLines = [];
  if (_smcEnabled && _smcData) {{
    (_smcData.levels || []).forEach(l => {{
      const pl = candleSeries.createPriceLine({{
        price: l.price, color: l.color, lineWidth: 1,
        lineStyle: 2, axisLabelVisible: true, title: l.title,
      }});
      _smcLines.push(pl);
    }});
  }}
  _refreshAllMarkers();
}}

function loadSMC(dataJson, enabled) {{
  try {{ _smcData = JSON.parse(dataJson); }} catch(e) {{ _smcData = null; }}
  _smcEnabled = enabled;
  _renderSMC();
}}

function toggleSMC(enabled) {{
  _smcEnabled = enabled;
  _renderSMC();
}}

function setPriceFormat(precision, minMove) {{
  if (!candleSeries) return;
  candleSeries.applyOptions({{ priceFormat: {{ type: 'price', precision: precision, minMove: minMove }} }});
  emaSeries.applyOptions({{ priceFormat: {{ type: 'price', precision: precision, minMove: minMove }} }});
}}

function setLogScale(enabled) {{
  if (!chart) return;
  chart.priceScale('right').applyOptions({{ mode: enabled ? 1 : 0 }});
}}

function updateLastBar(candleJson, emaJson, volJson, rsiJson) {{
  if (!window._chartReady || !candleSeries) return;
  try {{
    const candle = JSON.parse(candleJson);
    const ema    = JSON.parse(emaJson);
    const vol    = JSON.parse(volJson);
    const rsi    = JSON.parse(rsiJson);
    candleSeries.update(candle);
    emaSeries.update(ema);
    volSeries.update(vol);
    if (rsi !== null) rsiSeries.update(rsi);
  }} catch(e) {{
    console.log('updateLastBar error:', e.message);
  }}
}}
</script>
</body>
</html>
"""


def _build_html() -> str:
    js_code = _JS_PATH.read_text(encoding="utf-8") if _JS_PATH.exists() else ""
    js_tag = f"<script>{js_code}</script>" if js_code else \
             '<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>'
    return _HTML_TEMPLATE.format(
        bg=COLORS["bg_primary"],
        bg2=COLORS["bg_secondary"],
        border=COLORS["border"],
        muted=COLORS["text_muted"],
        green=COLORS["green"],
        red=COLORS["red"],
        purple=COLORS["purple"],
        js_url=js_tag,
    )


class TVChart(QWebEngineView):
    """
    TradingView lightweight-charts tabanlı grafik widget'ı.

    Kullanım:
        chart = TVChart()
        chart.load_df(df, symbol="BTCUSDT", tf="1h")
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(300)

        self._symbol = ""
        self._tf = ""
        self._ready = False
        self._pending_df: Optional[pd.DataFrame] = None
        self._pending_signal: Optional[dict] = None
        self._bar_state: Optional[dict] = None
        self._smc_enabled = False

        self.loadFinished.connect(self._on_load_finished)
        self.setHtml(_build_html(), QUrl("about:blank"))

    @pyqtSlot(bool)
    def _on_load_finished(self, ok: bool) -> None:
        self._ready = ok
        if ok and self._pending_df is not None:
            df, sym, tf = self._pending_df
            self._pending_df = None
            sig = self._pending_signal
            self._pending_signal = None
            QTimer.singleShot(600, lambda: self._send_data(df, sym, tf, sig))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._ready:
            w, h = event.size().width(), event.size().height()
            if w > 10 and h > 10:
                self.page().runJavaScript(f"resizeChart({w},{h})")

    def load_df(self, df: pd.DataFrame, symbol: str = "", tf: str = "", signal: Optional[dict] = None) -> None:
        self._symbol = symbol
        self._tf = tf
        if df is None or df.empty:
            return
        if not self._ready:
            self._pending_df = (df, symbol, tf)
            self._pending_signal = signal
            return
        self._send_data(df, symbol, tf, signal)

    def update_last_bar(self, df: pd.DataFrame) -> None:
        """Sadece son mumu günceller — fitContent çağırmaz, zoom/pan korunur."""
        if df is None or df.empty or not self._ready:
            return

        ts_series = df["timestamp"].astype("int64") // 10**9 + 3 * 3600

        # Bar[-2] değişmişse (yeni bar kapandı) → state'i yenile
        state_ts = int(ts_series.iloc[-2]) if len(ts_series) >= 2 else None
        if self._bar_state is None or self._bar_state["ts"] != state_ts:
            self._bar_state = self._compute_state(df.tail(70), ts_series.iloc[-2:-1])

        # Son bar için incremental EMA + RSI — 200 bar hesaplama yok
        t = int(ts_series.iloc[-1])
        row = df.iloc[-1]
        o = float(row["open"]); h = float(row["high"])
        l = float(row["low"]);  c = float(row["close"]); v = float(row["volume"])

        ema_val = _EMA_ALPHA * c + (1 - _EMA_ALPHA) * self._bar_state["ema"]

        delta = c - self._bar_state["close"]
        g  = max(delta, 0.0)
        lo = max(-delta, 0.0)
        avg_gain = (1 - _RSI_ALPHA) * self._bar_state["avg_gain"] + _RSI_ALPHA * g
        avg_loss = (1 - _RSI_ALPHA) * self._bar_state["avg_loss"] + _RSI_ALPHA * lo
        rsi_val  = 100.0 - 100.0 / (1.0 + avg_gain / (avg_loss + 1e-9))

        green = COLORS["green"]
        red   = COLORS["red"]
        candle  = {"time": t, "open": o, "high": h, "low": l, "close": c}
        ema_pt  = {"time": t, "value": round(ema_val, 6)}
        vol_pt  = {"time": t, "value": v, "color": (green + "99") if c >= o else (red + "99")}
        rsi_pt  = {"time": t, "value": round(rsi_val, 2)}

        js = (
            f"updateLastBar("
            f"{json.dumps(json.dumps(candle))},"
            f"{json.dumps(json.dumps(ema_pt))},"
            f"{json.dumps(json.dumps(vol_pt))},"
            f"{json.dumps(json.dumps(rsi_pt))})"
        )
        self.page().runJavaScript(js)

    def toggle_smc(self, enabled: bool) -> None:
        self._smc_enabled = enabled
        if self._ready:
            self.page().runJavaScript(f"toggleSMC({'true' if enabled else 'false'})")

    def set_log_scale(self, enabled: bool) -> None:
        self.page().runJavaScript(f"setLogScale({'true' if enabled else 'false'})")

    def set_signal_marker(self, signal_data: Optional[dict]) -> None:
        """Sinyal entry marker + SL/TP price line'larını çizer."""
        if signal_data is None:
            self.page().runJavaScript("clearSignalMarkers()")
            self._pending_signal = None
            return
        if not self._ready:
            self._pending_signal = signal_data
            return
        self._apply_signal_marker(signal_data)

    def clear_signal_marker(self) -> None:
        self._pending_signal = None
        if self._ready:
            self.page().runJavaScript("clearSignalMarkers()")

    def _apply_signal_marker(self, signal_data: dict) -> None:
        sig_type = signal_data.get("signal_type", "LONG")
        opened_at = signal_data.get("opened_at")
        entry = signal_data.get("entry_price")
        sl = signal_data.get("stop_loss_price")
        tp = signal_data.get("take_profit_price")
        trail = signal_data.get("trailing_stop_price")

        green = COLORS["green"]
        red   = COLORS["red"]

        markers = []
        if opened_at is not None and entry is not None:
            try:
                from datetime import datetime  # pylint: disable=import-outside-toplevel
                if isinstance(opened_at, str):
                    opened_at = datetime.fromisoformat(opened_at)
                t = int(opened_at.timestamp()) + 3 * 3600
                is_long = str(sig_type).upper() == "LONG"
                markers.append({
                    "time": t,
                    "position": "belowBar" if is_long else "aboveBar",
                    "color": green if is_long else red,
                    "shape": "arrowUp" if is_long else "arrowDown",
                    "text": f"{'▲' if is_long else '▼'} {float(entry):.4f}",
                    "size": 1,
                })
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        price_lines = []
        sl_price = trail if trail is not None else sl
        sl_label = "Trail" if trail is not None else "SL"
        if sl_price is not None:
            price_lines.append({"price": float(sl_price), "color": red, "title": sl_label})
        if tp is not None and trail is None:
            price_lines.append({"price": float(tp), "color": green, "title": "TP"})

        js = (
            f"setSignalMarkers("
            f"{json.dumps(json.dumps(markers))},"
            f"{json.dumps(json.dumps(price_lines))})"
        )
        self.page().runJavaScript(js)

    def _send_data(self, df: pd.DataFrame, _symbol: str, _tf: str, signal: Optional[dict] = None) -> None:
        candles, ema_data, vol_data, rsi_data = self._prepare(df)
        st_up, st_dn = self._prepare_supertrend(df)
        self._bar_state = self._compute_state(df)

        js = (
            f"loadData("
            f"{json.dumps(json.dumps(candles))},"
            f"{json.dumps(json.dumps(ema_data))},"
            f"{json.dumps(json.dumps(vol_data))},"
            f"{json.dumps(json.dumps(rsi_data))})"
        )
        self.page().runJavaScript(js)

        self.page().runJavaScript(
            f"loadSupertrend("
            f"{json.dumps(json.dumps(st_up))},"
            f"{json.dumps(json.dumps(st_dn))})"
        )

        smc_data = self._prepare_smc(df)
        self.page().runJavaScript(
            f"loadSMC("
            f"{json.dumps(json.dumps(smc_data))},"
            f"{'true' if self._smc_enabled else 'false'})"
        )

        last_price = float(df["close"].iloc[-1])
        precision, min_move = self._price_format(last_price)
        self.page().runJavaScript(f"setPriceFormat({precision}, {min_move})")

        if signal is not None:
            self._apply_signal_marker(signal)
        else:
            self.page().runJavaScript("clearSignalMarkers()")

    @staticmethod
    def _prepare_supertrend(df: pd.DataFrame) -> tuple[list, list]:
        """Supertrend(10,3) — uptrend (yeşil, lower band) ve downtrend (kırmızı, upper band)."""
        try:
            high  = df["high"].astype(float).values
            low   = df["low"].astype(float).values
            close = df["close"].astype(float).values
            ts    = (df["timestamp"].astype("int64") // 10**9 + 3 * 3600).values
            n = len(close)
            if n < _ST_PERIOD + 5:
                return [], []

            # ATR — Wilder's smoothing
            tr = np.zeros(n)
            tr[0] = high[0] - low[0]
            for i in range(1, n):
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr = np.zeros(n)
            atr[_ST_PERIOD - 1] = tr[:_ST_PERIOD].mean()
            for i in range(_ST_PERIOD, n):
                atr[i] = (atr[i-1] * (_ST_PERIOD - 1) + tr[i]) / _ST_PERIOD

            hl2 = (high + low) / 2.0
            bu = hl2 + _ST_MULT * atr
            bl = hl2 - _ST_MULT * atr

            fu = bu.copy()
            fl = bl.copy()
            for i in range(1, n):
                fu[i] = bu[i] if bu[i] < fu[i-1] or close[i-1] > fu[i-1] else fu[i-1]
                fl[i] = bl[i] if bl[i] > fl[i-1] or close[i-1] < fl[i-1] else fl[i-1]

            # Supertrend line: fl = uptrend, fu = downtrend
            st = fu.copy()
            for i in range(1, n):
                if abs(st[i-1] - fu[i-1]) < 1e-12:   # was downtrend
                    st[i] = fl[i] if close[i] > fu[i] else fu[i]
                else:                                   # was uptrend
                    st[i] = fu[i] if close[i] < fl[i] else fl[i]

            st_up, st_dn = [], []
            for i in range(_ST_PERIOD, n):
                t = int(ts[i])
                if abs(st[i] - fl[i]) < 1e-12:   # uptrend
                    st_up.append({"time": t, "value": round(float(fl[i]), 8)})
                else:                              # downtrend
                    st_dn.append({"time": t, "value": round(float(fu[i]), 8)})
            return st_up, st_dn
        except Exception:  # pylint: disable=broad-exception-caught
            return [], []

    @staticmethod
    def _prepare_fvg(df: pd.DataFrame, lookback: int = 50) -> list[dict]:
        """Son `lookback` barda dolmamış FVG zonlarını döner (grafik overlay için)."""
        try:
            high  = df["high"].astype(float).values
            low   = df["low"].astype(float).values
            close = df["close"].astype(float).values
            n = len(high)
            if n < 3:
                return []
            cur_close = close[-1]
            zones: list[dict] = []
            start = max(0, n - lookback)
            for i in range(start + 2, n):
                # Bullish FVG: gap between candle[i-2].high and candle[i].low
                gap_bot = high[i - 2]
                gap_top = low[i]
                if gap_top > gap_bot and cur_close >= gap_bot:
                    zones.append({"top": round(float(gap_top), 8), "bot": round(float(gap_bot), 8), "color": COLORS["green"], "label": "FVG↑"})
                # Bearish FVG: gap between candle[i-2].low and candle[i].high
                gap_top2 = low[i - 2]
                gap_bot2 = high[i]
                if gap_bot2 < gap_top2 and cur_close <= gap_top2:
                    zones.append({"top": round(float(gap_top2), 8), "bot": round(float(gap_bot2), 8), "color": COLORS["red"], "label": "FVG↓"})
            return zones
        except Exception:  # pylint: disable=broad-exception-caught
            return []

    @staticmethod
    def _prepare_smc(df: pd.DataFrame) -> dict:
        """Swing highs/lows + BOS/CHoCH verilerini grafik overlay için hazırlar."""
        try:
            from smartmoneyconcepts import smc as _smc  # pylint: disable=import-outside-toplevel

            ts = (df["timestamp"].astype("int64") // 10**9 + 3 * 3600).values
            df_smc = df[["open", "high", "low", "close", "volume"]].copy().reset_index(drop=True)
            for col in df_smc.columns:
                df_smc[col] = df_smc[col].astype(float)

            swing_df = _smc.swing_highs_lows(df_smc, swing_length=5)
            bos_df   = _smc.bos_choch(df_smc, swing_df, close_break=True)

            green  = COLORS["green"]
            red    = COLORS["red"]
            orange = "#ff9900"
            purple = COLORS["purple"]

            markers = []
            for i in range(len(swing_df)):
                hl = swing_df["HighLow"].iloc[i]
                if np.isnan(hl):
                    continue
                t = int(ts[i])
                if hl == 1.0:
                    markers.append({"time": t, "position": "aboveBar", "color": red,
                                    "shape": "arrowDown", "text": "H", "size": 0.8})
                else:
                    markers.append({"time": t, "position": "belowBar", "color": green,
                                    "shape": "arrowUp", "text": "L", "size": 0.8})

            levels = []
            for i in range(len(bos_df)):
                bos_val   = bos_df["BOS"].iloc[i]
                choch_val = bos_df["CHOCH"].iloc[i]
                lv        = bos_df["Level"].iloc[i]
                if not np.isnan(bos_val) and not np.isnan(lv) and lv > 0:
                    levels.append({"price": float(lv),
                                   "color": green if bos_val == 1 else red,
                                   "title": "BOS↑" if bos_val == 1 else "BOS↓"})
                elif not np.isnan(choch_val) and not np.isnan(lv) and lv > 0:
                    levels.append({"price": float(lv),
                                   "color": orange if choch_val == 1 else purple,
                                   "title": "CHoCH↑" if choch_val == 1 else "CHoCH↓"})

            return {"markers": markers, "levels": levels}
        except Exception:  # pylint: disable=broad-exception-caught
            return {"markers": [], "levels": []}

    @staticmethod
    def _price_format(price: float) -> tuple[int, float]:
        if price >= 1000:  return 2,  0.01
        if price >= 10:    return 3,  0.001
        if price >= 1:     return 4,  0.0001
        if price >= 0.1:   return 5,  0.00001
        if price >= 0.01:  return 6,  0.000001
        if price >= 0.001: return 7,  0.0000001
        if price >= 0.0001:return 8,  0.00000001
        return 10, 0.0000000001

    @staticmethod
    def _compute_state(df: pd.DataFrame, _ts_hint=None) -> dict:
        """Son kapalı bar'ın (bar[-2]) EMA/RSI durumunu döner."""
        closes = df["close"].astype(float).reset_index(drop=True)
        ts     = df["timestamp"].astype("int64") // 10**9 + 3 * 3600
        ema    = closes.ewm(span=_EMA_SPAN, adjust=False).mean()
        delta  = closes.diff()
        gain   = delta.clip(lower=0).ewm(alpha=_RSI_ALPHA, min_periods=_RSI_PERIOD).mean()
        loss   = (-delta.clip(upper=0)).ewm(alpha=_RSI_ALPHA, min_periods=_RSI_PERIOD).mean()
        idx    = -2 if len(closes) >= 2 else -1
        return {
            "ts":       int(ts.iloc[idx]),
            "ema":      float(ema.iloc[idx]),
            "avg_gain": float(gain.iloc[idx]),
            "avg_loss": float(loss.iloc[idx]),
            "close":    float(closes.iloc[idx]),
        }

    @staticmethod
    def _prepare(df: pd.DataFrame):
        ts     = df["timestamp"].astype("int64") // 10**9 + 3 * 3600
        opens  = df["open"].astype(float)
        highs  = df["high"].astype(float)
        lows   = df["low"].astype(float)
        closes = df["close"].astype(float)
        vols   = df["volume"].astype(float)

        ema   = closes.ewm(span=_EMA_SPAN, adjust=False).mean()
        delta = closes.diff()
        gain  = delta.clip(lower=0).ewm(alpha=_RSI_ALPHA, min_periods=_RSI_PERIOD).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=_RSI_ALPHA, min_periods=_RSI_PERIOD).mean()
        rsi   = 100 - 100 / (1 + gain / (loss + 1e-9))

        candles, ema_data, vol_data, rsi_data = [], [], [], []
        green = COLORS["green"]
        red   = COLORS["red"]

        for i in range(len(df)):
            t = int(ts.iloc[i])
            o = float(opens.iloc[i])
            h = float(highs.iloc[i])
            l = float(lows.iloc[i])
            c = float(closes.iloc[i])
            v = float(vols.iloc[i])
            r = float(rsi.iloc[i])

            candles.append({"time": t, "open": o, "high": h, "low": l, "close": c})
            ema_data.append({"time": t, "value": round(float(ema.iloc[i]), 6)})
            vol_data.append({
                "time": t, "value": v,
                "color": (green + "99") if c >= o else (red + "99"),
            })
            if r == r:  # NaN değil
                rsi_data.append({"time": t, "value": round(r, 2)})

        return candles, ema_data, vol_data, rsi_data
