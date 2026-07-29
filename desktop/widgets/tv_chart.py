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
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QSizePolicy

from desktop.theme import COLORS
from indicators.core import calculate_supertrend

_ASSETS_DIR = Path(__file__).parent.parent / "assets"
_JS_PATH = _ASSETS_DIR / "lightweight-charts.js"

_EMA_SPAN = 21
_EMA_ALPHA = 2 / (_EMA_SPAN + 1)  # ≈ 0.0909
_RSI_PERIOD = 14
_RSI_ALPHA = 1 / _RSI_PERIOD  # ≈ 0.0714
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
let _stSeries = [];

function safeApply(series, opts) {{
  try {{ const ps = series.priceScale(); if (ps) ps.applyOptions(opts); }}
  catch(e) {{ console.log('priceScale err:', e.message); }}
}}
let _priceLines = [];
let _smcData = null;
let _smcEnabled = false;
let _swingEnabled = false;
let _smcPlugin = null;
let _signalMarkers = [];

class SMCLevelPlugin {{
  constructor() {{
    this._levels = [];
    this._series = null;
    this._chart  = null;
  }}
  attached({{ series, chart }}) {{ this._series = series; this._chart = chart; }}
  detached() {{ this._series = null; this._chart = null; }}
  setLevels(levels) {{ this._levels = levels; }}
  updateAllViews() {{}}
  paneViews() {{
    const self = this;
    return [{{
      renderer() {{
        return {{
          draw(target) {{
            if (!self._series || !self._chart || !self._levels.length) return;
            target.useBitmapCoordinateSpace(scope => {{
              const ctx = scope.context;
              const w   = scope.bitmapSize.width;
              const vpr = scope.verticalPixelRatio;
              const hpr = scope.horizontalPixelRatio;
              self._levels.forEach(lv => {{
                const y = self._series.priceToCoordinate(lv.price);
                if (y === null) return;
                const by = y * vpr;
                const cssX = lv.time
                  ? self._chart.timeScale().timeToCoordinate(lv.time)
                  : 0;
                const x0 = cssX !== null ? cssX * hpr : 0;
                ctx.save();
                ctx.globalAlpha = 0.85;
                ctx.strokeStyle = lv.color;
                ctx.lineWidth   = 1.5 * vpr;
                ctx.setLineDash([6 * vpr, 3 * vpr]);
                ctx.beginPath();
                ctx.moveTo(x0, by); ctx.lineTo(w, by);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.globalAlpha = 1.0;
                ctx.fillStyle = lv.color;
                ctx.font = 'bold ' + (10 * vpr) + 'px sans-serif';
                ctx.fillText(lv.title, x0 + 6 * vpr, by - 4 * vpr);
                ctx.restore();
              }});
            }});
          }}
        }};
      }},
      zOrder() {{ return 'normal'; }}
    }}];
  }}
}}
let _fvgData = null;
let _fvgEnabled = false;
let _fvgPlugin = null;

class FVGBoxPlugin {{
  constructor() {{
    this._zones = [];
    this._series = null;
    this._chart  = null;
  }}
  attached({{ series, chart }}) {{ this._series = series; this._chart = chart; }}
  detached() {{ this._series = null; this._chart = null; }}
  setZones(zones) {{ this._zones = zones; }}
  updateAllViews() {{}}
  paneViews() {{
    const self = this;
    return [{{
      renderer() {{
        return {{
          draw(target) {{
            if (!self._series || !self._chart || !self._zones.length) return;
            target.useBitmapCoordinateSpace(scope => {{
              const ctx = scope.context;
              const w   = scope.bitmapSize.width;
              const vpr = scope.verticalPixelRatio;
              const hpr = scope.horizontalPixelRatio;
              self._zones.forEach(z => {{
                const topY = self._series.priceToCoordinate(z.top);
                const botY = self._series.priceToCoordinate(z.bot);
                if (topY === null || botY === null) return;
                const y1 = Math.min(topY, botY) * vpr;
                const y2 = Math.max(topY, botY) * vpr;
                const h  = y2 - y1;
                if (h < 1) return;
                const cssX = z.time
                  ? self._chart.timeScale().timeToCoordinate(z.time)
                  : 0;
                const x0 = cssX !== null ? cssX * hpr : 0;
                ctx.save();
                ctx.globalAlpha = 0.15;
                ctx.fillStyle = z.color;
                ctx.fillRect(x0, y1, w - x0, h);
                ctx.globalAlpha = 0.7;
                ctx.strokeStyle = z.color;
                ctx.lineWidth = vpr;
                ctx.setLineDash([5 * vpr, 3 * vpr]);
                ctx.beginPath();
                ctx.moveTo(x0, y1); ctx.lineTo(w, y1);
                ctx.moveTo(x0, y2); ctx.lineTo(w, y2);
                ctx.stroke();
                ctx.globalAlpha = 0.8;
                ctx.fillStyle = z.color;
                ctx.font = (10 * vpr) + 'px sans-serif';
                ctx.fillText(z.label, x0 + 6 * vpr, y1 + 12 * vpr);
                ctx.restore();
              }});
            }});
          }}
        }};
      }},
      zOrder() {{ return 'bottom'; }}
    }}];
  }}
}}
let _phlData = null;
let _phlEnabled = false;
let _phlLines = [];
let _pivotData = null;
let _pivotEnabled = false;
let _pivotLines = [];
let _wmyData = null;
let _wmyEnabled = false;
let _wmyLines = [];

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
  try {{
    _fvgPlugin = new FVGBoxPlugin();
    candleSeries.attachPrimitive(_fvgPlugin);
  }} catch(e) {{ console.log('FVG plugin attach err:', e.message); }}
  try {{
    _smcPlugin = new SMCLevelPlugin();
    candleSeries.attachPrimitive(_smcPlugin);
  }} catch(e) {{ console.log('SMC plugin attach err:', e.message); }}

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

  // Supertrend serileri loadSupertrend() icinde DINAMIK olarak (her renk
  // bloğu icin ayri bir seri) olusturuluyor - bkz. o fonksiyonun docstring'i.

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

function loadSupertrend(segmentsJson) {{
  // Her renk bloğu (ardisik ayni-yonlu barlar) AYRI bir line series olarak
  // cizilir - segmentler arasinda hicbir nokta paylasilmadigi icin uzak iki
  // noktayi birlestiren capraz cizgi FIZIKSEL OLARAK olusamaz. Onceki "iki
  // sabit seri + whitespace nokta" yontemi bu kutuphane paketinde guvenilir
  // calismiyordu (19 Tem 2026 bug'i - "whitespace" kelimesi bu js dosyasinda
  // hic gecmiyor).
  if (!chart) return;
  try {{
    let _removeFail = 0;
    _stSeries.forEach(s => {{ try {{ chart.removeSeries(s); }} catch(e) {{ _removeFail++; }} }});
    if (_removeFail > 0) console.log('SIZINTI-DEBUG removeSeries basarisiz:', _removeFail, '/', _stSeries.length);
    _stSeries = [];
    const segments = JSON.parse(segmentsJson);
    segments.forEach(seg => {{
      const s = chart.addLineSeries({{
        color: seg.up ? COLORS.green : COLORS.red,
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      }});
      safeApply(s, {{ scaleMargins: {{ top: 0.05, bottom: 0.35 }} }});
      s.setData(seg.points);
      _stSeries.push(s);
    }});
    console.log('SIZINTI-DEBUG loadSupertrend: yeni-segment=' + segments.length + ' _stSeries.length=' + _stSeries.length);
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

function _renderFVG() {{
  if (!_fvgPlugin) return;
  _fvgPlugin.setZones((_fvgEnabled && _fvgData) ? _fvgData : []);
  try {{ chart.applyOptions({{}}); }} catch(e) {{}}
}}

function loadFVG(fvgJson, enabled) {{
  try {{ _fvgData = JSON.parse(fvgJson); }} catch(e) {{ _fvgData = null; }}
  _fvgEnabled = enabled;
  _renderFVG();
}}

function toggleFVG(enabled) {{
  _fvgEnabled = enabled;
  _renderFVG();
}}

function _renderPreviousHL() {{
  _phlLines.forEach(pl => {{ try {{ candleSeries.removePriceLine(pl); }} catch(e) {{}} }});
  _phlLines = [];
  if (!_phlEnabled || !_phlData) return;
  const tfLabel = _phlData.tf ? ' (' + _phlData.tf + ')' : '';
  if (_phlData.high) {{
    _phlLines.push(candleSeries.createPriceLine({{
      price: _phlData.high, color: '#ff6b6b', lineWidth: 1,
      lineStyle: 1, axisLabelVisible: true, title: 'PDH' + tfLabel,
    }}));
  }}
  if (_phlData.low) {{
    _phlLines.push(candleSeries.createPriceLine({{
      price: _phlData.low, color: '#51cf66', lineWidth: 1,
      lineStyle: 1, axisLabelVisible: true, title: 'PDL' + tfLabel,
    }}));
  }}
  if (_phlData.pdo) {{
    _phlLines.push(candleSeries.createPriceLine({{
      price: _phlData.pdo, color: '#fcc419', lineWidth: 1,
      lineStyle: 2, axisLabelVisible: true, title: 'PDO',
    }}));
  }}
  if (_phlData.do_) {{
    _phlLines.push(candleSeries.createPriceLine({{
      price: _phlData.do_, color: '#74c0fc', lineWidth: 1,
      lineStyle: 2, axisLabelVisible: true, title: 'DO',
    }}));
  }}
}}

function loadPreviousHL(dataJson, enabled) {{
  try {{ _phlData = JSON.parse(dataJson); }} catch(e) {{ _phlData = null; }}
  _phlEnabled = enabled;
  _renderPreviousHL();
}}

function togglePreviousHL(enabled) {{
  _phlEnabled = enabled;
  _renderPreviousHL();
}}

function _renderFibPivots() {{
  _pivotLines.forEach(pl => {{ try {{ candleSeries.removePriceLine(pl); }} catch(e) {{}} }});
  _pivotLines = [];
  if (!_pivotEnabled || !_pivotData) return;
  const levels = [
    ['pp', 'PP', '#ffa94d', 0],
    ['r1', 'R1', '#ff6b6b', 2], ['r2', 'R2', '#ff6b6b', 2], ['r3', 'R3', '#ff6b6b', 2],
    ['s1', 'S1', '#51cf66', 2], ['s2', 'S2', '#51cf66', 2], ['s3', 'S3', '#51cf66', 2],
  ];
  levels.forEach(([key, title, color, style]) => {{
    const v = _pivotData[key];
    if (v === undefined || v === null) return;
    _pivotLines.push(candleSeries.createPriceLine({{
      price: v, color: color, lineWidth: 1,
      lineStyle: style, axisLabelVisible: true, title: title,
    }}));
  }});
}}

function loadFibPivots(dataJson, enabled) {{
  try {{ _pivotData = JSON.parse(dataJson); }} catch(e) {{ _pivotData = null; }}
  _pivotEnabled = enabled;
  _renderFibPivots();
}}

function toggleFibPivots(enabled) {{
  _pivotEnabled = enabled;
  _renderFibPivots();
}}

function _renderWMY() {{
  _wmyLines.forEach(pl => {{ try {{ candleSeries.removePriceLine(pl); }} catch(e) {{}} }});
  _wmyLines = [];
  if (!_wmyEnabled || !_wmyData) return;
  const levels = [
    ['w', 'W-Open', '#e599f7', 2],
    ['m', 'M-Open', '#da77f2', 2],
    ['y', 'Y-Open', '#be4bdb', 2],
  ];
  levels.forEach(([key, title, color, style]) => {{
    const v = _wmyData[key];
    if (v === undefined || v === null) return;
    _wmyLines.push(candleSeries.createPriceLine({{
      price: v, color: color, lineWidth: 1,
      lineStyle: style, axisLabelVisible: true, title: title,
    }}));
  }});
}}

function loadWMY(dataJson, enabled) {{
  try {{ _wmyData = JSON.parse(dataJson); }} catch(e) {{ _wmyData = null; }}
  _wmyEnabled = enabled;
  _renderWMY();
}}

function toggleWMY(enabled) {{
  _wmyEnabled = enabled;
  _renderWMY();
}}

function clearSignalMarkers() {{
  _signalMarkers = [];
  _priceLines.forEach(pl => {{ try {{ candleSeries.removePriceLine(pl); }} catch(e) {{}} }});
  _priceLines = [];
  _refreshAllMarkers();
}}

function _refreshAllMarkers() {{
  const swingM = (_swingEnabled && _smcData) ? (_smcData.markers || []) : [];
  const all = [..._signalMarkers, ...swingM].sort((a, b) => a.time - b.time);
  try {{ candleSeries.setMarkers(all); }} catch(e) {{}}
}}

function _renderSMC() {{
  if (_smcPlugin) {{
    _smcPlugin.setLevels((_smcEnabled && _smcData) ? (_smcData.levels || []) : []);
    try {{ chart.applyOptions({{}}); }} catch(e) {{}}
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

function toggleSwing(enabled) {{
  _swingEnabled = enabled;
  _refreshAllMarkers();
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
    js_tag = (
        f"<script>{js_code}</script>"
        if js_code
        else '<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>'
    )
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


class _ConsoleLoggingPage(QWebEnginePage):
    """JS console.log/error mesajlarını Python loguna akıtır — normalde
    Qt WebEngine'in kendi (görünmeyen) konsoluna gidiyordu, JS hatalarını
    teşhis edemiyorduk (19 Tem 2026, SuperTrend segment bug'ı sırasında
    eklendi)."""

    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):  # noqa: N802
        import logging

        logging.getLogger("TVChart.JS").info("[JS L%d] %s", lineNumber, message)


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
        self._swing_enabled = False
        self._fvg_enabled = False
        self._phl_enabled = False
        self._pivot_enabled = False
        self._wmy_enabled = False

        self.setPage(_ConsoleLoggingPage(self))
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

    def load_df(
        self, df: pd.DataFrame, symbol: str = "", tf: str = "", signal: Optional[dict] = None
    ) -> None:
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
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        v = float(row["volume"])

        ema_val = _EMA_ALPHA * c + (1 - _EMA_ALPHA) * self._bar_state["ema"]

        delta = c - self._bar_state["close"]
        g = max(delta, 0.0)
        lo = max(-delta, 0.0)
        avg_gain = (1 - _RSI_ALPHA) * self._bar_state["avg_gain"] + _RSI_ALPHA * g
        avg_loss = (1 - _RSI_ALPHA) * self._bar_state["avg_loss"] + _RSI_ALPHA * lo
        rsi_val = 100.0 - 100.0 / (1.0 + avg_gain / (avg_loss + 1e-9))

        green = COLORS["green"]
        red = COLORS["red"]
        candle = {"time": t, "open": o, "high": h, "low": l, "close": c}
        ema_pt = {"time": t, "value": round(ema_val, 6)}
        vol_pt = {"time": t, "value": v, "color": (green + "99") if c >= o else (red + "99")}
        rsi_pt = {"time": t, "value": round(rsi_val, 2)}

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

    def toggle_swing(self, enabled: bool) -> None:
        self._swing_enabled = enabled
        if self._ready:
            self.page().runJavaScript(f"toggleSwing({'true' if enabled else 'false'})")

    def toggle_fvg(self, enabled: bool) -> None:
        self._fvg_enabled = enabled
        if self._ready:
            self.page().runJavaScript(f"toggleFVG({'true' if enabled else 'false'})")

    def toggle_phl(self, enabled: bool) -> None:
        self._phl_enabled = enabled
        if self._ready:
            self.page().runJavaScript(f"togglePreviousHL({'true' if enabled else 'false'})")

    def toggle_pivots(self, enabled: bool) -> None:
        self._pivot_enabled = enabled
        if self._ready:
            self.page().runJavaScript(f"toggleFibPivots({'true' if enabled else 'false'})")

    def toggle_wmy(self, enabled: bool) -> None:
        self._wmy_enabled = enabled
        if self._ready:
            self.page().runJavaScript(f"toggleWMY({'true' if enabled else 'false'})")

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
        exit_price = signal_data.get("exit_price")
        closed_at = signal_data.get("closed_at")
        close_reason = signal_data.get("close_reason")

        green = COLORS["green"]
        red = COLORS["red"]

        markers = []
        if opened_at is not None and entry is not None:
            try:
                from datetime import datetime  # pylint: disable=import-outside-toplevel

                if isinstance(opened_at, str):
                    opened_at = datetime.fromisoformat(opened_at)
                t = int(opened_at.timestamp()) + 3 * 3600
                is_long = str(sig_type).upper() == "LONG"
                markers.append(
                    {
                        "time": t,
                        "position": "belowBar" if is_long else "aboveBar",
                        "color": green if is_long else red,
                        "shape": "arrowUp" if is_long else "arrowDown",
                        "text": f"{'▲' if is_long else '▼'} {float(entry):.4f}",
                        "size": 1,
                    }
                )
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        # Çıkış marker'ı — kapanmış işlemler için (paper_trade_panel'in
        # Kapalı İşlemler tablosundan tıklanınca) entry marker'ının yanına
        # ikinci bir işaret. Kazanç/kayıp entry-exit fiyat farkından, entry
        # marker'ının TERSİ konumda (çıkış = pozisyonu kapatma yönü).
        if closed_at is not None and exit_price is not None and entry is not None:
            try:
                from datetime import datetime  # pylint: disable=import-outside-toplevel

                if isinstance(closed_at, str):
                    closed_at = datetime.fromisoformat(closed_at)
                t_exit = int(closed_at.timestamp()) + 3 * 3600
                is_long = str(sig_type).upper() == "LONG"
                is_win = (
                    float(exit_price) > float(entry)
                    if is_long
                    else float(exit_price) < float(entry)
                )
                reason_txt = f" ({close_reason})" if close_reason else ""
                markers.append(
                    {
                        "time": t_exit,
                        "position": "aboveBar" if is_long else "belowBar",
                        "color": green if is_win else red,
                        "shape": "circle",
                        "text": f"✕ {float(exit_price):.4f}{reason_txt}",
                        "size": 1,
                    }
                )
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

    def _send_data(
        self, df: pd.DataFrame, _symbol: str, _tf: str, signal: Optional[dict] = None
    ) -> None:
        candles, ema_data, vol_data, rsi_data = self._prepare(df)
        st_segments = self._prepare_supertrend(df)
        self._bar_state = self._compute_state(df)

        js = (
            f"loadData("
            f"{json.dumps(json.dumps(candles))},"
            f"{json.dumps(json.dumps(ema_data))},"
            f"{json.dumps(json.dumps(vol_data))},"
            f"{json.dumps(json.dumps(rsi_data))})"
        )
        self.page().runJavaScript(js)

        self.page().runJavaScript(f"loadSupertrend({json.dumps(json.dumps(st_segments))})")

        smc_data = self._prepare_smc(df)
        self.page().runJavaScript(
            f"loadSMC("
            f"{json.dumps(json.dumps(smc_data))},"
            f"{'true' if self._smc_enabled else 'false'})"
        )

        fvg_data = self._prepare_fvg(df)
        self.page().runJavaScript(
            f"loadFVG("
            f"{json.dumps(json.dumps(fvg_data))},"
            f"{'true' if self._fvg_enabled else 'false'})"
        )

        phl_data = self._prepare_previous_hl(df, self._tf)
        self.page().runJavaScript(
            f"loadPreviousHL("
            f"{json.dumps(json.dumps(phl_data))},"
            f"{'true' if self._phl_enabled else 'false'})"
        )

        pivot_data = self._prepare_fib_pivots(df, self._tf)
        self.page().runJavaScript(
            f"loadFibPivots("
            f"{json.dumps(json.dumps(pivot_data))},"
            f"{'true' if self._pivot_enabled else 'false'})"
        )

        wmy_data = self._prepare_wmy_open(df)
        self.page().runJavaScript(
            f"loadWMY("
            f"{json.dumps(json.dumps(wmy_data))},"
            f"{'true' if self._wmy_enabled else 'false'})"
        )

        last_price = float(df["close"].iloc[-1])
        precision, min_move = self._price_format(last_price)
        self.page().runJavaScript(f"setPriceFormat({precision}, {min_move})")

        if signal is not None:
            self._apply_signal_marker(signal)
        else:
            self.page().runJavaScript("clearSignalMarkers()")

    @staticmethod
    def _prepare_supertrend(df: pd.DataFrame) -> list[dict]:
        """Supertrend(10,3) — indicators.core.calculate_supertrend ile
        HESAPLANIYOR (projenin tek doğrulanmış implementasyonu,
        signals/do_kirilimi.py da aynı kaynağı kullanıyor).

        Ardışık aynı-yönlü barlar SEGMENTLERE ayrılır, her segment JS
        tarafında AYRI bir line series olarak çizilir (19 Tem 2026 fix) —
        önceki iki-sabit-seri + "whitespace nokta" yöntemi bu projede
        paketli lightweight-charts sürümünde güvenilir çalışmıyordu ("çapraz
        çizgi" bug'ı: uzak iki noktanın düz çizgiyle birleşmesi). Segmentler
        arasında hiçbir nokta paylaşılmadığı için bu artık yapısal olarak
        imkansız."""
        try:
            if len(df) < _ST_PERIOD + 5:
                return []
            st_line, direction, _, _ = calculate_supertrend(
                df, atr_length=_ST_PERIOD, factor=_ST_MULT
            )
            ts = (df["timestamp"].astype("int64") // 10**9 + 3 * 3600).values

            segments: list[dict] = []
            current_up: Optional[bool] = None
            current_points: list[dict] = []
            for i in range(len(df)):
                d = direction.iloc[i]
                if pd.isna(d):
                    continue
                is_up = bool(d == -1)  # bullish/uptrend — numpy.bool_ değil, native bool (json.dumps için)
                if is_up != current_up:
                    if current_points:
                        segments.append({"up": current_up, "points": current_points})
                    current_points = []
                    current_up = is_up
                current_points.append(
                    {"time": int(ts[i]), "value": round(float(st_line.iloc[i]), 8)}
                )
            if current_points:
                segments.append({"up": current_up, "points": current_points})
            return segments
        except Exception:  # pylint: disable=broad-exception-caught
            return []

    @staticmethod
    def _prepare_fvg(df: pd.DataFrame) -> list[dict]:
        """Aktif (dolmamış) FVG zonlarını kütüphane ile tespit eder."""
        try:
            from smartmoneyconcepts import smc as _smc  # pylint: disable=import-outside-toplevel

            ts = (df["timestamp"].astype("int64") // 10**9 + 3 * 3600).values
            df_smc = df[["open", "high", "low", "close", "volume"]].copy().reset_index(drop=True)
            for col in df_smc.columns:
                df_smc[col] = df_smc[col].astype(float)

            fvg_df = _smc.fvg(df_smc)
            green = COLORS["green"]
            red = COLORS["red"]
            zones: list[dict] = []
            for i in range(len(fvg_df)):
                fvg_val = fvg_df["FVG"].iloc[i]
                if np.isnan(fvg_val):
                    continue
                mit = fvg_df["MitigatedIndex"].iloc[i]
                if not np.isnan(mit) and mit > 0:
                    continue  # dolduruldu
                top = float(fvg_df["Top"].iloc[i])
                bot = float(fvg_df["Bottom"].iloc[i])
                t = int(ts[i]) if i < len(ts) else None
                if fvg_val == 1:
                    zones.append(
                        {"top": top, "bot": bot, "color": green, "label": "FVG↑", "time": t}
                    )
                else:
                    zones.append({"top": top, "bot": bot, "color": red, "label": "FVG↓", "time": t})
            return zones
        except Exception:  # pylint: disable=broad-exception-caught
            return []

    @staticmethod
    def _prepare_previous_hl(df: pd.DataFrame, tf: str = "1h") -> dict:
        """TF'e göre önceki periyodun High/Low + sabit günlük DO/PDO seviyelerini hesaplar."""
        try:
            from smartmoneyconcepts import smc as _smc  # pylint: disable=import-outside-toplevel

            _TF_MAP = {
                "1m": "1h",
                "5m": "1h",
                "15m": "4h",
                "1h": "1D",
                "4h": "1W",
                "1d": "1W",
            }
            resample_tf = _TF_MAP.get(tf, "1D")

            df_smc = df[["open", "high", "low", "close", "volume"]].copy()
            df_smc.index = pd.to_datetime(df["timestamp"].values, utc=True)
            for col in df_smc.columns:
                df_smc[col] = df_smc[col].astype(float)

            phl_df = _smc.previous_high_low(df_smc, time_frame=resample_tf)

            ph = phl_df["PreviousHigh"].dropna()
            pl = phl_df["PreviousLow"].dropna()
            if ph.empty or pl.empty:
                return {"high": None, "low": None, "pdo": None, "do_": None, "tf": resample_tf}

            # DO / PDO — her zaman günlük (D), TF'ten bağımsız
            daily_open = df_smc["open"].resample("D").first().dropna()
            pdo = round(float(daily_open.iloc[-2]), 8) if len(daily_open) >= 2 else None
            do_ = round(float(daily_open.iloc[-1]), 8) if len(daily_open) >= 1 else None

            return {
                "high": round(float(ph.iloc[-1]), 8),
                "low": round(float(pl.iloc[-1]), 8),
                "pdo": pdo,
                "do_": do_,
                "tf": resample_tf,
            }
        except Exception:  # pylint: disable=broad-exception-caught
            return {"high": None, "low": None, "pdo": None, "do_": None, "tf": ""}

    _PIVOT_ANCHOR_MAP = {
        "1m": "D",
        "5m": "D",
        "15m": "D",  # TradingView "Auto": intraday, multiplier<=15 → günlük
        "1h": "W",
        "4h": "W",  # intraday, multiplier>15 → haftalık
        "1d": "ME",  # günlük grafik → aylık
    }

    @classmethod
    def _prepare_fib_pivots(cls, df: pd.DataFrame, tf: str = "15m") -> dict:
        """Önceki periyodun H/L/C'sinden Fibonacci Pivot Point seviyelerini hesaplar
        (indicators/core.py::calculate_fib_pivots — panel ve ileride pattern_lab
        backtest'lerinin AYNI fonksiyonu kullanması için orada tutuluyor).

        Periyot TF'e göre değişir — TradingView'ın resmi "Pivot Points Standard"
        indikatörünün "Auto" anchor mantığıyla BİREBİR aynı: 1m/5m/15m → günlük,
        1h/4h → haftalık, 1d → aylık. Önceden HER ZAMAN günlük kullanıyordu (DO/PDO
        emsaliyle), bu yanlış emsaldi — PDH/PDL'nin TF-bağımlı deseni doğruydu."""
        try:
            from indicators.core import (
                calculate_fib_pivots,  # pylint: disable=import-outside-toplevel
            )

            anchor = cls._PIVOT_ANCHOR_MAP.get(tf, "D")

            df_idx = df[["high", "low", "close"]].copy()
            df_idx.index = pd.to_datetime(df["timestamp"].values, utc=True)
            for col in df_idx.columns:
                df_idx[col] = df_idx[col].astype(float)

            period = (
                df_idx.resample(anchor).agg({"high": "max", "low": "min", "close": "last"}).dropna()
            )
            if len(period) < 2:
                return {}

            prev = period.iloc[-2]
            levels = calculate_fib_pivots(
                float(prev["high"]), float(prev["low"]), float(prev["close"])
            )
            return {k: round(v, 8) for k, v in levels.items()}
        except Exception:  # pylint: disable=broad-exception-caught
            return {}

    @staticmethod
    def _prepare_wmy_open(df: pd.DataFrame) -> dict:
        """Mevcut haftanın/ayın/yılın açılış seviyelerini hesaplar — DO/PDO'nun
        (_prepare_previous_hl) günlük resample deseninin W/M/Y'ye genişlemesi.
        Tlosx'un 'D/W/M/Y open = mıknatıs seviyesi' fikri (bkz. [[project_turtle_traders]])."""
        try:
            df_idx = df[["open"]].copy()
            df_idx.index = pd.to_datetime(df["timestamp"].values, utc=True)
            df_idx["open"] = df_idx["open"].astype(float)

            w_open = df_idx["open"].resample("W").first().dropna()
            m_open = df_idx["open"].resample("ME").first().dropna()
            y_open = df_idx["open"].resample("YE").first().dropna()

            return {
                "w": round(float(w_open.iloc[-1]), 8) if len(w_open) >= 1 else None,
                "m": round(float(m_open.iloc[-1]), 8) if len(m_open) >= 1 else None,
                "y": round(float(y_open.iloc[-1]), 8) if len(y_open) >= 1 else None,
            }
        except Exception:  # pylint: disable=broad-exception-caught
            return {}

    @staticmethod
    def _prepare_smc(df: pd.DataFrame, swing_limit: int = 10, level_limit: int = 3) -> dict:
        """Güncel SMC yapısı: son swing_limit pivot + son level_limit BOS/CHoCH seviyesi."""
        try:
            from smartmoneyconcepts import smc as _smc  # pylint: disable=import-outside-toplevel

            ts = (df["timestamp"].astype("int64") // 10**9 + 3 * 3600).values
            df_smc = df[["open", "high", "low", "close", "volume"]].copy().reset_index(drop=True)
            for col in df_smc.columns:
                df_smc[col] = df_smc[col].astype(float)

            swing_df = _smc.swing_highs_lows(df_smc, swing_length=5)
            bos_df = _smc.bos_choch(df_smc, swing_df, close_break=True)

            green = COLORS["green"]
            red = COLORS["red"]
            orange = "#ff9900"
            purple = COLORS["purple"]

            # Tüm pivotları topla, son swing_limit tanesini al
            all_markers = []
            for i in range(len(swing_df)):
                hl = swing_df["HighLow"].iloc[i]
                if np.isnan(hl):
                    continue
                t = int(ts[i])
                if hl == 1.0:
                    all_markers.append(
                        {
                            "time": t,
                            "position": "aboveBar",
                            "color": red,
                            "shape": "arrowDown",
                            "text": "H",
                            "size": 0.8,
                        }
                    )
                else:
                    all_markers.append(
                        {
                            "time": t,
                            "position": "belowBar",
                            "color": green,
                            "shape": "arrowUp",
                            "text": "L",
                            "size": 0.8,
                        }
                    )
            markers = all_markers[-swing_limit:]

            # Tüm BOS/CHoCH seviyelerini topla, son level_limit tanesini al
            all_levels = []
            for i in range(len(bos_df)):
                bos_val = bos_df["BOS"].iloc[i]
                choch_val = bos_df["CHOCH"].iloc[i]
                lv = bos_df["Level"].iloc[i]
                t = int(ts[i]) if i < len(ts) else None
                if not np.isnan(bos_val) and not np.isnan(lv) and lv > 0:
                    all_levels.append(
                        {
                            "price": float(lv),
                            "time": t,
                            "color": green if bos_val == 1 else red,
                            "title": "BOS↑" if bos_val == 1 else "BOS↓",
                        }
                    )
                elif not np.isnan(choch_val) and not np.isnan(lv) and lv > 0:
                    all_levels.append(
                        {
                            "price": float(lv),
                            "time": t,
                            "color": orange if choch_val == 1 else purple,
                            "title": "CHoCH↑" if choch_val == 1 else "CHoCH↓",
                        }
                    )
            levels = all_levels[-level_limit:]

            return {"markers": markers, "levels": levels}
        except Exception:  # pylint: disable=broad-exception-caught
            return {"markers": [], "levels": []}

    @staticmethod
    def _price_format(price: float) -> tuple[int, float]:
        if price >= 1000:
            return 2, 0.01
        if price >= 10:
            return 3, 0.001
        if price >= 1:
            return 4, 0.0001
        if price >= 0.1:
            return 5, 0.00001
        if price >= 0.01:
            return 6, 0.000001
        if price >= 0.001:
            return 7, 0.0000001
        if price >= 0.0001:
            return 8, 0.00000001
        return 10, 0.0000000001

    @staticmethod
    def _compute_state(df: pd.DataFrame, _ts_hint=None) -> dict:
        """Son kapalı bar'ın (bar[-2]) EMA/RSI durumunu döner."""
        closes = df["close"].astype(float).reset_index(drop=True)
        ts = df["timestamp"].astype("int64") // 10**9 + 3 * 3600
        ema = closes.ewm(span=_EMA_SPAN, adjust=False).mean()
        delta = closes.diff()
        gain = delta.clip(lower=0).ewm(alpha=_RSI_ALPHA, min_periods=_RSI_PERIOD).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=_RSI_ALPHA, min_periods=_RSI_PERIOD).mean()
        idx = -2 if len(closes) >= 2 else -1
        return {
            "ts": int(ts.iloc[idx]),
            "ema": float(ema.iloc[idx]),
            "avg_gain": float(gain.iloc[idx]),
            "avg_loss": float(loss.iloc[idx]),
            "close": float(closes.iloc[idx]),
        }

    @staticmethod
    def _prepare(df: pd.DataFrame):
        ts = df["timestamp"].astype("int64") // 10**9 + 3 * 3600
        opens = df["open"].astype(float)
        highs = df["high"].astype(float)
        lows = df["low"].astype(float)
        closes = df["close"].astype(float)
        vols = df["volume"].astype(float)

        ema = closes.ewm(span=_EMA_SPAN, adjust=False).mean()
        delta = closes.diff()
        gain = delta.clip(lower=0).ewm(alpha=_RSI_ALPHA, min_periods=_RSI_PERIOD).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=_RSI_ALPHA, min_periods=_RSI_PERIOD).mean()
        rsi = 100 - 100 / (1 + gain / (loss + 1e-9))

        candles, ema_data, vol_data, rsi_data = [], [], [], []
        green = COLORS["green"]
        red = COLORS["red"]

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
            vol_data.append(
                {
                    "time": t,
                    "value": v,
                    "color": (green + "99") if c >= o else (red + "99"),
                }
            )
            if r == r:  # NaN değil
                rsi_data.append({"time": t, "value": round(r, 2)})

        return candles, ema_data, vol_data, rsi_data
