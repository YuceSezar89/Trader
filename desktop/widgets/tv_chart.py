"""
TVChart — TradingView lightweight-charts tabanlı grafik widget'ı.

QWebEngineView içinde lightweight-charts v4 çalıştırır.
Yerel JS dosyası kullanılır: desktop/assets/lightweight-charts.js
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtCore import QTimer, QUrl, pyqtSlot
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QSizePolicy

from desktop.theme import COLORS

_ASSETS_DIR = Path(__file__).parent.parent / "assets"
_JS_PATH = _ASSETS_DIR / "lightweight-charts.js"

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
  }} catch(e) {{
    console.log('loadData error:', e.message);
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

        self.loadFinished.connect(self._on_load_finished)
        self.setHtml(_build_html(), QUrl("about:blank"))

    @pyqtSlot(bool)
    def _on_load_finished(self, ok: bool) -> None:
        self._ready = ok
        if ok and self._pending_df is not None:
            df, sym, tf = self._pending_df
            self._pending_df = None
            QTimer.singleShot(600, lambda: self._send_data(df, sym, tf))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._ready:
            w, h = event.size().width(), event.size().height()
            if w > 10 and h > 10:
                self.page().runJavaScript(f"resizeChart({w},{h})")

    def load_df(self, df: pd.DataFrame, symbol: str = "", tf: str = "") -> None:
        self._symbol = symbol
        self._tf = tf
        if df is None or df.empty:
            return
        if not self._ready:
            self._pending_df = (df, symbol, tf)
            return
        self._send_data(df, symbol, tf)

    def _send_data(self, df: pd.DataFrame, symbol: str, tf: str) -> None:
        candles, ema_data, vol_data, rsi_data = self._prepare(df)
        js = (
            f"loadData("
            f"{json.dumps(json.dumps(candles))},"
            f"{json.dumps(json.dumps(ema_data))},"
            f"{json.dumps(json.dumps(vol_data))},"
            f"{json.dumps(json.dumps(rsi_data))})"
        )
        self.page().runJavaScript(js)

    @staticmethod
    def _prepare(df: pd.DataFrame):
        ts     = df["timestamp"].astype("int64") // 10**9
        opens  = df["open"].astype(float)
        highs  = df["high"].astype(float)
        lows   = df["low"].astype(float)
        closes = df["close"].astype(float)
        vols   = df["volume"].astype(float)

        ema   = closes.ewm(span=21, adjust=False).mean()
        delta = closes.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
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
