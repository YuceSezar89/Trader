"""
RankingWorker — tüm takip edilen coinleri VPMV skoruna göre sıralar.

Her 3 dakikada bir:
  1. Redis'ten 5m / 15m / 1h kline verisi okunur
  2. Her TF için VPMV hesaplanır
  3. Ağırlıklı birleşik skor: 5m×0.50 + 15m×0.35 + 1h×0.15
  4. 200 coin içinde percentile rank uygulanır (cross-normalize)
  5. VS BTC farkı ayrı kolon olarak eklenir
"""

import logging
import threading
import time
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import redis
from PyQt6.QtCore import QThread, pyqtSignal  # pylint: disable=no-name-in-module

from config import Config
from indicators.core import calculate_atr, calculate_rsi
from utils.preprocessing import (
    normalize_momentum_0_100,
    normalize_price_0_100,
    normalize_volatility_0_100,
    normalize_volume_0_100,
)

logger = logging.getLogger(__name__)

_ARROW_MAGIC = b"ARDF"
_MIN_BARS    = 50
_UPDATE_SEC  = 180  # 3 dakika
_REF_SYMBOL  = "BTCUSDT"

_TF_WEIGHTS: dict[str, float] = {
    "5m":  0.50,
    "15m": 0.35,
    "1h":  0.15,
}
_Z_LOOKBACK = 100
_R_PERIOD   = 14


class RankingWorker(QThread):
    ranking_updated = pyqtSignal(object)  # list[dict]
    status_updated  = pyqtSignal(str)

    def __init__(self, redis_url: str, parent=None):
        super().__init__(parent)
        self._redis_url = redis_url
        self._running   = False
        self._redis: Optional[redis.Redis] = None
        self._wake = threading.Event()

    # ------------------------------------------------------------------
    def run(self) -> None:
        self._running = True
        try:
            self._redis = redis.Redis.from_url(
                self._redis_url, decode_responses=False, socket_connect_timeout=3
            )
            self._redis.ping()
        except redis.RedisError as exc:
            self.status_updated.emit(f"Redis bağlanamadı: {exc}")
            return

        while self._running:
            try:
                result = self._compute()
                if result:
                    self.ranking_updated.emit(result)
                    self.status_updated.emit(
                        f"{len(result)} sembol  •  {time.strftime('%H:%M:%S')}"
                    )
                else:
                    self.status_updated.emit("Veri hesaplanıyor…")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("Ranking hatası: %s", exc, exc_info=True)

            self._wake.wait(timeout=_UPDATE_SEC)
            self._wake.clear()

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        self.wait()

    def refresh(self) -> None:
        self._wake.set()

    # ------------------------------------------------------------------
    def _compute(self) -> list:
        symbols = self._all_symbols()
        if not symbols:
            return []

        scores: dict[str, dict] = {}

        for sym in symbols:
            tf_scores:  dict[str, float] = {}
            tf_dirs:    dict[str, int]   = {}
            tf_zscores: dict[str, float] = {}

            for tf in _TF_WEIGHTS:
                df = self._fetch_klines(sym, tf)
                if df is None or len(df) < _MIN_BARS:
                    continue
                vpmv, direction, z = self._vpmv(df)
                if vpmv is not None:
                    tf_scores[tf] = vpmv
                    tf_dirs[tf]   = direction
                    if z is not None:
                        tf_zscores[tf] = z

            if not tf_scores:
                continue

            total_w  = sum(_TF_WEIGHTS[tf] for tf in tf_scores)
            combined = sum(tf_scores[tf] * _TF_WEIGHTS[tf] for tf in tf_scores) / total_w

            if tf_zscores:
                total_zw      = sum(_TF_WEIGHTS[tf] for tf in tf_zscores)
                z_confluence  = round(sum(tf_zscores[tf] * _TF_WEIGHTS[tf] for tf in tf_zscores) / total_zw, 2)
            else:
                z_confluence  = None

            dirs    = list(tf_dirs.values())
            n_bull  = sum(1 for d in dirs if d > 0)
            n_bear  = len(dirs) - n_bull
            aligned = max(n_bull, n_bear) == len(dirs)

            df_1h = self._fetch_klines(sym, "1h")
            r_score = self._r_score(df_1h) if df_1h is not None and len(df_1h) >= _R_PERIOD else None

            scores[sym] = {
                "symbol":          sym,
                "score_5m":        tf_scores.get("5m"),
                "score_15m":       tf_scores.get("15m"),
                "score_1h":        tf_scores.get("1h"),
                "combined":        round(combined, 1),
                "z_confluence":    z_confluence,
                "r_score":         r_score,
                "aligned":         aligned,
                "alignment_count": max(n_bull, n_bear),
                "tf_count":        len(dirs),
                "direction":       "long" if n_bull >= n_bear else "short",
            }

        if not scores:
            return []

        # Cross-normalize: 200 coin içinde percentile rank
        all_combined = [v["combined"] for v in scores.values()]
        n = len(all_combined)
        for data in scores.values():
            data["rank_score"] = round(
                sum(1 for v in all_combined if v < data["combined"]) / n * 100, 1
            )

        # VS BTC
        btc_combined = scores.get(_REF_SYMBOL, {}).get("combined")
        for data in scores.values():
            data["vs_btc"] = (
                round(data["combined"] - btc_combined, 1)
                if btc_combined is not None else None
            )

        result = sorted(scores.values(), key=lambda x: x["rank_score"], reverse=True)
        for i, item in enumerate(result):
            item["rank"] = i + 1

        return result

    # ------------------------------------------------------------------
    def _all_symbols(self) -> list[str]:
        keys = self._redis.keys("live_kline_data:*:5m")
        symbols = []
        for k in keys:
            parts = k.decode().split(":")
            if len(parts) == 3:
                symbols.append(parts[1])
        return symbols

    def _fetch_klines(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        key = f"live_kline_data:{symbol}:{tf}".encode()
        raw = self._redis.get(key)
        if not raw:
            return None
        try:
            if raw[:4] == _ARROW_MAGIC:
                import pyarrow as pa  # pylint: disable=import-outside-toplevel
                return pa.ipc.open_stream(raw[4:]).read_pandas()
            return pd.read_json(StringIO(raw.decode()), orient="split")
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    def _r_score(self, df: pd.DataFrame) -> Optional[float]:
        try:
            closes  = df["close"].astype(float).values
            returns = np.diff(np.log(closes + 1e-12))[-_R_PERIOD:]
            if len(returns) < _R_PERIOD // 2:
                return None

            avg = returns.mean()
            std = returns.std() + 1e-12

            sharpe  = avg / std

            neg = returns[returns < 0]
            neg_std = neg.std() + 1e-12 if len(neg) > 1 else 1e-12
            sortino = avg / neg_std

            price_window = closes[-_R_PERIOD - 1:]
            max_dd = (price_window.max() - price_window.min()) / (price_window.max() + 1e-12)
            calmar  = avg / (max_dd + 1e-12)

            gains  = returns[returns >= 0].sum()
            losses = abs(returns[returns < 0].sum()) + 1e-12
            omega   = gains / losses

            r = sortino * 0.40 + omega * 0.30 + calmar * 0.20 + sharpe * 0.10
            return round(float(r), 3)
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    def _vpmv(self, df: pd.DataFrame) -> tuple[Optional[float], int, Optional[float]]:
        try:
            rsi_series    = calculate_rsi(df, period=14)
            rsi_centered  = rsi_series - 50
            atr_series    = calculate_atr(df, period=Config.ATR_PERIOD)
            price_pct     = df["close"].pct_change().fillna(0.0) * 100.0

            vpmv_series = (
                normalize_volume_0_100(df["volume"]) * 0.35 +
                normalize_momentum_0_100(rsi_centered) * 0.35 +
                normalize_volatility_0_100(atr_series) * 0.20 +
                normalize_price_0_100(price_pct) * 0.10
            )

            current   = float(vpmv_series.iloc[-1])
            direction = 1 if float(rsi_series.iloc[-1]) >= 50 else -1

            lookback = vpmv_series.iloc[-_Z_LOOKBACK - 1:-1]
            if len(lookback) >= 20:
                mean = float(lookback.mean())
                std  = float(lookback.std())
                z    = round((current - mean) / std, 2) if std > 0 else 0.0
            else:
                z = None

            return round(current, 1), direction, z

        except Exception:  # pylint: disable=broad-exception-caught
            return None, 0, None
