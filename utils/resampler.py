"""
BarResampler: 1m tick'lerini üst TF'lere (5m/15m/1h/4h) gerçek zamanlı yeniden örnekler.

Kullanım:
    resampler = BarResampler()
    # Her 1m tick geldiğinde:
    completed = resampler.update("BTCUSDT", tick_dict)  # tamamlanan tick döner veya None
    # Açık tick (incomplete):
    pending = resampler.get_pending("BTCUSDT", "1h")
"""

from __future__ import annotations

from typing import Dict, Optional

_TF_SECONDS: Dict[str, int] = {
    "5m":  300,
    "15m": 900,
    "1h":  3600,
    "4h":  14400,
}


def _bucket_start(ts_ms: int, tf_seconds: int) -> int:
    """ts_ms'yi tf_seconds'lık bucket başına hizalar (UTC epoch ms)."""
    ts_s = ts_ms // 1000
    return (ts_s // tf_seconds) * tf_seconds * 1000


class BarResampler:
    """Sembol başına her TF için aktif (açık) tick'ı tutar."""

    def __init__(self) -> None:
        # _pending[symbol][tf] = {"open_time", "open", "high", "low", "close", "volume"}
        self._pending: Dict[str, Dict[str, Optional[Dict]]] = {}

    def update(self, symbol: str, tick: Dict) -> Dict[str, Optional[Dict]]:
        """
        Yeni bir 1m tick gönderir. Döndürülen dict: TF → tamamlanan tick (veya None).
        tick anahtarları: open_time (ms int), open, high, low, close, volume (float).
        """
        if symbol not in self._pending:
            self._pending[symbol] = {tf: None for tf in _TF_SECONDS}

        ts_ms: int = int(tick["open_time"])
        completed: Dict[str, Optional[Dict]] = {}

        for tf, sec in _TF_SECONDS.items():
            bucket = _bucket_start(ts_ms, sec)
            pending = self._pending[symbol][tf]

            if pending is None:
                self._pending[symbol][tf] = self._new_tick(bucket, tick)
                completed[tf] = None
            elif pending["open_time"] == bucket:
                self._merge(pending, tick)
                completed[tf] = None
            else:
                completed[tf] = dict(pending)
                self._pending[symbol][tf] = self._new_tick(bucket, tick)

        return completed

    def get_pending(self, symbol: str, tf: str) -> Optional[Dict]:
        """Henüz kapanmamış (açık) tick'ı döndürür."""
        return self._pending.get(symbol, {}).get(tf)

    def get_all_pending(self, symbol: str) -> Dict[str, Optional[Dict]]:
        return dict(self._pending.get(symbol, {}))

    @staticmethod
    def _new_tick(open_time: int, tick: Dict) -> Dict:
        return {
            "open_time": open_time,
            "open":      float(tick["open"]),
            "high":      float(tick["high"]),
            "low":       float(tick["low"]),
            "close":     float(tick["close"]),
            "volume":    float(tick["volume"]),
        }

    @staticmethod
    def _merge(pending: Dict, tick: Dict) -> None:
        pending["high"]   = max(pending["high"],   float(tick["high"]))
        pending["low"]    = min(pending["low"],    float(tick["low"]))
        pending["close"]  = float(tick["close"])
        pending["volume"] += float(tick["volume"])
