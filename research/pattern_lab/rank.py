"""
Ayrışma = kesitsel sıra. Evren ızgarası ffill'li (son bilinen fiyat,
en fazla 3 bar = 15 dk bayat — canlı panelin davranışına en yakın).

Look-ahead güvenliği: rank_pct(t) yalnızca t ve öncesi ızgara satırlarını okur.
"""
import pandas as pd

from research.pattern_lab import config as C

MIN_UNIVERSE = 150   # bu sayının altında kesit → sıra güvenilmez (None)
FFILL_LIMIT = 3


class RankProvider:
    def __init__(self, layer_b: pd.DataFrame):
        grid = layer_b.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last")
        self.grid = grid.asfreq("5min").ffill(limit=FFILL_LIMIT)

    def universe_size(self, t: pd.Timestamp) -> int:
        if t not in self.grid.index:
            return 0
        return int(self.grid.loc[t].notna().sum())

    def rank_pct(self, symbol: str, t: pd.Timestamp, window_bars: int) -> float | None:
        """Sembolün t anındaki pencere getirisinin evren içindeki yüzdelik sırası (0-100)."""
        t = pd.Timestamp(t)
        t_prev = t - pd.Timedelta(minutes=5 * window_bars)
        if t not in self.grid.index or t_prev not in self.grid.index:
            return None
        now, prev = self.grid.loc[t], self.grid.loc[t_prev]
        rets = (now / prev - 1).dropna()
        if len(rets) < MIN_UNIVERSE or symbol not in rets.index:
            return None
        return float((rets < rets[symbol]).mean() * 100)

    def rank_series(self, symbol: str, t_end: pd.Timestamp, hours_back: int,
                    window_bars: int, step_bars: int = 6) -> pd.Series:
        """t_end'e kadar (dahil) geriye dönük sıra yörüngesi — 30 dk adımlarla."""
        t_end = pd.Timestamp(t_end)
        times = pd.date_range(t_end - pd.Timedelta(hours=hours_back), t_end,
                              freq=f"{5 * step_bars}min")
        vals = {t: self.rank_pct(symbol, t, window_bars) for t in times}
        return pd.Series(vals, dtype=float)
