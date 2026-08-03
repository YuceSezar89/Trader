"""
Trajectory Lab — Adım 2: 3 çekirdek görselleştirme.

v1 önceliği: kümeleme/motif algoritması değil, araştırmacının gözünü
güçlendiren araçlar (bkz. README). Hepsi corpus'un uzun/tidy formatını
(signal_id, symbol, t0, outcome, t_offset, metric, value) alır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless — figürler dosyaya yazılır, ekrana değil
import matplotlib.pyplot as plt  # noqa: E402

_COLORS = {"winner": "#2ca02c", "loser": "#d62728", "neutral": "#7f7f7f"}


def plot_overlay(corpus: pd.DataFrame, metric: str, group_col: str = "outcome", ax=None):
    """Her sinyal ince bir çizgi, outcome'a göre renkli — ham 'video' görünümü."""
    sub = corpus[corpus["metric"] == metric]
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    for signal_id, g in sub.groupby("signal_id"):
        g = g.sort_values("t_offset")
        color = _COLORS.get(g[group_col].iloc[0], "#1f77b4")
        ax.plot(g["t_offset"], g["value"], color=color, alpha=0.25, linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(f"{metric} — overlay (yeşil=winner, kırmızı=loser, gri=neutral)")
    ax.set_xlabel("t_offset (bar, 0=sinyal)")
    ax.set_ylabel(metric)
    return ax


def plot_mean_band(
    corpus: pd.DataFrame,
    metric: str,
    group_col: str = "outcome",
    band: tuple = (25, 75),
    n_boot: int = 500,
    ax=None,
    seed: int = 42,
):
    """Grup ortalaması + bootstrap güven bandı — 'gürültüde mi kayboluyor' testi."""
    sub = corpus[corpus["metric"] == metric]
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    rng = np.random.default_rng(seed)

    for group_name, g in sub.groupby(group_col):
        pivot = g.pivot_table(index="signal_id", columns="t_offset", values="value")
        offsets = pivot.columns.to_numpy()
        mean_line = pivot.mean(axis=0).to_numpy()

        n_signals = len(pivot)
        if n_signals >= 5:
            boot_means = np.empty((n_boot, len(offsets)))
            arr = pivot.to_numpy()
            for i in range(n_boot):
                sample_idx = rng.integers(0, n_signals, size=n_signals)
                boot_means[i] = np.nanmean(arr[sample_idx], axis=0)
            lo = np.nanpercentile(boot_means, band[0], axis=0)
            hi = np.nanpercentile(boot_means, band[1], axis=0)
            color = _COLORS.get(group_name, "#1f77b4")
            ax.fill_between(offsets, lo, hi, color=color, alpha=0.15)

        color = _COLORS.get(group_name, "#1f77b4")
        ax.plot(offsets, mean_line, color=color, linewidth=2, label=f"{group_name} (n={n_signals})")

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.legend()
    ax.set_title(f"{metric} — grup ortalaması + %{band[0]}-{band[1]} bootstrap bandı")
    ax.set_xlabel("t_offset (bar, 0=sinyal)")
    ax.set_ylabel(metric)
    return ax


def plot_signal_story(corpus: pd.DataFrame, signal_id, metrics: list[str] | None = None):
    """Tek sinyal, çoklu metrik, ortak zaman ekseninde alt alta panel —
    'davranış hikâyesi' replay'i. metrics=None ise corpus'ta o sinyal için
    bulunan tüm metrikler kullanılır (VPMV alt bileşenlerinin drill-down'ı
    da bu fonksiyonla, ayrı bir çağrıyla yapılabilir — corpus'a dahil
    edilmeden)."""
    sub = corpus[corpus["signal_id"] == signal_id]
    if metrics is None:
        metrics = sorted(sub["metric"].unique())

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 2.2 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    outcome = sub["outcome"].iloc[0] if not sub.empty else "?"
    symbol = sub["symbol"].iloc[0] if not sub.empty else "?"
    fig.suptitle(f"Sinyal #{signal_id} — {symbol} ({outcome})")

    for ax, metric in zip(axes, metrics):
        g = sub[sub["metric"] == metric].sort_values("t_offset")
        ax.plot(g["t_offset"], g["value"], color=_COLORS.get(outcome, "#1f77b4"))
        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_ylabel(metric, fontsize=9)

    axes[-1].set_xlabel("t_offset (bar, 0=sinyal)")
    fig.tight_layout()
    return fig
