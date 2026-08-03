"""
Trajectory Lab — Adım 3: kullanım noktası.

Kullanım:
    python -m research.trajectory_lab.explore --corpus research/trajectory_corpus/HA_Cross_Long.parquet --metric evol
    python -m research.trajectory_lab.explore --corpus ... --metric evol --story SIGNAL_ID
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.trajectory_lab import config as C  # noqa: E402
from research.trajectory_lab import viz  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="corpus_builder.py çıktısı .parquet yolu")
    parser.add_argument("--metric", required=True, help="ör. evol, vpmv, cvd_slope, price_return")
    parser.add_argument("--story", type=int, default=None, help="tek bir signal_id için hikâye paneli")
    args = parser.parse_args()

    corpus = pd.read_parquet(args.corpus)
    os.makedirs(C.REPORT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.corpus))[0]

    if args.story is not None:
        fig = viz.plot_signal_story(corpus, args.story)
        out = os.path.join(C.REPORT_DIR, f"{base}_signal{args.story}_story.png")
        fig.savefig(out, dpi=120)
        print(f"-> {out}")
        return

    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12.5))
    viz.plot_overlay(corpus, args.metric, ax=ax1)
    viz.plot_mean_band(corpus, args.metric, ax=ax2)
    _, peak_offset, peak_value = viz.plot_divergence(corpus, args.metric, ax=ax3)
    fig.tight_layout()
    out = os.path.join(C.REPORT_DIR, f"{base}_{args.metric}.png")
    fig.savefig(out, dpi=120)
    print(f"-> {out} (en güçlü ayrışma: t={peak_offset}, değer={peak_value:+.3f})")


if __name__ == "__main__":
    main()
