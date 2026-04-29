#!/usr/bin/env python3
"""
Phase 1: Run chatfreq analysis on all 4 stream samples.

Produces per-stream:
  - <stream>_frequency.csv        : normalized sliding-window time series
  - <stream>_regions_z0.csv       : CBS high-engagement regions (Z-score >= 0.0)
  - <stream>_regions_z05.csv      : CBS high-engagement regions (Z-score >= 0.5)
  - <stream>_frequency.png        : engagement plot

Parameters:
  window=60s, step=10s, degree=3 polynomial trend, normalize=True (trend-normalized)
  CBS: t-threshold=2.5, min_duration=30s, max_gap=30s
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Make sure the package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatfreq.core.analyzer import ChatFrequencyAnalyzer
from chatfreq.core.time_utils import seconds_to_hms

# ── Configuration ────────────────────────────────────────────────────────────
WINDOW   = 60.0
STEP     = 10.0
DEGREE   = 3
CBS_T    = 2.5
MIN_DUR  = 30.0
MAX_GAP  = 30.0
NORMALIZE = True

SAMPLES_DIR = PROJECT_ROOT / "samples"
RESEARCH_DIR = Path(__file__).resolve().parent

STREAMS = {
    "gura-first-minecraft": {
        "chat_tsv": SAMPLES_DIR / "gura-first-minecraft" / "[MINECRAFT] BUILD ATLANTIS #GAWRGURA #HololiveEnglish [OlJQItn5Z2o].live_chat_reduced.tsv",
    },
    "gura-solo-minecraft0": {
        "chat_tsv": SAMPLES_DIR / "gura-solo-minecraft0" / "[MINECRAFT] I Want A Trident!!! [tQcV9eEH7fk].live_chat_reduced.tsv",
    },
    "nimi-retro-rewind": {
        "chat_tsv": SAMPLES_DIR / "nimi-retro-rewind" / "【Retro Rewind】 Legally distinct Blockbuster simulator [7Px9qClCzt8].live_chat_reduced.tsv",
    },
    "nimi-vampire": {
        "chat_tsv": SAMPLES_DIR / "nimi-vampire" / "【Vampire： The Masquerade - Bloodlines】 Living out my sick and twisted vampire dreams [Db54iFFWLWc].live_chat_reduced.tsv",
    },
}


def regions_to_df(regions, label: str) -> pd.DataFrame:
    rows = []
    for i, (start, end) in enumerate(regions):
        rows.append({
            "region_index": i + 1,
            "start_seconds": start,
            "end_seconds": end,
            "start_hms": seconds_to_hms(start),
            "end_hms": seconds_to_hms(end),
            "duration_seconds": end - start,
            "z_threshold": label,
        })
    return pd.DataFrame(rows)


def plot_engagement(stream_name: str, analyzer: ChatFrequencyAnalyzer,
                    time_axis: np.ndarray, signal: np.ndarray,
                    regions_z0, regions_z05, out_path: Path):
    fig, ax = plt.subplots(figsize=(18, 5))

    # Raw signal
    ax.plot(time_axis, signal, color="#4477aa", linewidth=0.8, alpha=0.85,
            label="Normalized frequency")
    ax.axhline(1.0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)

    # Shade regions
    for start, end in regions_z0:
        ax.axvspan(start, end, alpha=0.15, color="red", label="_nolegend_")
    for start, end in regions_z05:
        ax.axvspan(start, end, alpha=0.25, color="crimson", label="_nolegend_")

    # Legend patches
    p1 = mpatches.Patch(color="red", alpha=0.3, label=f"High-engagement Z≥0 ({len(regions_z0)})")
    p2 = mpatches.Patch(color="crimson", alpha=0.5, label=f"High-engagement Z≥0.5 ({len(regions_z05)})")

    # Nice x-axis: HH:MM:SS ticks every 10 minutes
    max_t = time_axis[-1]
    tick_every = 600  # 10 min
    ticks = np.arange(0, max_t + tick_every, tick_every)
    ax.set_xticks(ticks)
    ax.set_xticklabels([seconds_to_hms(t) for t in ticks], rotation=45, ha="right", fontsize=8)

    ax.set_xlabel("Stream time")
    ax.set_ylabel("Normalized chat rate (rolling/trend)")
    ax.set_title(f"Chat engagement — {stream_name}")
    ax.legend(handles=[
        plt.Line2D([0], [0], color="#4477aa", linewidth=1.5, label="Normalized freq"),
        p1, p2,
    ])
    ax.set_xlim(0, max_t)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved plot → {out_path.name}")


def run_stream(stream_name: str, cfg: dict):
    print(f"\n{'='*60}")
    print(f"  {stream_name}")
    print(f"{'='*60}")

    out_dir = RESEARCH_DIR / stream_name
    out_dir.mkdir(parents=True, exist_ok=True)

    chat_tsv = cfg["chat_tsv"]
    if not chat_tsv.exists():
        print(f"  ERROR: chat TSV not found: {chat_tsv}")
        return

    print(f"  Loading chat data…")
    analyzer = ChatFrequencyAnalyzer(str(chat_tsv))
    print(f"  Messages: {len(analyzer.df):,}   Duration: {seconds_to_hms(analyzer.max_seconds)}")

    # ── Normalized signal ────────────────────────────────────────────────────
    print(f"  Computing normalized sliding window (window={WINDOW}s, step={STEP}s, degree={DEGREE})…")
    time_axis, signal = analyzer.compute_normalized_signal(STEP, WINDOW, DEGREE)

    freq_df = pd.DataFrame({
        "time_seconds": time_axis,
        "time_hms": [seconds_to_hms(t) for t in time_axis],
        "normalized_frequency": signal,
    })
    freq_path = out_dir / f"{stream_name}_frequency.csv"
    freq_df.to_csv(freq_path, index=False)
    print(f"  Saved → {freq_path.name}  ({len(freq_df):,} rows)")

    # ── High-engagement regions ──────────────────────────────────────────────
    for z_thresh, suffix in [(0.0, "z0"), (0.5, "z05")]:
        print(f"  Detecting regions (Z≥{z_thresh})…")
        regions = analyzer.detect_high_engagement_regions(
            step=STEP,
            window_size=WINDOW,
            cbs_threshold=CBS_T,
            z_threshold=z_thresh,
            min_duration=MIN_DUR,
            max_gap=MAX_GAP,
            normalize=NORMALIZE,
            degree=DEGREE,
        )
        df = regions_to_df(regions, label=str(z_thresh))
        path = out_dir / f"{stream_name}_regions_{suffix}.csv"
        df.to_csv(path, index=False)
        total_dur = sum(e - s for s, e in regions)
        pct = 100 * total_dur / analyzer.max_seconds if analyzer.max_seconds > 0 else 0
        print(f"  Saved → {path.name}   {len(regions)} regions, "
              f"{seconds_to_hms(total_dur)} total ({pct:.1f}% of stream)")

    # ── Plot ─────────────────────────────────────────────────────────────────
    regions_z0  = analyzer.detect_high_engagement_regions(STEP, WINDOW, CBS_T, 0.0, MIN_DUR, MAX_GAP, NORMALIZE, DEGREE)
    regions_z05 = analyzer.detect_high_engagement_regions(STEP, WINDOW, CBS_T, 0.5, MIN_DUR, MAX_GAP, NORMALIZE, DEGREE)
    plot_path = out_dir / f"{stream_name}_frequency.png"
    plot_engagement(stream_name, analyzer, time_axis, signal, regions_z0, regions_z05, plot_path)


def main():
    print("chatfreq analysis — all streams")
    print(f"Parameters: window={WINDOW}s  step={STEP}s  degree={DEGREE}  normalize={NORMALIZE}")
    print(f"CBS: t={CBS_T}  min_dur={MIN_DUR}s  max_gap={MAX_GAP}s")

    for stream_name, cfg in STREAMS.items():
        run_stream(stream_name, cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
