#!/usr/bin/env python3
"""
Phase 4: Visualizations.

Per-stream:
  1. timeline.png   – engagement signal + high-engagement regions (Z≥0 and Z≥0.5)
                      + clipped intervals for each clipper (stacked tracks)
  2. region_coverage.png – bar chart: coverage fraction per region (Z≥0.5)
  3. correlation.png     – scatter: engagement vs. P(clipped) in sliding windows

Aggregate:
  4. aggregate_summary.png – multi-panel comparison across all streams
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from chatfreq.core.time_utils import seconds_to_hms

RESEARCH_DIR = Path(__file__).resolve().parent

STREAMS = [
    "gura-first-minecraft",
    "gura-solo-minecraft0",
    "nimi-retro-rewind",
    "nimi-vampire",
]

CLIPPER_COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261"]
REGION_Z0_COLOR  = "#ff9999"
REGION_Z05_COLOR = "#e63946"
SIGNAL_COLOR     = "#4a4e69"

# ── Time axis helpers ─────────────────────────────────────────────────────────

def nice_ticks(max_s: float, n_ticks: int = 10) -> tuple[np.ndarray, list[str]]:
    """Return (tick_positions_s, tick_labels_hms) spaced every ~10 min."""
    step = max(60, int(max_s / n_ticks / 60) * 60)
    ticks = np.arange(0, max_s + step, step)
    labels = [seconds_to_hms(t) for t in ticks]
    return ticks, labels


def intervals_union(segs):
    if not segs:
        return []
    segs = sorted(segs)
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


# ── Plot 1: timeline ──────────────────────────────────────────────────────────

def plot_timeline(stream_name: str, d: Path):
    freq_path = d / f"{stream_name}_frequency.csv"
    if not freq_path.exists():
        return

    freq_df   = pd.read_csv(freq_path)
    time_s    = freq_df["time_seconds"].values
    signal    = freq_df["normalized_frequency"].values
    max_t     = float(time_s[-1])

    # Load regions
    reg_z0  = pd.read_csv(d / f"{stream_name}_regions_z0.csv")  if (d / f"{stream_name}_regions_z0.csv").exists()  else pd.DataFrame()
    reg_z05 = pd.read_csv(d / f"{stream_name}_regions_z05.csv") if (d / f"{stream_name}_regions_z05.csv").exists() else pd.DataFrame()

    # Load alignment
    aln_path = d / f"{stream_name}_alignment.csv"
    if not aln_path.exists():
        return
    aln_df   = pd.read_csv(aln_path)
    clippers = sorted(aln_df["highlight_file"].unique())

    n_clipper_rows = len(clippers)
    fig_height = 4 + n_clipper_rows * 0.7
    fig, axes = plt.subplots(
        nrows=2 + n_clipper_rows, ncols=1,
        figsize=(20, fig_height),
        gridspec_kw={"height_ratios": [3, 0.4] + [0.55] * n_clipper_rows},
        sharex=True,
    )
    ax_freq = axes[0]
    ax_reg  = axes[1]
    ax_clips = axes[2:]

    # ── Top panel: frequency signal ──────────────────────────────────────────
    ax_freq.fill_between(time_s, 0, signal, color=SIGNAL_COLOR, alpha=0.25, linewidth=0)
    ax_freq.plot(time_s, signal, color=SIGNAL_COLOR, linewidth=0.7, alpha=0.9, label="Norm. chat freq.")
    ax_freq.axhline(1.0, color="grey", linewidth=0.5, linestyle="--", alpha=0.6)

    # Shade Z≥0 regions in background
    for _, r in reg_z0.iterrows():
        ax_freq.axvspan(r.start_seconds, r.end_seconds, alpha=0.12, color=REGION_Z0_COLOR, zorder=0)
    # Shade Z≥0.5 overlap
    for _, r in reg_z05.iterrows():
        ax_freq.axvspan(r.start_seconds, r.end_seconds, alpha=0.22, color=REGION_Z05_COLOR, zorder=1)

    ax_freq.set_ylabel("Norm. freq.\n(rolling/trend)", fontsize=9)
    ax_freq.set_ylim(bottom=0)
    ax_freq.set_title(f"Chat engagement vs. clipping decisions — {stream_name}", fontsize=12, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=SIGNAL_COLOR, alpha=0.6, label="Normalised chat frequency"),
        mpatches.Patch(color=REGION_Z0_COLOR, alpha=0.5, label=f"High-engagement Z≥0 ({len(reg_z0)})"),
        mpatches.Patch(color=REGION_Z05_COLOR, alpha=0.7, label=f"High-engagement Z≥0.5 ({len(reg_z05)})"),
    ]
    ax_freq.legend(handles=legend_handles, loc="upper right", fontsize=8)

    # ── Region panel (compact bar view) ──────────────────────────────────────
    ax_reg.set_facecolor("#f8f8f8")
    ax_reg.set_ylim(0, 1)
    ax_reg.set_yticks([])
    ax_reg.set_ylabel("Regions", fontsize=8)
    for _, r in reg_z0.iterrows():
        ax_reg.axvspan(r.start_seconds, r.end_seconds, ymin=0.05, ymax=0.5,
                       alpha=0.5, color=REGION_Z0_COLOR)
    for _, r in reg_z05.iterrows():
        ax_reg.axvspan(r.start_seconds, r.end_seconds, ymin=0.5, ymax=0.95,
                       alpha=0.7, color=REGION_Z05_COLOR)

    # ── Clipper panels ────────────────────────────────────────────────────────
    short_names = [c.split("[")[0].strip()[:40] for c in clippers]

    for i, (c, sname) in enumerate(zip(clippers, short_names)):
        ax = ax_clips[i]
        ax.set_facecolor("#f0f4ff")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(sname, fontsize=7, rotation=0, ha="right", labelpad=80)
        color = CLIPPER_COLORS[i % len(CLIPPER_COLORS)]
        rows = aln_df[aln_df["highlight_file"] == c]
        merged = intervals_union([(float(r.stream_start_s), float(r.stream_end_s))
                                  for _, r in rows.iterrows()])
        for s, e in merged:
            ax.axvspan(s, e, ymin=0.1, ymax=0.9, alpha=0.75, color=color)

    # X axis
    axes[-1].set_xlabel("Stream time", fontsize=9)
    ticks, labels = nice_ticks(max_t)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[-1].set_xlim(0, max_t)

    plt.tight_layout(h_pad=0.3)
    out = d / f"{stream_name}_timeline.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ── Plot 2: region coverage bar chart ─────────────────────────────────────────

def plot_region_coverage(stream_name: str, d: Path):
    for z_label, z_thresh in [("z0", "0.0"), ("z05", "0.5")]:
        cov_path = d / f"{stream_name}_region_coverage_{z_label}.csv"
        if not cov_path.exists():
            continue
        cov = pd.read_csv(cov_path)
        if len(cov) == 0:
            continue

        aln_df = pd.read_csv(d / f"{stream_name}_alignment.csv") if (d / f"{stream_name}_alignment.csv").exists() else pd.DataFrame()
        clippers = sorted(aln_df["highlight_file"].unique()) if len(aln_df) > 0 else []

        n = len(cov)
        fig, ax = plt.subplots(figsize=(max(10, n * 0.55), 5))

        x = np.arange(n)
        bar_w = 0.6

        # Stacked bar: coverage fraction for each clipper
        clip_cols = [c for c in cov.columns if c.startswith("coverage_frac_") and c != "coverage_frac_any"]

        if clip_cols:
            bottoms = np.zeros(n)
            for j, col in enumerate(clip_cols):
                vals = cov[col].fillna(0).values
                # Cap each stack layer at remaining space up to 1.0
                vals_plot = np.minimum(vals, 1.0 - bottoms)
                color = CLIPPER_COLORS[j % len(CLIPPER_COLORS)]
                lbl = col.replace("coverage_frac_", "").strip()[:30]
                ax.bar(x, vals_plot, bar_w, bottom=bottoms, color=color, alpha=0.75, label=lbl)
                bottoms = np.minimum(bottoms + vals, 1.0)

        ax.axhline(0.50, color="black", linewidth=1.2, linestyle="--", alpha=0.7, label="50% threshold")
        ax.axhline(0.25, color="grey",  linewidth=0.8, linestyle=":",  alpha=0.6, label="25% threshold")

        # Region duration as dot marker on secondary axis
        ax2 = ax.twinx()
        ax2.scatter(x, cov["duration_s"].values / 60, color="black", s=25, alpha=0.5,
                    zorder=5, label="Duration (min)")
        ax2.set_ylabel("Region duration (min)", fontsize=9)
        ax2.yaxis.label.set_color("grey")

        ax.set_xticks(x)
        ax.set_xticklabels([f"R{int(i+1)}\n{seconds_to_hms(float(cov.iloc[i].start_seconds))}"
                             for i in range(n)], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Fraction of region covered by any clipper", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{stream_name} — Region coverage (Z≥{z_thresh})", fontsize=11)
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        out = d / f"{stream_name}_region_coverage_{z_label}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out.name}")


# ── Plot 3: engagement vs. clipped probability ────────────────────────────────

def plot_correlation(stream_name: str, d: Path):
    bin_path = d / f"{stream_name}_per_bin.csv"
    if not bin_path.exists():
        return
    df = pd.read_csv(bin_path)
    if "clipped_any" not in df.columns:
        return

    freq = df["normalized_frequency"].values
    clipped = df["clipped_any"].values.astype(float)

    # Bin the frequency values into 15 equal-width bins and compute mean P(clipped)
    q_edges = np.percentile(freq, np.linspace(0, 100, 16))
    q_edges = np.unique(q_edges)  # remove dupes
    if len(q_edges) < 4:
        return

    bin_indices = np.digitize(freq, q_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(q_edges) - 2)

    bin_centers = []
    bin_probs   = []
    bin_counts  = []
    for b in range(len(q_edges) - 1):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_centers.append(freq[mask].mean())
            bin_probs.append(clipped[mask].mean())
            bin_counts.append(mask.sum())

    bin_centers = np.array(bin_centers)
    bin_probs   = np.array(bin_probs)
    bin_counts  = np.array(bin_counts)

    fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter (jitter for binary y)
    jitter = np.random.default_rng(42).uniform(-0.02, 0.02, size=len(freq))
    ax_scatter.scatter(freq, clipped + jitter, alpha=0.03, s=3, color=SIGNAL_COLOR)
    ax_scatter.set_xlabel("Normalised chat frequency", fontsize=10)
    ax_scatter.set_ylabel("Clipped (0/1)", fontsize=10)
    ax_scatter.set_title(f"{stream_name}\nPoint-biserial correlation", fontsize=10)

    # Trend line (logistic-looking, just smoothed means)
    ax_scatter.scatter(bin_centers, bin_probs, color="red", s=60, zorder=5, label="Bin mean P(clipped)")
    z = np.polyfit(bin_centers, bin_probs, 1)
    xp = np.linspace(freq.min(), freq.max(), 100)
    ax_scatter.plot(xp, np.poly1d(z)(xp), "r--", linewidth=1.5, alpha=0.8)

    # Point-biserial r annotation
    from scipy.stats import pointbiserialr
    if clipped.sum() > 5:
        r, p = pointbiserialr(clipped, freq)
        ax_scatter.annotate(f"r = {r:.3f},  p = {p:.4f}", xy=(0.05, 0.93),
                            xycoords="axes fraction", fontsize=10,
                            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax_scatter.legend(fontsize=9)

    # Bar chart: binned P(clipped)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(bin_centers)))
    bars = ax_bar.bar(range(len(bin_centers)), bin_probs, color=colors, edgecolor="grey", linewidth=0.5)
    ax_bar.set_xticks(range(len(bin_centers)))
    ax_bar.set_xticklabels([f"{c:.2f}" for c in bin_centers], rotation=45, ha="right", fontsize=8)
    ax_bar.set_xlabel("Normalised frequency (bin mean)", fontsize=10)
    ax_bar.set_ylabel("P(clipped)", fontsize=10)
    ax_bar.set_title(f"P(clipped) by engagement quantile\n{stream_name}", fontsize=10)

    fig.tight_layout()
    out = d / f"{stream_name}_correlation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ── Plot 4: aggregate summary ─────────────────────────────────────────────────

def plot_aggregate():
    agg_path = RESEARCH_DIR / "aggregate_metrics.csv"
    if not agg_path.exists():
        return
    agg = pd.read_csv(agg_path)

    streams = agg["stream"].tolist()
    short_names = [s.replace("-", "\n") for s in streams]
    x = np.arange(len(streams))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ((ax1, ax2), (ax3, ax4)) = axes

    # ── Panel A: recall at Z≥0.0 and Z≥0.5 ──────────────────────────────────
    w = 0.35
    if "recall_s_z0" in agg.columns:
        ax1.bar(x - w/2, agg["recall_s_z0"].astype(float), w, color="#4a90d9", label="Z≥0.0")
    if "recall_s_z05" in agg.columns:
        ax1.bar(x + w/2, agg["recall_s_z05"].astype(float), w, color="#e63946", label="Z≥0.5", alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(short_names, fontsize=9)
    ax1.set_ylabel("Recall (engagement-seconds covered)"); ax1.set_ylim(0, 1)
    ax1.legend(fontsize=9); ax1.set_title("A. Recall: fraction of engagement-seconds clipped")
    ax1.axhline(0.5, color="grey", linestyle="--", alpha=0.5)

    # ── Panel B: regions covered at 50% threshold ─────────────────────────────
    if "regions_covered_at_50pct_z0" in agg.columns and "n_regions_z0" in agg.columns:
        frac_z0  = agg["regions_covered_at_50pct_z0"].astype(float) / agg["n_regions_z0"].astype(float)
        frac_z05 = agg["regions_covered_at_50pct_z05"].astype(float) / agg["n_regions_z05"].astype(float)
        ax2.bar(x - w/2, frac_z0, w, color="#4a90d9", label="Z≥0.0")
        ax2.bar(x + w/2, frac_z05, w, color="#e63946", label="Z≥0.5", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(short_names, fontsize=9)
    ax2.set_ylabel("Fraction of regions ≥50% covered"); ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9); ax2.set_title("B. Regions ≥50% covered by any clipper")
    ax2.axhline(0.5, color="grey", linestyle="--", alpha=0.5)

    # ── Panel C: point-biserial r ─────────────────────────────────────────────
    if "pointbiserial_r" in agg.columns:
        r_vals = agg["pointbiserial_r"].astype(float)
        colors_r = ["#4a90d9" if v >= 0 else "#e63946" for v in r_vals]
        ax3.bar(x, r_vals, color=colors_r, alpha=0.8)
    ax3.set_xticks(x); ax3.set_xticklabels(short_names, fontsize=9)
    ax3.set_ylabel("Point-biserial r (engagement vs. clipped)")
    ax3.set_title("C. Correlation: engagement vs. P(clipped)")
    ax3.axhline(0, color="black", linewidth=0.8)

    # ── Panel D: region coverage distribution heat-like bar chart ─────────────
    thr_cols_z0 = [(t, f"regions_covered_at_{t}pct_z0") for t in [0, 25, 50, 75]]
    n_reg_z0    = agg.get("n_regions_z0", pd.Series([1] * len(agg))).astype(float)
    fracs = {}
    for thr, col in thr_cols_z0:
        if col in agg.columns:
            fracs[thr] = (agg[col].astype(float) / n_reg_z0).values
        else:
            fracs[thr] = np.zeros(len(agg))

    bar_w_d = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    cmap = ["#d4f1f4", "#75e6da", "#189ab4", "#05445e"]
    for i, (thr, offset) in enumerate(zip([0, 25, 50, 75], offsets)):
        ax4.bar(x + offset * bar_w_d, fracs.get(thr, np.zeros(len(agg))),
                bar_w_d, color=cmap[i], label=f"≥{thr}%", alpha=0.9)
    ax4.set_xticks(x); ax4.set_xticklabels(short_names, fontsize=9)
    ax4.set_ylabel("Fraction of Z≥0.0 regions covered")
    ax4.set_title("D. Region coverage at multiple thresholds (Z≥0.0)")
    ax4.legend(fontsize=9, title="Coverage")
    ax4.set_ylim(0, 1.05)

    fig.suptitle("Chat Engagement vs. Human Clipping — Aggregate Results", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = RESEARCH_DIR / "aggregate_summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ── Coverage threshold detail plot ────────────────────────────────────────────

def plot_coverage_histogram(stream_name: str, d: Path, z_label: str = "z05"):
    cov_path = d / f"{stream_name}_region_coverage_{z_label}.csv"
    if not cov_path.exists():
        return
    cov = pd.read_csv(cov_path)
    if len(cov) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    z_thresh = "0.0" if z_label == "z0" else "0.5"

    # Coverage fraction histogram
    fracs = cov["coverage_frac_any"].values
    ax1.hist(fracs, bins=20, range=(0, 1), color=REGION_Z05_COLOR, alpha=0.7, edgecolor="white")
    ax1.axvline(0.50, color="black", linestyle="--", linewidth=1.5, label="50% threshold")
    ax1.axvline(0.25, color="grey",  linestyle=":",  linewidth=1.2, label="25% threshold")
    ax1.set_xlabel("Coverage fraction (clipped / region duration)")
    ax1.set_ylabel("# regions")
    ax1.set_title(f"{stream_name}\nCoverage fraction distribution (Z≥{z_thresh})")
    ax1.legend(fontsize=9)

    # Engagement vs coverage scatter
    ax2.scatter(cov["mean_engagement"], cov["coverage_frac_any"],
                s=cov["duration_s"] / 10, alpha=0.7, color=CLIPPER_COLORS[0],
                edgecolors="grey", linewidths=0.5)
    ax2.axhline(0.50, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.set_xlabel("Mean engagement in region (norm. freq.)")
    ax2.set_ylabel("Coverage fraction by any clipper")
    ax2.set_title(f"Engagement vs. coverage\n(bubble size = duration)")
    ax2.set_ylim(-0.05, 1.10)

    fig.tight_layout()
    out = d / f"{stream_name}_coverage_histogram_{z_label}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating visualizations...\n")

    for stream_name in STREAMS:
        d = RESEARCH_DIR / stream_name
        if not d.exists():
            continue
        print(f"  {stream_name}:")
        plot_timeline(stream_name, d)
        plot_region_coverage(stream_name, d)
        plot_correlation(stream_name, d)
        plot_coverage_histogram(stream_name, d, "z0")
        plot_coverage_histogram(stream_name, d, "z05")

    print("\nAggregate:")
    plot_aggregate()

    print("\nDone.")


if __name__ == "__main__":
    main()
