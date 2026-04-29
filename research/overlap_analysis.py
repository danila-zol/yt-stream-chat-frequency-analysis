#!/usr/bin/env python3
"""
Phase 3: Overlap analysis — high-engagement regions vs. clipped intervals.

For each stream:
  1. Load alignment CSV  → per-second "clipped" binary mask (by any highlight)
  2. Load regions CSVs (Z≥0.0 and Z≥0.5) → high-engagement intervals
  3. Load frequency CSV  → per-bin normalized chat frequency
  4. Compute:
     a. Per-region coverage fraction (by any clipper / by each clipper)
     b. Per-10s-bin clipped flag + engagement value  → used for correlation
     c. Summary metrics: recall, precision, point-biserial correlation

Outputs per stream (in research/<stream>/):
  <stream>_region_coverage_z0.csv   – per-region details at Z≥0.0
  <stream>_region_coverage_z05.csv  – per-region details at Z≥0.5
  <stream>_per_second.csv           – per-10s-bin: time, norm_freq, clipped_any, clipped_c1…
  <stream>_metrics.csv              – summary statistics
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

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

COVERAGE_THRESHOLDS = [0.0, 0.25, 0.50, 0.75]   # fraction of region that must be clipped


# ── Interval utilities ────────────────────────────────────────────────────────

def intervals_union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the union (merged) of a list of (start, end) intervals."""
    if not intervals:
        return []
    segs = sorted(intervals)
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def overlap_seconds(a_start: float, a_end: float,
                    intervals: list[tuple[float, float]]) -> float:
    """Total seconds of [a_start, a_end] covered by the union of intervals."""
    total = 0.0
    for s, e in intervals:
        lo = max(a_start, s)
        hi = min(a_end, e)
        if hi > lo:
            total += hi - lo
    return total


def build_second_mask(
    alignment_df: pd.DataFrame,
    stream_duration_s: float,
    clipper_col: str = "highlight_file",
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """
    Build per-second binary masks:
      - one mask per clipper (unique highlight_file values)
      - one combined 'any' mask

    Returns:
      (per_clipper_masks, any_mask, clipper_names)
      where each mask has length ceil(stream_duration_s) + 1
    """
    n = int(np.ceil(stream_duration_s)) + 2
    any_mask = np.zeros(n, dtype=np.int8)
    per_clipper: dict[str, np.ndarray] = {}

    clippers = sorted(alignment_df[clipper_col].unique())
    for c in clippers:
        per_clipper[c] = np.zeros(n, dtype=np.int8)

    for _, row in alignment_df.iterrows():
        s = max(0, int(row["stream_start_s"]))
        e = min(n - 1, int(np.ceil(row["stream_end_s"])))
        any_mask[s:e] = 1
        c = row["highlight_file"]
        if c in per_clipper:
            per_clipper[c][s:e] = 1

    return per_clipper, any_mask, clippers


def get_clipped_intervals(
    alignment_df: pd.DataFrame,
    clipper: str | None = None,
) -> list[tuple[float, float]]:
    """Return merged list of (stream_start_s, stream_end_s) tuples for one or all clippers."""
    if clipper is not None:
        rows = alignment_df[alignment_df["highlight_file"] == clipper]
    else:
        rows = alignment_df
    intervals = [(float(r.stream_start_s), float(r.stream_end_s)) for _, r in rows.iterrows()]
    return intervals_union(intervals)


# ── Region coverage ───────────────────────────────────────────────────────────

def compute_region_coverage(
    regions_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    freq_df: pd.DataFrame,
    clippers: list[str],
) -> pd.DataFrame:
    """For each region, compute clipped fraction by each clipper and combined."""
    all_clipped_intervals = get_clipped_intervals(alignment_df)
    per_clipper_intervals = {c: get_clipped_intervals(alignment_df, c) for c in clippers}

    rows = []
    for _, reg in regions_df.iterrows():
        rs = float(reg["start_seconds"])
        re = float(reg["end_seconds"])
        dur = re - rs
        if dur <= 0:
            continue

        covered_any = overlap_seconds(rs, re, all_clipped_intervals)
        frac_any = covered_any / dur

        # Mean engagement over the region
        mask = (freq_df["time_seconds"] >= rs) & (freq_df["time_seconds"] <= re)
        mean_eng = float(freq_df.loc[mask, "normalized_frequency"].mean()) if mask.any() else np.nan
        max_eng  = float(freq_df.loc[mask, "normalized_frequency"].max()) if mask.any() else np.nan

        row = {
            "region_index": int(reg["region_index"]),
            "start_seconds": rs,
            "end_seconds": re,
            "start_hms": reg["start_hms"],
            "end_hms": reg["end_hms"],
            "duration_s": dur,
            "mean_engagement": round(mean_eng, 4),
            "max_engagement": round(max_eng, 4),
            "covered_seconds_any": round(covered_any, 1),
            "coverage_frac_any": round(frac_any, 4),
        }
        for c in clippers:
            covered = overlap_seconds(rs, re, per_clipper_intervals[c])
            row[f"coverage_frac_{c[:20]}"] = round(covered / dur, 4)

        rows.append(row)

    df = pd.DataFrame(rows)
    for thresh in COVERAGE_THRESHOLDS:
        # Use strict > for 0% threshold to mean "any coverage at all"
        if thresh == 0.0:
            df["covered_at_0pct"] = (df["coverage_frac_any"] > 0.0).astype(int)
        else:
            df[f"covered_at_{int(thresh*100)}pct"] = (df["coverage_frac_any"] >= thresh).astype(int)
    return df


# ── Per-bin analysis ──────────────────────────────────────────────────────────

def compute_per_bin(
    freq_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    clippers: list[str],
    stream_duration_s: float,
) -> pd.DataFrame:
    """
    Join the frequency time-series with the clipped mask at 10s bin resolution.
    Produces per-bin: time_seconds, normalized_frequency, clipped_any, clipped_c1, …
    """
    all_intervals  = get_clipped_intervals(alignment_df)
    per_c_intervals = {c: get_clipped_intervals(alignment_df, c) for c in clippers}

    result = freq_df[["time_seconds", "time_hms", "normalized_frequency"]].copy()

    # Mark each bin as clipped if ANY part of the bin's 10s window is clipped
    def is_clipped(t, intervals):
        # t is the center of the bin; half-width = 5s
        lo, hi = t - 5, t + 5
        for s, e in intervals:
            if s < hi and e > lo:
                return 1
        return 0

    result["clipped_any"] = result["time_seconds"].apply(lambda t: is_clipped(t, all_intervals))
    for c in clippers:
        col = f"clipped_{c[:25]}"
        result[col] = result["time_seconds"].apply(lambda t: is_clipped(t, per_c_intervals[c]))

    return result


# ── Summary metrics ───────────────────────────────────────────────────────────

def compute_metrics(
    region_coverage_z0: pd.DataFrame,
    region_coverage_z05: pd.DataFrame,
    per_bin: pd.DataFrame,
    stream_name: str,
    stream_duration_s: float,
    alignment_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(name, value, note=""):
        rows.append({"metric": name, "value": value, "note": note})

    add("stream_name", stream_name)
    add("stream_duration_s", round(stream_duration_s, 1))
    add("n_unique_clippers", len(alignment_df["highlight_file"].unique()))

    # Total clipped seconds
    all_intervals = get_clipped_intervals(alignment_df)
    total_clipped = sum(e - s for s, e in all_intervals)
    add("total_clipped_s", round(total_clipped, 1))
    add("clipped_pct_of_stream", round(100 * total_clipped / stream_duration_s, 2) if stream_duration_s > 0 else np.nan)

    for z_label, rcov in [("z0", region_coverage_z0), ("z05", region_coverage_z05)]:
        n = len(rcov)
        if n == 0:
            continue
        dur_total = rcov["duration_s"].sum()
        add(f"n_regions_{z_label}", n)
        add(f"total_region_s_{z_label}", round(dur_total, 1))
        add(f"region_pct_of_stream_{z_label}", round(100 * dur_total / stream_duration_s, 2))

        for thresh in COVERAGE_THRESHOLDS:
            col = f"covered_at_{int(thresh*100)}pct"
            n_covered = int(rcov[col].sum())
            add(f"regions_covered_at_{int(thresh*100)}pct_{z_label}",
                n_covered, f"{n_covered}/{n} = {100*n_covered/n:.1f}%")

        add(f"mean_coverage_frac_{z_label}", round(rcov["coverage_frac_any"].mean(), 4))
        add(f"recall_s_{z_label}", round(rcov["covered_seconds_any"].sum() / dur_total, 4),
            "engagement-seconds covered / total engagement-seconds")

    # Precision: of all clipped seconds, what fraction falls in a high-engagement region?
    for z_label, rcov in [("z0", region_coverage_z0), ("z05", region_coverage_z05)]:
        if len(rcov) == 0 or len(all_intervals) == 0:
            continue
        region_intervals = [(float(r.start_seconds), float(r.end_seconds)) for _, r in rcov.iterrows()]
        region_intervals_merged = intervals_union(region_intervals)
        clipped_in_region = sum(
            overlap_seconds(s, e, region_intervals_merged)
            for s, e in all_intervals
        )
        total_clipped_local = sum(e - s for s, e in all_intervals)
        precision = clipped_in_region / total_clipped_local if total_clipped_local > 0 else np.nan
        add(f"precision_{z_label}", round(precision, 4),
            "clipped-in-region / total-clipped")

    # Point-biserial correlation: normalized frequency vs. clipped_any
    if "clipped_any" in per_bin.columns and len(per_bin) > 10:
        freq_vals = per_bin["normalized_frequency"].values
        clip_vals = per_bin["clipped_any"].values
        if clip_vals.sum() > 5 and (1 - clip_vals).sum() > 5:
            r, p = pointbiserialr(clip_vals, freq_vals)
            add("pointbiserial_r", round(r, 4))
            add("pointbiserial_p", round(p, 6))
        else:
            add("pointbiserial_r", np.nan, "not enough clipped/unclipped bins")
            add("pointbiserial_p", np.nan)

    return pd.DataFrame(rows)


# ── Per-stream orchestration ──────────────────────────────────────────────────

def analyze_stream(stream_name: str):
    print(f"\n{'='*62}")
    print(f"  {stream_name}")
    print(f"{'='*62}")

    d = RESEARCH_DIR / stream_name

    # Load frequency
    freq_path = d / f"{stream_name}_frequency.csv"
    if not freq_path.exists():
        print(f"  SKIP: frequency CSV not found")
        return None
    freq_df = pd.read_csv(freq_path)
    stream_duration_s = float(freq_df["time_seconds"].max()) + 60  # approx

    # Load alignment
    aln_path = d / f"{stream_name}_alignment.csv"
    if not aln_path.exists():
        print(f"  SKIP: alignment CSV not found")
        return None
    aln_df = pd.read_csv(aln_path)
    aln_df["stream_start_s"] = aln_df["stream_start_s"].clip(lower=0)
    clippers = sorted(aln_df["highlight_file"].unique())
    print(f"  Duration ≈ {seconds_to_hms(stream_duration_s)}")
    print(f"  Alignment: {len(aln_df)} segments  |  {len(clippers)} clipper(s)")
    for c in clippers:
        n = len(aln_df[aln_df["highlight_file"] == c])
        print(f"    • {c[:60]}  ({n} segs)")

    all_metrics = []

    for z_label, z_thresh in [("z0", 0.0), ("z05", 0.5)]:
        reg_path = d / f"{stream_name}_regions_{z_label}.csv"
        if not reg_path.exists():
            print(f"  SKIP: regions CSV not found: {reg_path.name}")
            continue
        regions_df = pd.read_csv(reg_path)
        print(f"\n  Regions Z≥{z_thresh}: {len(regions_df)} regions")

        rcov = compute_region_coverage(regions_df, aln_df, freq_df, clippers)
        out_path = d / f"{stream_name}_region_coverage_{z_label}.csv"
        rcov.to_csv(out_path, index=False)
        print(f"  Saved → {out_path.name}")

        # Quick coverage summary
        for thresh in COVERAGE_THRESHOLDS:
            col = f"covered_at_{int(thresh*100)}pct"
            n_cov = int(rcov[col].sum())
            print(f"    Coverage ≥{int(thresh*100)}%: {n_cov}/{len(rcov)} regions "
                  f"({100*n_cov/len(rcov):.0f}%)")

    # Per-bin table
    per_bin = compute_per_bin(freq_df, aln_df, clippers, stream_duration_s)
    bin_path = d / f"{stream_name}_per_bin.csv"
    per_bin.to_csv(bin_path, index=False)
    total_bins = len(per_bin)
    clipped_bins = int(per_bin["clipped_any"].sum())
    print(f"\n  Per-bin: {total_bins} bins, {clipped_bins} clipped "
          f"({100*clipped_bins/total_bins:.1f}%)")

    # Correlation
    freq_vals = per_bin["normalized_frequency"].values
    clip_vals = per_bin["clipped_any"].values
    if clip_vals.sum() > 5 and (1 - clip_vals).sum() > 5:
        r, p = pointbiserialr(clip_vals, freq_vals)
        print(f"  Point-biserial r = {r:.4f}  (p = {p:.4g})")

    # Summary metrics
    reg_z0_df  = pd.read_csv(d / f"{stream_name}_regions_z0.csv")  if (d / f"{stream_name}_regions_z0.csv").exists()  else pd.DataFrame()
    reg_z05_df = pd.read_csv(d / f"{stream_name}_regions_z05.csv") if (d / f"{stream_name}_regions_z05.csv").exists() else pd.DataFrame()

    rcov_z0  = pd.read_csv(d / f"{stream_name}_region_coverage_z0.csv")  if (d / f"{stream_name}_region_coverage_z0.csv").exists()  else pd.DataFrame()
    rcov_z05 = pd.read_csv(d / f"{stream_name}_region_coverage_z05.csv") if (d / f"{stream_name}_region_coverage_z05.csv").exists() else pd.DataFrame()

    metrics_df = compute_metrics(
        rcov_z0, rcov_z05, per_bin, stream_name, stream_duration_s, aln_df
    )
    met_path = d / f"{stream_name}_metrics.csv"
    metrics_df.to_csv(met_path, index=False)
    print(f"\n  Saved → {met_path.name}")

    # Print key metrics
    def get_metric(name):
        row = metrics_df[metrics_df["metric"] == name]
        return row["value"].values[0] if len(row) else None

    for z_label in ["z0", "z05"]:
        r  = get_metric(f"recall_s_{z_label}")
        pr = get_metric(f"precision_{z_label}")
        cov50 = get_metric(f"regions_covered_at_50pct_{z_label}")
        n_reg  = get_metric(f"n_regions_{z_label}")
        if r is not None and n_reg is not None:
            note = metrics_df.loc[metrics_df["metric"] == f"regions_covered_at_50pct_{z_label}", "note"].values
            print(f"  [Z≥{z_label[1:].replace('05','0.5').replace('0','0.0')}]  recall={r:.3f}  precision={pr:.3f}  "
                  + (f"regions@50%={note[0]}" if len(note) else ""))

    return metrics_df


def build_aggregate():
    """Collect key metrics from all streams into a single summary CSV."""
    all_rows = []
    for stream in STREAMS:
        d = RESEARCH_DIR / stream
        met_path = d / f"{stream}_metrics.csv"
        if not met_path.exists():
            continue
        df = pd.read_csv(met_path)
        row = {"stream": stream}
        for _, r in df.iterrows():
            row[r["metric"]] = r["value"]
        all_rows.append(row)

    if all_rows:
        agg = pd.DataFrame(all_rows)
        agg_path = RESEARCH_DIR / "aggregate_metrics.csv"
        agg.to_csv(agg_path, index=False)
        print(f"\n  Aggregate metrics → {agg_path}")
        return agg
    return None


def main():
    print("Overlap analysis — all streams")

    for stream_name in STREAMS:
        analyze_stream(stream_name)

    print("\n" + "="*62)
    agg = build_aggregate()
    if agg is not None:
        print("\nAggregate summary:")
        cols = [c for c in agg.columns if c in [
            "stream", "n_regions_z0", "recall_s_z0", "precision_z0",
            "regions_covered_at_50pct_z0",
            "n_regions_z05", "recall_s_z05", "precision_z05",
            "regions_covered_at_50pct_z05",
            "pointbiserial_r",
        ]]
        if cols:
            print(agg[cols].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
