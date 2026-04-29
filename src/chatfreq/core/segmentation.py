"""
Circular Binary Segmentation (CBS) for chat frequency data.

This module implements a fast, threshold-based CBS algorithm inspired by
the Olshen et al. (2004) method for DNA copy-number analysis.
"""
from typing import List, Tuple

import numpy as np


def cbs_segment(data: np.ndarray, threshold: float = 2.5, min_seg: int = 5) -> List[Tuple[int, int]]:
    """
    Fast circular binary segmentation using a t-statistic threshold.

    Recursively splits the data into segments by finding the split point
    that maximises the two-sample t-statistic between the left and right
    sub-segments.  A segment is only split if the best t-statistic exceeds
    ``threshold``.

    Parameters
    ----------
    data : np.ndarray
        1-D array of sliding-window frequency values.
    threshold : float
        Minimum t-statistic required to accept a split (default 2.5).
    min_seg : int
        Minimum number of points that each sub-segment must contain.

    Returns
    -------
    List[Tuple[int, int]]
        List of (start, end) index pairs (end is exclusive).
    """
    n = len(data)
    if n < 2 * min_seg:
        return [(0, n)]

    segments: List[Tuple[int, int]] = [(0, n)]
    changed = True

    while changed:
        changed = False
        new_segments: List[Tuple[int, int]] = []

        for start, end in segments:
            seg_len = end - start
            if seg_len < 2 * min_seg:
                new_segments.append((start, end))
                continue

            seg_data = data[start:end]
            seg_n = len(seg_data)

            # Prefix sums for O(1) range queries
            cumsum = np.cumsum(seg_data)
            cumsum_sq = np.cumsum(seg_data ** 2)

            splits = np.arange(min_seg, seg_n - min_seg)

            s1 = cumsum[splits - 1]
            s2 = cumsum[-1] - s1
            sq1 = cumsum_sq[splits - 1]
            sq2 = cumsum_sq[-1] - sq1
            n1 = splits.astype(float)
            n2 = (seg_n - splits).astype(float)

            m1 = s1 / n1
            m2 = s2 / n2
            v1 = sq1 / n1 - m1 ** 2
            v2 = sq2 / n2 - m2 ** 2

            # Pooled variance; guard against zero-variance segments
            pooled_var = (n1 * v1 + n2 * v2) / (n1 + n2 - 2)
            pooled_var = np.where(pooled_var <= 1e-12, np.nan, pooled_var)

            t_stats = np.full(len(splits), np.nan)
            valid = ~np.isnan(pooled_var)
            t_stats[valid] = np.abs(m1[valid] - m2[valid]) / np.sqrt(
                pooled_var[valid] * (1.0 / n1[valid] + 1.0 / n2[valid])
            )

            if np.all(np.isnan(t_stats)):
                new_segments.append((start, end))
                continue

            best_idx = int(np.nanargmax(t_stats))
            best_t = t_stats[best_idx]
            best_split = int(splits[best_idx])

            if best_t > threshold:
                mid = start + best_split
                new_segments.extend([(start, mid), (mid, end)])
                changed = True
            else:
                new_segments.append((start, end))

        segments = new_segments

    return segments


def classify_and_merge_regions(
    time_axis: np.ndarray,
    rolling_sum: np.ndarray,
    segments: List[Tuple[int, int]],
    z_threshold: float = 0.0,
    min_duration: float = 30.0,
    max_gap: float = 30.0,
    step: float = 10.0,
) -> List[Tuple[float, float]]:
    """
    Classify CBS segments and merge them into high-engagement regions.

    Parameters
    ----------
    time_axis : np.ndarray
        Time coordinates for each point in ``rolling_sum``.
    rolling_sum : np.ndarray
        Sliding-window frequency values.
    segments : List[Tuple[int, int]]
        Output from :func:`cbs_segment`.
    z_threshold : float
        Segments whose mean is >= global_mean + z_threshold * global_std
        are labelled high-engagement.
    min_duration : float
        Minimum duration (seconds) for a region to be retained.
    max_gap : float
        Adjacent regions separated by <= ``max_gap`` seconds are merged.
    step : float
        Step size in seconds (used to convert index spans to time spans).

    Returns
    -------
    List[Tuple[float, float]]
        Merged, filtered (start_time, end_time) regions.
    """
    if len(rolling_sum) == 0 or not segments:
        return []

    global_mean = float(np.mean(rolling_sum))
    global_std = float(np.std(rolling_sum))
    if global_std == 0:
        return []

    # 1. Classify segments
    high_regions_idx: List[Tuple[int, int]] = []
    for s, e in segments:
        seg_mean = float(np.mean(rolling_sum[s:e]))
        if seg_mean >= global_mean + z_threshold * global_std:
            high_regions_idx.append((s, e))

    if not high_regions_idx:
        return []

    # 2. Merge adjacent high-engagement segments
    merged_idx = [high_regions_idx[0]]
    for s, e in high_regions_idx[1:]:
        last_s, last_e = merged_idx[-1]
        if s <= last_e:
            merged_idx[-1] = (last_s, e)
        else:
            merged_idx.append((s, e))

    # 3. Convert to time coordinates
    # time_axis[i] is the centre of the i-th bin.
    # A segment [s, e) spans from the left edge of bin s to the right edge of bin e-1.
    regions: List[Tuple[float, float]] = []
    for s, e in merged_idx:
        start_t = float(time_axis[s] - step / 2)
        end_t = float(time_axis[e - 1] + step / 2) if e > 0 else float(time_axis[s] + step / 2)
        regions.append((max(0.0, start_t), end_t))

    # 4. Merge small gaps
    if len(regions) > 1:
        final_regions = [regions[0]]
        for r_start, r_end in regions[1:]:
            last_start, last_end = final_regions[-1]
            if r_start - last_end <= max_gap:
                final_regions[-1] = (last_start, r_end)
            else:
                final_regions.append((r_start, r_end))
        regions = final_regions

    # 5. Filter by minimum duration
    regions = [(s, e) for s, e in regions if e - s >= min_duration]

    return regions
