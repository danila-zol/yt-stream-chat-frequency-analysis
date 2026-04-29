"""
Core chat frequency analyzer with LRU caching for interactive updates.

This module contains only the processing logic with no CLI or web dependencies.
"""
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .data_utils import load_chat_data
from .segmentation import cbs_segment, classify_and_merge_regions


class ChatFrequencyAnalyzer:
    """
    Core analyzer for chat message frequency using sliding window convolution.

    Uses LRU caching for expensive computations to enable responsive
    interactive updates.
    """

    CACHE_SIZE = 128

    def __init__(self, csv_path: str):
        """
        Initialize analyzer with chat data from TSV file.

        Args:
            csv_path: Path to tab-separated CSV file with columns:
                     video_time, author, message
        """
        self.csv_path = csv_path
        self.df = load_chat_data(csv_path)
        self.timestamps = self.df["seconds"].values
        self.max_seconds = self.timestamps.max()
        self.timestamps_set = set(self.timestamps)

    def filter_by_time(
        self,
        data: np.ndarray,
        time_axis: np.ndarray,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter data by time range.

        Args:
            data: Data array to filter
            time_axis: Time values corresponding to data points
            start: Start time in seconds (inclusive), or None for beginning
            end: End time in seconds (inclusive), or None for end

        Returns:
            Tuple of (filtered_time_axis, filtered_data)
        """
        if start is None and end is None:
            return time_axis, data

        mask = np.ones(len(time_axis), dtype=bool)
        if start is not None:
            mask &= time_axis >= start
        if end is not None:
            mask &= time_axis <= end

        return time_axis[mask], data[mask]

    @lru_cache(maxsize=CACHE_SIZE)
    def compute_histogram(self, step: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (bin_edges, hist) for given step size.

        Args:
            step: Bin size in seconds

        Returns:
            Tuple of (bin_edges array, histogram counts array)
        """
        if step <= 0:
            raise ValueError("Step must be positive")

        bin_edges = np.arange(0, self.max_seconds + step, step)
        hist, _ = np.histogram(self.timestamps, bins=bin_edges)
        return bin_edges, hist

    @lru_cache(maxsize=CACHE_SIZE)
    def compute_sliding_window(
        self, step: float, window_size: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (time_axis, rolling_sum) for given parameters.

        Args:
            step: Bin size in seconds
            window_size: Sliding window size in seconds

        Returns:
            Tuple of (time_axis centers, rolling sum values)
        """
        bin_edges, hist = self.compute_histogram(step)

        if window_size % step != 0:
            window_size = round(window_size / step) * step

        window_bins = int(window_size / step)
        if window_bins < 1:
            raise ValueError(f"Window size {window_size}s smaller than step {step}s")
        if window_bins > len(hist):
            raise ValueError("Window size exceeds stream duration")

        rolling_sum = np.convolve(hist, np.ones(window_bins), mode="valid")
        time_axis = bin_edges[: len(hist) - window_bins + 1] + window_size / 2

        if len(time_axis) != len(rolling_sum):
            raise RuntimeError("Length mismatch between time_axis and rolling_sum")

        return time_axis, rolling_sum

    @lru_cache(maxsize=CACHE_SIZE)
    def compute_polynomial_trend(
        self, step: float, window_size: float, degree: int
    ) -> Optional[np.ndarray]:
        """
        Return trend line values for given parameters.

        Args:
            step: Bin size in seconds
            window_size: Sliding window size in seconds
            degree: Polynomial degree (0 = no trend)

        Returns:
            Array of trend values, or None if degree <= 0
        """
        if degree <= 0:
            return None

        time_axis, rolling_sum = self.compute_sliding_window(step, window_size)
        if len(time_axis) < degree + 1:
            return np.zeros_like(time_axis)

        coeff = np.polyfit(time_axis, rolling_sum, degree)
        poly = np.poly1d(coeff)
        return poly(time_axis)

    @lru_cache(maxsize=CACHE_SIZE)
    def compute_normalized_signal(
        self,
        step: float,
        window_size: float,
        degree: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (time_axis, normalized_signal) where signal is rolling_sum / trend.

        Args:
            step: Bin size in seconds
            window_size: Sliding window size in seconds
            degree: Polynomial degree for trend (must be > 0)

        Returns:
            Tuple of (time_axis, normalized values). Normalized values are
            dimensionless ratios (1.0 = on-trend, 2.0 = 2x expected).
        """
        time_axis, rolling_sum = self.compute_sliding_window(step, window_size)
        trend = self.compute_polynomial_trend(step, window_size, degree)

        if trend is None or len(trend) == 0:
            return time_axis, rolling_sum

        mean_trend = float(np.mean(trend))
        floor = max(1e-6, 0.1 * mean_trend)
        safe_trend = np.where(trend < floor, floor, trend)

        normalized = rolling_sum / safe_trend
        return time_axis, normalized

    def detect_peaks(
        self,
        step: float,
        window_size: float,
        prominence: float = 5.0,
        min_distance_ratio: float = 0.5,
        normalize: bool = False,
        degree: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect local maxima in sliding window frequency.

        Args:
            step: Bin size in seconds
            window_size: Sliding window size in seconds
            prominence: Minimum prominence for peak detection
            min_distance_ratio: Minimum distance between peaks as ratio of window_size
            normalize: If True, detect peaks on trend-normalized signal
            degree: Polynomial degree for trend when normalize=True

        Returns:
            Tuple of (peak_times, peak_values) arrays
        """
        if normalize:
            time_axis, signal = self.compute_normalized_signal(step, window_size, degree)
        else:
            time_axis, signal = self.compute_sliding_window(step, window_size)

        if len(signal) < 3:
            return np.array([]), np.array([])

        min_distance = max(1, int((window_size * min_distance_ratio) / step))

        peaks_idx, _ = find_peaks(signal, prominence=prominence, distance=min_distance)
        peak_times = time_axis[peaks_idx]
        peak_values = signal[peaks_idx]

        return peak_times, peak_values

    @lru_cache(maxsize=CACHE_SIZE)
    def detect_high_engagement_regions(
        self,
        step: float,
        window_size: float,
        cbs_threshold: float = 2.5,
        z_threshold: float = 0.0,
        min_duration: float = 30.0,
        max_gap: float = 30.0,
        normalize: bool = False,
        degree: int = 3,
    ) -> List[Tuple[float, float]]:
        """
        Detect high-engagement regions using circular binary segmentation.

        Args:
            step: Bin size in seconds
            window_size: Sliding window size in seconds
            cbs_threshold: Minimum t-statistic required to accept a CBS split
            z_threshold: Segments with mean >= global_mean + z_threshold * global_std
                         are labelled high-engagement
            min_duration: Minimum duration (seconds) for a region to be retained
            max_gap: Adjacent regions separated by <= max_gap seconds are merged
            normalize: If True, run CBS on trend-normalized signal
            degree: Polynomial degree for trend when normalize=True

        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        if normalize:
            time_axis, signal = self.compute_normalized_signal(step, window_size, degree)
        else:
            time_axis, signal = self.compute_sliding_window(step, window_size)

        if len(signal) < 20:
            return []

        min_seg = max(3, int(min_duration / step / 2))
        segments = cbs_segment(signal, threshold=cbs_threshold, min_seg=min_seg)

        return classify_and_merge_regions(
            time_axis=time_axis,
            rolling_sum=signal,
            segments=segments,
            z_threshold=z_threshold,
            min_duration=min_duration,
            max_gap=max_gap,
            step=step,
        )

    def get_messages_near_time(
        self, center_time: float, margin: float = 5.0, limit: int = 5
    ) -> pd.DataFrame:
        """
        Return messages within ±margin seconds of center_time.

        Args:
            center_time: Center time in seconds
            margin: Time margin in seconds
            limit: Maximum number of messages to return

        Returns:
            DataFrame of messages near the specified time
        """
        mask = (
            (self.df["seconds"] >= center_time - margin)
            & (self.df["seconds"] <= center_time + margin)
        )
        return self.df[mask].head(limit)

    def get_sample_messages_at_peaks(
        self,
        peak_times: np.ndarray,
        margin: float = 10.0,
        max_msg_length: int = 50,
    ) -> List[str]:
        """
        Get sample messages near each peak time.

        Args:
            peak_times: Array of peak times
            margin: Time margin around each peak
            max_msg_length: Maximum length of each message sample

        Returns:
            List of message samples, one per peak
        """
        samples = []
        for t in peak_times:
            messages = self.get_messages_near_time(t, margin=margin, limit=3)
            sample = "; ".join(
                str(row["message"])[:max_msg_length] for _, row in messages.iterrows()
            )
            samples.append(sample)
        return samples

    def export_sliding_window(
        self,
        step: float = 10.0,
        window_size: float = 60.0,
        start: Optional[int] = None,
        end: Optional[int] = None,
        include_trend: bool = False,
        degree: int = 3,
    ) -> pd.DataFrame:
        """
        Export sliding window frequency data.

        Args:
            step: Bin size in seconds
            window_size: Sliding window size in seconds
            start: Start time filter in seconds
            end: End time filter in seconds
            include_trend: Whether to include polynomial trend column
            degree: Polynomial degree for trend

        Returns:
            DataFrame with columns: time_seconds, time_hms, frequency, (trend)
        """
        time_axis, rolling_sum = self.compute_sliding_window(step, window_size)

        from .time_utils import seconds_to_hms

        if start is not None or end is not None:
            time_axis, rolling_sum = self.filter_by_time(
                rolling_sum, time_axis, start, end
            )

        df = pd.DataFrame(
            {
                "time_seconds": time_axis,
                "time_hms": [seconds_to_hms(t) for t in time_axis],
                "frequency": rolling_sum,
            }
        )

        if include_trend and degree > 0:
            trend = self.compute_polynomial_trend(step, window_size, degree)
            if trend is not None:
                _, trend_filtered = self.filter_by_time(trend, time_axis, start, end)
                df["trend"] = trend_filtered

        return df

    def export_peaks(
        self,
        step: float = 10.0,
        window_size: float = 60.0,
        prominence: float = 5.0,
        start: Optional[int] = None,
        end: Optional[int] = None,
        margin: float = 10.0,
        normalize: bool = False,
        degree: int = 3,
    ) -> pd.DataFrame:
        """
        Export detected peaks with sample messages.

        Args:
            step: Bin size in seconds
            window_size: Sliding window size in seconds
            prominence: Minimum prominence for peak detection
            start: Start time filter in seconds
            end: End time filter in seconds
            margin: Time margin for sample messages
            normalize: If True, detect peaks on trend-normalized signal
            degree: Polynomial degree for trend when normalize=True

        Returns:
            DataFrame with columns: peak_time_seconds, peak_time_hms, peak_frequency, sample_messages
        """
        peak_times, peak_values = self.detect_peaks(
            step, window_size, prominence, normalize=normalize, degree=degree
        )

        if start is not None or end is not None:
            mask = np.ones(len(peak_times), dtype=bool)
            if start is not None:
                mask &= peak_times >= start
            if end is not None:
                mask &= peak_times <= end
            peak_times = peak_times[mask]
            peak_values = peak_values[mask]

        from .time_utils import seconds_to_hms

        samples = self.get_sample_messages_at_peaks(peak_times, margin=margin)

        return pd.DataFrame(
            {
                "peak_time_seconds": peak_times,
                "peak_time_hms": [seconds_to_hms(t) for t in peak_times],
                "peak_frequency": peak_values,
                "sample_messages": samples,
            }
        )

    def export_timestamps(
        self,
        step: float = 10.0,
        window_size: float = 60.0,
        prominence: float = 5.0,
        clip_before: int = 30,
        clip_after: int = 30,
        start: Optional[int] = None,
        end: Optional[int] = None,
        normalize: bool = False,
        degree: int = 3,
        intelligent_cutter = None,
        max_snap_distance: float = 3.0,
        silence_tolerance: float = 1.0,
        fallback_expand: bool = False,
    ) -> pd.DataFrame:
        """
        Export timestamps for video editing.

        Args:
            step: Bin size in seconds
            window_size: Sliding window size in seconds
            prominence: Minimum prominence for peak detection
            clip_before: Seconds before peak for start time
            clip_after: Seconds after peak for end time
            start: Start time filter in seconds
            end: End time filter in seconds
            normalize: If True, detect peaks on trend-normalized signal
            degree: Polynomial degree for trend when normalize=True
            intelligent_cutter: Optional IntelligentCutter instance
            max_snap_distance: Max distance to snap a boundary
            silence_tolerance: Tolerance for matching silence edges
            fallback_expand: If True, expand search if no boundary found

        Returns:
            DataFrame with columns: clip_index, start_seconds, peak_seconds, end_seconds, start_hms, peak_hms, end_hms
        """
        peak_times, peak_values = self.detect_peaks(
            step, window_size, prominence, normalize=normalize, degree=degree
        )

        if start is not None or end is not None:
            mask = np.ones(len(peak_times), dtype=bool)
            if start is not None:
                mask &= peak_times >= start
            if end is not None:
                mask &= peak_times <= end
            peak_times = peak_times[mask]
            peak_values = peak_values[mask]

        from .time_utils import seconds_to_hms

        clips = []
        for t in peak_times:
            clips.append((max(0, t - clip_before), t + clip_after))

        if intelligent_cutter is not None:
            clips = intelligent_cutter.refine_clips(
                clips,
                max_snap_distance=max_snap_distance,
                silence_tolerance=silence_tolerance,
                fallback_expand=fallback_expand,
            )

        rows = []
        for i, ((c_start, c_end), t, v) in enumerate(zip(clips, peak_times, peak_values)):
            rows.append(
                {
                    "clip_index": i + 1,
                    "start_seconds": c_start,
                    "peak_seconds": t,
                    "end_seconds": c_end,
                    "start_hms": seconds_to_hms(c_start),
                    "peak_hms": seconds_to_hms(t),
                    "end_hms": seconds_to_hms(c_end),
                    "peak_frequency": v,
                }
            )

        return pd.DataFrame(rows)

    def generate_ffmpeg_commands(
        self,
        video_path: str,
        step: float = 10.0,
        window_size: float = 60.0,
        prominence: float = 5.0,
        clip_before: int = 30,
        clip_after: int = 30,
        start: Optional[int] = None,
        end: Optional[int] = None,
        use_regions: bool = False,
        cbs_threshold: float = 2.5,
        z_threshold: float = 0.0,
        min_duration: float = 30.0,
        max_gap: float = 30.0,
        normalize: bool = False,
        degree: int = 3,
        intelligent_cutter = None,
        max_snap_distance: float = 3.0,
        silence_tolerance: float = 1.0,
        fallback_expand: bool = False,
    ) -> List[str]:
        """
        Generate FFmpeg commands for highlight reel.

        Args:
            video_path: Path to video file
            step: Bin size in seconds
            window_size: Sliding window size in seconds
            prominence: Minimum prominence for peak detection
            clip_before: Seconds before peak for clip start
            clip_after: Seconds after peak for clip end
            start: Start time filter in seconds
            end: End time filter in seconds
            use_regions: If True, use one cut per high-engagement region instead of peaks
            cbs_threshold: CBS t-statistic threshold (when use_regions=True)
            z_threshold: Z-score threshold for high engagement (when use_regions=True)
            min_duration: Minimum region duration (when use_regions=True)
            max_gap: Max gap to merge regions (when use_regions=True)
            normalize: If True, use trend-normalized signal for detection
            degree: Polynomial degree for trend when normalize=True
            intelligent_cutter: Optional IntelligentCutter instance
            max_snap_distance: Max distance to snap a boundary
            silence_tolerance: Tolerance for matching silence edges
            fallback_expand: If True, expand search if no boundary found

        Returns:
            List of command lines for bash script
        """
        if use_regions:
            regions = self.detect_high_engagement_regions(
                step, window_size, cbs_threshold, z_threshold, min_duration, max_gap,
                normalize=normalize, degree=degree,
            )

            if start is not None or end is not None:
                filtered = []
                for r_start, r_end in regions:
                    if start is not None and r_end < start:
                        continue
                    if end is not None and r_start > end:
                        continue
                    filtered.append((max(r_start, start) if start else r_start, min(r_end, end) if end else r_end))
                regions = filtered

            # Expand by clip_before/clip_after and merge overlaps
            clips = []
            for r_start, r_end in regions:
                c_start = max(0, r_start - clip_before)
                c_end = r_end + clip_after
                clips.append((c_start, c_end))

            if intelligent_cutter is not None:
                clips = intelligent_cutter.refine_clips(
                    clips,
                    max_snap_distance=max_snap_distance,
                    silence_tolerance=silence_tolerance,
                    fallback_expand=fallback_expand,
                )

            # Merge overlapping/adjacent clips
            if clips:
                clips.sort(key=lambda x: x[0])
                merged = [list(clips[0])]
                for c_start, c_end in clips[1:]:
                    if c_start <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], c_end)
                    else:
                        merged.append([c_start, c_end])
                clip_times = merged
            else:
                clip_times = []

            label = "high-engagement regions"
        else:
            peak_times, _ = self.detect_peaks(
                step, window_size, prominence, normalize=normalize, degree=degree
            )

            if start is not None or end is not None:
                mask = np.ones(len(peak_times), dtype=bool)
                if start is not None:
                    mask &= peak_times >= start
                if end is not None:
                    mask &= peak_times <= end
                peak_times = peak_times[mask]

            clips = []
            for t in peak_times:
                c_start = max(0, t - clip_before)
                c_end = t + clip_after
                clips.append((c_start, c_end))

            if intelligent_cutter is not None:
                clips = intelligent_cutter.refine_clips(
                    clips,
                    max_snap_distance=max_snap_distance,
                    silence_tolerance=silence_tolerance,
                    fallback_expand=fallback_expand,
                )

            clip_times = []
            for c_start, c_end in clips:
                clip_times.append([c_start, c_end])

            label = "peaks"

        filter_parts = []
        concat_inputs = []

        for i, (c_start, c_end) in enumerate(clip_times):
            v_filter = f"[0:v]trim={c_start}:{c_end},setpts=PTS-STARTPTS[v{i}]"
            a_filter = f"[0:a]atrim={c_start}:{c_end},asetpts=PTS-STARTPTS[a{i}]"

            filter_parts.append(v_filter)
            filter_parts.append(a_filter)
            concat_inputs.extend([f"[v{i}]", f"[a{i}]"])

        n_clips = len(clip_times)
        concat_str = "".join(concat_inputs)
        concat_filter = f"{concat_str}concat=n={n_clips}:v=1:a=1[outv][outa]"
        filter_parts.append(concat_filter)

        filter_complex = "; ".join(filter_parts)

        import time as time_module

        output_file = f"highlight_reel_{int(time_module.time())}.mp4"
        cmd = (
            f'ffmpeg -i "{video_path}" -filter_complex "{filter_complex}" '
            f'-map "[outv]" -map "[outa]" -c:v libx264 -preset fast -crf 22 '
            f'-c:a aac -b:a 192k "{output_file}"'
        )

        commands = [
            "#!/bin/bash",
            f"# FFmpeg command generated from chat frequency {label}",
            f"# Video: {video_path}",
            f"# Clips: {clip_before}s before, {clip_after}s after each {label}",
            f"# Number of clips: {n_clips}",
            "",
            cmd,
        ]

        return commands
