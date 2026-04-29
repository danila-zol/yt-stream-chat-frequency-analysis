"""
Audio silence detection using Python signal analysis.

FFmpeg is used only to demux audio into a temporary WAV file; all analysis
(RMS energy, thresholding, gap detection) is done in NumPy so we can tune
sensitivity precisely.
"""
import json
import os
import subprocess
import tempfile
from typing import List, Optional, Tuple

import numpy as np
from scipy.io import wavfile


def _cache_path(video_path: str) -> str:
    return video_path + ".chatfreq_silence.json"


def _load_cached(
    video_path: str, frame_ms: int, hop_ms: int, threshold_db: float, min_silence_ms: float
) -> Optional[List[Tuple[float, float]]]:
    cache = _cache_path(video_path)
    if not os.path.exists(cache):
        return None
    try:
        with open(cache, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != 1:
            return None
        p = data.get("params", {})
        if (
            p.get("frame_ms") != frame_ms
            or p.get("hop_ms") != hop_ms
            or abs(p.get("threshold_db", 0) - threshold_db) > 0.01
            or abs(p.get("min_silence_ms", 0) - min_silence_ms) > 0.1
        ):
            return None
        return [tuple(pair) for pair in data["intervals"]]
    except Exception:
        return None


def _save_cached(
    video_path: str,
    intervals: List[Tuple[float, float]],
    frame_ms: int,
    hop_ms: int,
    threshold_db: float,
    min_silence_ms: float,
) -> None:
    cache = _cache_path(video_path)
    data = {
        "version": 1,
        "params": {
            "frame_ms": frame_ms,
            "hop_ms": hop_ms,
            "threshold_db": threshold_db,
            "min_silence_ms": min_silence_ms,
        },
        "intervals": intervals,
    }
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _extract_audio(video_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Use ffmpeg to write mono s16le WAV to a temp file, return float32 samples in [-1,1]."""
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                tmp_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        sr, audio = wavfile.read(tmp_path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype in (np.float32, np.float64):
            audio = audio.astype(np.float32)
        else:
            raise RuntimeError(f"Unexpected WAV dtype {audio.dtype}")
        return audio
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _rms_energy(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Return RMS for each frame.  Pure NumPy with chunked vectorisation."""
    num_frames = (len(audio) - frame_length) // hop_length + 1
    if num_frames <= 0:
        return np.array([], dtype=np.float32)
    rms = np.empty(num_frames, dtype=np.float32)
    chunk_size = 100_000  # frames per batch
    # Create a [chunk_size, frame_length] index grid once per chunk
    for start in range(0, num_frames, chunk_size):
        end = min(start + chunk_size, num_frames)
        idx = np.arange(start, end)[:, None] * hop_length + np.arange(frame_length)
        frames = audio[idx]
        # Use float64 for the mean to avoid underflow on very quiet signals
        rms[start:end] = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)).astype(np.float32)
    return rms


def detect_silence(
    video_path: str,
    sample_rate: int = 16000,
    frame_ms: int = 30,
    hop_ms: int = 10,
    threshold_db: float = -45.0,
    min_silence_ms: float = 200.0,
) -> List[Tuple[float, float]]:
    """
    Detect silent gaps in *video_path*.

    Parameters
    ----------
    threshold_db : float
        dB relative to the 95th-percentile loudness.  More negative = stricter.
    min_silence_ms : float
        Minimum gap length to be considered a silence.

    Returns
    -------
    List of (start_sec, end_sec) silence intervals.
    """
    cached = _load_cached(video_path, frame_ms, hop_ms, threshold_db, min_silence_ms)
    if cached is not None:
        return cached

    audio = _extract_audio(video_path, sample_rate)
    frame_length = int(sample_rate * frame_ms / 1000)
    hop_length = int(sample_rate * hop_ms / 1000)

    rms = _rms_energy(audio, frame_length, hop_length)
    if rms.size == 0:
        return []

    # Normalise to 95th percentile so the threshold is video-independent.
    ref = float(np.percentile(rms, 95))
    if ref <= 0:
        ref = 1.0
    rms_safe = np.where(rms == 0, 1e-10, rms)
    db = 20.0 * np.log10(rms_safe / ref)

    is_silence = db < threshold_db
    min_frames = max(1, int(min_silence_ms / hop_ms))

    diffs = np.diff(is_silence.astype(np.int8))
    silence_starts = (np.where(diffs == 1)[0] + 1).tolist()
    silence_ends = (np.where(diffs == -1)[0] + 1).tolist()

    if is_silence[0]:
        silence_starts.insert(0, 0)
    if is_silence[-1]:
        silence_ends.append(len(is_silence))

    intervals: List[Tuple[float, float]] = []
    for s, e in zip(silence_starts, silence_ends):
        if e - s >= min_frames:
            t_start = s * hop_ms / 1000.0
            t_end = e * hop_ms / 1000.0
            intervals.append((t_start, t_end))

    _save_cached(video_path, intervals, frame_ms, hop_ms, threshold_db, min_silence_ms)
    return intervals


class AudioProcessor:
    """Thin wrapper around detect_silence with configurable parameters."""

    def __init__(
        self,
        video_path: str,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        hop_ms: int = 10,
        threshold_db: float = -45.0,
        min_silence_ms: float = 200.0,
    ):
        self.video_path = video_path
        self.params = {
            "sample_rate": sample_rate,
            "frame_ms": frame_ms,
            "hop_ms": hop_ms,
            "threshold_db": threshold_db,
            "min_silence_ms": min_silence_ms,
        }
        self._intervals: Optional[List[Tuple[float, float]]] = None

    def get_silence_intervals(self) -> List[Tuple[float, float]]:
        if self._intervals is None:
            self._intervals = detect_silence(self.video_path, **self.params)
        return self._intervals

    def find_nearest_silence_edge(
        self,
        t: float,
        edge: str = "start",  # or "end"
        tolerance: float = 1.0,
    ) -> Optional[float]:
        """
        Find a silence interval whose *edge* is within *tolerance* of *t*.
        Returns the edge time, or None.
        """
        best_dist = float("inf")
        best_time: Optional[float] = None
        for s, e in self.get_silence_intervals():
            candidate = s if edge == "start" else e
            dist = abs(candidate - t)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_time = candidate
        return best_time
