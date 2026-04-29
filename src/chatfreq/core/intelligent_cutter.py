"""
Intelligent clip-boundary refiner.

Combines subtitle sentence boundaries with audio silence gaps to snap rough
cut times to clean speech boundaries.
"""
from typing import List, Tuple

from .audio_processor import AudioProcessor
from .subtitle_processor import SubtitleProcessor


class IntelligentCutter:
    """
    Refines rough clip times using subtitles + audio silence.

    Typical usage::

        cutter = IntelligentCutter("video.mp4", "subs.srt")
        refined = cutter.refine_clips(
            [(120.0, 180.0), (300.0, 360.0)],
            max_snap_distance=3.0,
            silence_tolerance=1.0,
        )
    """

    def __init__(
        self,
        video_path: str,
        subtitle_path: str,
        audio_sample_rate: int = 16000,
        audio_frame_ms: int = 30,
        audio_hop_ms: int = 10,
        silence_threshold_db: float = -45.0,
        min_silence_ms: float = 200.0,
    ):
        self.video_path = video_path
        self.subtitle = SubtitleProcessor(subtitle_path)
        self.audio = AudioProcessor(
            video_path,
            sample_rate=audio_sample_rate,
            frame_ms=audio_frame_ms,
            hop_ms=audio_hop_ms,
            threshold_db=silence_threshold_db,
            min_silence_ms=min_silence_ms,
        )

    def _refine_start(
        self,
        t: float,
        max_snap: float,
        silence_tolerance: float,
        fallback_expand: bool,
    ) -> float:
        # 1) find nearest sentence start within max_snap
        sent = self.subtitle.find_nearest_sentence_start(t, max_snap)
        if sent is None:
            if fallback_expand:
                sent = self.subtitle.find_nearest_sentence_start(t, float("inf"))
            if sent is None:
                return t
        candidate = sent.start
        # 2) snap to silence edge (end of silence == speech resumes)
        silence_end = self.audio.find_nearest_silence_edge(
            candidate, edge="end", tolerance=silence_tolerance
        )
        if silence_end is not None:
            candidate = silence_end
        return candidate

    def _refine_end(
        self,
        t: float,
        max_snap: float,
        silence_tolerance: float,
        fallback_expand: bool,
    ) -> float:
        # 1) find nearest sentence end within max_snap
        sent = self.subtitle.find_nearest_sentence_end(t, max_snap)
        if sent is None:
            if fallback_expand:
                sent = self.subtitle.find_nearest_sentence_end(t, float("inf"))
            if sent is None:
                return t
        candidate = sent.end
        # 2) snap to silence edge (start of silence == speech pauses)
        silence_start = self.audio.find_nearest_silence_edge(
            candidate, edge="start", tolerance=silence_tolerance
        )
        if silence_start is not None:
            candidate = silence_start
        return candidate

    def refine_clips(
        self,
        rough_clips: List[Tuple[float, float]],
        max_snap_distance: float = 3.0,
        silence_tolerance: float = 1.0,
        fallback_expand: bool = False,
    ) -> List[Tuple[float, float]]:
        """
        Snap each rough (start, end) to the best boundary found.

        Sanity checks ensure start < end and that the clip hasn't collapsed
        to zero length.
        """
        refined: List[Tuple[float, float]] = []
        for rs, re in rough_clips:
            new_start = self._refine_start(
                rs, max_snap_distance, silence_tolerance, fallback_expand
            )
            new_end = self._refine_end(
                re, max_snap_distance, silence_tolerance, fallback_expand
            )
            # If refinement inverted the clip, fall back to original.
            if new_start >= new_end:
                new_start, new_end = rs, re
            # Also enforce a tiny minimum duration so we don't produce 0-length cuts
            if new_end - new_start < 0.5:
                new_start, new_end = rs, re
            refined.append((new_start, new_end))
        return refined
