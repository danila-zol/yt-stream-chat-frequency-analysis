"""
Subtitle parsing and sentence-boundary extraction for intelligent video cutting.

Handles YouTube-style rolling-word SRT files where consecutive cues overlap
heavily.  Deduplicates them into clean chunks, then uses NLTK Punkt to find
sentence boundaries and map them back to approximate timestamps.
"""
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


def _ensure_nltk_punkt() -> None:
    try:
        import nltk
    except ImportError as exc:  # pragma: no cover
        raise ImportError("nltk is required for subtitle processing. pip install nltk") from exc
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)


@dataclass(frozen=True)
class SrtCue:
    index: int
    start: float  # seconds
    end: float  # seconds
    text: str


@dataclass(frozen=True)
class SentenceInterval:
    start: float
    end: float
    text: str


def _srt_time_to_seconds(timestr: str) -> float:
    """Convert '00:03:08,949' -> 188.949."""
    timestr = timestr.strip().replace(",", ".")
    # some files use '.' instead of ','
    parts = timestr.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0
        m, s = parts
    else:
        raise ValueError(f"Bad SRT time: {timestr}")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_srt(path: str) -> List[SrtCue]:
    """Parse a standard SubRip file into a list of cues."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Subtitle file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]

    cues: List[SrtCue] = []
    i = 0
    n = len(lines)
    while i < n:
        # skip empty / whitespace-only lines
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break

        # optional index line
        idx = 0
        if re.match(r"^\s*\d+\s*$", lines[i]):
            idx = int(lines[i].strip())
            i += 1
            while i < n and not lines[i].strip():
                i += 1

        if i >= n:
            break

        # timecode line
        time_line = lines[i].strip()
        i += 1
        m = re.match(r"([\d:,.]+)\s*-->\s*([\d:,.]+)", time_line)
        if not m:
            # malformed – skip forward to next blank line to recover
            while i < n and lines[i].strip():
                i += 1
            continue

        start = _srt_time_to_seconds(m.group(1))
        end = _srt_time_to_seconds(m.group(2))

        # Some YouTube SRTs insert a blank line between the timecode and the text.
        if i < n and not lines[i].strip():
            i += 1

        # collect text lines until blank line or EOF
        text_lines: List[str] = []
        while i < n and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1

        text = "\n".join(text_lines)
        cues.append(SrtCue(index=idx, start=start, end=end, text=text))
        # loop will skip the trailing blank line(s)

    cues.sort(key=lambda c: c.start)
    return cues


def _clean_text(text: str) -> str:
    """Normalise whitespace, strip music/sfx tags? Keep them – they mark pauses."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def deduplicate_rolling_cues(cues: List[SrtCue], short_cue_threshold: float = 0.05) -> List[Tuple[float, float, str]]:
    """
    Turn overlapping YouTube-style cues into non-overlapping text chunks.

    Returns list of (chunk_start, chunk_end, text).
    """
    # Filter out empty cues first
    nonempty = [c for c in cues if _clean_text(c.text)]

    chunks: List[Tuple[float, str]] = []  # (start, text)
    prev_text = ""
    for cue in nonempty:
        text = _clean_text(cue.text)
        # Skip very short transition cues – they are just markers.
        if cue.end - cue.start < short_cue_threshold:
            # still update prev_text so the diff works for the next real cue
            prev_text = text
            continue

        if text == prev_text:
            continue

        if prev_text and text.startswith(prev_text):
            new_part = text[len(prev_text) :].strip()
            if new_part:
                chunks.append((cue.start, new_part))
        else:
            # New block or after a reset
            chunks.append((cue.start, text))

        prev_text = text

    if not chunks:
        return []

    # Assign end times from the next chunk's start; last chunk gets last cue's end.
    last_cue_end = nonempty[-1].end if nonempty else 0.0
    result = []
    for i, (start, text) in enumerate(chunks):
        end = chunks[i + 1][0] if i + 1 < len(chunks) else last_cue_end
        result.append((start, end, text))
    return result


def _split_chunks_by_sentence(chunks: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    """Break long multi-sentence chunks into smaller pieces with interpolated times."""
    _ensure_nltk_punkt()
    from nltk.tokenize import PunktSentenceTokenizer
    tokenizer = PunktSentenceTokenizer()
    out: List[Tuple[float, float, str]] = []
    for t_start, t_end, text in chunks:
        sents = list(tokenizer.tokenize(text))
        if len(sents) <= 1:
            out.append((t_start, t_end, text))
            continue
        duration = t_end - t_start
        sub_dur = duration / len(sents)
        for i, sent in enumerate(sents):
            sub_start = t_start + i * sub_dur
            sub_end = t_start + (i + 1) * sub_dur
            out.append((sub_start, sub_end, sent))
    return out


def build_sentence_intervals(chunks: List[Tuple[float, float, str]]) -> List[SentenceInterval]:
    """
    Run NLTK Punkt over the concatenated chunk text and map sentence ends
    back to chunk timestamps.
    """
    if not chunks:
        return []

    _ensure_nltk_punkt()
    from nltk.tokenize import PunktSentenceTokenizer

    # Build one long string and a char-offset -> time mapping via chunk spans.
    full_text = ""
    chunk_spans: List[Tuple[int, int, float, float]] = []  # char_start, char_end, t_start, t_end
    for t_start, t_end, text in chunks:
        if full_text and not full_text.endswith(" "):
            full_text += " "
        char_start = len(full_text)
        full_text += text
        char_end = len(full_text)
        chunk_spans.append((char_start, char_end, t_start, t_end))

    tokenizer = PunktSentenceTokenizer()
    try:
        spans = list(tokenizer.span_tokenize(full_text))
    except Exception:
        # If tokenization fails (e.g. empty or only symbols), treat whole text as one sentence
        spans = [(0, len(full_text))]

    intervals: List[SentenceInterval] = []
    chunk_idx = 0
    for s_start, s_end in spans:
        # advance chunk_idx so that s_start falls inside (or at the start of) chunk_idx
        while (
            chunk_idx < len(chunk_spans)
            and chunk_spans[chunk_idx][1] <= s_start
        ):
            chunk_idx += 1
        if chunk_idx < len(chunk_spans):
            sent_start = chunk_spans[chunk_idx][2]
        else:
            sent_start = chunk_spans[-1][3] if chunk_spans else 0.0

        # advance from chunk_idx to find the chunk containing s_end
        j = chunk_idx
        while j < len(chunk_spans) and chunk_spans[j][1] < s_end:
            j += 1
        if j < len(chunk_spans):
            sent_end = chunk_spans[j][3]
        else:
            sent_end = chunk_spans[-1][3] if chunk_spans else 0.0

        sent_text = full_text[s_start:s_end]
        intervals.append(
            SentenceInterval(
                start=float(sent_start),
                end=float(sent_end),
                text=sent_text,
            )
        )

    # Sort and deduplicate by time (Punkt shouldn't overlap, but be safe)
    intervals.sort(key=lambda x: (x.start, x.end))
    deduped: List[SentenceInterval] = []
    for iv in intervals:
        if deduped and abs(iv.start - deduped[-1].start) < 0.01 and abs(iv.end - deduped[-1].end) < 0.01:
            continue
        deduped.append(iv)
    return deduped


class SubtitleProcessor:
    """High-level wrapper around SRT parsing -> dedup -> sentence intervals."""

    def __init__(self, path: str):
        self.path = path
        cues = parse_srt(path)
        self.chunks = deduplicate_rolling_cues(cues)
        self.split_chunks = _split_chunks_by_sentence(self.chunks)
        self.sentences = build_sentence_intervals(self.split_chunks)

    def get_sentence_intervals(self) -> List[SentenceInterval]:
        return self.sentences

    def find_nearest_sentence_start(
        self, t: float, max_distance: float
    ) -> Optional[SentenceInterval]:
        """Return the sentence whose start time is closest to *t* within max_distance."""
        best: Optional[SentenceInterval] = None
        best_dist = float("inf")
        for sent in self.sentences:
            dist = abs(sent.start - t)
            if dist <= max_distance and dist < best_dist:
                best = sent
                best_dist = dist
        return best

    def find_nearest_sentence_end(
        self, t: float, max_distance: float
    ) -> Optional[SentenceInterval]:
        """Return the sentence whose end time is closest to *t* within max_distance."""
        best: Optional[SentenceInterval] = None
        best_dist = float("inf")
        for sent in self.sentences:
            dist = abs(sent.end - t)
            if dist <= max_distance and dist < best_dist:
                best = sent
                best_dist = dist
        return best
