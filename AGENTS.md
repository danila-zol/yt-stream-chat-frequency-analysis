# Agent Instructions – Chat Frequency Analysis Tools

This repository contains Python tools for analyzing chat message frequency in video streams using a sliding‑window convolution algorithm.

## Architecture

The project is organized as a proper Python package (`chatfreq`) with separated concerns:

```
chatfreq/
├── src/chatfreq/
│   ├── core/           # Pure processing logic (no web/CLI deps)
│   │   ├── analyzer.py # ChatFrequencyAnalyzer class
│   │   ├── segmentation.py  # Circular Binary Segmentation (CBS)
│   │   ├── subtitle_processor.py  # SRT parsing & sentence boundaries
│   │   ├── audio_processor.py     # Python-based silence detection
│   │   ├── intelligent_cutter.py  # Subtitle + audio boundary refiner
│   │   ├── time_utils.py
│   │   └── data_utils.py
│   ├── cli/            # Click-based CLI commands
│   │   ├── commands.py
│   │   └── renderer.py
│   └── web/            # Dash/Plotly web visualization
│       ├── app.py      # Dash app factory
│       ├── layout.py   # UI layout
│       ├── callbacks.py
│       └── components.py
├── research/           # Validation study: chat engagement vs. human highlights
│   ├── clipping_research.md    # Study report (manuscript-quality)
│   ├── generate_pdf.sh        # Pandoc → XeLaTeX script
│   └── *.py                  # Alignment, overlap, and visualisation scripts
├── pyproject.toml      # Package configuration
└── requirements*.txt    # Dependencies
```

## Setup & Dependencies

**Installation (editable mode):**
```bash
pip install -e .
```

**Core dependencies:**
- `pandas`, `numpy`, `scipy` – required for processing
- `nltk` – sentence tokenization for subtitle analysis

**CLI dependencies:**
- `click` – for command-line interface

**Web dependencies:**
- `dash`, `plotly`, `kaleido` – for interactive dashboard

**All dependencies:**
```bash
pip install -r requirements.txt
```

## Data Format & Preprocessing

**Input format:**
- Tab‑separated CSV (TSV) with columns: `video_time`, `author`, `message`
- `video_time` must be in `mm:ss`, `h:mm:ss`, or `hh:mm:ss` format

**Preprocessing YouTube chat logs:**
```bash
python reduce_yt_chat_metadata.py "video.live_chat.json" --csv
```
Outputs `video.live_chat_reduced.tsv` ready for analysis.

## CLI Commands

After installation (`pip install -e .`), use the `chatfreq` command:

### Serve (Web Dashboard)
```bash
chatfreq serve chat.csv [OPTIONS]

Options:
  --port, -p          Port (default: 8050)
  --host              Host (default: 127.0.0.1)
  --no-browser        Don't auto-open browser
  --debug             Enable debug mode
  --video PATH        Preload video path
  --window SECONDS    Default window size (default: 60)
  --step SECONDS      Default step size (default: 10)
  --degree N          Default polynomial degree (default: 3)
  --prominence N      Default peak prominence (default: 5)
```

**Web dashboard trace toggles:**
- `Histogram Bars` / `Sliding Window` / `Trend Line` / `Peak Markers`
- `High engagement regions` – shows CBS-derived regions as a red semi-transparent track (IGV-style)
- `Normalize by trend` – replaces sliding-window line with `rolling_sum / trend` ratio; auto-hides trend line and rescales prominence slider to ratio-friendly range (0.1–2.0). Both peaks and CBS regions are computed on the normalized signal.

**Additional web sliders (when regions are enabled):**
- `CBS Sensitivity (t-threshold)` – default 2.5
- `Region Z-Score Threshold` – default 0.0
- `Min Region Duration (s)` – default 30
- `Max Gap to Merge (s)` – default 30

### Plot (Static PNG)
```bash
chatfreq plot chat.csv [OPTIONS]

Options:
  --output, -o PATH   Output PNG file
  --window SECONDS    Window size (default: 60)
  --step SECONDS     Step size (default: 10)
  --degree N         Polynomial trend degree (default: 3)
  --dpi N           Output DPI (default: 100)
  --no-bars          Hide histogram bars
  --no-trend         Hide trend line
  --start TIME       Start time filter
  --end TIME         End time filter
```

### Export Commands
```bash
chatfreq export sliding chat.csv [OPTIONS]
chatfreq export peaks chat.csv [OPTIONS]
chatfreq export timestamps chat.csv [OPTIONS]
chatfreq export ffmpeg chat.csv [OPTIONS]

Common Options:
  --window SECONDS    Window size (default: 60)
  --step SECONDS     Step size (default: 10)
  --prominence N     Peak prominence (default: 5)
  --start TIME       Start time filter (mm:ss, h:mm:ss, or seconds)
  --end TIME         End time filter

Export-specific options:
  sliding:  --degree N (polynomial trend, 0=skip)
  peaks:    --normalize, --degree N
  timestamps: --clip-before SEC, --clip-after SEC, --normalize, --degree N
  ffmpeg:   --video PATH (required), --clip-before SEC, --clip-after SEC,
            --region-cuts (use one cut per region instead of peaks),
            --cbs-threshold N, --z-threshold N,
            --min-duration SEC, --max-gap SEC,
            --normalize, --degree N,
            --subtitle PATH, --intelligent-cut, --max-snap SEC,
            --silence-threshold-db N, --min-silence-duration SEC,
            --fallback-expand
```

**Time format examples:**
- `5:00` → 5 minutes
- `1:30:00` → 1 hour 30 minutes
- `5400` → raw seconds (also valid)

## Video Clipping Workflow

**Prerequisite:** FFmpeg must be installed and available in PATH.

### Intelligent Cutting
Instead of the simple "±X seconds around peak" approach, you can enable **Intelligent Cutting** which uses the subtitle transcript to locate sentence boundaries and then snaps the cut to the nearest silence gap in the audio. This avoids cutting the streamer mid-sentence or mid-word.

**How it works:**
1. Parse the SRT file and deduplicate overlapping YouTube-style rolling captions.
2. Detect sentence boundaries with NLTK Punkt.
3. Demux the video audio to WAV via FFmpeg, then analyse short-time RMS energy in NumPy.
4. For each rough cut time, find the nearest sentence boundary within the max-snap radius, then snap to the nearest silence edge (start or end of a quiet gap).
5. If no boundary is found, fall back to the original rough cut (or expand the search radius if the toggle is on).

Audio silence detection is cached next to the video (`video.mp4.chatfreq_silence.json`) so repeated exports are instant.

### CLI Export Example
```bash
# Export peaks from 5:00 to 15:00
chatfreq export peaks chat.csv --start 5:00 --end 15:00

# Generate FFmpeg script for highlight reel (simple cuts)
chatfreq export ffmpeg chat.csv --video stream.mp4 \
    --clip-before 30 --clip-after 60 \
    --start 5:00 --end 15:00

# Intelligent cutting (requires --subtitle)
chatfreq export ffmpeg chat.csv --video stream.mp4 --subtitle stream.srt \
    --intelligent-cut --max-snap 5.0 \
    --silence-threshold-db -45 --min-silence-duration 0.3 \
    --clip-before 30 --clip-after 60
```

### Web Dashboard Workflow
1. Enable peaks in trace toggles, adjust prominence
2. Optionally enable `High engagement regions` to visualize CBS-derived segments
3. Tune region sliders (CBS sensitivity, Z-score, duration, gap) as needed
4. Set video path via CLI `--video` or dropdown; subtitle file auto-detects if it shares the video basename
5. Configure clip durations (before/after)
6. Check **"Use intelligent cutting"** to reveal subtitle/audio snap controls
7. Adjust **Max Snap Distance**, **Silence Threshold (dB)**, and **Min Silence Duration** sliders
8. Optionally check **"Expand search if no boundary found"**
9. Optionally check "Only include peaks in current zoom region"
10. Optionally check "Use one cut per high-engagement region" for region-based FFmpeg cuts
11. Click export buttons (PNG, CSV, Peaks CSV, FFmpeg, Timestamps)

## Implementation Notes

- **Caching:** `ChatFrequencyAnalyzer` uses `@lru_cache` for histogram, sliding‑window, polynomial, peak, and high-engagement-region computations
- **Algorithm:** Sliding window uses `np.convolve(hist, np.ones(window_bins), mode='valid')`
- **Segmentation:** Circular Binary Segmentation (`cbs_segment`) recursively splits the signal by maximising the two-sample t-statistic at each split, stopping when the statistic falls below a tunable threshold
- **Normalization:** `compute_normalized_signal()` divides the sliding-window sum by a polynomial trend (with a floor at 10% of mean trend) to produce dimensionless ratios. This corrects for stream-wide timing biases (e.g. start/end hype) so peaks and regions reflect *relative* rather than absolute engagement
- **Time formatting:** `seconds_to_hms()` converts seconds to `H:MM:SS` or `M:SS`
- **Package entry point:** `chatfreq` console script in pyproject.toml
- **Intelligent Cutting:** `IntelligentCutter` refines rough clip boundaries by (1) locating the nearest sentence start/end from `SubtitleProcessor` within a user-defined snap radius, then (2) snapping to the closest audio silence edge from `AudioProcessor` for precise cut timing. Subtitle deduplication handles YouTube's overlapping rolling-word SRT format; audio analysis uses NumPy RMS on FFmpeg-demuxed mono WAV with 95th-percentile relative dB thresholding.

## Research

A validation study in `research/` examines how well chat-engagement signals (CBS high-engagement regions) predict which moments human clippers select for independent highlight reels.

- **`research/clipping_research.md`** – Full study report written in publication-ready Markdown.
- **`research/generate_pdf.sh`** – Pandoc → XeLaTeX (DejaVu fonts) script to regenerate the PDF; run from within `research/`.
- **Supporting scripts:** `run_chatfreq.py`, `align_audio_mel.py`, `overlap_analysis.py`, `visualize.py`.

The study covers 4 streams (2 VTubers) and 8 independent highlight reels. The preferred alignment method is a 32-bin mel-spectrogram NCC with offset clustering; earlier subtitle n-gram and raw-RMS approaches are kept for comparison but are documented as superseded.

## Testing & Quality

**Run tests:**
```bash
pip install -r requirements-dev.txt
pytest
```

**Lint:**
```bash
ruff check src/
```

## Backward Compatibility

The original scripts still work:
```bash
python interactive_chat_frequency.py chat.csv --port 8050
python plot_chat_frequency.py chat.csv --output plot.png
```

## Git

- Repository uses standard git workflow
- Run `ruff check src/` before committing
