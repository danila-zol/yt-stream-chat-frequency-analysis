# Chat Frequency Analysis Tools

Interactive and static tools for analyzing chat message frequency throughout video streams using a sliding‑window convolution algorithm.

## Overview

These Python tools analyze YouTube/Twitch chat logs to visualize how chat activity changes over time. They use a **convolution-based sliding window algorithm** to produce smoothed frequency curves, with options for polynomial trend fitting, peak detection, and advanced highlight generation that respects speech boundaries.

## Research

A validation study examining the correlation between live-chat engagement and human highlight-selection decisions is documented in the `research/` directory. It compares Circular Binary Segmentation engagement regions against eight independently produced highlight reels (four streams, two VTubers) using mel-spectrogram audio alignment.

- **Study report:** [`research/clipping_research.md`](research/clipping_research.md)
- **PDF export script:** [`research/generate_pdf.sh`](research/generate_pdf.sh) (re-runs pandoc + XeLaTeX with DejaVu fonts)

## Features

- **Sliding window convolution**: Smoothed frequency tracking via NumPy convolution
- **Interactive web dashboard**: Real-time parameter adjustment, zooming, hover previews
- **Static PNG plotting**: Batch processing for fixed parameters
- **Full CLI export**: Sliding window CSV, peaks, timestamps, FFmpeg highlight reels
- **Time filtering**: Limit exports to specific time ranges via CLI or UI zoom
- **High engagement regions**: Circular Binary Segmentation finds sustained hype stretches, not just single peaks
- **Intelligent cutting** (NEW): Snap clip boundaries to subtitle sentence ends and audio silence gaps so you never cut the streamer mid-word
- **LRU caching**: Fast parameter updates for responsive interaction

## Installation

```bash
pip install -e .
```

This installs the `chatfreq` CLI tool with all core dependencies.

**Optional extras:**
```bash
pip install -e ".[all]"   # includes web dashboard (dash, plotly, kaleido) + matplotlib
pip install -e ".[web]"   # web dashboard only
pip install -e ".[cli]"   # CLI only
```

## Quick Start

```bash
# Launch interactive dashboard
chatfreq serve chat.csv

# Generate static plot
chatfreq plot chat.csv --output plot.png

# Export peaks from 5:00 to 15:00
chatfreq export peaks chat.csv --start 5:00 --end 15:00

# Generate highlight reel script (simple cuts)
chatfreq export ffmpeg chat.csv --video stream.mp4 --clip-before 30 --clip-after 60

# Intelligent cutting (snap to sentence/silence boundaries)
chatfreq export ffmpeg chat.csv --video stream.mp4 --subtitle stream.srt \
    --intelligent-cut --max-snap 5.0 \
    --silence-threshold-db -45 --min-silence-duration 0.3
```

## Input Format

The tools expect a **tab-separated CSV** with columns:
- `video_time`: Timestamp in format `mm:ss`, `h:mm:ss`, or `hh:mm:ss`
- `author`: Username
- `message`: Chat message text

Example:
```csv
video_time	author	message
0:01	@aix_bert	:_nimiHeart:
0:01	@saidaht6	Yeah, YT been doing the subscriber chat bug
0:02	@C.Fanbert	thanks for all the cool ore
```

## Preprocessing YouTube Chat Data

Use `reduce_yt_chat_metadata.py` to convert YouTube's `.live_chat.json` format:
```bash
python3 reduce_yt_chat_metadata.py "video.live_chat.json" --csv
```

## CLI Commands

### `chatfreq serve` - Web Dashboard

```bash
chatfreq serve chat.csv [OPTIONS]

Options:
  --port, -p          Port (default: 8050)
  --host              Host (default: 127.0.0.1)
  --no-browser        Don't auto-open browser
  --debug             Enable debug mode
  --video PATH        Preload video path for clipping
  --window SECONDS    Default window size (default: 60)
  --step SECONDS      Default step size (default: 10)
  --degree N          Default polynomial degree (default: 3)
  --prominence N      Default peak prominence (default: 5)
```

### `chatfreq plot` - Static PNG

```bash
chatfreq plot chat.csv [OPTIONS]

Options:
  --output, -o PATH   Output PNG file
  --window SECONDS   Window size (default: 60)
  --step SECONDS     Step size (default: 10)
  --degree N        Polynomial trend degree (default: 3)
  --dpi N           Output DPI (default: 100)
  --no-bars         Hide histogram bars
  --no-trend        Hide trend line
  --start TIME      Start time filter
  --end TIME        End time filter
```

### `chatfreq export` - Data Export

```bash
# Sliding window frequency CSV
chatfreq export sliding chat.csv [OPTIONS]

# Peak timestamps with sample messages
chatfreq export peaks chat.csv [OPTIONS]

# Timestamps for video editing
chatfreq export timestamps chat.csv [OPTIONS]

# FFmpeg highlight reel bash script
chatfreq export ffmpeg chat.csv --video stream.mp4 [OPTIONS]
```

**Common export options:**
- `--window SECONDS`  Window size (default: 60)
- `--step SECONDS`    Step size (default: 10)
- `--prominence N`    Peak prominence (default: 5)
- `--start TIME`      Start time filter (mm:ss, h:mm:ss, or seconds)
- `--end TIME`        End time filter

**ffmpeg-specific options:**
- `--video PATH`              Video file (required)
- `--clip-before SEC`         Seconds before peak (default: 30)
- `--clip-after SEC`          Seconds after peak (default: 30)
- `--region-cuts`             Use one cut per high-engagement region instead of peaks
- `--cbs-threshold N`         CBS sensitivity (default: 2.5)
- `--z-threshold N`           Region Z-score threshold (default: 0.0)
- `--min-duration SEC`        Minimum region duration (default: 30)
- `--max-gap SEC`             Max gap to merge regions (default: 30)
- `--normalize`               Use trend-normalized signal
- `--degree N`                Polynomial degree for trend
- `--subtitle PATH`           Subtitle file for intelligent cutting
- `--intelligent-cut`         Enable boundary snapping to sentence/silence edges
- `--max-snap SEC`            Max distance to snap a cut (default: 3.0)
- `--silence-threshold-db N`  Silence threshold in dB (default: -45)
- `--min-silence-duration SEC` Minimum silence duration (default: 0.3)
- `--fallback-expand`         Expand search if no boundary found

## Web Dashboard

Launch the interactive Dash/Plotly dashboard with:
```bash
chatfreq serve chat.csv --video stream.mp4
```

### Trace toggles
- **Histogram Bars** / **Sliding Window** / **Trend Line** / **Peak Markers**
- **High engagement regions** - Shows CBS-derived regions as a red semi-transparent track
- **Normalize by trend** - Replaces sliding-window line with `rolling_sum / trend` ratio; auto-hides trend line and rescales prominence slider

### Parameters
| Parameter | Effect | Typical Range | Default |
|-----------|--------|---------------|---------|
| Window Size | Smoothing. Larger = smoother | 10-300s | 60s |
| Step Size | Resolution. Smaller = more detail | 1-30s | 10s |
| Polynomial Degree | Trend complexity | 0-5 | 3 |
| Peak Prominence | Minimum peak height | 1-20 | 5 |
| CBS Sensitivity | T-stat threshold for region splits | 1.0-5.0 | 2.5 |
| Region Z-Score | Mean above this is "high engagement" | -1.0-2.0 | 0.0 |
| Min Region Duration | Drop regions shorter than this | 0-120s | 30s |
| Max Gap to Merge | Merge regions closer than this | 0-120s | 30s |

### Video Clipping Panel
1. Set video path via CLI `--video` or dropdown
2. Configure clip durations (**Clip Before** / **Clip After**)
3. Optionally check **"Use one cut per high-engagement region"** for longer highlights
4. Check **"Use intelligent cutting"** to reveal:
   - **Subtitle file** (auto-detected if same basename as video)
   - **Max Snap Distance** (default 3 s)
   - **Silence Threshold (dB)** (default -45)
   - **Min Silence Duration** (default 0.3 s)
   - **"Expand search if no boundary found"** fallback toggle
5. Click **Generate FFmpeg Commands** or **Export Timestamps**

## Intelligent Cutting

Instead of naively cutting at `peak - clip_before` / `peak + clip_after`, **Intelligent Cutting** uses two signals to find natural speech boundaries:

1. **Subtitle sentence boundaries** - Parses the SRT, deduplicates YouTube's overlapping rolling captions, and uses NLTK Punkt to detect sentence starts/ends.
2. **Audio silence gaps** - Demuxes the video audio to WAV via FFmpeg, then computes short-time RMS energy in NumPy to find silent pauses. Thresholds are relative to the 95th percentile loudness so they adapt per-video.

**Algorithm for each rough cut time:**
1. Find the nearest sentence boundary within the **max snap distance**
2. Snap to the nearest silence edge (start of silence for end-cuts, end of silence for start-cuts) within a small tolerance
3. Fall back to the original rough cut if nothing suitable is found (or expand the search radius if enabled)

Audio silence detection is cached next to the video (`video.mp4.chatfreq_silence.json`) so repeated exports are instant.

## Algorithm

1. **Histogram binning**: Messages are counted in fixed-size intervals (`step` parameter)
2. **Sliding window convolution**: A window of size `W` slides across the histogram, summing counts
3. **Implementation**: `np.convolve(hist, np.ones(window_bins), mode='valid')`
4. **Trend fitting**: Optional polynomial regression (degree 0-5)
5. **Peak detection**: Local maxima with adjustable prominence threshold
6. **Region detection**: Circular Binary Segmentation splits the signal into statistically different segments, then classifies and merges high-engagement ones

## Architecture

```
chatfreq/
├── src/chatfreq/
│   ├── core/                # Pure processing (no web/CLI deps)
│   │   ├── analyzer.py      # ChatFrequencyAnalyzer
│   │   ├── segmentation.py  # Circular Binary Segmentation
│   │   ├── subtitle_processor.py  # SRT parsing & sentence boundaries
│   │   ├── audio_processor.py     # Silence detection
│   │   ├── intelligent_cutter.py  # Boundary refiner
│   │   ├── time_utils.py
│   │   └── data_utils.py
│   ├── cli/                 # Click CLI commands
│   │   ├── commands.py
│   │   └── renderer.py
│   └── web/                 # Dash/Plotly visualization
│       ├── app.py
│       ├── layout.py
│       ├── callbacks.py
│       └── components.py
└── pyproject.toml          # Package configuration
```

The **core module** contains only the processing logic with no dependencies on web frameworks or CLI libraries.

## Troubleshooting

- **"Missing columns" error**: Ensure CSV has `video_time`, `author`, `message` columns (tab-separated)
- **PNG export fails**: Install `kaleido`: `pip install kaleido`
- **No peaks detected**: Lower prominence or increase window size
- **Slow updates**: Use larger step size or window for fewer computation points
- **Subtitle not found**: Check that the SRT/VTT file shares the same basename as the video, or select it manually

## License

This repository uses a dual-license model:

- **Software (all code)** is released under the [BSD 3-Clause License](LICENSE).
- **Research report** (`research/clipping_research.md`, `research/clipping_research.pdf`, and related figures) is released under the [Creative Commons Attribution-ShareAlike 4.0 International License](LICENSE-CC-BY-SA).

In other words: you may freely use, modify, and redistribute the code under BSD-3 terms, while the report and its derivatives are governed by CC BY-SA 4.0.
