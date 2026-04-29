#!/usr/bin/env python3
"""
Audio alignment: mel-spectrogram NCC with offset clustering.

Synthesises the best elements of prior attempts:

  Feature:    32-bin mel-spectrogram (dB) at 1 Hz, per-frame z-scored
              across frequency bins.  Z-scoring makes each frame invariant
              to absolute loudness while preserving spectral shape — the
              key property needed so that BGM amplitude variation doesn't
              dominate the correlation.

  Similarity: Proper normalised cross-correlation (NCC) in [−1, 1].
              The previous attempt that worked used raw summed mel
              correlation with a threshold of 125 on 16 bins × 10 s = 160
              max → effective threshold ≈ 0.78.  That is extremely
              conservative; it only accepted near-perfect spectral copies.
              Using explicit NCC lets us set a physically meaningful
              threshold of 0.30 ("moderate match"), which is appropriate
              when speech + game sounds dominate but some BGM is mixed in.

  Efficiency: Stream mel-FFT is pre-computed once per stream and reused
              for every reel window.  All features are cached as .npy
              files next to the source video.

  BGM guard:  Offset clustering (same insight as the v2 script that worked)
              — a window is only accepted inside a segment if ≥ MIN_WINDOWS
              consecutive reel windows all independently find the SAME
              stream offset (within OFFSET_TOL seconds).  A looping BGM
              produces high NCC for an isolated window but cannot create
              a consistent cluster because consecutive highlight windows
              contain different speech/game content while the BGM advances
              at a different phase in the stream.

  Min clip:   With W=10 s, hop=1 s, MIN_WINDOWS=3 the minimum detectable
              clip span is ~12 s.  Much shorter than the prior v2 approach
              (which required ≥ 4 windows with a 2 s step = ≥ 40 s).

Usage:
    python research/align_audio_mel.py [--stream NAME] [--force]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from scipy.fft import rfft, irfft, next_fast_len

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from chatfreq.core.time_utils import seconds_to_hms

SAMPLES_DIR  = PROJECT_ROOT / "samples"
RESEARCH_DIR = Path(__file__).resolve().parent

# ── Feature parameters ────────────────────────────────────────────────────────
SR          = 22050       # decode sample rate
HOP_LENGTH  = SR          # hop = 1 s → 1 frame per second
N_MELS      = 32          # frequency bins (vs 16 in the prior v2 attempt)
FMAX        = 8000        # max frequency to model
FEAT_RATE   = SR // HOP_LENGTH   # 1 Hz

# ── Alignment parameters ──────────────────────────────────────────────────────
WINDOW_SECS = 10          # seconds per alignment window
W           = WINDOW_SECS * FEAT_RATE   # frames per window

# ── Clustering parameters ─────────────────────────────────────────────────────
NCC_THRESHOLD   = 0.30    # minimum NCC per window  (vs ~0.78 in other approach)
OFFSET_TOL      = 6       # maximum offset drift (frames = seconds) within cluster
MAX_REEL_GAP    = 6       # maximum gap in reel time (frames) between cluster members
MIN_WINDOWS     = 3       # minimum windows per accepted cluster
MIN_COVERAGE    = 8.0     # minimum reel-time covered per segment (seconds)

# ── Stream manifest ───────────────────────────────────────────────────────────
STREAMS = {
    "gura-first-minecraft": {
        "stream_video": SAMPLES_DIR / "gura-first-minecraft" / "[MINECRAFT] BUILD ATLANTIS #GAWRGURA #HololiveEnglish [OlJQItn5Z2o].mkv",
        "highlights": [
            SAMPLES_DIR / "gura-first-minecraft" / "highlights" / "Gawr Gura Begins her MINECRAFT Journey! Best Moments! - Hololive EN Highlights [kwoZkGGZlv4].mkv",
            SAMPLES_DIR / "gura-first-minecraft" / "highlights" / "Gawr Gura Builds Atlantis - First stream highlights 【HoloEN】【Minecraft】 [ZS6xn7fWIHQ].mkv",
            SAMPLES_DIR / "gura-first-minecraft" / "highlights" / "Gura's First Minecraft Stream ｜ Highlights [DHSnBVGTVes].mkv",
        ],
    },
    "gura-solo-minecraft0": {
        "stream_video": SAMPLES_DIR / "gura-solo-minecraft0" / "[MINECRAFT] I Want A Trident!!! [tQcV9eEH7fk].mkv",
        "highlights": [
            SAMPLES_DIR / "gura-solo-minecraft0" / "highlights" / "HololiveEN Gura's 10th solo Minecraft Highlights [MVykDx60kUE].mkv",
        ],
    },
    "nimi-retro-rewind": {
        "stream_video": SAMPLES_DIR / "nimi-retro-rewind" / "【Retro Rewind】 Legally distinct Blockbuster simulator [7Px9qClCzt8].mkv",
        "highlights": [
            SAMPLES_DIR / "nimi-retro-rewind" / "highlights" / "Nimi Island Gets a New Resident and It\u2019s Nimi\u2019s Daughter [Bwmg1cpkUGk].mkv",
            SAMPLES_DIR / "nimi-retro-rewind" / "highlights" / "Nimi Opened a New Store and She Couldn\u2019t Stop Yapping [sCKzzi9HkNg].mkv",
        ],
    },
    "nimi-vampire": {
        "stream_video": SAMPLES_DIR / "nimi-vampire" / "【Vampire： The Masquerade - Bloodlines】 Living out my sick and twisted vampire dreams [Db54iFFWLWc].mkv",
        "highlights": [
            SAMPLES_DIR / "nimi-vampire" / "highlights" / "Nimi gets scammed by a lady of the night, commits a crime, dies then dances (Vampire：The Masquerade) [I-xB_W1pl0Q].mkv",
            SAMPLES_DIR / "nimi-vampire" / "highlights" / "Nimi Hilariously Caused Havoc After Turning Into a Vampire [K0LAOOsc61g].mkv",
        ],
    },
}

# ── Feature extraction ────────────────────────────────────────────────────────

def _feat_cache(video: Path) -> Path:
    return video.parent / (video.name + ".chatfreq_mel32.npy")


def extract_features(video: Path, force: bool = False) -> np.ndarray:
    """
    Decode audio → 32-bin mel-spectrogram at 1 Hz, per-frame z-scored.

    Returns float32 array of shape (32, n_frames).

    Per-frame z-score (axis=0, across the 32 mel bins): removes absolute
    loudness while preserving which bins are active relative to each other.
    """
    cache = _feat_cache(video)
    if cache.exists() and not force:
        feat = np.load(str(cache))
        print(f"    cached  {feat.shape[1]} frames ({feat.shape[1]/FEAT_RATE:.0f} s)  ← {cache.name}")
        return feat

    print(f"    decode  {video.name[:70]} … ", end="", flush=True)
    t0 = time.time()

    cmd = ["ffmpeg", "-y", "-i", str(video),
           "-vn", "-acodec", "pcm_f32le", "-ar", str(SR),
           "-ac", "1", "-f", "f32le", "-loglevel", "error", "-"]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{r.stderr.decode()[:300]}")

    y = np.frombuffer(r.stdout, dtype=np.float32).copy()

    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                       hop_length=HOP_LENGTH, fmax=FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)                # (32, T)

    # Per-frame z-score across the 32 frequency bins
    mu  = S_db.mean(axis=0, keepdims=True)
    sig = S_db.std( axis=0, keepdims=True) + 1e-8
    feat = ((S_db - mu) / sig).astype(np.float32)             # (32, T)

    np.save(str(cache), feat)
    elapsed = time.time() - t0
    print(f"{feat.shape[1]} fr. ({feat.shape[1]/FEAT_RATE:.0f} s)  [{elapsed:.1f} s]")
    return feat


# ── NCC engine ────────────────────────────────────────────────────────────────

class StreamNCCEngine:
    """
    Pre-computes the stream FFT once; answers per-window NCC queries in O(D log N).

    NCC(k) = Σ_d Σ_j T_z[d,j] · S_z[d, k+j]  /  (D · W)

    Since both T_z and S_z are per-frame z-scored, the mean is ≈ 0 and std ≈ 1
    for every column, so the dot product inner term ≈ D · Pearson(T[:,j], S[:,k+j]).
    Dividing by D·W gives the mean Pearson correlation over the window ∈ [−1, 1].
    """

    def __init__(self, stream_feat: np.ndarray):
        """stream_feat: float32 (D, N_stream)"""
        D, N = stream_feat.shape
        self.D = D
        self.N = N
        self.N_fft = next_fast_len(N + W - 1)
        # Pre-compute FFT of stream along the time axis, for every mel bin
        self.S_fft = rfft(stream_feat.astype(np.float64), n=self.N_fft, axis=1)
        # shape: (D, N_fft//2 + 1)

    def ncc(self, template: np.ndarray) -> np.ndarray:
        """
        template: float32 (D, W)
        Returns NCC array of shape (N - W + 1,) with values in [−1, 1].
        """
        # Reverse template along time axis (for cross-correlation via convolution)
        T_rev = template[:, ::-1].astype(np.float64)
        # Zero-pad and FFT
        T_fft = rfft(T_rev, n=self.N_fft, axis=1)             # (D, N_fft//2+1)
        # Summed cross-correlation across all mel bins
        cross_fft = (self.S_fft * T_fft).sum(axis=0)           # (N_fft//2+1)
        cross = irfft(cross_fft, n=self.N_fft)                 # (N_fft,)
        # Extract 'valid' region: lags 0 … N-W  (length N-W+1)
        ncc_raw = cross[: self.N - W + 1]                      # (N-W+1,)
        # Normalise: D*W is the theoretical maximum (all bins perfectly correlated)
        return (ncc_raw / (self.D * W)).astype(np.float32)


# ── Per-reel alignment ────────────────────────────────────────────────────────

def align_one(stream_engine: StreamNCCEngine,
              hl_feat: np.ndarray,
              hl_path: Path) -> list[dict]:
    """
    Slide W-second windows (hop=1 s) over the highlight, find best stream
    offset per window, cluster into segments.
    """
    N_hl = hl_feat.shape[1]
    N_str = stream_engine.N

    # ── Step 1: per-window best offset ──────────────────────────────────────
    candidates = []
    for reel_start in range(0, N_hl - W + 1):          # hop = 1 frame = 1 s
        template = hl_feat[:, reel_start : reel_start + W]
        ncc_arr = stream_engine.ncc(template)            # (N_str - W + 1,)

        best_k   = int(np.argmax(ncc_arr))
        best_ncc = float(ncc_arr[best_k])

        if best_ncc >= NCC_THRESHOLD:
            candidates.append({
                "reel_start": reel_start,
                "reel_end":   reel_start + W,
                "vod_start":  best_k,
                "vod_end":    best_k + W,
                "offset":     best_k - reel_start,
                "ncc":        best_ncc,
            })

    if not candidates:
        return []

    # ── Step 2: cluster by near-constant offset ──────────────────────────────
    candidates.sort(key=lambda x: x["reel_start"])
    clusters, cur = [], [candidates[0]]
    for c in candidates[1:]:
        last = cur[-1]
        offset_ok = abs(c["offset"] - last["offset"]) <= OFFSET_TOL
        gap_ok    = (c["reel_start"] - last["reel_end"]) <= MAX_REEL_GAP
        if offset_ok and gap_ok:
            cur.append(c)
        else:
            clusters.append(cur)
            cur = [c]
    clusters.append(cur)

    # ── Step 3: filter and emit segments ────────────────────────────────────
    segments = []
    for cl in clusters:
        n_win    = len(cl)
        coverage = float(cl[-1]["reel_end"] - cl[0]["reel_start"])  # in frames = seconds
        avg_ncc  = float(np.mean([c["ncc"] for c in cl]))

        if n_win < MIN_WINDOWS or coverage < MIN_COVERAGE:
            continue

        median_offset = float(np.median([c["offset"] for c in cl]))
        reel_s = float(cl[0]["reel_start"])
        reel_e = float(cl[-1]["reel_end"])
        vod_s  = max(0.0, reel_s + median_offset)
        vod_e  = reel_e + median_offset

        segments.append({
            "highlight_file":    hl_path.name,
            "segment_idx":       len(segments) + 1,
            "highlight_start_s": round(reel_s, 1),
            "highlight_end_s":   round(reel_e, 1),
            "stream_start_s":    round(vod_s,  1),
            "stream_end_s":      round(vod_e,  1),
            "duration_s":        round(reel_e - reel_s, 1),
            "mean_ncc":          round(avg_ncc, 4),
            "n_windows":         n_win,
            "method":            "audio_mel_ncc",
            "highlight_start_hms": seconds_to_hms(reel_s),
            "highlight_end_hms":   seconds_to_hms(reel_e),
            "stream_start_hms":  seconds_to_hms(vod_s),
            "stream_end_hms":    seconds_to_hms(vod_e),
        })

    return segments


# ── Per-stream orchestration ──────────────────────────────────────────────────

def align_stream(stream_name: str, cfg: dict, force: bool = False):
    print(f"\n{'='*62}")
    print(f"  {stream_name}")
    print(f"{'='*62}")

    out_dir = RESEARCH_DIR / stream_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stream_video = cfg["stream_video"]
    if not stream_video.exists():
        print(f"  ERROR: stream not found: {stream_video}")
        return None

    print("  Stream:")
    stream_feat = extract_features(stream_video, force=force)
    stream_dur  = stream_feat.shape[1] / FEAT_RATE
    print(f"    duration = {seconds_to_hms(stream_dur)}")

    print("  Building NCC engine (pre-computing stream FFT) … ", end="", flush=True)
    t0 = time.time()
    engine = StreamNCCEngine(stream_feat)
    print(f"done [{time.time()-t0:.1f} s]")

    all_segs = []

    for hl_path in cfg["highlights"]:
        if not hl_path.exists():
            print(f"\n  WARNING: file not found: {hl_path.name}")
            continue

        print(f"\n  Highlight: {hl_path.name[:65]}")
        hl_feat = extract_features(hl_path, force=force)
        hl_dur  = hl_feat.shape[1] / FEAT_RATE
        print(f"    duration = {seconds_to_hms(hl_dur)}")

        print(f"    aligning (window={WINDOW_SECS}s, hop=1s, NCC≥{NCC_THRESHOLD}) … ",
              end="", flush=True)
        t0 = time.time()
        segs = align_one(engine, hl_feat, hl_path)
        elapsed = time.time() - t0

        # Re-number segment indices
        for i, s in enumerate(segs):
            s["segment_idx"] = i + 1

        # Coverage stats
        hl_covered_s   = sum(s["highlight_end_s"] - s["highlight_start_s"] for s in segs)
        hl_pct         = 100 * hl_covered_s / hl_dur if hl_dur > 0 else 0

        # Check for overlapping reel segments (should not happen with 1s hop + consistent offset)
        reel_segs = sorted((s["highlight_start_s"], s["highlight_end_s"]) for s in segs)
        merged_hl = []
        for rs, re in reel_segs:
            if merged_hl and rs < merged_hl[-1][1]:
                merged_hl[-1] = (merged_hl[-1][0], max(merged_hl[-1][1], re))
            else:
                merged_hl.append((rs, re))
        net_hl_s = sum(e - s for s, e in merged_hl)

        print(f"{len(segs)} segments, {net_hl_s:.0f}/{hl_dur:.0f} s ({100*net_hl_s/hl_dur:.0f}%) reel covered  [{elapsed:.1f} s]")

        for s in segs:
            print(f"       seg {s['segment_idx']:02d}: clip [{s['highlight_start_hms']} – {s['highlight_end_hms']}] "
                  f"→ stream [{s['stream_start_hms']} – {s['stream_end_hms']}]  "
                  f"NCC={s['mean_ncc']:.3f}  n={s['n_windows']}")

        all_segs.extend(segs)

    if not all_segs:
        print("  No segments found.")
        return None

    df = pd.DataFrame(all_segs)
    out_path = out_dir / f"{stream_name}_alignment.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path.name}  ({len(df)} segments)")

    # Coverage summary
    print("\n  Summary — reel coverage:")
    for hl_name, grp in df.groupby("highlight_file"):
        hl_segs = sorted(zip(grp["highlight_start_s"], grp["highlight_end_s"]))
        merged = []
        for s, e in hl_segs:
            if merged and s < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        net = sum(e - s for s, e in merged)
        print(f"    {hl_name[:60]:60s}: {len(grp)} segs, {seconds_to_hms(net)} net")

    # Sanity check: stream-side inflation ratio
    all_stream_segs = sorted(zip(df["stream_start_s"], df["stream_end_s"]))
    m = []
    for s, e in all_stream_segs:
        if m and s < m[-1][1]:
            m[-1] = (m[-1][0], max(m[-1][1], e))
        else:
            m.append((s, e))
    net_stream = sum(e - s for s, e in m)

    hl_total = sum(df["highlight_end_s"] - df["highlight_start_s"])
    print(f"\n  Inflation check: stream_net={net_stream:.0f}s  reel_sum={hl_total:.0f}s  "
          f"ratio={net_stream/max(hl_total,1):.2f}  (should be ≤ 1.0 for perfect alignment)")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", default=None)
    parser.add_argument("--force", action="store_true", help="Re-extract audio even if cached")
    args = parser.parse_args()

    streams = STREAMS
    if args.stream:
        if args.stream not in STREAMS:
            print(f"Unknown stream '{args.stream}'. Available: {list(STREAMS)}")
            sys.exit(1)
        streams = {args.stream: STREAMS[args.stream]}

    print("Audio mel-NCC alignment")
    print(f"  features: {N_MELS}-bin mel @ {FEAT_RATE} Hz, per-frame z-scored")
    print(f"  window:   {WINDOW_SECS} s   hop: 1 s")
    print(f"  NCC threshold: {NCC_THRESHOLD}   offset_tol: {OFFSET_TOL} s")
    print(f"  cluster:  min_windows={MIN_WINDOWS}  min_coverage={MIN_COVERAGE} s")

    for sname, cfg in streams.items():
        align_stream(sname, cfg, args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
