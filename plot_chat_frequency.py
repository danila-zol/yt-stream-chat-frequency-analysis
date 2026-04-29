#!/usr/bin/env python3
"""
Plot chat frequency over stream duration using sliding window algorithm.
Shows raw histogram, sliding window frequency, and polynomial trend line.
Exports frequency data as CSV if requested.
"""

import argparse
import sys
import os

# Check for required packages
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except ImportError as e:
    print(f"Missing required package: {e}", file=sys.stderr)
    print("Please install pandas, numpy, matplotlib.", file=sys.stderr)
    sys.exit(1)

def parse_time(timestr):
    """
    Convert timestamp string to total seconds.
    Supports formats:
        "mm:ss"
        "h:mm:ss"
        "hh:mm:ss"
    """
    parts = timestr.strip().split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        hours = 0
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Invalid time format: {timestr}")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)

def main():
    parser = argparse.ArgumentParser(
        description='Plot chat frequency over stream using sliding window.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input', help='Path to chat CSV file (tab‑separated)')
    parser.add_argument('--window', type=float, default=60.0,
                        help='Sliding window size in seconds')
    parser.add_argument('--step', type=float, default=10.0,
                        help='Step size for histogram bins and window sliding')
    parser.add_argument('--fit-degree', type=int, default=3,
                        help='Degree of polynomial trend line (0 to disable)')
    parser.add_argument('--output', help='Output image path (default: <input>_chat_frequency.png)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figure')
    parser.add_argument('--no-bars', action='store_true',
                        help='Omit histogram bars')
    parser.add_argument('--no-sliding', action='store_true',
                        help='Omit sliding window line')
    parser.add_argument('--no-trend', action='store_true',
                        help='Omit polynomial trend line')
    parser.add_argument('--export-csv',
                        help='Export sliding window data to CSV file')
    parser.add_argument('--export-histogram-csv',
                        help='Export raw histogram (per step bin) to CSV file')
    
    args = parser.parse_args()
    
    # Read CSV (tab‑separated)
    try:
        df = pd.read_csv(args.input, sep='\t')
    except Exception as e:
        print(f"Error reading {args.input}: {e}", file=sys.stderr)
        sys.exit(1)
    
    required = ['video_time', 'author', 'message']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    
    # Parse timestamps to seconds
    df['seconds'] = df['video_time'].apply(parse_time)
    timestamps = df['seconds'].values
    
    if len(timestamps) == 0:
        print("No messages found.", file=sys.stderr)
        sys.exit(1)
    
    max_seconds = timestamps.max()
    window_size = args.window
    step = args.step
    
    if window_size <= 0 or step <= 0:
        print("Window and step must be positive.", file=sys.stderr)
        sys.exit(1)
    
    # Ensure window_size is a multiple of step (simplifies convolution)
    if window_size % step != 0:
        window_size = round(window_size / step) * step
        print(f"Adjusted window size to {window_size}s (multiple of step)")
    
    # Create bin edges for histogram with step size
    bin_edges = np.arange(0, max_seconds + step, step)
    hist, _ = np.histogram(timestamps, bins=bin_edges)
    
    # Rolling sum over window_size/step bins (sliding window frequency)
    window_bins = int(window_size / step)
    if window_bins < 1:
        print(f"Error: window size {window_size}s is smaller than step {step}s", file=sys.stderr)
        sys.exit(1)
    if window_bins > len(hist):
        print(f"Error: window size {window_size}s ({window_bins} bins) exceeds total stream duration", file=sys.stderr)
        sys.exit(1)
    rolling_sum = np.convolve(hist, np.ones(window_bins), mode='valid')
    # Time axis: center of each window (in seconds)
    # For window starting at bin i, center = bin_edges[i] + window_size/2
    # where i ranges from 0 to len(hist) - window_bins
    time_axis = bin_edges[:len(hist) - window_bins + 1] + window_size / 2
    # Sanity check
    if len(time_axis) != len(rolling_sum):
        print(f"Error: length mismatch: time_axis {len(time_axis)} != rolling_sum {len(rolling_sum)}", file=sys.stderr)
        print(f"hist length: {len(hist)}, window_bins: {window_bins}", file=sys.stderr)
        sys.exit(1)
    
    # Polynomial trend line (fit on sliding window data)
    trend_line = None
    if args.fit_degree > 0 and not args.no_trend and len(time_axis) >= args.fit_degree + 1:
        coeff = np.polyfit(time_axis, rolling_sum, args.fit_degree)
        poly = np.poly1d(coeff)
        trend_line = poly(time_axis)
    
    # Export raw histogram CSV
    if args.export_histogram_csv:
        hist_df = pd.DataFrame({
            'bin_start': bin_edges[:-1],
            'bin_end': bin_edges[1:],
            'raw_count': hist
        })
        try:
            hist_df.to_csv(args.export_histogram_csv, index=False)
            print(f"Raw histogram saved to {args.export_histogram_csv}")
        except Exception as e:
            print(f"Error saving histogram CSV: {e}", file=sys.stderr)
    
    # Export sliding window CSV
    if args.export_csv:
        sw_df = pd.DataFrame({
            'window_center': time_axis,
            'window_frequency': rolling_sum
        })
        if trend_line is not None:
            sw_df['trend'] = trend_line
        try:
            sw_df.to_csv(args.export_csv, index=False)
            print(f"Sliding window data saved to {args.export_csv}")
        except Exception as e:
            print(f"Error saving sliding window CSV: {e}", file=sys.stderr)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Raw histogram bars (per step bin)
    if not args.no_bars:
        ax.bar(bin_edges[:-1], hist, width=step, alpha=0.5,
               label=f'Messages per {step}s', align='edge', color='lightgray')
    
    # Sliding window frequency line
    if not args.no_sliding:
        ax.plot(time_axis, rolling_sum, color='red', linewidth=2,
                label=f'{window_size}s sliding window')
    
    # Polynomial trend line
    if trend_line is not None:
        ax.plot(time_axis, trend_line, color='blue', linewidth=2, linestyle='--',
                label=f'Polynomial degree {args.fit_degree}')
    
    ax.set_xlabel('Stream Time')
    ax.set_ylabel('Message Frequency')
    ax.set_title('Chat Frequency Throughout Stream')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x‑axis to show hours:minutes:seconds
    def sec_to_hms(x, pos):
        h = int(x // 3600)
        m = int((x % 3600) // 60)
        s = int(x % 60)
        if h > 0:
            return f'{h}:{m:02d}:{s:02d}'
        else:
            return f'{m}:{s:02d}'
    ax.xaxis.set_major_formatter(FuncFormatter(sec_to_hms))
    
    # Auto‑rotate x‑axis labels
    fig.autofmt_xdate()
    
    # Save figure
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(args.input)[0]
        out_path = f'{base}_chat_frequency.png'
    
    try:
        plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
        print(f'Plot saved to {out_path}')
    except Exception as e:
        print(f"Error saving figure: {e}", file=sys.stderr)
        sys.exit(1)
    
    plt.close()

if __name__ == '__main__':
    main()