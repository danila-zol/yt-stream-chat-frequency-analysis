"""
CLI commands using Click framework.
"""
import os
import webbrowser

import click
import numpy as np

from ..core import ChatFrequencyAnalyzer, parse_time, seconds_to_hms
from ..core.time_utils import find_video_files as find_videos
from .renderer import die, info, print_data_summary, success


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Chat Frequency Analysis Tool

    Analyze chat message frequency in video streams using sliding window
    convolution algorithm. Provides both interactive web visualization
    and CLI export capabilities.
    """
    pass


@cli.command("serve")
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "--port", "-p", default=8050, help="Port to run Dash server on", show_default=True
)
@click.option("--host", default="127.0.0.1", help="Host to bind server to")
@click.option("--no-browser", is_flag=True, help="Do not automatically open browser")
@click.option("--debug", is_flag=True, help="Run in debug mode")
@click.option("--video", default="", help="Path to video file to preload")
@click.option(
    "--window", default=60, type=int, help="Default window size in seconds"
)
@click.option("--step", default=10, type=int, help="Default step size in seconds")
@click.option("--degree", default=3, type=int, help="Default polynomial degree")
@click.option(
    "--prominence", default=5, type=int, help="Default peak prominence threshold"
)
def serve(
    input,
    port,
    host,
    no_browser,
    debug,
    video,
    window,
    step,
    degree,
    prominence,
):
    """
    Start interactive web dashboard.

    Launches a Dash/Plotly web application for interactive chat frequency
    visualization with zooming, parameter adjustment, and export capabilities.
    """
    try:
        from ..web.app import create_app
    except ImportError:
        die(
            "Web dependencies not installed. Run: pip install dash plotly kaleido"
        )

    try:
        analyzer = ChatFrequencyAnalyzer(input)
    except Exception as e:
        die(f"Failed to load data: {e}")

    print_data_summary(analyzer, input)

    base_dir = os.path.dirname(os.path.abspath(input)) if os.path.dirname(input) else "."
    video_files = find_videos(base_dir)

    info(f"\nStarting Dash server on http://{host}:{port}")
    info("Press Ctrl+C to stop\n")

    app = create_app(
        analyzer,
        input,
        video_files,
        default_window=window,
        default_step=step,
        default_degree=degree,
        default_prominence=prominence,
    )

    if not no_browser:
        webbrowser.open(f"http://{host}:{port}")

    app.run(host=host, port=port, debug=debug, dev_tools_hot_reload=False)


@cli.command("plot")
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output PNG file path")
@click.option("--window", default=60, type=int, help="Window size in seconds")
@click.option("--step", default=10, type=int, help="Step size in seconds")
@click.option("--degree", default=3, type=int, help="Polynomial trend degree")
@click.option("--dpi", default=100, type=int, help="Output DPI")
@click.option("--no-bars", is_flag=True, help="Hide histogram bars")
@click.option("--no-trend", is_flag=True, help="Hide trend line")
@click.option("--start", help="Start time filter (mm:ss, h:mm:ss, or seconds)")
@click.option("--end", help="End time filter (mm:ss, h:mm:ss, or seconds)")
def plot(
    input, output, window, step, degree, dpi, no_bars, no_trend, start, end
):
    """
    Generate static PNG plot.

    Creates a matplotlib-based frequency plot with histogram bars,
    sliding window line, and optional polynomial trend.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        die("matplotlib not installed. Run: pip install matplotlib")

    try:
        analyzer = ChatFrequencyAnalyzer(input)
    except Exception as e:
        die(f"Failed to load data: {e}")

    start_sec = parse_time(start) if start else None
    end_sec = parse_time(end) if end else None

    print_data_summary(analyzer, input)
    if start_sec or end_sec:
        info(
            f"Time filter: {seconds_to_hms(start_sec) if start_sec else 'start'} "
            f"to {seconds_to_hms(end_sec) if end_sec else 'end'}"
        )

    time_axis, rolling_sum = analyzer.compute_sliding_window(step, window)

    if start_sec or end_sec:
        time_axis, rolling_sum = analyzer.filter_by_time(
            rolling_sum, time_axis, start_sec, end_sec
        )

    bin_edges, hist = analyzer.compute_histogram(step)
    bin_centers = bin_edges[:-1] + step / 2

    if start_sec or end_sec:
        mask = np.ones(len(bin_centers), dtype=bool)
        if start_sec:
            mask &= bin_centers >= start_sec
        if end_sec:
            mask &= bin_centers <= end_sec
        bin_centers = bin_centers[mask]
        hist = hist[mask]

    fig, ax = plt.subplots(figsize=(14, 6))

    if not no_bars:
        ax.bar(bin_centers, hist, width=step * 0.9, alpha=0.5, label="Raw", color="gray")

    ax.plot(
        time_axis,
        rolling_sum,
        color="red",
        linewidth=2,
        label=f"Sliding Window ({window}s)",
    )

    if not no_trend and degree > 0:
        trend = analyzer.compute_polynomial_trend(step, window, degree)
        if trend is not None:
            if start_sec or end_sec:
                _, trend = analyzer.filter_by_time(trend, time_axis, start_sec, end_sec)
            ax.plot(
                time_axis,
                trend,
                color="blue",
                linewidth=2,
                linestyle="--",
                label=f"Trend (deg {degree})",
            )

    ax.set_xlabel("Stream Time")
    ax.set_ylabel("Message Frequency")
    ax.set_title(f"Chat Frequency Analysis: {os.path.basename(input)}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    tick_count = 8
    tick_positions = np.linspace(
        time_axis.min(), time_axis.max(), tick_count
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([seconds_to_hms(p) for p in tick_positions])

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=dpi)
        success(f"Plot saved to {output}")
    else:
        plt.show()


@cli.group("export")
def export():
    """
    Export data in various formats.

    Subcommands:
      sliding    Export sliding window frequency CSV
      peaks      Export peak timestamps with sample messages
      timestamps Export timestamps for video editors
      ffmpeg     Generate FFmpeg highlight reel bash script
    """
    pass


@export.command("sliding")
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output CSV file path")
@click.option("--window", default=60, type=int, help="Window size in seconds")
@click.option("--step", default=10, type=int, help="Step size in seconds")
@click.option("--degree", default=0, type=int, help="Polynomial trend degree (0=skip)")
@click.option("--start", help="Start time filter (mm:ss, h:mm:ss, or seconds)")
@click.option("--end", help="End time filter (mm:ss, h:mm:ss, or seconds)")
def export_sliding(input, output, window, step, degree, start, end):
    """
    Export sliding window frequency data to CSV.
    """
    try:
        analyzer = ChatFrequencyAnalyzer(input)
    except Exception as e:
        die(f"Failed to load data: {e}")

    start_sec = parse_time(start) if start else None
    end_sec = parse_time(end) if end else None

    df = analyzer.export_sliding_window(
        step=step,
        window_size=window,
        start=start_sec,
        end=end_sec,
        include_trend=degree > 0,
        degree=degree,
    )

    if output:
        df.to_csv(output, index=False)
        success(f"Sliding window data exported to {output}")
    else:
        print(df.to_string(index=False))


@export.command("peaks")
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output CSV file path")
@click.option("--window", default=60, type=int, help="Window size in seconds")
@click.option("--step", default=10, type=int, help="Step size in seconds")
@click.option(
    "--prominence", default=5, type=float, help="Peak prominence threshold"
)
@click.option("--start", help="Start time filter (mm:ss, h:mm:ss, or seconds)")
@click.option("--end", help="End time filter (mm:ss, h:mm:ss, or seconds)")
@click.option(
    "--normalize", is_flag=True, help="Detect peaks on trend-normalized signal"
)
@click.option(
    "--degree", default=3, type=int, help="Polynomial degree for trend when normalizing"
)
def export_peaks(input, output, window, step, prominence, start, end, normalize, degree):
    """
    Export detected peaks with sample messages to CSV.
    """
    try:
        analyzer = ChatFrequencyAnalyzer(input)
    except Exception as e:
        die(f"Failed to load data: {e}")

    start_sec = parse_time(start) if start else None
    end_sec = parse_time(end) if end else None

    df = analyzer.export_peaks(
        step=step,
        window_size=window,
        prominence=prominence,
        start=start_sec,
        end=end_sec,
        normalize=normalize,
        degree=degree,
    )

    if len(df) == 0:
        info("No peaks found with current parameters.")
        return

    info(f"Found {len(df)} peaks")

    if output:
        df.to_csv(output, index=False)
        success(f"Peaks exported to {output}")
    else:
        print(df.to_string(index=False))


@export.command("timestamps")
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output CSV file path")
@click.option("--window", default=60, type=int, help="Window size in seconds")
@click.option("--step", default=10, type=int, help="Step size in seconds")
@click.option("--prominence", default=5, type=float, help="Peak prominence threshold")
@click.option("--clip-before", default=30, type=int, help="Seconds before peak for clip start")
@click.option("--clip-after", default=30, type=int, help="Seconds after peak for clip end")
@click.option("--start", help="Start time filter (mm:ss, h:mm:ss, or seconds)")
@click.option("--end", help="End time filter (mm:ss, h:mm:ss, or seconds)")
@click.option("--normalize", is_flag=True, help="Detect peaks on trend-normalized signal")
@click.option("--degree", default=3, type=int, help="Polynomial degree for trend when normalizing")
@click.option("--video", help="Path to video file (required for intelligent cutting)")
@click.option("--subtitle", help="Path to subtitle file (.srt or .vtt)")
@click.option("--intelligent-cut", is_flag=True, help="Enable intelligent boundary snapping")
@click.option("--max-snap", default=3.0, type=float, help="Max snap distance in seconds")
@click.option("--silence-threshold-db", default=-45.0, type=float, help="Silence threshold in dB")
@click.option("--min-silence-duration", default=0.3, type=float, help="Minimum silence duration in seconds")
@click.option("--fallback-expand", is_flag=True, help="Expand search if no boundary found")
def export_timestamps(
    input, output, window, step, prominence, clip_before, clip_after, start, end,
    normalize, degree, video, subtitle, intelligent_cut, max_snap,
    silence_threshold_db, min_silence_duration, fallback_expand
):
    """Export timestamps for video editors to CSV."""
    try:
        analyzer = ChatFrequencyAnalyzer(input)
    except Exception as e:
        die(f"Failed to load data: {e}")

    start_sec = parse_time(start) if start else None
    end_sec = parse_time(end) if end else None

    intelligent_cutter = None
    if intelligent_cut:
        if not video or not subtitle:
            die("--video and --subtitle are required for --intelligent-cut")
        if not os.path.exists(video):
            die(f"Video file not found: {video}")
        if not os.path.exists(subtitle):
            die(f"Subtitle file not found: {subtitle}")
        from ..core.intelligent_cutter import IntelligentCutter
        intelligent_cutter = IntelligentCutter(
            video,
            subtitle,
            silence_threshold_db=silence_threshold_db,
            min_silence_ms=int(min_silence_duration * 1000),
        )

    df = analyzer.export_timestamps(
        step=step,
        window_size=window,
        prominence=prominence,
        clip_before=clip_before,
        clip_after=clip_after,
        start=start_sec,
        end=end_sec,
        normalize=normalize,
        degree=degree,
        intelligent_cutter=intelligent_cutter,
        max_snap_distance=max_snap,
        silence_tolerance=1.0,
        fallback_expand=fallback_expand,
    )

    if len(df) == 0:
        info("No peaks found with current parameters.")
        return

    info(f"Generated {len(df)} clip timestamps")

    if output:
        df.to_csv(output, index=False)
        success(f"Timestamps exported to {output}")
    else:
        print(df.to_string(index=False))


@export.command("ffmpeg")
@click.argument("input", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output bash script path")
@click.option("--video", required=True, help="Path to video file")
@click.option("--window", default=60, type=int, help="Window size in seconds")
@click.option("--step", default=10, type=int, help="Step size in seconds")
@click.option("--prominence", default=5, type=float, help="Peak prominence threshold")
@click.option("--clip-before", default=30, type=int, help="Seconds before peak for clip start")
@click.option("--clip-after", default=30, type=int, help="Seconds after peak for clip end")
@click.option("--start", help="Start time filter (mm:ss, h:mm:ss, or seconds)")
@click.option("--end", help="End time filter (mm:ss, h:mm:ss, or seconds)")
@click.option("--region-cuts", is_flag=True, help="Use one cut per high-engagement region instead of peaks")
@click.option("--cbs-threshold", default=2.5, type=float, help="CBS t-statistic threshold")
@click.option("--z-threshold", default=0.0, type=float, help="Z-score threshold for high engagement")
@click.option("--min-duration", default=30, type=int, help="Minimum region duration in seconds")
@click.option("--max-gap", default=30, type=int, help="Max gap to merge regions in seconds")
@click.option("--normalize", is_flag=True, help="Use trend-normalized signal for detection")
@click.option("--degree", default=3, type=int, help="Polynomial degree for trend when normalizing")
@click.option("--subtitle", help="Path to subtitle file (.srt or .vtt)")
@click.option("--intelligent-cut", is_flag=True, help="Enable intelligent boundary snapping")
@click.option("--max-snap", default=3.0, type=float, help="Max snap distance in seconds")
@click.option("--silence-threshold-db", default=-45.0, type=float, help="Silence threshold in dB")
@click.option("--min-silence-duration", default=0.3, type=float, help="Minimum silence duration in seconds")
@click.option("--fallback-expand", is_flag=True, help="Expand search if no boundary found")
def export_ffmpeg(
    input, output, video, window, step, prominence, clip_before, clip_after, start, end,
    region_cuts, cbs_threshold, z_threshold, min_duration, max_gap, normalize, degree,
    subtitle, intelligent_cut, max_snap, silence_threshold_db, min_silence_duration,
    fallback_expand
):
    """Generate FFmpeg highlight reel bash script."""
    if not os.path.exists(video):
        die(f"Video file not found: {video}")

    try:
        analyzer = ChatFrequencyAnalyzer(input)
    except Exception as e:
        die(f"Failed to load data: {e}")

    start_sec = parse_time(start) if start else None
    end_sec = parse_time(end) if end else None

    intelligent_cutter = None
    if intelligent_cut:
        if not subtitle:
            die("--subtitle is required for --intelligent-cut")
        if not os.path.exists(subtitle):
            die(f"Subtitle file not found: {subtitle}")
        from ..core.intelligent_cutter import IntelligentCutter
        intelligent_cutter = IntelligentCutter(
            video,
            subtitle,
            silence_threshold_db=silence_threshold_db,
            min_silence_ms=int(min_silence_duration * 1000),
        )

    commands = analyzer.generate_ffmpeg_commands(
        video_path=video,
        step=step,
        window_size=window,
        prominence=prominence,
        clip_before=clip_before,
        clip_after=clip_after,
        start=start_sec,
        end=end_sec,
        use_regions=region_cuts,
        cbs_threshold=cbs_threshold,
        z_threshold=z_threshold,
        min_duration=min_duration,
        max_gap=max_gap,
        normalize=normalize,
        degree=degree,
        intelligent_cutter=intelligent_cutter,
        max_snap_distance=max_snap,
        silence_tolerance=1.0,
        fallback_expand=fallback_expand,
    )

    if output:
        with open(output, "w") as f:
            f.write("\n".join(commands))
            f.write("\n")
        os.chmod(output, 0o755)
        success(f"FFmpeg script saved to {output}")
    else:
        print("\n".join(commands))


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("--port", "-p", default=8050, help="Port to run Dash server on")
@click.option("--host", default="127.0.0.1", help="Host to bind server to")
@click.option("--no-browser", is_flag=True, help="Do not automatically open browser")
@click.option("--debug", is_flag=True, help="Run in debug mode")
@click.option("--video", default="", help="Path to video file to preload")
def web(input, port, host, no_browser, debug, video):
    """Alias for 'serve' command - start interactive web dashboard."""
    serve.callback(
        input=input,
        port=port,
        host=host,
        no_browser=no_browser,
        debug=debug,
        video=video,
        window=60,
        step=10,
        degree=3,
        prominence=5,
    )


if __name__ == "__main__":
    cli()
