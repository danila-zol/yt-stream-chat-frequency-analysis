"""
Dash callbacks for the chat frequency analyzer web app.
"""
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import callback_context, no_update
from plotly.subplots import make_subplots

try:
    import kaleido

    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False


def seconds_to_hms(seconds):
    """Convert seconds to H:MM:SS or M:SS format."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def register_callbacks(app, analyzer):
    """
    Register all Dash callbacks for the app.

    Args:
        app: Dash app instance
        analyzer: ChatFrequencyAnalyzer instance
    """

    @app.callback(
        [
            Output("prominence-slider", "min"),
            Output("prominence-slider", "max"),
            Output("prominence-slider", "step"),
            Output("prominence-slider", "value"),
            Output("prominence-slider", "marks"),
        ],
        [Input("trace-checklist", "value")],
        [State("prominence-slider", "value")],
    )
    def adjust_prominence_slider(trace_options, current_value):
        """Auto-adjust prominence slider range when normalize mode changes."""
        normalize = "normalize" in trace_options
        if normalize:
            # Scale-friendly defaults for ratio mode
            min_val, max_val, step_val = 0.1, 2.0, 0.1
            default_val = 0.3
            marks = {round(i, 1): str(round(i, 1)) for i in [0.1, 0.5, 1.0, 1.5, 2.0]}
        else:
            min_val, max_val, step_val = 1, 20, 1
            default_val = 5
            marks = {i: str(i) for i in [1, 5, 10, 15, 20]}

        # Try to preserve current value if it still fits in the new range
        if current_value is not None:
            if min_val <= current_value <= max_val:
                return min_val, max_val, step_val, current_value, marks

        return min_val, max_val, step_val, default_val, marks

    @app.callback(
        [
            Output("main-graph", "figure"),
            Output("current-data-store", "data"),
            Output("figure-json-store", "data"),
        ],
        [
            Input("window-slider", "value"),
            Input("step-slider", "value"),
            Input("degree-slider", "value"),
            Input("prominence-slider", "value"),
            Input("trace-checklist", "value"),
            Input("zoom-range-store", "data"),
            Input("cbs-slider", "value"),
            Input("z-slider", "value"),
            Input("min-duration-slider", "value"),
            Input("max-gap-slider", "value"),
        ],
    )
    def update_main_graph(
        window_size,
        step,
        degree,
        prominence,
        trace_options,
        zoom_range,
        cbs_threshold,
        z_threshold,
        min_duration,
        max_gap,
    ):
        """Update main graph with current parameters."""
        normalize = "normalize" in trace_options

        time_axis, rolling_sum = analyzer.compute_sliding_window(step, window_size)
        time_formatted = [seconds_to_hms(t) for t in time_axis]

        trend = None
        if "trend" in trace_options and degree > 0 and not normalize:
            trend = analyzer.compute_polynomial_trend(step, window_size, degree)

        # Determine which signal to plot and detect on
        if normalize:
            plot_axis, plot_signal = analyzer.compute_normalized_signal(step, window_size, degree)
            plot_label = "Normalized"
            y_axis_title = "Relative Frequency (ratio to trend)"
        else:
            plot_axis, plot_signal = time_axis, rolling_sum
            plot_label = f"Sliding ({window_size}s)"
            y_axis_title = "Message Frequency"

        peak_times, peak_values = None, None
        if "peaks" in trace_options:
            peak_times, peak_values = analyzer.detect_peaks(
                step, window_size, prominence, normalize=normalize, degree=degree
            )
            peak_times_formatted = (
                [seconds_to_hms(t) for t in peak_times] if peak_times is not None else None
            )

        bin_edges, hist = analyzer.compute_histogram(step)
        bin_centers_formatted = [seconds_to_hms(b) for b in bin_edges[:-1]]

        show_regions = "regions" in trace_options

        if show_regions:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                row_heights=[0.88, 0.12],
                vertical_spacing=0.02,
                specs=[[{"secondary_y": False}], [{}]],
            )
        else:
            fig = make_subplots(specs=[[{"secondary_y": False}]])

        template = "plotly_dark" if "dark" in trace_options else "plotly_white"

        def _add_trace(trace, row=1, col=1):
            fig.add_trace(trace, row=row, col=col)

        if "bars" in trace_options and not normalize:
            _add_trace(
                go.Bar(
                    x=bin_edges[:-1],
                    y=hist,
                    width=step,
                    name=f"Raw ({step}s bins)",
                    marker_color="rgba(200, 200, 200, 0.6)",
                    opacity=0.5,
                    hoverinfo="x+y",
                    customdata=[[ft] for ft in bin_centers_formatted],
                    hovertemplate="Time: %{customdata[0]}<br>Count: %{y}<extra></extra>",
                )
            )

        if "sliding" in trace_options:
            _add_trace(
                go.Scatter(
                    x=plot_axis,
                    y=plot_signal,
                    mode="lines",
                    name=plot_label,
                    line=dict(color="red", width=2),
                    hoverinfo="x+y",
                    customdata=[[ft] for ft in time_formatted],
                    hovertemplate="Time: %{customdata[0]}<br>Freq: %{y:.2f}<extra></extra>",
                )
            )

        if trend is not None:
            _add_trace(
                go.Scatter(
                    x=time_axis,
                    y=trend,
                    mode="lines",
                    name=f"Trend (deg {degree})",
                    line=dict(color="blue", width=2, dash="dash"),
                    hoverinfo="x+y",
                    customdata=[[ft] for ft in time_formatted],
                    hovertemplate="Time: %{customdata[0]}<br>Trend: %{y:.1f}<extra></extra>",
                )
            )

        if peak_times is not None and len(peak_times) > 0:
            _add_trace(
                go.Scatter(
                    x=peak_times,
                    y=peak_values,
                    mode="markers",
                    name="Peaks",
                    marker=dict(color="orange", size=10, symbol="triangle-up"),
                    hoverinfo="x+y",
                    customdata=[[ft] for ft in peak_times_formatted],
                    hovertemplate="PEAK! Time: %{customdata[0]}<br>Freq: %{y:.2f}<extra></extra>",
                )
            )

        regions = analyzer.detect_high_engagement_regions(
            step, window_size, cbs_threshold, z_threshold, min_duration, max_gap,
            normalize=normalize, degree=degree,
        )

        if show_regions:
            for r_start, r_end in regions:
                fig.add_shape(
                    type="rect",
                    x0=r_start,
                    x1=r_end,
                    y0=0,
                    y1=1,
                    fillcolor="rgba(255, 0, 0, 0.3)",
                    line=dict(width=0),
                    row=2,
                    col=1,
                )
            fig.update_yaxes(visible=False, range=[0, 1], row=2, col=1)
            fig.update_xaxes(title_text="Stream Time", row=2, col=1)

        fig.update_layout(
            template=template,
            title=f"Chat Frequency (Window: {window_size}s, Step: {step}s)",
            xaxis_title="Stream Time",
            yaxis_title=y_axis_title,
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=50, r=20, t=50, b=50),
        )

        if zoom_range and "x_min" in zoom_range:
            fig.update_xaxes(range=[zoom_range["x_min"], zoom_range["x_max"]])
            x_min = zoom_range["x_min"]
            x_max = zoom_range["x_max"]
        else:
            x_min = 0
            x_max = analyzer.max_seconds

        tick_count = 8
        tick_positions = np.linspace(x_min, x_max, tick_count)
        tick_labels = [seconds_to_hms(pos) for pos in tick_positions]

        fig.update_xaxes(
            tickvals=tick_positions.tolist(),
            ticktext=tick_labels,
            tickmode="array",
            nticks=tick_count,
        )

        export_data = {
            "window_size": window_size,
            "step": step,
            "degree": degree,
            "time_axis": time_axis.tolist(),
            "rolling_sum": rolling_sum.tolist(),
            "trend": trend.tolist() if trend is not None else None,
            "peak_times": peak_times.tolist() if peak_times is not None else None,
            "peak_values": peak_values.tolist() if peak_values is not None else None,
            "regions": regions,
            "cbs_threshold": cbs_threshold,
            "z_threshold": z_threshold,
            "min_duration": min_duration,
            "max_gap": max_gap,
            "normalize": normalize,
        }

        fig_dict = fig.to_dict()
        return fig, export_data, fig_dict

    @app.callback(
        Output("overview-graph", "figure"),
        [
            Input("window-slider", "value"),
            Input("step-slider", "value"),
            Input("degree-slider", "value"),
            Input("trace-checklist", "value"),
            Input("zoom-range-store", "data"),
        ],
    )
    def update_overview_graph(window_size, step, degree, trace_options, zoom_range):
        """Update overview graph (full stream)."""
        normalize = "normalize" in trace_options
        if normalize:
            time_axis, signal = analyzer.compute_normalized_signal(step, window_size, degree)
        else:
            time_axis, signal = analyzer.compute_sliding_window(step, window_size)

        template = "plotly_dark" if "dark" in trace_options else "plotly_white"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal,
                mode="lines",
                line=dict(color="green", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        if zoom_range and "x_min" in zoom_range:
            fig.add_shape(
                type="rect",
                x0=zoom_range["x_min"],
                x1=zoom_range["x_max"],
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="RoyalBlue", width=2),
                fillcolor="LightSkyBlue",
                opacity=0.3,
            )

        fig.update_layout(
            template=template,
            title="Overview (drag to zoom)",
            height=200,
            margin=dict(l=40, r=20, t=40, b=20),
            dragmode="select",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
        )

        return fig

    @app.callback(
        Output("hover-info", "children"),
        [Input("main-graph", "hoverData")],
    )
    def update_hover_info(hover_data):
        """Show message preview on hover."""
        if not hover_data:
            return "Hover over the graph to see messages from that time."

        point = hover_data["points"][0]
        x_val = point["x"]

        messages = analyzer.get_messages_near_time(x_val, margin=10)

        if len(messages) == 0:
            return f"No messages near {seconds_to_hms(x_val)}"

        preview_items = []
        for _, row in messages.iterrows():
            time_str = seconds_to_hms(row["seconds"])
            author = str(row["author"])[:30]
            msg = str(row["message"])[:100]
            if len(str(row["message"])) > 100:
                msg += "..."

            preview_items.append(
                html.Div(
                    [
                        html.Strong(f"{time_str} - {author}:"),
                        html.Br(),
                        html.Span(msg, style={"color": "#666"}),
                    ],
                    style={
                        "borderBottom": "1px solid #eee",
                        "padding": "5px 0",
                        "marginBottom": "5px",
                    },
                )
            )

        return [
            html.Strong(f"Messages near {seconds_to_hms(x_val)}:"),
            html.Br(),
            html.Br(),
        ] + preview_items

    @app.callback(
        Output("zoom-range-store", "data"),
        [
            Input("overview-graph", "selectedData"),
            Input("overview-graph", "relayoutData"),
            Input("main-graph", "relayoutData"),
            Input("reset-zoom-btn", "n_clicks"),
        ],
    )
    def update_zoom_range(selected_data, overview_relayout, main_relayout, reset_clicks):
        """Extract zoom range from graph interactions."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        trigger = ctx.triggered[0]["prop_id"]
        trigger_id, trigger_prop = trigger.split(".") if "." in trigger else (trigger, None)

        if trigger_id == "reset-zoom-btn" and reset_clicks > 0:
            return None

        if (
            trigger_id == "overview-graph"
            and trigger_prop == "selectedData"
            and selected_data
        ):
            if "range" in selected_data:
                x_range = selected_data["range"]["x"]
                if x_range and len(x_range) == 2:
                    return {"x_min": x_range[0], "x_max": x_range[1]}

        relayout_data = (
            overview_relayout
            if trigger_id == "overview-graph" and trigger_prop == "relayoutData"
            else main_relayout
            if trigger_id == "main-graph" and trigger_prop == "relayoutData"
            else None
        )

        if relayout_data:
            if "xaxis.range[0]" in relayout_data:
                return {
                    "x_min": relayout_data["xaxis.range[0]"],
                    "x_max": relayout_data["xaxis.range[1]"],
                }
            if "xaxis.autorange" in relayout_data:
                return None

        return no_update

    @app.callback(
        [Output("video-path-input", "value"), Output("subtitle-path-input", "value")],
        [Input("video-dropdown", "value")],
    )
    def update_video_path_from_dropdown(selected_video):
        """Update video and auto-detect subtitle when dropdown selection changes."""
        if not selected_video:
            return no_update, no_update
        base = os.path.splitext(selected_video)[0]
        for ext in (".srt", ".vtt"):
            spath = base + ext
            if os.path.isfile(spath):
                return selected_video, spath
        return selected_video, ""

    @app.callback(
        Output("subtitle-path-input", "value"),
        [Input("subtitle-dropdown", "value")],
    )
    def update_subtitle_path_from_dropdown(selected_subtitle):
        """Update subtitle path input when subtitle dropdown selection changes."""
        return selected_subtitle if selected_subtitle else no_update

    @app.callback(
        [
            Output("export-status", "children"),
            Output("export-png-btn", "n_clicks"),
            Output("export-csv-btn", "n_clicks"),
            Output("export-peaks-btn", "n_clicks"),
            Output("export-regions-btn", "n_clicks"),
        ],
        [
            Input("export-png-btn", "n_clicks"),
            Input("export-csv-btn", "n_clicks"),
            Input("export-peaks-btn", "n_clicks"),
            Input("export-regions-btn", "n_clicks"),
        ],
        [
            State("current-data-store", "data"),
            State("figure-json-store", "data"),
            State("trace-checklist", "value"),
            State("zoom-range-store", "data"),
            State("zoom-peaks-only-check", "value"),
        ],
    )
    def handle_exports(
        png_clicks, csv_clicks, peaks_clicks, regions_clicks,
        data, fig_data, trace_options, zoom_range, zoom_peaks_only
    ):
        """Handle PNG, CSV, Peaks, and Regions export."""
        ctx = callback_context
        if not ctx.triggered:
            return "", 0, 0, peaks_clicks, regions_clicks

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "export-png-btn" and png_clicks > 0:
            if not KALEIDO_AVAILABLE:
                return "PNG export requires kaleido: pip install kaleido", 0, csv_clicks, peaks_clicks, regions_clicks

            if not fig_data:
                return "No figure data available", 0, csv_clicks, peaks_clicks, regions_clicks

            fig = go.Figure(fig_data)
            fig.update_layout(width=1200, height=600)

            fname = f"chat_frequency_{int(time.time())}.png"
            fig.write_image(fname)

            return f"PNG exported to {fname}", 0, csv_clicks, peaks_clicks, regions_clicks

        elif button_id == "export-csv-btn" and csv_clicks > 0:
            if not data:
                return "No data to export", png_clicks, 0, peaks_clicks, regions_clicks

            df_export = pd.DataFrame(
                {
                    "window_center_seconds": data["time_axis"],
                    "window_frequency": data["rolling_sum"],
                }
            )

            if data["trend"]:
                df_export["trend"] = data["trend"]

            if data["peak_times"]:
                df_export["is_peak"] = df_export["window_center_seconds"].isin(
                    data["peak_times"]
                )

            fname = f"chat_frequency_{int(time.time())}.csv"
            df_export.to_csv(fname, index=False)

            return f"CSV exported to {fname}", png_clicks, 0, peaks_clicks, regions_clicks

        elif button_id == "export-peaks-btn" and peaks_clicks > 0:
            if not data or not data["peak_times"]:
                return "No peak data available", png_clicks, csv_clicks, 0, regions_clicks

            peak_times = np.array(data["peak_times"])
            peak_values = (
                np.array(data["peak_values"])
                if data["peak_values"]
                else np.full_like(peak_times, None)
            )

            original_count = len(peak_times)
            status_prefix = ""
            if (
                zoom_peaks_only
                and "zoom" in zoom_peaks_only
                and zoom_range
                and "x_min" in zoom_range
            ):
                x_min, x_max = zoom_range["x_min"], zoom_range["x_max"]
                mask = (peak_times >= x_min) & (peak_times <= x_max)
                peak_times = peak_times[mask]
                peak_values = peak_values[mask]

                if len(peak_times) == 0:
                    return (
                        f"No peaks found in zoom region ({seconds_to_hms(x_min)} to {seconds_to_hms(x_max)})",
                        png_clicks,
                        csv_clicks,
                        0,
                        regions_clicks,
                    )

                filtered_count = len(peak_times)
                status_prefix = f"Using {filtered_count} of {original_count} peaks (zoom region) - "
            else:
                status_prefix = f"Using all {original_count} peaks - "

            rows = []
            for i, (t, v) in enumerate(zip(peak_times, peak_values)):
                messages = analyzer.get_messages_near_time(t, margin=10)
                sample = "; ".join(
                    str(row["message"])[:50] for _, row in messages.head(3).iterrows()
                )

                rows.append(
                    {
                        "peak_time_seconds": t,
                        "peak_time_hms": seconds_to_hms(t),
                        "peak_frequency": v,
                        "sample_messages": sample,
                    }
                )

            df_peaks = pd.DataFrame(rows)
            fname = f"chat_peaks_{int(time.time())}.csv"
            df_peaks.to_csv(fname, index=False)

            return f"{status_prefix}Peaks CSV exported to {fname}", png_clicks, csv_clicks, 0, regions_clicks

        elif button_id == "export-regions-btn" and regions_clicks > 0:
            if not data or not data.get("regions"):
                return "No region data available", png_clicks, csv_clicks, peaks_clicks, 0

            regions = data["regions"]
            original_count = len(regions)
            status_prefix = ""

            if (
                zoom_peaks_only
                and "zoom" in zoom_peaks_only
                and zoom_range
                and "x_min" in zoom_range
            ):
                x_min, x_max = zoom_range["x_min"], zoom_range["x_max"]
                filtered = []
                for r_start, r_end in regions:
                    if r_end < x_min or r_start > x_max:
                        continue
                    filtered.append((max(r_start, x_min), min(r_end, x_max)))
                regions = filtered

                if len(regions) == 0:
                    return (
                        f"No regions found in zoom region ({seconds_to_hms(x_min)} to {seconds_to_hms(x_max)})",
                        png_clicks,
                        csv_clicks,
                        peaks_clicks,
                        0,
                    )

                filtered_count = len(regions)
                status_prefix = f"Using {filtered_count} of {original_count} regions (zoom region) - "
            else:
                status_prefix = f"Using all {original_count} regions - "

            rows = []
            for i, (r_start, r_end) in enumerate(regions):
                rows.append(
                    {
                        "region_index": i + 1,
                        "start_seconds": r_start,
                        "end_seconds": r_end,
                        "start_hms": seconds_to_hms(r_start),
                        "end_hms": seconds_to_hms(r_end),
                        "duration_seconds": r_end - r_start,
                    }
                )

            df_regions = pd.DataFrame(rows)
            fname = f"chat_regions_{int(time.time())}.csv"
            df_regions.to_csv(fname, index=False)

            return f"{status_prefix}Regions CSV exported to {fname}", png_clicks, csv_clicks, peaks_clicks, 0

        return "", png_clicks, csv_clicks, peaks_clicks, regions_clicks

    @app.callback(
        [
            Output("export-status", "children"),
            Output("ffmpeg-btn", "n_clicks"),
            Output("timestamps-btn", "n_clicks"),
        ],
        [
            Input("ffmpeg-btn", "n_clicks"),
            Input("timestamps-btn", "n_clicks"),
        ],
        [
            State("video-path-input", "value"),
            State("clip-before-input", "value"),
            State("clip-after-input", "value"),
            State("current-data-store", "data"),
            State("zoom-range-store", "data"),
            State("zoom-peaks-only-check", "value"),
            State("region-cuts-check", "value"),
            State("intelligent-cut-check", "value"),
            State("subtitle-path-input", "value"),
            State("max-snap-slider", "value"),
            State("silence-threshold-slider", "value"),
            State("min-silence-slider", "value"),
            State("fallback-expand-check", "value"),
        ],
    )
    def handle_video_clipping(
        ffmpeg_clicks,
        timestamps_clicks,
        video_path,
        clip_before,
        clip_after,
        data,
        zoom_range,
        zoom_peaks_only,
        region_cuts_check,
        intelligent_cut_check,
        subtitle_path,
        max_snap_distance,
        silence_threshold_db,
        min_silence_duration,
        fallback_expand_check,
    ):
        """Handle video clipping exports."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, 0, 0

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        use_regions = region_cuts_check and "region-cuts" in region_cuts_check
        use_intelligent = intelligent_cut_check and "intelligent" in intelligent_cut_check
        fallback_expand = fallback_expand_check and "expand" in fallback_expand_check

        clip_before = clip_before or 30
        clip_after = clip_after or 30
        max_snap_distance = max_snap_distance if max_snap_distance is not None else 3.0
        silence_threshold_db = silence_threshold_db if silence_threshold_db is not None else -45.0
        min_silence_duration = min_silence_duration if min_silence_duration is not None else 0.3

        def _filter_by_zoom(items, is_peak=True):
            """Filter peaks or regions by zoom range."""
            if not (
                zoom_peaks_only
                and "zoom" in zoom_peaks_only
                and zoom_range
                and "x_min" in zoom_range
            ):
                return items, ""

            x_min, x_max = zoom_range["x_min"], zoom_range["x_max"]
            filtered = []
            for item in items:
                if is_peak:
                    if x_min <= item <= x_max:
                        filtered.append(item)
                else:
                    r_start, r_end = item
                    if r_end < x_min or r_start > x_max:
                        continue
                    filtered.append((max(r_start, x_min), min(r_end, x_max)))

            original = len(items)
            kept = len(filtered)
            if kept == 0:
                return None, f"No {'peaks' if is_peak else 'regions'} found in zoom region"
            prefix = f"Using {kept} of {original} {'peaks' if is_peak else 'regions'} (zoom region) - "
            return filtered, prefix

        def _merge_overlapping(clips):
            """Merge overlapping or adjacent clips."""
            if not clips:
                return clips
            clips = sorted(clips, key=lambda x: x[0])
            merged = [list(clips[0])]
            for c_start, c_end in clips[1:]:
                if c_start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], c_end)
                else:
                    merged.append([c_start, c_end])
            return merged

        def _maybe_refine_clips(clip_times):
            """Apply intelligent boundary refinement if enabled."""
            if not use_intelligent:
                return clip_times
            if not video_path or not subtitle_path:
                return clip_times
            if not os.path.exists(video_path):
                return clip_times
            if not os.path.exists(subtitle_path):
                return clip_times
            try:
                from ..core.intelligent_cutter import IntelligentCutter
                cutter = IntelligentCutter(
                    video_path,
                    subtitle_path,
                    silence_threshold_db=silence_threshold_db,
                    min_silence_ms=int(min_silence_duration * 1000),
                )
                return cutter.refine_clips(
                    clip_times,
                    max_snap_distance=max_snap_distance,
                    silence_tolerance=1.0,
                    fallback_expand=fallback_expand,
                )
            except Exception:
                # Log gracefully and fall back to rough cuts
                return clip_times

        def _build_ffmpeg(clip_times, label):
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

            output_file = f"highlight_reel_{int(time.time())}.mp4"
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

            fname = f"make_highlight_reel_{int(time.time())}.sh"
            with open(fname, "w") as f:
                f.write("\n".join(commands))

            return fname

        if button_id == "ffmpeg-btn" and ffmpeg_clicks > 0:
            if not video_path:
                return "Please provide video file path", 0, timestamps_clicks

            if use_regions:
                if not data or not data.get("regions"):
                    return "No region data available", 0, timestamps_clicks

                regions = data["regions"]
                regions, status_prefix = _filter_by_zoom(regions, is_peak=False)
                if regions is None:
                    return status_prefix, 0, timestamps_clicks

                clips = []
                for r_start, r_end in regions:
                    clips.append((max(0, r_start - clip_before), r_end + clip_after))
                clip_times = _maybe_refine_clips(clips)
                clip_times = _merge_overlapping(clip_times)
                label = "regions"
            else:
                if not data or not data["peak_times"]:
                    return "No peak data available", 0, timestamps_clicks

                peak_times = np.array(data["peak_times"])
                peak_times, status_prefix = _filter_by_zoom(peak_times.tolist(), is_peak=True)
                if peak_times is None:
                    return status_prefix, 0, timestamps_clicks

                clips = []
                for t in peak_times:
                    clips.append([max(0, t - clip_before), t + clip_after])
                clip_times = _maybe_refine_clips(clips)
                clip_times = _merge_overlapping(clip_times)
                label = "peaks"

            fname = _build_ffmpeg(clip_times, label)
            return f"{status_prefix}Highlight reel script saved to {fname}", 0, timestamps_clicks

        elif button_id == "timestamps-btn" and timestamps_clicks > 0:
            if use_regions:
                if not data or not data.get("regions"):
                    return "No region data available", ffmpeg_clicks, 0

                regions = data["regions"]
                regions, status_prefix = _filter_by_zoom(regions, is_peak=False)
                if regions is None:
                    return status_prefix, ffmpeg_clicks, 0

                clips = []
                for r_start, r_end in regions:
                    clips.append((max(0, r_start - clip_before), r_end + clip_after))
                clip_times = _maybe_refine_clips(clips)
                clip_times = _merge_overlapping(clip_times)

                rows = []
                for i, (c_start, c_end) in enumerate(clip_times):
                    rows.append(
                        {
                            "clip_index": i + 1,
                            "start_time_seconds": c_start,
                            "end_time_seconds": c_end,
                            "start_hms": seconds_to_hms(c_start),
                            "end_hms": seconds_to_hms(c_end),
                        }
                    )

                df = pd.DataFrame(rows)
                fname = f"region_timestamps_{int(time.time())}.csv"
                df.to_csv(fname, index=False)
                return f"{status_prefix}Region timestamps CSV exported to {fname}", ffmpeg_clicks, 0
            else:
                if not data or not data["peak_times"]:
                    return "No peak data available", ffmpeg_clicks, 0

                peak_times = np.array(data["peak_times"])
                peak_values = (
                    np.array(data["peak_values"])
                    if data["peak_values"]
                    else np.full_like(peak_times, None)
                )

                peak_times_list, status_prefix = _filter_by_zoom(peak_times.tolist(), is_peak=True)
                if peak_times_list is None:
                    return status_prefix, ffmpeg_clicks, 0

                # Rebuild filtered peak_values aligned with filtered peak_times
                peak_values_filtered = []
                for t in peak_times_list:
                    idx = np.where(peak_times == t)[0]
                    if len(idx) > 0:
                        peak_values_filtered.append(peak_values[idx[0]])
                    else:
                        peak_values_filtered.append(None)

                clips = []
                for t in peak_times_list:
                    clips.append([max(0, t - clip_before), t + clip_after])
                clip_times = _maybe_refine_clips(clips)
                clip_times = _merge_overlapping(clip_times)

                rows = []
                for i, (c_start, c_end) in enumerate(clip_times):
                    # recover peak time for this clip (closest original peak)
                    peak_t = c_start + clip_before  # rough approximation for metadata
                    v = None
                    # try to find the original peak that landed in this clip
                    for pt, pv in zip(peak_times_list, peak_values_filtered):
                        if c_start <= pt <= c_end:
                            peak_t = pt
                            v = pv
                            break
                    rows.append(
                        {
                            "peak_index": i + 1,
                            "start_time_seconds": c_start,
                            "peak_time_seconds": peak_t,
                            "end_time_seconds": c_end,
                            "peak_frequency": v,
                        }
                    )

                df = pd.DataFrame(rows)
                fname = f"peak_timestamps_{int(time.time())}.csv"
                df.to_csv(fname, index=False)

                return f"{status_prefix}Timestamps CSV exported to {fname}", ffmpeg_clicks, 0

        return no_update, ffmpeg_clicks, timestamps_clicks


from dash import Input, Output, State, html
