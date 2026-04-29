"""
App layout definition for the Dash web application.
"""
import os

from dash import dcc, html

try:
    import kaleido

    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False


def _find_subtitle_files(video_files):
    """Build dropdown options for subtitle files paired with video basenames."""
    subs = []
    seen = set()
    for vf in video_files:
        path = vf.get("value", "")
        base = os.path.splitext(path)[0]
        for ext in (".srt", ".vtt"):
            spath = base + ext
            if os.path.isfile(spath) and spath not in seen:
                seen.add(spath)
                subs.append({"label": os.path.basename(spath), "value": spath})
    return subs


def _guess_subtitle_for_video(video_path):
    if not video_path:
        return ""
    base = os.path.splitext(video_path)[0]
    for ext in (".srt", ".vtt"):
        spath = base + ext
        if os.path.isfile(spath):
            return spath
    return ""


def create_layout(
    app,
    csv_path,
    video_files,
    default_video="",
    default_window=60,
    default_step=10,
    default_degree=3,
    default_prominence=5,
    default_cbs_threshold=2.5,
    default_z_threshold=0.0,
    default_min_duration=30,
    default_max_gap=30,
):
    """
    Create the full app layout.

    Args:
        app: Dash app instance
        csv_path: Path to loaded CSV file
        video_files: List of video file options for dropdown
        default_video: Default video path
        default_window: Default window size slider value
        default_step: Default step size slider value
        default_degree: Default polynomial degree
        default_prominence: Default peak prominence

    Returns:
        Dash HTML component
    """
    app.title = f"Chat Frequency: {os.path.basename(csv_path)}"

    return html.Div(
        [
            html.H1(
                f"Chat Frequency Analyzer: {os.path.basename(csv_path)}",
                style={"textAlign": "center", "marginBottom": "20px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Parameters", style={"marginTop": "0"}),
                            _create_parameter_controls(
                                default_window,
                                default_step,
                                default_degree,
                                default_prominence,
                                default_cbs_threshold,
                                default_z_threshold,
                                default_min_duration,
                                default_max_gap,
                            ),
                            html.Hr(),
                            _create_trace_controls(),
                            html.Hr(),
                            _create_action_buttons(),
                            html.Hr(),
                            _create_video_clipping_section(
                                csv_path,
                                video_files,
                                default_video,
                                _find_subtitle_files(video_files),
                                _guess_subtitle_for_video(default_video),
                            ),
                            html.Hr(),
                            _create_hover_info_section(),
                        ],
                        style={
                            "width": "25%",
                            "padding": "20px",
                            "borderRight": "1px solid #ccc",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                id="main-graph",
                                style={"height": "70vh"},
                                config={
                                    "displaylogo": False,
                                    "modeBarButtonsToAdd": [
                                        "drawline",
                                        "drawopenpath",
                                        "drawclosedpath",
                                        "drawcircle",
                                        "drawrect",
                                        "eraseshape",
                                    ],
                                },
                            ),
                            dcc.Graph(
                                id="overview-graph",
                                style={"height": "20vh"},
                                config={"displayModeBar": False},
                            ),
                            dcc.Store(id="current-data-store"),
                            dcc.Store(id="figure-json-store"),
                            dcc.Store(id="zoom-range-store"),
                        ],
                        style={"width": "75%", "padding": "20px"},
                    ),
                ],
                style={"display": "flex", "flexDirection": "row"},
            ),
        ]
    )


def _create_parameter_controls(
    window, step, degree, prominence, cbs_threshold, z_threshold, min_duration, max_gap
):
    """Create parameter slider controls."""
    return html.Div(
        [
            html.Label("Window Size (seconds):"),
            dcc.Slider(
                id="window-slider",
                min=10,
                max=300,
                step=10,
                value=window,
                marks={i: str(i) for i in [10, 60, 120, 180, 240, 300]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label("Step Size (seconds):", style={"marginTop": "20px"}),
            dcc.Slider(
                id="step-slider",
                min=1,
                max=30,
                step=1,
                value=step,
                marks={
                    i: str(i)
                    for i in [1, 5, 10, 15, 20, 25, 30]
                },
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label("Polynomial Degree:", style={"marginTop": "20px"}),
            dcc.Slider(
                id="degree-slider",
                min=0,
                max=5,
                step=1,
                value=degree,
                marks={i: str(i) for i in range(6)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label("Peak Prominence:", style={"marginTop": "20px"}),
            dcc.Slider(
                id="prominence-slider",
                min=1,
                max=20,
                step=1,
                value=prominence,
                marks={i: str(i) for i in [1, 5, 10, 15, 20]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label("CBS Sensitivity (t-threshold):", style={"marginTop": "20px"}),
            dcc.Slider(
                id="cbs-slider",
                min=1.0,
                max=5.0,
                step=0.5,
                value=cbs_threshold,
                marks={i: str(i) for i in [1.0, 2.0, 2.5, 3.0, 4.0, 5.0]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label("Region Z-Score Threshold:", style={"marginTop": "20px"}),
            dcc.Slider(
                id="z-slider",
                min=-1.0,
                max=2.0,
                step=0.1,
                value=z_threshold,
                marks={i: str(i) for i in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label("Min Region Duration (s):", style={"marginTop": "20px"}),
            dcc.Slider(
                id="min-duration-slider",
                min=0,
                max=120,
                step=10,
                value=min_duration,
                marks={i: str(i) for i in [0, 30, 60, 90, 120]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label("Max Gap to Merge (s):", style={"marginTop": "20px"}),
            dcc.Slider(
                id="max-gap-slider",
                min=0,
                max=120,
                step=10,
                value=max_gap,
                marks={i: str(i) for i in [0, 30, 60, 90, 120]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ]
    )


def _create_trace_controls():
    """Create trace visibility checklist and theme toggle."""
    return html.Div(
        [
            dcc.Checklist(
                id="trace-checklist",
                options=[
                    {"label": " Histogram Bars", "value": "bars"},
                    {"label": " Sliding Window", "value": "sliding"},
                    {"label": " Trend Line", "value": "trend"},
                    {"label": " Peak Markers", "value": "peaks"},
                    {"label": " High engagement regions", "value": "regions"},
                    {"label": " Normalize by trend", "value": "normalize"},
                    {"label": " Dark Theme", "value": "dark"},
                ],
                value=["bars", "sliding", "trend"],
                style={"marginTop": "20px"},
            ),
        ]
    )


def _create_action_buttons():
    """Create export action buttons."""
    return html.Div(
        [
            html.Button(
                "Reset Zoom",
                id="reset-zoom-btn",
                n_clicks=0,
                style={"marginRight": "10px", "marginTop": "20px"},
            ),
            html.Button(
                "Export PNG",
                id="export-png-btn",
                n_clicks=0,
                disabled=not KALEIDO_AVAILABLE,
                style={"marginRight": "10px", "marginTop": "20px"},
            ),
            html.Button(
                "Export CSV",
                id="export-csv-btn",
                n_clicks=0,
                style={"marginRight": "10px", "marginTop": "20px"},
            ),
            html.Button(
                "Export Peaks",
                id="export-peaks-btn",
                n_clicks=0,
                style={"marginRight": "10px", "marginTop": "20px"},
            ),
            html.Button(
                "Export Regions",
                id="export-regions-btn",
                n_clicks=0,
                style={"marginTop": "20px"},
            ),
        ]
    )


def _create_video_clipping_section(
    csv_path, video_files, default_video, subtitle_files, default_subtitle
):
    """Create video clipping controls."""
    base_dir = (
        os.path.dirname(os.path.abspath(csv_path)) if os.path.dirname(csv_path) else "."
    )

    return html.Div(
        [
            html.H4("Video Clipping"),
            html.Label("Video File Path:"),
            dcc.Input(
                id="video-path-input",
                type="text",
                placeholder="/path/to/video.mp4",
                value=default_video,
                style={"width": "100%", "marginBottom": "10px"},
            ),
            html.Div(
                [
                    html.Label("Or select video from directory:"),
                    dcc.Dropdown(
                        id="video-dropdown",
                        options=video_files,
                        placeholder="Select a video file...",
                        style={"marginBottom": "5px"},
                    ),
                    html.Small(
                        f"Directory: {base_dir}",
                        style={
                            "color": "#666",
                            "display": "block",
                            "marginBottom": "10px",
                        },
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label(
                        "Clip Before (s):",
                        style={"display": "inline-block", "marginRight": "10px"},
                    ),
                    dcc.Input(
                        id="clip-before-input",
                        type="number",
                        value=30,
                        min=0,
                        step=5,
                        style={"width": "80px", "marginRight": "20px"},
                    ),
                    html.Label(
                        "Clip After (s):",
                        style={"display": "inline-block", "marginRight": "10px"},
                    ),
                    dcc.Input(
                        id="clip-after-input",
                        type="number",
                        value=30,
                        min=0,
                        step=5,
                        style={"width": "80px"},
                    ),
                ]
            ),
            html.Hr(),
            html.H5("Intelligent Cutting"),
            dcc.Checklist(
                id="intelligent-cut-check",
                options=[
                    {
                        "label": " Use intelligent cutting (subtitles + audio silence)",
                        "value": "intelligent",
                    }
                ],
                value=[],
                style={"marginBottom": "10px"},
            ),
            html.Label("Subtitle File Path:"),
            dcc.Input(
                id="subtitle-path-input",
                type="text",
                placeholder="/path/to/subs.srt",
                value=default_subtitle,
                style={"width": "100%", "marginBottom": "10px"},
            ),
            html.Div(
                [
                    html.Label("Or select subtitle from directory:"),
                    dcc.Dropdown(
                        id="subtitle-dropdown",
                        options=subtitle_files,
                        placeholder="Select a subtitle file...",
                        style={"marginBottom": "5px"},
                    ),
                ]
            ),
            html.Label("Max Snap Distance (s):", style={"marginTop": "10px"}),
            dcc.Slider(
                id="max-snap-slider",
                min=0,
                max=15,
                step=0.5,
                value=3.0,
                marks={i: str(i) for i in [0, 3, 5, 10, 15]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label("Silence Threshold (dB):", style={"marginTop": "10px"}),
            dcc.Slider(
                id="silence-threshold-slider",
                min=-60,
                max=-20,
                step=1,
                value=-45,
                marks={i: str(i) for i in [-60, -50, -45, -40, -30, -20]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Label("Min Silence Duration (s):", style={"marginTop": "10px"}),
            dcc.Slider(
                id="min-silence-slider",
                min=0.1,
                max=2.0,
                step=0.1,
                value=0.3,
                marks={round(i, 1): str(round(i, 1)) for i in [0.1, 0.5, 1.0, 1.5, 2.0]},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            dcc.Checklist(
                id="fallback-expand-check",
                options=[
                    {
                        "label": " Expand search if no boundary found (instead of falling back to rough cut)",
                        "value": "expand",
                    }
                ],
                value=[],
                style={"marginTop": "10px", "marginBottom": "10px"},
            ),
            html.Hr(),
            html.Div(
                [
                    dcc.Checklist(
                        id="zoom-peaks-only-check",
                        options=[
                            {
                                "label": " Only include peaks in current zoom region",
                                "value": "zoom",
                            }
                        ],
                        value=[],
                        style={"marginTop": "10px", "marginBottom": "10px"},
                    ),
                    dcc.Checklist(
                        id="region-cuts-check",
                        options=[
                            {
                                "label": " Use one cut per high-engagement region",
                                "value": "region-cuts",
                            }
                        ],
                        value=[],
                        style={"marginTop": "5px", "marginBottom": "10px"},
                    ),
                ]
            ),
            html.Button(
                "Generate FFmpeg Commands",
                id="ffmpeg-btn",
                n_clicks=0,
                style={"marginTop": "10px", "marginRight": "10px"},
            ),
            html.Button(
                "Export Timestamps for Editing",
                id="timestamps-btn",
                n_clicks=0,
                style={"marginTop": "10px"},
            ),
            html.Div(id="export-status", style={"marginTop": "10px", "color": "green"}),
        ],
        style={"marginTop": "20px"},
    )


def _create_hover_info_section():
    """Create hover info display section."""
    return html.Div(
        [
            html.H4("Hover Info"),
            html.Div(
                id="hover-info",
                style={
                    "border": "1px solid #ccc",
                    "padding": "10px",
                    "maxHeight": "200px",
                    "overflowY": "auto",
                    "fontSize": "12px",
                },
            ),
        ]
    )
