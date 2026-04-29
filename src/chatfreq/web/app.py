"""
Dash app factory for the chat frequency analyzer.
"""
import dash

from .callbacks import register_callbacks
from .layout import create_layout


def create_app(
    analyzer,
    csv_path,
    video_files,
    default_window=60,
    default_step=10,
    default_degree=3,
    default_prominence=5,
    default_video="",
):
    """
    Create and configure the Dash application.

    Args:
        analyzer: ChatFrequencyAnalyzer instance
        csv_path: Path to loaded CSV file
        video_files: List of video file options for dropdown
        default_window: Default window size slider value
        default_step: Default step size slider value
        default_degree: Default polynomial degree
        default_prominence: Default peak prominence
        default_video: Default video path

    Returns:
        Configured Dash app instance
    """
    app = dash.Dash(
        __name__,
        title=f"Chat Frequency: {csv_path}",
        update_title=None,
    )

    app.layout = create_layout(
        app,
        csv_path,
        video_files,
        default_video=default_video,
        default_window=default_window,
        default_step=default_step,
        default_degree=default_degree,
        default_prominence=default_prominence,
    )

    register_callbacks(app, analyzer)

    return app
