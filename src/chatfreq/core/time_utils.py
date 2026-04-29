"""
Time parsing and formatting utilities for chat frequency analysis.
"""
from typing import Union


def parse_time(timestr: Union[str, int, float]) -> int:
    """
    Convert timestamp string to total seconds.

    Supports formats:
        "mm:ss"
        "h:mm:ss"
        "hh:mm:ss"
        raw int/float seconds

    Args:
        timestr: Time string or numeric seconds

    Returns:
        Total seconds as integer

    Raises:
        ValueError: If time format is invalid
    """
    if isinstance(timestr, (int, float)):
        return int(timestr)

    timestr = str(timestr).strip()

    try:
        return int(float(timestr))
    except ValueError:
        pass

    parts = timestr.split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Invalid time format: {timestr}")

    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def seconds_to_hms(seconds: float) -> str:
    """
    Convert seconds to H:MM:SS or M:SS format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "1:23:45" or "5:30")
    """
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def find_video_files(directory: str):
    """
    Return list of video files in directory for dropdown options.

    Args:
        directory: Path to directory to search

    Returns:
        List of dicts with 'label' and 'value' keys
    """
    import os

    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg"}
    files = []

    try:
        for f in os.listdir(directory):
            full_path = os.path.join(directory, f)
            if os.path.isfile(full_path) and os.path.splitext(f)[1].lower() in video_extensions:
                files.append({"label": f, "value": full_path})
    except OSError:
        pass

    files.sort(key=lambda x: x["label"].lower())
    return files
