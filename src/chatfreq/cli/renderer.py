"""
Terminal output helpers for CLI.
"""
import sys


def info(msg: str) -> None:
    """Print info message to stdout."""
    print(msg)


def success(msg: str) -> None:
    """Print success message to stdout."""
    print(f"\033[92m{msg}\033[0m")


def warning(msg: str) -> None:
    """Print warning message to stderr."""
    print(f"\033[93mWARNING: {msg}\033[0m", file=sys.stderr)


def error(msg: str) -> None:
    """Print error message to stderr."""
    print(f"\033[91mERROR: {msg}\033[0m", file=sys.stderr)


def die(msg: str, exit_code: int = 1) -> None:
    """Print error message and exit."""
    error(msg)
    sys.exit(exit_code)


def format_duration(seconds: float) -> str:
    """Format duration for display."""
    from ..core.time_utils import seconds_to_hms

    return seconds_to_hms(seconds)


def print_data_summary(analyzer, csv_path: str) -> None:
    """Print summary of loaded chat data."""
    info(f"Loaded {len(analyzer.timestamps)} messages from {csv_path}")
    info(f"Stream duration: {format_duration(analyzer.max_seconds)}")
