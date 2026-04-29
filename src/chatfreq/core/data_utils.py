"""
Data loading and validation utilities.
"""
import pandas as pd


def load_chat_data(csv_path: str) -> pd.DataFrame:
    """
    Load chat data from a TSV file.

    Args:
        csv_path: Path to tab-separated CSV file

    Returns:
        DataFrame with columns: video_time, author, message, seconds

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing or no messages found
    """
    df = pd.read_csv(csv_path, sep="\t")

    required = ["video_time", "author", "message"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    from .time_utils import parse_time

    df["seconds"] = df["video_time"].apply(parse_time)

    if len(df) == 0:
        raise ValueError("No messages found in file")

    return df
