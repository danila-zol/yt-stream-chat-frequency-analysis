#!/usr/bin/env python3
"""
Backward compatibility wrapper for interactive_chat_frequency.py

This script provides compatibility with the old command-line interface:
    python interactive_chat_frequency.py CHAT_FILE.csv [OPTIONS]

It delegates to the new chatfreq package's serve command.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

sys.argv = ["interactive_chat_frequency.py", "serve"] + sys.argv[1:]

from chatfreq.cli.commands import cli

if __name__ == "__main__":
    cli()
