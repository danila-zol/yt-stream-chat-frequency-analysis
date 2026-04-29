"""
Chat Frequency Analysis Tool

Analyze chat message frequency in video streams using sliding window
convolution algorithm. Provides both interactive web visualization
and CLI export capabilities.
"""
from .core import ChatFrequencyAnalyzer

__version__ = "1.0.0"
__all__ = ["ChatFrequencyAnalyzer"]
