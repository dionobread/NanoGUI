"""
NanoGUI Data Loading Utilities

This package provides unified data loading for testing and evaluation.
"""

from .test_data_loader import (
    load_test_sample,
    load_local_screenspot,
    get_dataset_stats,
)

__all__ = [
    "load_test_sample",
    "load_local_screenspot",
    "get_dataset_stats",
]
