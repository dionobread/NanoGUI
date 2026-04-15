"""
Cross-platform screen capture using mss.

Provides:
- capture_screen: Capture full screen or a region as PIL Image
- get_screen_size: Get monitor resolution
"""

import logging
from typing import Tuple

import mss
import PIL.Image
import numpy as np

logger = logging.getLogger(__name__)


def capture_screen(
    monitor: int = 1,
    region: Tuple[int, int, int, int] | None = None,
) -> PIL.Image.Image:
    """
    Capture the screen as a PIL Image.

    Args:
        monitor: Monitor index (1-based, primary is 1).
        region: Optional (left, top, right, bottom) crop region in pixels.
                If None, captures the full monitor.

    Returns:
        PIL Image in RGB mode.
    """
    with mss.mss() as sct:
        # mss uses 0-based index for monitor list;
        # index 0 is the virtual bounding box, index 1+ are physical monitors.
        monitor_idx = min(monitor, len(sct.monitors) - 1)
        monitor_idx = max(monitor_idx, 1)

        if region is not None:
            left, top, right, bottom = region
            capture_area = {
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top,
            }
        else:
            capture_area = sct.monitors[monitor_idx]

        screenshot = sct.grab(capture_area)

    # mss returns BGRA; convert to RGB PIL Image
    img = PIL.Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
    logger.debug("Captured screen: %dx%d", img.width, img.height)
    return img


def get_screen_size(monitor: int = 1) -> Tuple[int, int]:
    """
    Get the resolution of a monitor.

    Args:
        monitor: Monitor index (1-based).

    Returns:
        (width, height) in pixels.
    """
    with mss.mss() as sct:
        monitor_idx = min(monitor, len(sct.monitors) - 1)
        monitor_idx = max(monitor_idx, 1)
        m = sct.monitors[monitor_idx]
        return m["width"], m["height"]
