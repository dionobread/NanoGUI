"""Core modules for screen capture and action execution."""

from .screen_capture import capture_screen, get_screen_size
from .action_executor import ActionExecutor

__all__ = ["capture_screen", "get_screen_size", "ActionExecutor"]
