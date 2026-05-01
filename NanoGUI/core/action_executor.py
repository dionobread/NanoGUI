"""
Action executor for real GUI automation using pyautogui.

Converts normalized coordinates from the Grounder to pixel-level OS actions.
Supports a dry-run mode for safe testing without real clicks.
"""

import asyncio
import logging
from typing import Tuple

import pyautogui
import PIL.Image

from ..agents.base import GroundedAction, ActionType

logger = logging.getLogger(__name__)

# Safety: pyautogui will raise if the cursor hits a screen corner.
pyautogui.FAILSAFE = True


class ActionExecutor:
    """Execute grounded GUI actions on the operating system."""

    def __init__(
        self,
        action_delay: float = 0.5,
        dry_run: bool = False,
    ):
        """
        Args:
            action_delay: Seconds to pause between actions.
            dry_run: If True, log actions without executing them.
        """
        self.action_delay = action_delay
        self.dry_run = dry_run

    async def execute_action(
        self,
        action: GroundedAction,
        screen_width: int,
        screen_height: int,
    ) -> None:
        """
        Execute a GroundedAction, denormalizing coordinates first.

        Args:
            action: The grounded action from the Grounder agent.
            screen_width: Screen width in pixels.
            screen_height: Screen height in pixels.
        """
        match action.action_type:
            case ActionType.CLICK:
                if action.coordinate is None:
                    raise ValueError("Click action requires a coordinate")
                x, y = action.coordinate.to_pixels(screen_width, screen_height)
                await self.click(x, y)

            case ActionType.TYPE:
                if action.text is None:
                    raise ValueError("Type action requires text")
                await self.type_text(action.text)

            case ActionType.SCROLL:
                direction = action.direction or "down"
                amount = 3  # default scroll clicks
                await self.scroll(direction, amount)

            case ActionType.KEY:
                if action.text is None:
                    raise ValueError("Key action requires a key name")
                await self.press_key(action.text)

            case ActionType.WAIT:
                await asyncio.sleep(self.action_delay * 2)

            case _:
                logger.warning("Unsupported action type: %s", action.action_type)

    # -- Low-level async wrappers --

    async def click(self, x: int, y: int) -> None:
        """Click at absolute pixel coordinates."""
        logger.info("CLICK at (%d, %d)%s", x, y, " [DRY RUN]" if self.dry_run else "")
        if self.dry_run:
            return
        pyautogui.click(x, y)
        await asyncio.sleep(self.action_delay)

    async def type_text(self, text: str) -> None:
        """Type a string using the keyboard."""
        logger.info("TYPE '%s'%s", text[:50], " [DRY RUN]" if self.dry_run else "")
        if self.dry_run:
            return
        pyautogui.typewrite(text, interval=0.05)
        await asyncio.sleep(self.action_delay)

    async def scroll(self, direction: str, amount: int = 3) -> None:
        """Scroll the mouse wheel. direction: 'up' or 'down'."""
        ticks = amount if direction == "up" else -amount
        logger.info("SCROLL %s (%d)%s", direction, amount, " [DRY RUN]" if self.dry_run else "")
        if self.dry_run:
            return
        pyautogui.scroll(ticks)
        await asyncio.sleep(self.action_delay)

    async def press_key(self, key: str) -> None:
        """Press a single key (e.g. 'enter', 'tab', 'escape')."""
        logger.info("KEY '%s'%s", key, " [DRY RUN]" if self.dry_run else "")
        if self.dry_run:
            return
        pyautogui.press(key)
        await asyncio.sleep(self.action_delay)
