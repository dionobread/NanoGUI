"""
GrounderAgent — visual grounding of GUI elements.

Given a screenshot and a natural-language sub-goal from the Planner, the
Grounder predicts the exact (x, y) screen coordinate to click, or produces a
structured action object for other action types (type text, scroll, press key).

This is the performance-critical path: every action step calls the Grounder,
so it uses the SMALLER model (2-3B) while the Planner uses the larger model.
In Phase 2, this agent will be fine-tuned with LoRA on ScreenSpot data to
dramatically improve its grounding accuracy.

Output format (enforced via system prompt): 
{
  "action_type": "click" | "type" | "scroll" | "key" | "wait",
  "coordinate":  [x, y],          // normalized [0,1] — click/scroll only
  "text":        "...",            // type/key only
  "direction":   "up"|"down",     // scroll only
  "reasoning":   "..."            // brief explanation (aids debugging)
}
"""

from __future__ import annotations

import logging
from typing import Optional

import PIL.Image
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image as AGImage
from autogen_core.models import ChatCompletionClient

from .base import (
    BaseAgent,
    ActionResult,
    GroundedAction,
    ActionType,
    Coordinate,
    ParsingError,
    log_agent_call,
)

logger = logging.getLogger(__name__)

class GrounderAgent(BaseAgent):
    """
    Visual grounding agent — maps a sub-goal + screenshot → GroundedAction.

    Inherits from BaseAgent for common functionality including:
    - AutoGen AssistantAgent management
    - JSON parsing
    - Logging and error handling
    - Resource cleanup

    Parameters
    ----------
    model_client : AutoGen-compatible ChatCompletionClient
                   Should be the SMALLER model (2-3B) for efficiency
                   In Phase 2, the underlying model will be LoRA-fine-tuned
    """

    def __init__(
        self,
        model_client: ChatCompletionClient,
    ) -> None:
        """
        Initialize the Grounder agent.

        Parameters
        ----------
        model_client : ChatCompletionClient - AutoGen-compatible model client
        """
        # Initialize BaseAgent with grounder-specific configuration
        super().__init__(
            name="grounder",
            model_client=model_client,
            reflect_on_tool_use=False,
        )

    async def execute(self, *args, **kwargs) -> ActionResult:
        """
        Execute method required by BaseAgent.

        Routes to ground() method with the provided parameters.
        For direct use, prefer the ground() method.
        """
        sub_goal = kwargs.get("sub_goal") or (args[0] if args else None)
        screenshot = kwargs.get("screenshot") or (args[1] if len(args) > 1 else None)

        if not sub_goal or not screenshot:
            return ActionResult(
                success=False,
                content="",
                error="Missing required parameters: sub_goal and screenshot"
            )

        action = await self.ground(sub_goal, screenshot)
        return ActionResult(
            success=True,
            content=str(action),
            metadata={
                "action_type": action.action_type.value,
                "coordinate": [action.coordinate.x, action.coordinate.y] if action.coordinate else None,
                "text": action.text,
                "direction": action.direction,
                "reasoning": action.reasoning
            }
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def ground(
        self,
        sub_goal: str,
        screenshot: PIL.Image.Image,
    ) -> GroundedAction:
        """
        Predict the action required to execute sub_goal on the given screenshot.

        Parameters: 
        sub_goal   : Single atomic instruction from the Planner
        screenshot : Current GUI screenshot

        Returns:
        GroundedAction with parsed action_type, coordinate, text, etc.

        Raises:
        ParsingError - If model output cannot be parsed
        """
        log_agent_call(self._name, "grounding", sub_goal=sub_goal)

        prompt = (
            f"Sub-goal: {sub_goal}\n\n"
            "Identify the target element in the screenshot and output the action."
        )
        message = MultiModalMessage(
            content=[prompt, AGImage(screenshot)],
            source="user",
        )

        result = await self._agent.run(task=message)
        raw = self.extract_last_text(result)

        try:
            action = self._parse_action(raw)
            logger.info("[%s] %s", self._name, action)
            return action
        except Exception as exc:
            logger.error("[%s] Failed to parse action: %s", self._name, exc)
            # Return a fallback action instead of raising
            return GroundedAction(
                action_type=ActionType.CLICK,
                coordinate=Coordinate(0.5, 0.5),  # Center of screen
                text=None,
                direction=None,
                reasoning=f"PARSE_ERROR: {exc}. Defaulted to screen centre."
            )

    # ── Helper methods ─────────────────────────────────────────────────────────

    def _parse_action(self, raw: str) -> GroundedAction:
        """
        Parse model JSON output into a GroundedAction.

        Uses BaseAgent.parse_json_output() for robust parsing.
        Handles markdown fences and gracefully falls back on parse errors.

        Parameters
        ----------
        raw : str - Raw model output

        Returns
        -------
        GroundedAction

        Raises
        ------
        ParsingError - If JSON is invalid and cannot be recovered
        """
        try:
            data = self.parse_json_output(raw)

            # Parse action type
            action_type_str = data.get("action_type", "click").lower()
            try:
                action_type = ActionType(action_type_str)
            except ValueError:
                logger.warning("[%s] Unknown action_type '%s', defaulting to click", self._name, action_type_str)
                action_type = ActionType.CLICK

            # Parse coordinate
            coord_raw = data.get("coordinate")
            coordinate = None
            if coord_raw and isinstance(coord_raw, list) and len(coord_raw) >= 2:
                try:
                    coordinate = Coordinate(float(coord_raw[0]), float(coord_raw[1]))
                except (ValueError, TypeError) as exc:
                    logger.warning("[%s] Invalid coordinate %s: %s", self._name, coord_raw, exc)

            return GroundedAction(
                action_type=action_type,
                coordinate=coordinate,
                text=data.get("text"),
                direction=data.get("direction"),
                reasoning=data.get("reasoning", ""),
            )
        except ValueError as exc:
            # Re-raise parsing errors with context
            raise ParsingError(
                f"Failed to parse Grounder output: {exc}. Raw: {raw[:200]}"
            ) from exc

    # ── Convenience methods ─────────────────────────────────────────────────────

    async def ground_and_execute(
        self,
        sub_goal: str,
        screenshot: PIL.Image.Image,
        screen_width: int,
        screen_height: int,
        executor,  # ActionExecutor instance
    ) -> bool:
        """
        Ground the action and immediately execute it.

        Convenience method that combines grounding and execution.

        Parameters
        ----------
        sub_goal : str - The sub-goal to ground
        screenshot : PIL.Image - Current screenshot
        screen_width : int - Screen width in pixels
        screen_height : int - Screen height in pixels
        executor : ActionExecutor - Executor instance to perform the action

        Returns
        -------
        bool - True if action was executed successfully
        """
        action = await self.ground(sub_goal, screenshot)

        try:
            if action.action_type == ActionType.CLICK and action.coordinate:
                x, y = action.to_pixel_coords(screen_width, screen_height)
                await executor.click(x, y)
            elif action.action_type == ActionType.TYPE and action.text:
                await executor.type_text(action.text)
            elif action.action_type == ActionType.KEY and action.text:
                await executor.press_key(action.text)
            elif action.action_type == ActionType.SCROLL and action.direction:
                await executor.scroll(action.direction)
            elif action.action_type == ActionType.WAIT:
                import asyncio
                await asyncio.sleep(1)
            else:
                logger.warning("[%s] Cannot execute action: %s", self._name, action)
                return False

            return True
        except Exception as exc:
            logger.error("[%s] Failed to execute action %s: %s", self._name, action, exc)
            return False
