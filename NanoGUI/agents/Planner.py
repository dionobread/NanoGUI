"""
PlannerAgent — high-level task decomposition.

Given a natural-language task description and the current screenshot of the
GUI, the Planner decomposes the task into a numbered list of atomic sub-goals,
each expressed as a single actionable instruction (e.g. "Click the search bar",
"Type 'flights to NYC'").

The Planner runs once per high-level step (or when the Verifier reports a
failure and requests replanning with an error description).

It does NOT predict pixel coordinates — that is the Grounder's job.

Design: 
- Inherits from BaseAgent for common functionality
- Works with any vision-capable model client
- Exposes a clean async plan() method that returns list[SubGoal]
- Exposes a replan() method for Verifier-triggered error recovery

Output format (enforced via system prompt): 

The model must respond with a JSON object:
{
  "sub_goals": [
    "Click the address bar",
    "Type 'google.com'",
    "Press Enter"
  ]
}
"""

import logging
from typing import List

import PIL.Image
from autogen_core import Image as AGImage
from autogen_agentchat.messages import MultiModalMessage
from autogen_core.models import ChatCompletionClient

from .base import (
    BaseAgent,
    ActionResult,
    SubGoal,
    ParsingError,
    log_agent_call,
)

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """
    High-level task planner that produces a list of atomic GUI sub-goals.

    Inherits from BaseAgent for common functionality including:
    - AutoGen AssistantAgent management
    - JSON parsing
    - Logging and error handling
    - Resource cleanup

    model_client : Any AutoGen-compatible ChatCompletionClient (GLM or Qwen)
    max_steps    : Hard cap on number of sub-goals generated (default 20)
    """

    def __init__(
        self,
        model_client: ChatCompletionClient,
        max_steps: int = 20,
    ) -> None:
        """
        Initialize the Planner agent.

        model_client : ChatCompletionClient - AutoGen-compatible model client
        max_steps : int - Maximum number of sub-goals to generate
        """
        self._max_steps = max_steps

        super().__init__(
            name="planner",
            model_client=model_client,
            reflect_on_tool_use=False,
        )

    async def execute(self, *args, **kwargs) -> ActionResult:
        """
        Execute method required by BaseAgent.

        This routes to the appropriate method (plan or replan) based on kwargs.
        For direct use, prefer plan() or replan() methods.
        """
        if "replan" in kwargs and kwargs["replan"]:
            # Replanning mode
            task = kwargs.get("task")
            completed = kwargs.get("completed", [])
            failed_step = kwargs.get("failed_step")
            error = kwargs.get("error")
            screenshot = kwargs.get("screenshot")

            if not all([task, failed_step, error, screenshot]):
                return ActionResult(
                    success=False,
                    content="",
                    error="Missing required parameters for replan"
                )

            sub_goals = await self.replan(task, completed, failed_step, error, screenshot)
            return ActionResult(
                success=True,
                content=f"Generated {len(sub_goals)} remaining sub-goals",
                metadata={"sub_goals": sub_goals}
            )
        else:
            # Initial planning mode
            task = kwargs.get("task") or (args[0] if args else None)
            screenshot = kwargs.get("screenshot") or (args[1] if len(args) > 1 else None)

            if not task or not screenshot:
                return ActionResult(
                    success=False,
                    content="",
                    error="Missing required parameters: task and screenshot"
                )

            sub_goals = await self.plan(task, screenshot)
            return ActionResult(
                success=True,
                content=f"Generated {len(sub_goals)} sub-goals",
                metadata={"sub_goals": sub_goals}
            )

    # Public API 

    async def plan(
        self,
        task: str,
        screenshot: PIL.Image.Image,
    ) -> List[str]:
        """
        Decompose a task into a list of sub-goals given the current screenshot.

        Parameters: 
        task       : Natural-language task description
        screenshot : Current GUI screenshot as a PIL Image

        Returns: 
        list[str] — ordered sub-goals, capped at max_steps
        """
        log_agent_call(self._name, "planning", task=task)

        prompt = f"Task: {task}\n\nAnalyse the screenshot and produce the sub-goals."
        message = self._build_multimodal_message(prompt, screenshot)

        result = await self._agent.run(task=message)
        raw = self.extract_last_text(result)

        try:
            sub_goals = self._parse_sub_goals(raw)
            logger.info(
                "[%s] Generated %d sub-goals for task: %s",
                self._name, len(sub_goals), task
            )
            return sub_goals[:self._max_steps]
        except Exception as exc:
            raise ParsingError(
                f"Failed to parse sub-goals: {exc}"
            ) from exc

    async def replan(
        self,
        task: str,
        completed: List[str],
        failed_step: str,
        error: str,
        screenshot: PIL.Image.Image,
    ) -> List[str]:
        """
        Produce a revised plan after a Verifier-reported failure.

        Parameters:
        task        : Original task description
        completed   : Sub-goals already executed successfully
        failed_step : The sub-goal that just failed
        error       : Error description from the Verifier
        screenshot  : Current screenshot after the failed action

        Returns:
        list[str] — revised remaining sub-goals
        """
        log_agent_call(
            self._name,
            "replanning",
            task=task,
            num_completed=len(completed),
            failed_step=failed_step
        )

        completed_str = "\n".join(f"  ✓ {s}" for s in completed) or "  (none yet)"
        prompt = (
            f"Original task: {task}\n\n"
            f"Completed steps:\n{completed_str}\n\n"
            f"Failed step: {failed_step}\n"
            f"Verifier error: {error}\n\n"
            f"Produce a revised list of remaining sub-goals."
        )

        # Create a temporary agent with replan system prompt
        from autogen_agentchat.agents import AssistantAgent

        replan_agent = AssistantAgent(
            name=f"{self._name}_replan",
            model_client=self._model_client,
            system_message=self.load_system_prompt("planner", "replan"),
            reflect_on_tool_use=False,
        )

        message = self._build_multimodal_message(prompt, screenshot)
        result = await replan_agent.run(task=message)

        raw = self.extract_last_text(result)
        sub_goals = self._parse_sub_goals(raw)

        logger.info(
            "[%s] Replanned — %d remaining sub-goals",
            self._name, len(sub_goals)
        )
        return sub_goals[:self._max_steps]

    # Helper methods 

    @staticmethod
    def _build_multimodal_message(text: str, image: PIL.Image.Image) -> MultiModalMessage:
        """Build a multimodal message with text and image."""
        return MultiModalMessage(
            content=[text, AGImage(image)],
            source="user",
        )

    def _parse_sub_goals(self, raw: str) -> List[str]:
        """
        Parse the model's JSON response into a clean list of sub-goal strings.

        Uses BaseAgent.parse_json_output() for robust parsing.

        Falls back gracefully if JSON is malformed by extracting lines
        that look like list items.
        """
        try:
            data = self.parse_json_output(raw)
            goals = data.get("sub_goals", [])
            return [str(g).strip() for g in goals if str(g).strip()]
        except ValueError:
            # Fallback: extract lines that look like numbered list items
            logger.warning("[%s] JSON parse failed — attempting line-by-line fallback", self._name)
            import re
            lines = []
            for line in raw.splitlines():
                line = re.sub(r"^\s*[\d\-\*\.]+\s*", "", line).strip()
                if line and len(line) > 3:
                    lines.append(line)
            return lines

    # Convenience methods 

    async def plan_with_metadata(
        self,
        task: str,
        screenshot: PIL.Image.Image,
    ) -> List[SubGoal]:
        """
        Plan and return SubGoal objects with metadata.

        Convenience method that wraps plan() to return structured SubGoal objects.
        """
        descriptions = await self.plan(task, screenshot)
        return [
            SubGoal(
                description=desc,
                step_number=i + 1,
                completed=False
            )
            for i, desc in enumerate(descriptions)
        ]