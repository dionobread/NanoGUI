"""
PlannerAgent — high-level task decomposition.

Given a natural-language task description and the current screenshot of the
GUI, the Planner decomposes the task into a numbered list of atomic sub-goals,
each expressed as a single actionable instruction (e.g. "Click the search bar",
"Type 'flights to NYC'").

The Planner runs once per high-level task (and again whenever the Verifier
reports a failure and requests replanning with an error description).

It does NOT predict pixel coordinates — that is the Grounder's job.

Output format (enforced via system prompt):

    {
      "sub_goals": [
        "Click the address bar",
        "Type 'google.com'",
        "Press Enter"
      ]
    }
"""

import logging
import re
from typing import List, Optional

import PIL.Image
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image as AGImage
from autogen_core.models import ChatCompletionClient

from .base import (
    ActionResult,
    BaseAgent,
    ParsingError,
    SubGoal,
    log_agent_call,
)

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """
    High-level task planner that produces an ordered list of atomic GUI sub-goals.

    Inherits from BaseAgent, which provides:
      - AutoGen AssistantAgent lifecycle management
      - Robust JSON parsing (parse_json_output)
      - System-prompt loading (load_system_prompt)
      - Shared logging helpers

    Parameters
    ----------
    model_client : ChatCompletionClient
        Any AutoGen-compatible client (GLM cloud, local Qwen, etc.).
    max_steps : int
        Hard cap on the number of sub-goals returned.  Defaults to 20.
    """

    def __init__(
        self,
        model_client: ChatCompletionClient,
        max_steps: int = 20,
    ) -> None:
        self._max_steps = max_steps

        super().__init__(
            name="planner",
            model_client=model_client,
            reflect_on_tool_use=False,
        )

    # ------------------------------------------------------------------ #
    # BaseAgent contract                                                   #
    # ------------------------------------------------------------------ #

    async def execute(self, *args, **kwargs) -> ActionResult:
        """
        Route to plan() or replan() based on kwargs, returning an ActionResult.

        Prefer calling plan() / replan() directly for typed results.
        """
        if kwargs.get("replan"):
            task        = kwargs.get("task")
            completed   = kwargs.get("completed", [])
            failed_step = kwargs.get("failed_step")
            error       = kwargs.get("error")
            screenshot  = kwargs.get("screenshot")

            if not all([task, failed_step, error, screenshot]):
                return ActionResult(
                    success=False,
                    content="",
                    error="replan requires: task, failed_step, error, screenshot",
                )

            sub_goals = await self.replan(task, completed, failed_step, error, screenshot)
            return ActionResult(
                success=True,
                content=f"Replanned — {len(sub_goals)} remaining sub-goals",
                metadata={"sub_goals": sub_goals},
            )

        # --- initial plan ---
        task       = kwargs.get("task")       or (args[0] if args else None)
        screenshot = kwargs.get("screenshot") or (args[1] if len(args) > 1 else None)

        if not task or not screenshot:
            return ActionResult(
                success=False,
                content="",
                error="plan requires: task, screenshot",
            )

        sub_goals = await self.plan(task, screenshot)
        return ActionResult(
            success=True,
            content=f"Generated {len(sub_goals)} sub-goals",
            metadata={"sub_goals": sub_goals},
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def plan(
        self,
        task: str,
        screenshot: PIL.Image.Image,
    ) -> List[str]:
        """
        Decompose *task* into an ordered list of atomic sub-goals.

        The shared AssistantAgent (created in __init__) is reused across
        calls so its conversation history accumulates naturally.

        Parameters
        ----------
        task       : Natural-language description of the user's goal.
        screenshot : Current GUI state as a PIL image.

        Returns
        -------
        list[str]  Ordered sub-goals, capped at self._max_steps.

        Raises
        ------
        ParsingError  If the model response cannot be parsed at all.
        """
        log_agent_call(self._name, "planning", task=task)

        prompt  = f"Task: {task}\n\nAnalyse the screenshot and produce the sub-goals."
        message = _build_multimodal_message(prompt, screenshot)

        result = await self._agent.run(task=message)
        raw    = self.extract_last_text(result)

        try:
            sub_goals = self._parse_sub_goals(raw)
        except Exception as exc:
            raise ParsingError(f"Failed to parse sub-goals: {exc}") from exc

        logger.info(
            "[%s] plan() — %d sub-goals for task: %r",
            self._name, len(sub_goals), task,
        )
        return sub_goals[: self._max_steps]

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

        A *fresh* AssistantAgent is created for each replan call so it
        has no memory of previous (possibly mis-leading) exchanges, and
        is cleaned up immediately afterwards.

        Parameters
        ----------
        task        : Original task description.
        completed   : Sub-goals already executed successfully.
        failed_step : The sub-goal that just failed.
        error       : Human-readable error description from the Verifier.
        screenshot  : Current screenshot (state after the failed action).

        Returns
        -------
        list[str]  Revised remaining sub-goals.
        """
        log_agent_call(
            self._name,
            "replanning",
            task=task,
            num_completed=len(completed),
            failed_step=failed_step,
        )

        completed_str = (
            "\n".join(f"  ✓ {s}" for s in completed) or "  (none yet)"
        )
        prompt = (
            f"Original task: {task}\n\n"
            f"Completed steps:\n{completed_str}\n\n"
            f"Failed step: {failed_step}\n"
            f"Verifier error: {error}\n\n"
            "Produce a revised list of remaining sub-goals."
        )

        # Use a fresh, stateless agent so stale context does not mislead it.
        replan_agent: Optional[AssistantAgent] = None
        try:
            replan_agent = AssistantAgent(
                name=f"{self._name}_replan",
                model_client=self._model_client,
                system_message=self.load_system_prompt("replan"),
                reflect_on_tool_use=False,
            )

            message   = _build_multimodal_message(prompt, screenshot)
            result    = await replan_agent.run(task=message)
            raw       = self.extract_last_text(result)
            sub_goals = self._parse_sub_goals(raw)

        finally:
            # Release the ephemeral agent regardless of success / failure.
            if replan_agent is not None:
                await replan_agent.close()

        logger.info(
            "[%s] replan() — %d remaining sub-goals",
            self._name, len(sub_goals),
        )
        return sub_goals[: self._max_steps]

    # ------------------------------------------------------------------ #
    # Convenience methods                                                  #
    # ------------------------------------------------------------------ #

    async def plan_with_metadata(
        self,
        task: str,
        screenshot: PIL.Image.Image,
    ) -> List[SubGoal]:
        """
        Like plan(), but returns structured SubGoal objects with step numbers.
        """
        descriptions = await self.plan(task, screenshot)
        return [
            SubGoal(description=desc, step_number=i + 1, completed=False)
            for i, desc in enumerate(descriptions)
        ]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _parse_sub_goals(self, raw: str) -> List[str]:
        """
        Parse the model's JSON response into a clean list of sub-goal strings.

        Primary path  : extract the "sub_goals" array from JSON.
        Fallback path : strip list-item prefixes (numbers, bullets) line-by-line.
                        Used when the model ignores the JSON instruction.

        Raises
        ------
        ParsingError  If neither path yields any non-empty strings.
        """
        # --- primary: JSON ---
        try:
            data  = self.parse_json_output(raw)
            goals = data.get("sub_goals", [])
            clean = [str(g).strip() for g in goals if str(g).strip()]
            if clean:
                return clean
            logger.warning("[%s] JSON parsed but 'sub_goals' was empty", self._name)
        except ValueError:
            logger.warning(
                "[%s] JSON parse failed — attempting line-by-line fallback", self._name
            )

        # --- fallback: numbered / bulleted lines ---
        clean = []
        for line in raw.splitlines():
            line = re.sub(r"^\s*[\d\-\*\.\u2022]+\s*", "", line).strip()
            if len(line) > 3:
                clean.append(line)

        if not clean:
            raise ParsingError(
                "Could not extract sub-goals from model output. "
                f"Raw response was:\n{raw[:500]}"
            )

        return clean


# ------------------------------------------------------------------ #
# Module-level helper (shared with Grounder / Verifier if needed)    #
# ------------------------------------------------------------------ #

def _build_multimodal_message(text: str, image: PIL.Image.Image) -> MultiModalMessage:
    """
    Wrap a text prompt and a PIL image into an AutoGen MultiModalMessage.

    Keeping this as a module-level function (rather than a static method)
    makes it easy to import into Grounder.py and Verifier.py without
    inheriting PlannerAgent.
    """
    return MultiModalMessage(
        content=[text, AGImage(image)],
        source="user",
    )