"""
VerifierAgent — closed-loop action success verification.

After the Grounder executes an action, the Verifier receives:
  - The before-screenshot (state before the action)
  - The after-screenshot  (state after the action)
  - The sub-goal that was attempted
  - The action that was taken

It decides whether the action SUCCEEDED, FAILED, or produced an UNCERTAIN
result, and provides a human-readable reason.  On failure, the pipeline hands
the reason to the Planner for replanning.

Two-stage verification
──────────────────────
Stage 1 — Fast pixel-similarity check (no model call):
  If the before and after screenshots are >95 % similar (measured via mean
  absolute pixel difference), the screen has not changed — the action almost
  certainly had no effect.  This stage runs in milliseconds and avoids an
  unnecessary model call for obvious no-ops.

Stage 2 — VLM semantic check:
  If the screens differ, the VLM examines both images together with the
  sub-goal and action, and reasons about whether the change represents a
  successful execution.

Output format (enforced via system prompt)
─────────────────────────────────────────
{
  "status": "success" | "failure" | "uncertain",
  "reason": "..."
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
    VerificationResult,
    VerificationStatus,
    ParsingError,
    log_agent_call,
)

logger = logging.getLogger(__name__)

class VerifierAgent(BaseAgent):
    """
    Closed-loop action verifier.

    Inherits from BaseAgent for common functionality including:
    - AutoGen AssistantAgent management
    - JSON parsing
    - Logging and error handling
    - Resource cleanup

    Two-stage verification process:
    1. Fast pixel-similarity check (no model call)
    2. VLM semantic check if screens differ

    Parameters
    ----------
    model_client : AutoGen-compatible ChatCompletionClient
    pixel_similarity_threshold : If before/after MAE is below this value
                                 (0–255 scale), treat as no-change failure
                                 without calling the model. Default 5.0.
    """

    def __init__(
        self,
        model_client: ChatCompletionClient,
        pixel_similarity_threshold: float = 5.0,
    ) -> None:
        """
        Initialize the Verifier agent.

        Parameters
        ----------
        model_client : ChatCompletionClient - AutoGen-compatible model client
        pixel_similarity_threshold : float - MAE threshold for fast-fail (default 5.0)
        """
        self._threshold = pixel_similarity_threshold

        # Initialize BaseAgent with verifier-specific configuration
        super().__init__(
            name="verifier",
            model_client=model_client,
            reflect_on_tool_use=False,
        )

    async def execute(self, *args, **kwargs) -> ActionResult:
        """
        Execute method required by BaseAgent.

        Routes to verify() method with the provided parameters.
        For direct use, prefer the verify() method.
        """
        sub_goal = kwargs.get("sub_goal") or (args[0] if args else None)
        action_description = kwargs.get("action_description") or (args[1] if len(args) > 1 else None)
        before = kwargs.get("before") or (args[2] if len(args) > 2 else None)
        after = kwargs.get("after") or (args[3] if len(args) > 3 else None)

        if not all([sub_goal, action_description, before, after]):
            return ActionResult(
                success=False,
                content="",
                error="Missing required parameters: sub_goal, action_description, before, after"
            )

        result = await self.verify(sub_goal, action_description, before, after)
        return ActionResult(
            success=result.succeeded,
            content=result.reason,
            metadata={
                "status": result.status.value,
                "confidence": result.confidence
            }
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def verify(
        self,
        sub_goal: str,
        action_description: str,
        before: PIL.Image.Image,
        after: PIL.Image.Image,
    ) -> VerificationResult:
        """
        Verify whether an action succeeded.

        Parameters
        ----------
        sub_goal           : The sub-goal that was attempted
        action_description : Human-readable description of what the Grounder did
                             (e.g. "Clicked at (0.45, 0.12)")
        before             : Screenshot captured BEFORE the action
        after              : Screenshot captured AFTER the action

        Returns
        -------
        VerificationResult with status and reason
        """
        log_agent_call(
            self._name,
            "verifying",
            sub_goal=sub_goal,
            action_description=action_description
        )

        # ── Stage 1: fast pixel-similarity check ──────────────────────────────
        mae = self._mean_absolute_error(before, after)
        logger.debug(
            "[%s] Before/after MAE=%.2f (threshold=%.2f)",
            self._name, mae, self._threshold
        )

        if mae < self._threshold:
            reason = (
                f"Screen did not change after action (MAE={mae:.2f} < {self._threshold}). "
                "The action had no visible effect."
            )
            logger.info("[%s] Stage-1 FAILURE (no screen change): %s", self._name, reason)
            return VerificationResult(status=VerificationStatus.FAILURE, reason=reason)

        # ── Stage 2: VLM semantic check ───────────────────────────────────────
        prompt = (
            f"Sub-goal: {sub_goal}\n"
            f"Action taken: {action_description}\n\n"
            "Image 1 is BEFORE the action. Image 2 is AFTER the action.\n"
            "Did the action succeed? Output JSON only."
        )
        message = MultiModalMessage(
            content=[prompt, AGImage(before), AGImage(after)],
            source="user",
        )

        result = await self._agent.run(task=message)
        raw = self.extract_last_text(result)

        try:
            verification = self._parse_result(raw)
            logger.info("[%s] %s", self._name, verification)
            return verification
        except Exception as exc:
            logger.error("[%s] Failed to parse verification: %s", self._name, exc)
            return VerificationResult(
                status=VerificationStatus.UNCERTAIN,
                reason=f"Could not parse verifier response: {exc}"
            )

    # ── Helper methods ─────────────────────────────────────────────────────────

    @staticmethod
    def _mean_absolute_error(img_a: PIL.Image.Image, img_b: PIL.Image.Image) -> float:
        """
        Compute mean absolute pixel difference between two images.

        Images are resized to a common size before comparison for speed.

        Parameters
        ----------
        img_a, img_b : PIL.Image - Images to compare

        Returns
        -------
        float - MAE in [0, 255] range
        """
        try:
            import numpy as np
            size = (320, 180)  # downsample for speed
            a = np.array(img_a.resize(size).convert("RGB"), dtype=float)
            b = np.array(img_b.resize(size).convert("RGB"), dtype=float)
            return float(np.mean(np.abs(a - b)))
        except Exception as exc:
            logger.warning("[%s] MAE computation failed: %s", exc)
            return 255.0  # assume screens differ if we can't compare

    def _parse_result(self, raw: str) -> VerificationResult:
        """
        Parse model output into a VerificationResult.

        Uses BaseAgent.parse_json_output() for robust parsing.

        Parameters
        ----------
        raw : str - Raw model output

        Returns
        -------
        VerificationResult

        Raises
        ------
        ParsingError - If JSON is invalid
        """
        try:
            data = self.parse_json_output(raw)

            status_str = data.get("status", "uncertain").lower()
            try:
                status = VerificationStatus(status_str)
            except ValueError:
                logger.warning("[%s] Unknown status '%s', defaulting to uncertain", self._name, status_str)
                status = VerificationStatus.UNCERTAIN

            return VerificationResult(
                status=status,
                reason=data.get("reason", ""),
                confidence=data.get("confidence")  # Optional confidence score
            )
        except ValueError as exc:
            raise ParsingError(
                f"Failed to parse verification result: {exc}. Raw: {raw[:200]}"
            ) from exc

    # ── Convenience methods ─────────────────────────────────────────────────────

    async def verify_and_retry(
        self,
        sub_goal: str,
        action_description: str,
        before: PIL.Image.Image,
        after: PIL.Image.Image,
        max_retries: int = 3,
        grounder=None,  # GrounderAgent instance
        screenshot=None,
    ) -> tuple[bool, VerificationResult]:
        """
        Verify and optionally trigger a retry with the Grounder.

        Convenience method that combines verification with retry logic.

        Parameters
        ----------
        sub_goal : str - The sub-goal being verified
        action_description : str - Description of action taken
        before : PIL.Image - Before screenshot
        after : PIL.Image - After screenshot
        max_retries : int - Maximum number of retry attempts
        grounder : GrounderAgent - Optional grounder for retry
        screenshot : PIL.Image - Current screenshot for retry

        Returns
        -------
        tuple[bool, VerificationResult] - (success, final_verification_result)
        """
        result = await self.verify(sub_goal, action_description, before, after)

        # If succeeded or uncertain, return as-is
        if result.succeeded or result.uncertain:
            return True, result

        # If failed and we have a grounder, retry
        if result.failed and grounder and screenshot and max_retries > 0:
            logger.info("[%s] Action failed, retrying... (%d attempts left)", self._name, max_retries)

            # Retry with feedback
            new_action = await grounder.ground(
                f"{sub_goal} (Previous attempt failed: {result.reason})",
                screenshot
            )

            # Note: Caller should execute the new_action and call verify again
            # This is a simplified version - full implementation would loop
            return False, result

        return result.succeeded, result

