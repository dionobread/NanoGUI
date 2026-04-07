"""
NanoGUI Pipeline - Phase 1 Implementation

A simple, clean orchestrator for the multi-agent GUI automation framework.
Coordinates: Planner → Grounder → Verifier

Phase 1 Features:
- Agent coordination and state management
- ScreenSpot dataset integration
- Simulated action execution
- Basic error handling and retry logic
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import PIL.Image
import numpy as np

from agents.Planner import PlannerAgent
from agents.Grounder import GrounderAgent, GroundedAction
from agents.Verifier import VerifierAgent, VerificationResult
from models.glm_client import create_glm_client
from data.test_data_loader import load_test_sample, get_dataset_stats


# Configuration
@dataclass
class PipelineConfig:
    """Configuration for the NanoGUI pipeline."""
    max_steps: int = 10
    max_retries: int = 2
    step_delay: float = 0.3
    log_level: str = "INFO"


@dataclass
class StepResult:
    """Result of a single pipeline step."""
    step_number: int
    sub_goal: str
    action: Optional[GroundedAction]
    verification: Optional[VerificationResult]
    success: bool
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    task: str
    success: bool
    total_steps: int
    completed_steps: int
    steps: List[StepResult] = field(default_factory=list)
    execution_time: float = 0.0
    error: Optional[str] = None


class NanoGUIPipeline:
    """
    Main pipeline orchestrating Planner → Grounder → Verifier.
    """

    def __init__(self, config: PipelineConfig = None):
        """Initialize the pipeline with configuration."""
        self.config = config or PipelineConfig()
        self.model_client = None
        self.planner = None
        self.grounder = None
        self.verifier = None
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the pipeline."""
        log_level = getattr(logging, self.config.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        return logging.getLogger(__name__)

    async def initialize(self):
        """Initialize all agents."""
        self.logger.info("Initializing NanoGUI Pipeline...")
        self.model_client = create_glm_client()
        self.planner = PlannerAgent(model_client=self.model_client)
        self.grounder = GrounderAgent(model_client=self.model_client)
        self.verifier = VerifierAgent(model_client=self.model_client)
        self.logger.info("All agents initialized")

    async def execute_task(
        self,
        task: str,
        initial_screenshot: PIL.Image.Image,
    ) -> PipelineResult:
        """
        Execute a complete task using the agent pipeline.

        Args:
            task: Natural language task description
            initial_screenshot: Starting screenshot

        Returns:
            PipelineResult with execution details
        """
        start_time = datetime.now()
        steps = []
        current_screenshot = initial_screenshot

        self.logger.info(f"Executing task: {task[:100]}...")

        try:
            # Stage 1: Planning
            sub_goals = await self.planner.plan(task=task, screenshot=initial_screenshot)
            self.logger.info(f"Planner generated {len(sub_goals)} sub-goals")

            # Stage 2: Execute each sub-goal
            for i, sub_goal in enumerate(sub_goals[:self.config.max_steps], 1):
                step_result = await self._execute_step(
                    step_number=i,
                    sub_goal=sub_goal,
                    screenshot=current_screenshot
                )
                steps.append(step_result)

                if not step_result.success:
                    self.logger.error(f"Step {i} failed: {step_result.error}")
                    if i >= len(sub_goals) or step_result.step_number >= self.config.max_retries:
                        break
                    continue

                # Simulate action execution (update screenshot for next step)
                if step_result.action:
                    current_screenshot = self._simulate_action(
                        current_screenshot,
                        step_result.action
                    )

            # Stage 3: Final assessment
            success = all(s.success for s in steps)
            execution_time = (datetime.now() - start_time).total_seconds()

            return PipelineResult(
                task=task,
                success=success,
                total_steps=len(sub_goals),
                completed_steps=len([s for s in steps if s.success]),
                steps=steps,
                execution_time=execution_time
            )

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return PipelineResult(
                task=task,
                success=False,
                total_steps=0,
                completed_steps=0,
                steps=steps,
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e)
            )

    async def _execute_step(
        self,
        step_number: int,
        sub_goal: str,
        screenshot: PIL.Image.Image,
        retry_count: int = 0
    ) -> StepResult:
        """Execute a single step: Ground → Verify."""
        try:
            # Ground the action
            action = await self.grounder.ground(sub_goal=sub_goal, screenshot=screenshot)

            # Simulate execution
            after_screenshot = self._simulate_action(screenshot, action)

            # Verify the action
            verification = await self.verifier.verify(
                sub_goal=sub_goal,
                action_description=f"Execute {action.action_type.value}",
                before=screenshot,
                after=after_screenshot
            )

            success = verification.succeeded

            return StepResult(
                step_number=step_number,
                sub_goal=sub_goal,
                action=action,
                verification=verification,
                success=success
            )

        except Exception as e:
            return StepResult(
                step_number=step_number,
                sub_goal=sub_goal,
                action=None,
                verification=None,
                success=False,
                error=str(e)
            )

    def _simulate_action(
        self,
        screenshot: PIL.Image.Image,
        action: GroundedAction
    ) -> PIL.Image.Image:
        """
        Simulate action execution by adding visual changes.

        For Phase 1: Adds a red circle at click coordinates for visual feedback.
        """
        screenshot = screenshot.convert("RGB")
        img_array = np.array(screenshot)
        height, width = img_array.shape[:2]

        if action.coordinate and action.action_type.value == "click":
            # Convert normalized coordinates to pixels
            x = int(action.coordinate.x * width)
            y = int(action.coordinate.y * height)

            # Draw a red circle (simplified - just mark the area)
            radius = 20
            y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
            mask = x_grid**2 + y_grid**2 <= radius**2

            # Apply red circle
            y_start = max(0, y - radius)
            y_end = min(height, y + radius + 1)
            x_start = max(0, x - radius)
            x_end = min(width, x + radius + 1)

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx**2 + dy**2 <= radius**2:
                        py, px = y + dy, x + dx
                        if 0 <= py < height and 0 <= px < width:
                            img_array[py, px] = [255, 0, 0]  # Red

        return PIL.Image.fromarray(img_array)

    async def close(self):
        """Clean up resources."""
        if self.model_client:
            await self.model_client.close()
        self.logger.info("Pipeline closed")


# Main execution function
async def main():
    """Run Phase 1 pipeline test."""

    print("=" * 70)
    print("NanoGUI Pipeline - Phase 1")
    print("=" * 70)

    # Check dataset availability
    print("\nChecking dataset availability...")
    stats = get_dataset_stats()
    if stats['available']:
        print(f"[OK] Local ScreenSpot dataset found:")
        for split_name, split_info in stats['splits'].items():
            print(f"    {split_name}: {split_info['count']} samples")
    else:
        print("[INFO] No local dataset found, will use synthetic test data")

    # Load test data
    print("\nLoading test data...")
    screenshot, instruction, metadata = load_test_sample()
    task = f"Navigate to and interact with: {instruction}"

    print(f"Task: {task}")
    print(f"Screenshot size: {screenshot.size}")

    # Create and run pipeline
    config = PipelineConfig(
        max_steps=5,
        max_retries=1,
        step_delay=0.3,
        log_level="INFO"
    )

    pipeline = NanoGUIPipeline(config=config)
    await pipeline.initialize()

    print("\n" + "=" * 70)
    print("Starting pipeline execution...")
    print("=" * 70)

    result = await pipeline.execute_task(
        task=task,
        initial_screenshot=screenshot
    )

    # Print results
    print("\n" + "=" * 70)
    print("EXECUTION RESULTS")
    print("=" * 70)
    print(f"Task: {result.task[:80]}...")
    print(f"Success: {'[OK] YES' if result.success else '[ERROR] NO'}")
    print(f"Completed: {result.completed_steps}/{result.total_steps} steps")
    print(f"Execution time: {result.execution_time:.2f}s")

    if result.steps:
        print("\nStep-by-step breakdown:")
        for step in result.steps:
            status = "[OK]" if step.success else "[ERROR]"
            print(f"  {status} Step {step.step_number}: {step.sub_goal[:60]}...")
            if step.verification:
                print(f"      Verification: {step.verification.status.value}")
                if step.verification.reason:
                    print(f"      Reason: {step.verification.reason[:80]}...")

    await pipeline.close()

    print("\n" + "=" * 70)
    if result.success:
        print("[OK] PIPELINE COMPLETED SUCCESSFULLY")
    else:
        print("[ERROR] PIPELINE COMPLETED WITH ERRORS")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
