"""
End-to-end pipeline test using real NanoGUI agents with local models.

Uses the actual agent classes from NanoGUI.agents:
  - PlannerAgent   (task decomposition)
  - GrounderAgent  (visual grounding)
  - VerifierAgent  (pixel + semantic verification)

Fits in 8GB VRAM by loading one model at a time via LocalVLMClient.

Usage:
    python scripts/test_pipeline_e2e.py --task "Open Chrome and search for cats"
    python scripts/test_pipeline_e2e.py --model-planner Qwen2.5-VL-3B-Instruct \
                                         --model-grounder GUI-Actor-3B-Qwen2.5-VL
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import PIL.Image

# Add NanoGUI to path
sys.path.insert(0, str(Path(__file__).parent.parent / "NanoGUI"))

from agents.Planner import PlannerAgent
from agents.Grounder import GrounderAgent
from agents.Verifier import VerifierAgent
from agents.base import GroundedAction, ActionType, VerificationStatus
from models.local_vlm_client import LocalVLMClient
from core.screen_capture import capture_screen, get_screen_size
from core.action_executor import ActionExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def discover_models() -> List[str]:
    models_dir = get_project_root() / "models"
    if not models_dir.exists():
        return []
    return sorted([d.name for d in models_dir.iterdir()
                   if d.is_dir() and (d / "config.json").exists()])


@dataclass
class StepResult:
    step_num: int
    subtask: str
    action: GroundedAction
    pixel_diff: float
    verification: VerificationStatus
    reason: str = ""


async def run_pipeline(
    task: str,
    model_planner: str,
    model_grounder: str,
    max_steps: int = 10,
    pixel_threshold: float = 5.0,
    dry_run: bool = True,
) -> List[StepResult]:
    """Run the full pipeline using real agents."""

    results: List[StepResult] = []
    executor = ActionExecutor(action_delay=0.5, dry_run=dry_run)
    screen_w, screen_h = get_screen_size()

    # ── Phase 1: Planning ──
    logger.info("=" * 60)
    logger.info("PHASE 1: PLANNING")
    logger.info("=" * 60)

    screenshot = capture_screen()
    logger.info("Screenshot: %dx%d", screenshot.width, screenshot.height)

    planner_client = LocalVLMClient(model_planner)
    planner = PlannerAgent(model_client=planner_client, max_steps=max_steps)

    try:
        subtasks = await planner.plan(task, screenshot)
        logger.info("Plan (%d steps):", len(subtasks))
        for i, st in enumerate(subtasks, 1):
            logger.info("  %d. %s", i, st)
    finally:
        await planner_client.close()

    if not subtasks:
        logger.error("Planner returned no subtasks.")
        return results

    # ── Phase 2: Execution ──
    logger.info("=" * 60)
    logger.info("PHASE 2: EXECUTION")
    logger.info("=" * 60)

    grounder_client = LocalVLMClient(model_grounder)
    grounder = GrounderAgent(model_client=grounder_client)

    # Verifier uses the same client (reuses loaded model) or skip to save VRAM
    # For 8GB, skip VLM verifier and use pixel-based only
    use_vlm_verifier = False
    verifier = None
    verifier_client = None

    try:
        current_screenshot = screenshot

        for i, subtask in enumerate(subtasks, 1):
            logger.info("-" * 40)
            logger.info("Step %d/%d: %s", i, len(subtasks), subtask)

            # Ground the action
            action = await grounder.ground(subtask, current_screenshot)
            logger.info("Grounded: %s", action)

            # Execute
            if action.action_type == ActionType.CLICK and action.coordinate:
                px, py = action.coordinate.to_pixels(screen_w, screen_h)
                await executor.click(px, py)
            elif action.action_type == ActionType.TYPE and action.text:
                await executor.type_text(action.text)
            elif action.action_type == ActionType.KEY and action.text:
                await executor.press_key(action.text)
            elif action.action_type == ActionType.SCROLL and action.direction:
                await executor.scroll(action.direction)
            elif action.action_type == ActionType.WAIT:
                await asyncio.sleep(1.0)

            # Capture after screenshot
            after_screenshot = capture_screen()

            # Verify
            if dry_run:
                # No real action taken — pixel diff is meaningless
                status = VerificationStatus.UNCERTAIN
                reason = "dry_run: no real action taken"
                diff = 0.0
            else:
                if use_vlm_verifier and verifier:
                    vresult = await verifier.verify(
                        sub_goal=subtask,
                        action_description=str(action),
                        before=current_screenshot,
                        after=after_screenshot,
                    )
                    status = vresult.status
                    reason = vresult.reason
                else:
                    # Fast pixel-based check
                    diff = _pixel_mae(current_screenshot, after_screenshot)
                    if diff < pixel_threshold:
                        status = VerificationStatus.FAILURE
                        reason = f"No screen change (MAE={diff:.1f})"
                    else:
                        status = VerificationStatus.SUCCESS
                        reason = f"Screen changed (MAE={diff:.1f})"

                logger.info("Verify: %s — %s", status.value, reason)

            results.append(StepResult(
                step_num=i,
                subtask=subtask,
                action=action,
                pixel_diff=diff if not dry_run else 0.0,
                verification=status,
                reason=reason,
            ))

            current_screenshot = after_screenshot
            await asyncio.sleep(0.3)

    finally:
        await grounder_client.close()
        if verifier_client:
            await verifier_client.close()

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    return results


def _pixel_mae(img1: PIL.Image.Image, img2: PIL.Image.Image) -> float:
    """Mean absolute pixel difference (0-255 scale)."""
    import numpy as np
    a = np.array(img1.resize((320, 180)).convert("RGB"), dtype=float)
    b = np.array(img2.resize((320, 180)).convert("RGB"), dtype=float)
    return float(np.mean(np.abs(a - b)))


def print_results(results: List[StepResult], task: str, dry_run: bool):
    print("\n" + "=" * 70)
    print("PIPELINE RESULTS")
    print("=" * 70)
    print(f"Task: {task}")
    print(f"Steps: {len(results)}")
    if dry_run:
        print("Note: Actions were SIMULATED. Verification is N/A.")
    print()

    for r in results:
        status = "[ ? ]" if r.verification == VerificationStatus.UNCERTAIN else \
                 "[OK]" if r.verification == VerificationStatus.SUCCESS else "[FAIL]"
        coord = r.action.coordinate
        coord_str = f"({coord.x:.3f}, {coord.y:.3f})" if coord else "N/A"
        print(f"  {status} Step {r.step_num}: {r.subtask[:50]}")
        print(f"       Action: {r.action.action_type.value} @ {coord_str}")
        if r.action.text:
            print(f"       Text: {r.action.text}")
        if not dry_run:
            print(f"       MAE: {r.pixel_diff:.1f} | {r.reason}")
        print()

    print("=" * 70)
    if dry_run:
        print(f"Executed {len(results)} steps in dry_run mode.")
        print("Use --real-actions for real GUI interactions (requires human supervision).")
    else:
        ok = sum(1 for r in results if r.verification == VerificationStatus.SUCCESS)
        print(f"Success: {ok}/{len(results)} steps passed")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline using real NanoGUI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_pipeline_e2e.py
  python scripts/test_pipeline_e2e.py --task "Open Chrome"
  python scripts/test_pipeline_e2e.py --model-grounder GUI-Actor-2B-Qwen2-VL
        """,
    )
    parser.add_argument("--task", default="Open the Start menu and launch Notepad")
    parser.add_argument("--model-planner", default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--model-grounder", default="GUI-Actor-3B-Qwen2.5-VL")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--pixel-threshold", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--real-actions", action="store_true")

    args = parser.parse_args()

    available = discover_models()
    for name, label in [(args.model_planner, "Planner"), (args.model_grounder, "Grounder")]:
        if name not in available:
            print(f"ERROR: {label} model '{name}' not found.")
            print(f"Available: {available}")
            raise SystemExit(1)

    dry_run = not args.real_actions

    if not dry_run:
        print("\n" + "!" * 60)
        print("WARNING: REAL ACTIONS ENABLED")
        print("!" * 60)
        if input("Type 'yes' to continue: ").lower() != "yes":
            raise SystemExit(0)

    print("\n" + "=" * 70)
    print("NanoGUI Pipeline — Real Agents")
    print("=" * 70)
    print(f"Task:      {args.task}")
    print(f"Planner:   {args.model_planner}")
    print(f"Grounder:  {args.model_grounder}")
    print(f"Actions:   {'SIMULATED' if dry_run else 'REAL'}")
    print("=" * 70 + "\n")

    results = asyncio.run(run_pipeline(
        task=args.task,
        model_planner=args.model_planner,
        model_grounder=args.model_grounder,
        max_steps=args.steps,
        pixel_threshold=args.pixel_threshold,
        dry_run=dry_run,
    ))

    print_results(results, args.task, dry_run)


if __name__ == "__main__":
    main()
