"""
Simple test for Planner Agent.

Tests basic task planning functionality using ScreenSpot dataset.
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.Planner import PlannerAgent
from models.glm_client import create_glm_client
from data.test_data_loader import load_test_sample


async def test_planner():
    """Test the Planner agent with a sample from ScreenSpot."""

    print("=" * 60)
    print("TEST: Planner Agent")
    print("=" * 60)

    # Load test data
    print("\n1. Loading test data...")
    image, instruction, metadata = load_test_sample()

    task = f"Navigate to and interact with: {instruction}"
    print(f"   Task: {task}")
    print(f"   Image size: {image.size}")
    print(f"   Metadata: {metadata}")

    # Create planner agent
    print("\n2. Creating Planner agent...")
    client = create_glm_client()
    planner = PlannerAgent(model_client=client)
    print("   [OK] Planner agent created")

    # Test planning
    print("\n3. Testing task planning...")
    try:
        sub_goals = await planner.plan(task=task, screenshot=image)

        print(f"   [OK] Planning completed successfully!")
        print(f"\n   Generated {len(sub_goals)} sub-goals:")
        for i, sub_goal in enumerate(sub_goals, 1):
            print(f"      {i}. {sub_goal}")

        # Cleanup
        await planner.close()

        print("\n" + "=" * 60)
        print("[OK] PLANNER TEST PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n   [ERROR] Planning failed: {e}")
        print("\n" + "=" * 60)
        print("[ERROR] PLANNER TEST FAILED")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_planner())
    sys.exit(0 if success else 1)
