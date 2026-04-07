"""
Simple test for Grounder Agent.

Tests visual grounding functionality using ScreenSpot dataset.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.Grounder import GrounderAgent
from models.glm_client import create_glm_client
from data.test_data_loader import load_test_sample


async def test_grounder():
    """Test the Grounder agent with a sample from ScreenSpot."""

    print("=" * 60)
    print("TEST: Grounder Agent")
    print("=" * 60)

    # Load test data
    print("\n1. Loading test data...")
    image, instruction, metadata = load_test_sample()

    sub_goal = instruction  # Use the instruction as a sub-goal
    print(f"   Sub-goal: {sub_goal}")
    print(f"   Image size: {image.size}")
    print(f"   Metadata: {metadata}")

    # Create grounder agent
    print("\n2. Creating Grounder agent...")
    client = create_glm_client()
    grounder = GrounderAgent(model_client=client)
    print("   [OK] Grounder agent created")

    # Test grounding
    print("\n3. Testing visual grounding...")
    try:
        action = await grounder.ground(sub_goal=sub_goal, screenshot=image)

        print(f"   [OK] Grounding completed successfully!")
        print(f"\n   Generated action:")
        print(f"      Action Type: {action.action_type.value}")
        if action.coordinate:
            print(f"      Coordinate: ({action.coordinate.x:.3f}, {action.coordinate.y:.3f})")
        if action.text:
            print(f"      Text: {action.text}")
        if action.direction:
            print(f"      Direction: {action.direction}")
        print(f"      Reasoning: {action.reasoning}")

        # Cleanup
        await grounder.close()

        print("\n" + "=" * 60)
        print("[OK] GROUNDER TEST PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n   [ERROR] Grounding failed: {e}")
        print("\n" + "=" * 60)
        print("[ERROR] GROUNDER TEST FAILED")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_grounder())
    sys.exit(0 if success else 1)
