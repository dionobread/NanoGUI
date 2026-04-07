"""
Simple test for Verifier Agent.

Tests action verification functionality using ScreenSpot dataset.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.Verifier import VerifierAgent
from models.glm_client import create_glm_client
from data.test_data_loader import load_test_sample


async def test_verifier():
    """Test the Verifier agent with samples from ScreenSpot."""

    print("=" * 60)
    print("TEST: Verifier Agent")
    print("=" * 60)

    # Load test data
    print("\n1. Loading test data...")
    before_image, instruction, metadata = load_test_sample()

    # Load another sample for "after" state
    after_image, _, _ = load_test_sample()

    sub_goal = instruction
    action_description = "Simulated click action on target element"

    print(f"   Sub-goal: {sub_goal}")
    print(f"   Action: {action_description}")
    print(f"   Before image size: {before_image.size}")
    print(f"   After image size: {after_image.size}")
    print(f"   Metadata: {metadata}")

    # Create verifier agent
    print("\n2. Creating Verifier agent...")
    client = create_glm_client()
    verifier = VerifierAgent(model_client=client)
    print("   [OK] Verifier agent created")

    # Test verification
    print("\n3. Testing action verification...")
    try:
        verification = await verifier.verify(
            sub_goal=sub_goal,
            action_description=action_description,
            before=before_image,
            after=after_image
        )

        print(f"   [OK] Verification completed successfully!")
        print(f"\n   Verification result:")
        print(f"      Status: {verification.status.value}")
        print(f"      Reason: {verification.reason}")
        if verification.confidence is not None:
            print(f"      Confidence: {verification.confidence:.2f}")

        # Cleanup
        await verifier.close()

        print("\n" + "=" * 60)
        print("[OK] VERIFIER TEST PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n   [ERROR] Verification failed: {e}")
        print("\n" + "=" * 60)
        print("[ERROR] VERIFIER TEST FAILED")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_verifier())
    sys.exit(0 if success else 1)
