"""
Run all agent tests sequentially.

Simple test runner that executes all agent tests in order.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_planner import test_planner
from tests.test_grounder import test_grounder
from tests.test_verifier import test_verifier


async def run_all_tests():
    """Run all agent tests sequentially."""

    print("\n" + "=" * 70)
    print("NanoGUI - Running All Agent Tests")
    print("=" * 70)

    results = {}

    # Test Planner
    print("\n")
    results['planner'] = await test_planner()

    # Small delay between tests
    await asyncio.sleep(1)

    # Test Grounder
    print("\n")
    results['grounder'] = await test_grounder()

    # Small delay between tests
    await asyncio.sleep(1)

    # Test Verifier
    print("\n")
    results['verifier'] = await test_verifier()

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for agent, passed in results.items():
        status = "[OK] PASSED" if passed else "[ERROR] FAILED"
        print(f"  {agent.capitalize():15s} : {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("[OK] ALL TESTS PASSED")
    else:
        print("[ERROR] SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
