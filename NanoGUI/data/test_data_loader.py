"""
Unified Test Data Loader for NanoGUI Agents

Provides simple functions to load test data from local ScreenSpot dataset.
All test scripts should use this utility for consistent data loading.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple
import PIL.Image

logger = logging.getLogger(__name__)


def load_local_screenspot(
    data_dir: str = "./data/screenspot",
    split: str = "test",
    sample_idx: int = 0
) -> Tuple[Optional[PIL.Image.Image], Optional[str], Optional[dict]]:
    """
    Load a single sample from local ScreenSpot dataset.

    Args:
        data_dir: Path to the ScreenSpot data directory
        split: Dataset split to load from (test, train, validation)
        sample_idx: Index of the sample to load

    Returns:
        Tuple of (image, instruction, metadata) or (None, None, None) if not found
    """
    data_path = Path(data_dir)
    annotations_file = data_path / "annotations" / f"{split}_annotations.json"

    if not annotations_file.exists():
        logger.warning(f"Annotations file not found: {annotations_file}")
        logger.info("Run: python NanoGUI/datasets/download_screenspot.py")
        return None, None, None

    try:
        with open(annotations_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        if sample_idx >= len(annotations):
            logger.warning(f"Sample index {sample_idx} out of range (total: {len(annotations)})")
            sample_idx = 0

        sample = annotations[sample_idx]
        image_path = sample.get("image_path")

        if not image_path or not Path(image_path).exists():
            logger.warning(f"Image path not found or invalid: {image_path}")
            return None, None, None

        # Load image
        image = PIL.Image.open(image_path).convert("RGB")

        # Extract data
        instruction = sample.get("instruction", "")
        bbox = sample.get("bbox", [0, 0, 0, 0])
        data_type = sample.get("data_type", "unknown")

        metadata = {
            "bbox": bbox,
            "data_type": data_type,
            "image_path": image_path,
            "sample_id": sample.get("id", f"{split}_{sample_idx}")
        }

        logger.info(f"Loaded ScreenSpot sample {sample_idx} from {split} split")
        logger.info(f"  Instruction: {instruction[:60]}...")
        logger.info(f"  Image size: {image.size}")
        logger.info(f"  Data type: {data_type}")

        return image, instruction, metadata

    except Exception as e:
        logger.error(f"Error loading local ScreenSpot data: {e}")
        return None, None, None


def load_test_sample(
    fallback_to_synthetic: bool = True
) -> Tuple[PIL.Image.Image, str, dict]:
    """
    Load a test sample (from local ScreenSpot or fallback).

    This is the recommended function for test scripts.

    Args:
        fallback_to_synthetic: If True, create synthetic data when local data unavailable

    Returns:
        Tuple of (image, instruction, metadata)
    """
    # Try local ScreenSpot first
    image, instruction, metadata = load_local_screenspot()

    if image is not None:
        return image, instruction, metadata or {}

    # Fallback to synthetic data
    if fallback_to_synthetic:
        logger.warning("Using synthetic test data")
        import numpy as np

        # Create a simple test image
        arr = np.full((720, 1280, 3), 240, dtype=np.uint8)
        image = PIL.Image.fromarray(arr)
        instruction = "Click the search bar at the top of the page"

        metadata = {
            "synthetic": True,
            "image_size": (1280, 720),
            "instruction": instruction
        }

        logger.info(f"Created synthetic test image: {image.size}")
        return image, instruction, metadata

    raise RuntimeError("No test data available and fallback disabled")


def get_dataset_stats(data_dir: str = "./data/screenspot") -> dict:
    """
    Get statistics about the local ScreenSpot dataset.

    Args:
        data_dir: Path to the ScreenSpot data directory

    Returns:
        Dictionary with dataset statistics
    """
    data_path = Path(data_dir)
    stats = {
        "available": False,
        "splits": {}
    }

    annotations_dir = data_path / "annotations"
    if not annotations_dir.exists():
        return stats

    stats["available"] = True

    for split_file in annotations_dir.glob("*_annotations.json"):
        split_name = split_file.stem.replace("_annotations", "")

        try:
            with open(split_file, "r", encoding="utf-8") as f:
                annotations = json.load(f)

            stats["splits"][split_name] = {
                "count": len(annotations),
                "file": str(split_file)
            }

        except Exception as e:
            logger.warning(f"Error reading {split_file}: {e}")

    return stats


# Convenience functions for backward compatibility
def load_screenspot_sample():
    """Alias for load_local_screenspot with default parameters."""
    return load_local_screenspot()


if __name__ == "__main__":
    """Test the data loader."""
    import sys

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("Testing ScreenSpot data loader...")
    print("=" * 50)

    # Check dataset availability
    stats = get_dataset_stats()
    print(f"Dataset available: {stats['available']}")
    if stats['available']:
        print("Splits:")
        for split_name, split_info in stats['splits'].items():
            print(f"  {split_name}: {split_info['count']} samples")

    # Load a test sample
    image, instruction, metadata = load_test_sample()

    print(f"\nTest sample loaded:")
    print(f"  Image size: {image.size}")
    print(f"  Instruction: {instruction}")
    print(f"  Metadata: {metadata}")

    print("\nData loader test completed!")