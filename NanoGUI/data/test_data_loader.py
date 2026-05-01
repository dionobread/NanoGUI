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

# Project root (NanoGUI repo root, one level up from this package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default data directory — annotations reference 'data\screenspot\...' but
# the actual location is 'datasets/screenspot/...'
SCREENSPOT_DIR = str(PROJECT_ROOT / "datasets" / "screenspot")


def _resolve_image_path(image_path: str) -> Optional[Path]:
    """Resolve an image path from annotation, handling path mismatches."""
    p = Path(image_path)

    # 1. Try as-is
    if p.exists():
        return p

    # 2. Try relative to project root
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return candidate

    # 3. ScreenSpot annotations use 'data\screenspot\images\...' but
    #    repo uses 'datasets/screenspot/images/...' — rewrite
    parts = p.parts
    if "data" in parts:
        idx = parts.index("data")
        new_parts = parts[:idx] + ("datasets",) + parts[idx + 1:]
        candidate = PROJECT_ROOT.joinpath(*new_parts)
        if candidate.exists():
            return candidate

    return None


def load_local_screenspot(
    data_dir: str = SCREENSPOT_DIR,
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
        return None, None, None

    try:
        with open(annotations_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        if sample_idx >= len(annotations):
            logger.warning(f"Sample index {sample_idx} out of range (total: {len(annotations)})")
            sample_idx = 0

        sample = annotations[sample_idx]
        raw_image_path = sample.get("image_path", "")

        # Resolve image path (handles data/ → datasets/ rewrite)
        resolved = _resolve_image_path(raw_image_path)
        if resolved is None:
            logger.warning(f"Image not found: {raw_image_path}")
            return None, None, None

        image = PIL.Image.open(resolved).convert("RGB")
        instruction = sample.get("instruction", "")
        bbox = sample.get("bbox", [0, 0, 0, 0])
        data_type = sample.get("data_type", "unknown")

        metadata = {
            "bbox": bbox,
            "data_type": data_type,
            "image_path": str(resolved),
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
    image, instruction, metadata = load_local_screenspot()

    if image is not None:
        return image, instruction, metadata or {}

    if fallback_to_synthetic:
        logger.warning("Using synthetic test data")
        import numpy as np

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


def get_dataset_stats(data_dir: str = SCREENSPOT_DIR) -> dict:
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


def load_screenspot_sample():
    """Alias for load_local_screenspot with default parameters."""
    return load_local_screenspot()
