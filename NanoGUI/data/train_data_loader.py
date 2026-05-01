"""
Unified Training Data Loader for NanoGUI Agents

Provides functions to load training data from the local OmniAct dataset.
Primarily used by the Grounder training script.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict
import PIL.Image
from datasets import Dataset

logger = logging.getLogger(__name__)


def load_local_omniact(
    data_dir: str = "./datasets/omniact",
    split: str = "train",
    load_images: bool = True,
) -> Dict[str, Dataset]:
    """
    Load OmniAct splits as HuggingFace Datasets (for training).

    Args:
        data_dir: Path to the OmniAct data directory.
        split: Single split name to load (e.g. "train", "validation").
               If None, loads all splits found in the annotations directory.
        load_images: Whether to attach PIL Images.

    Returns:
        dict mapping split name → Dataset
    """
    data_dir = Path(data_dir)
    annotations_dir = data_dir / "annotations"

    if not annotations_dir.exists():
        logger.warning("Annotations directory not found: %s", annotations_dir)
        logger.info("Run: python -m NanoGUI.data.download_all_datasets omniact --no-images")
        return {}

    # Determine which splits to load
    if split is not None:
        splits = [split]
    else:
        splits = [
            p.stem.replace("_annotations", "")
            for p in annotations_dir.glob("*_annotations.json")
        ]

    result = {}
    for split_name in splits:
        annotations_path = annotations_dir / f"{split_name}_annotations.json"
        if not annotations_path.exists():
            logger.warning("Annotations file not found: %s", annotations_path)
            continue

        with open(annotations_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        # Optionally attach PIL Images
        if load_images:
            for sample in annotations:
                if "image_path" in sample and Path(sample["image_path"]).exists():
                    sample["image"] = PIL.Image.open(sample["image_path"]).convert("RGB").copy()

        result[split_name] = Dataset.from_list(annotations)

    logger.info("Loaded OmniAct splits: %s", list(result.keys()))
    return result


# Convenience alias
load_omniact_sample = load_local_omniact
