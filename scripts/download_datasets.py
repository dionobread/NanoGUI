#!/usr/bin/env python3
"""
Download additional datasets for NanoGUI training.

Downloads:
- ScreenSpot-v2 (1272 samples, updated from ScreenSpot)
- SeeClick subset (500 images, ~8.5K elements)

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --seeclick-only
    python scripts/download_datasets.py --screenspot-v2-only
"""

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def get_project_root():
    return Path(__file__).resolve().parent.parent


def download_screenspot_v2(root: Path):
    """Download ScreenSpot-v2 with images."""
    out_dir = root / "datasets" / "screenspot_v2"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    anno_path = out_dir / "annotations.json"
    if anno_path.exists():
        print(f"ScreenSpot-v2 already exists at {anno_path}")
        return

    print("Downloading ScreenSpot-v2...")
    ds = load_dataset("HongxinLi/ScreenSpot_v2", split="test")

    annotations = []
    for i, item in enumerate(tqdm(ds, desc="Saving images")):
        img = item["image"]
        if img is None:
            continue

        img_filename = f"ssv2_{i:05d}.png"
        img_path = img_dir / img_filename
        img.save(img_path)

        annotations.append({
            "id": f"ssv2_{i}",
            "image_path": str(img_path),
            "bbox": item["bbox"],
            "instruction": item["instruction"],
            "data_type": item["data_type"],
            "data_source": item["data_source"],
        })

    with open(anno_path, "w") as f:
        json.dump(annotations, f, indent=2)

    from collections import Counter
    sources = Counter(a["data_source"] for a in annotations)
    print(f"ScreenSpot-v2: {len(annotations)} samples")
    print(f"  Sources: {dict(sources)}")


def download_seeclick(root: Path, num_images: int = 500):
    """Download SeeClick subset with images."""
    out_dir = root / "datasets" / "seeclick"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    anno_path = out_dir / "annotations.json"
    if anno_path.exists():
        print(f"SeeClick already exists at {anno_path}")
        return

    print(f"Downloading SeeClick ({num_images} images)...")
    ds = load_dataset("moondream/seeclick", split="train", streaming=True)

    annotations = []
    img_count = 0
    elem_count = 0

    for sample in tqdm(ds, desc="Downloading"):
        if img_count >= num_images:
            break

        img = sample["image"]
        if img is None:
            continue

        img_filename = f"seeclick_{img_count:05d}.png"
        img_path = img_dir / img_filename
        img.save(img_path)

        for elem in sample.get("elements", []):
            bbox = elem.get("bbox")
            instruction = elem.get("instruction", "")
            data_type = elem.get("data_type", "unknown")

            if not bbox or not instruction:
                continue

            annotations.append({
                "image_path": str(img_path),
                "bbox": bbox,
                "instruction": instruction,
                "data_type": data_type,
                "source": "seeclick",
            })
            elem_count += 1

        img_count += 1

    with open(anno_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"SeeClick: {img_count} images, {elem_count} elements")


def create_screenspot_splits(root: Path):
    """Create train/val/test splits from ScreenSpot."""
    anno_path = root / "datasets" / "screenspot" / "annotations" / "test_annotations.json"
    out_dir = root / "datasets" / "screenspot" / "annotations"

    train_path = out_dir / "train_annotations.json"
    if train_path.exists():
        print("ScreenSpot splits already exist")
        return

    if not anno_path.exists():
        print(f"ScreenSpot test annotations not found: {anno_path}")
        return

    with open(anno_path) as f:
        data = json.load(f)

    random.seed(42)
    random.shuffle(data)

    n = len(data)
    n_test = int(0.1 * n)
    n_val = int(0.1 * n)
    n_train = n - n_test - n_val

    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]

    for name, split_data in [("train", train), ("val", val), ("test_split", test)]:
        path = out_dir / f"{name}_annotations.json"
        with open(path, "w") as f:
            json.dump(split_data, f, indent=2)

    print(f"ScreenSpot splits: train={len(train)}, val={len(val)}, test={len(test)}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--seeclick-only", action="store_true")
    parser.add_argument("--screenspot-v2-only", action="store_true")
    parser.add_argument("--seeclick-images", type=int, default=500)
    args = parser.parse_args()

    root = get_project_root()

    if not args.seeclick_only:
        download_screenspot_v2(root)
        create_screenspot_splits(root)

    if not args.screenspot_v2_only:
        download_seeclick(root, args.seeclick_images)

    print("\nDone! Datasets saved to datasets/")


if __name__ == "__main__":
    main()
