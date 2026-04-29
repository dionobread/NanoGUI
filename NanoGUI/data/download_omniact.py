"""
OmniAct Dataset Download Pipeline

Downloads the rootsautomation/OmniAct dataset from Hugging Face
and organizes it for training GUI grounding models.

Dataset Description:
- OmniAct contains 6750+ GUI grounding samples
- Covers Web and Desktop interfaces
- Each sample includes: screenshot, instruction, and bounding box annotation

NOTE: OmniAct already has a validation set split called "val"; should 
we use that instead? It's only 991 datapoints compared to train's 6788...
"""

import json
from PIL import Image
from pathlib import Path
from typing import Dict, Any
from datasets import load_dataset, Dataset


class OmniActDownloader:
    """Downloads and processes the OmniAct dataset."""

    def __init__(self, save_dir: str = "./data/omniact"):
        """
        Initialize the downloader.

        Args:
            save_dir: Directory to save the dataset
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.save_dir / "images").mkdir(exist_ok=True)
        (self.save_dir / "annotations").mkdir(exist_ok=True)
        (self.save_dir / "processed").mkdir(exist_ok=True)

    def download_dataset(self) -> Dataset:
        """
        Download OmniAct dataset from Hugging Face.

        Returns:
            The downloaded dataset
        """
        print("Downloading OmniAct dataset from Hugging Face...")
        print("   Repository: Writer/OmniAct")

        try:
            dataset = load_dataset("Writer/OmniAct")
            print(f"Successfully downloaded dataset!")

            # Print dataset info
            print("\nDataset Information:")
            for split in dataset.keys():
                print(f"   {split}: {len(dataset[split])} samples")

            return dataset

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Try: pip install datasets")
            raise

    def analyze_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Analyze the dataset structure and content.

        Args:
            dataset: The dataset to analyze

        Returns:
            Statistics dictionary
        """
        print("\n🔍 Analyzing dataset structure...")

        stats = {}

        for split_name, split_data in dataset.items():
            print(f"\n   {split_name.upper()} SPLIT:")

            # Get basic info
            num_samples = len(split_data)
            stats[split_name] = {"num_samples": num_samples}

            # Check columns
            print(f"   Columns: {split_data.column_names}")

            # Show sample
            if num_samples > 0:
                sample = split_data[0]
                print(f"   Sample keys: {list(sample.keys())}")

                # Print first sample instruction
                if "task" in sample:
                    print(f"   Example instruction: \"{sample['task']}\"")

                # Check bounding box format
                if "box" in sample:
                    print(f"   Bounding box format: {sample['box']}")

                # Check image format
                if "image" in sample:
                    img = sample["image"]
                    if hasattr(img, "size"):
                        print(f"   Image size: {img.size}")

        return stats

    def save_dataset_locally(self, dataset: Dataset, save_images: bool = True):
        """
        Save dataset locally in organized format.

        Args:
            dataset: The dataset to save
            save_images: Whether to save individual images
        """
        print("\nSaving dataset locally...")

        for split_name, split_data in dataset.items():
            print(f"   Processing {split_name} split...")

            # Save metadata
            metadata = {
                "split": split_name,
                "num_samples": len(split_data),
                "columns": split_data.column_names
            }

            metadata_path = self.save_dir / "annotations" / f"{split_name}_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            # Save individual samples
            annotations = []

            for idx, sample in enumerate(split_data):
                # Create annotation entry
                annotation = {
                    "id": f"{split_name}_{idx}",
                    "instruction": sample.get("task", ""),
                    "data_type": sample.get("data_type", "unknown"),
                }

                # Add bounding box info
                if "box" in sample:
                    annotation["bbox"] = sample["box"]

                # Save image if requested
                if save_images and "image" in sample:
                    img = sample["image"]
                    img_filename = f"{split_name}_{idx:05d}.png"
                    img_path = self.save_dir / "images" / img_filename

                    # Convert to PIL Image if needed
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(img)

                    img.save(img_path)
                    annotation["image_path"] = str(img_path)

                annotations.append(annotation)

            # Save annotations as JSON
            annotations_path = self.save_dir / "annotations" / f"{split_name}_annotations.json"
            with open(annotations_path, "w", encoding="utf-8") as f:
                json.dump(annotations, f, indent=2)

            print(f"   Saved {len(annotations)} samples from {split_name} split")

        print(f"\nDataset saved to: {self.save_dir}")

    def create_train_val_split_omniact(self, dataset: Dataset, val_ratio: float = 0.1):
        """
        Create a custom train/validation split from the dataset.

        Args:
            dataset: The dataset to split
            val_ratio: Ratio of validation data
        """
        print("\n🔪 Creating custom train/validation split...")

        if "train" not in dataset:
            print("No 'train' split found in dataset")
            return

        train_data = dataset["train"]
        num_samples = len(train_data)
        num_val = int(num_samples * val_ratio)
        num_train = num_samples - num_val

        # Create splits
        split_dataset = train_data.train_test_split(test_size=val_ratio, seed=42)

        print(f"   Train samples: {num_train}")
        print(f"   Val samples: {num_val}")

        return split_dataset


def main():
    """Main execution function."""

    print("=" * 60)
    print("OmniAct Dataset Download Pipeline")
    print("=" * 60)

    # Initialize downloader
    downloader = OmniActDownloader(save_dir="./data/omniact")

    # Download dataset
    dataset = downloader.download_dataset()

    # Analyze dataset
    stats = downloader.analyze_dataset(dataset)

    # Save locally
    downloader.save_dataset_locally(dataset, save_images=True)

    print("\n" + "=" * 60)
    print("Download pipeline completed successfully!")
    print("=" * 60)
    print(f"\nDataset location: {downloader.save_dir.absolute()}")

    return dataset


if __name__ == "__main__":
    dataset = main()
