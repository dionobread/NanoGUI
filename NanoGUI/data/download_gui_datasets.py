"""
Download GUI-agent datasets for NanoGUI.

This script keeps all downloads explicit and local. Small/medium Hugging Face
datasets can be materialized into NanoGUI's simple local format:

    data/<dataset_name>/
      images/
      annotations/<split>_annotations.json
      annotations/<split>_metadata.json

Large raw repositories such as OS-Atlas are downloaded with file patterns so you
can start with JSON annotations before pulling image archives.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    hf_id: str
    output_dir: str
    instruction_fields: tuple[str, ...] = ("instruction", "task", "intent", "confirmed_task")
    bbox_fields: tuple[str, ...] = ("bbox", "box", "target_bbox")
    image_fields: tuple[str, ...] = ("image", "screenshot")
    default_split: str | None = None
    notes: str = ""
    load_kwargs: dict[str, Any] = field(default_factory=dict)


DATASETS: dict[str, DatasetSpec] = {
    "screenspot": DatasetSpec(
        key="screenspot",
        hf_id="rootsautomation/ScreenSpot",
        output_dir="screenspot",
        default_split="test",
        notes="Grounder benchmark. Keep as evaluation data when possible.",
    ),
    "screenspot_pro": DatasetSpec(
        key="screenspot_pro",
        hf_id="Voxel51/ScreenSpot-Pro",
        output_dir="screenspot_pro",
        default_split="train",
        notes="Hard high-resolution GUI grounding benchmark.",
    ),
    "omniact": DatasetSpec(
        key="omniact",
        hf_id="Writer/OmniAct",
        output_dir="omniact",
        instruction_fields=("task", "instruction"),
        bbox_fields=("box", "bbox"),
        notes="Grounding/action dataset for desktop and web UIs.",
    ),
    "salesforce_grounding": DatasetSpec(
        key="salesforce_grounding",
        hf_id="Salesforce/grounding_dataset",
        output_dir="salesforce_grounding",
        instruction_fields=("instruction", "description"),
        bbox_fields=("bbox",),
        notes="Combined GUI grounding data from several sources.",
    ),
    "mind2web": DatasetSpec(
        key="mind2web",
        hf_id="osunlp/Mind2Web",
        output_dir="mind2web",
        instruction_fields=("confirmed_task", "task", "annotation_id"),
        notes="Planner/trajectory data for web agents.",
    ),
    "multimodal_mind2web": DatasetSpec(
        key="multimodal_mind2web",
        hf_id="osunlp/Multimodal-Mind2Web",
        output_dir="multimodal_mind2web",
        instruction_fields=("confirmed_task", "task", "annotation_id"),
        notes="Mind2Web paired with screenshots where available.",
    ),
    "online_mind2web": DatasetSpec(
        key="online_mind2web",
        hf_id="osunlp/Online-Mind2Web",
        output_dir="online_mind2web",
        instruction_fields=("confirmed_task", "task", "annotation_id"),
        notes="Online Mind2Web benchmark. May require accepting HF terms.",
    ),
    "aitw_single": DatasetSpec(
        key="aitw_single",
        hf_id="cjfcsjt/AITW_Single",
        output_dir="aitw_single",
        instruction_fields=("goal", "instruction", "task"),
        notes="Convenient HF mirror/subset of Android in the Wild style data.",
    ),
}


OS_ATLAS_PATTERNS = {
    "annotations": [
        "*.json",
        "**/*.json",
        "prompts.json",
        "README.md",
    ],
    "mobile_annotations": [
        "mobile_domain/*.json",
        "prompts.json",
        "README.md",
    ],
    "desktop_annotations": [
        "desktop_domain/*.json",
        "prompts.json",
        "README.md",
    ],
    "web_annotations": [
        "web_domain/*.json",
        "prompts.json",
        "README.md",
    ],
}


def _first_present(sample: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field_name in fields:
        if field_name in sample and sample[field_name] is not None:
            return sample[field_name]
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _as_pil_image(value: Any) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "path" in value:
        return Image.open(value["path"]).convert("RGB")
    try:
        return Image.fromarray(value).convert("RGB")
    except Exception:
        return None


def _select_splits(dataset: Any, requested: list[str] | None) -> list[str]:
    if requested:
        missing = [split for split in requested if split not in dataset]
        if missing:
            raise ValueError(f"Missing split(s) {missing}; available: {list(dataset.keys())}")
        return requested
    return list(dataset.keys())


def save_hf_dataset(
    spec: DatasetSpec,
    root_dir: Path,
    splits: list[str] | None,
    max_samples: int | None,
    save_images: bool,
    streaming: bool,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the Hugging Face datasets package first: pip install datasets") from exc

    out_dir = root_dir / spec.output_dir
    images_dir = out_dir / "images"
    annotations_dir = out_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {spec.key}: {spec.hf_id}")
    if spec.notes:
        print(f"Note: {spec.notes}")

    dataset = load_dataset(spec.hf_id, streaming=streaming, **spec.load_kwargs)
    selected_splits = _select_splits(dataset, splits)

    for split_name in selected_splits:
        split_data = dataset[split_name]
        annotations: list[dict[str, Any]] = []
        columns = getattr(split_data, "column_names", None)
        print(f"Processing split '{split_name}'")

        for idx, sample in enumerate(split_data):
            if max_samples is not None and idx >= max_samples:
                break

            sample_dict = dict(sample)
            instruction = _first_present(sample_dict, spec.instruction_fields)
            bbox = _first_present(sample_dict, spec.bbox_fields)
            image_value = _first_present(sample_dict, spec.image_fields)

            annotation: dict[str, Any] = {
                "id": f"{split_name}_{idx}",
                "dataset": spec.key,
                "source": spec.hf_id,
                "split": split_name,
                "instruction": instruction or "",
            }
            if bbox is not None:
                annotation["bbox"] = _json_safe(bbox)

            for key, value in sample_dict.items():
                if key in set(spec.image_fields):
                    continue
                if key not in annotation and key not in {"instruction", "task", "bbox", "box"}:
                    annotation[key] = _json_safe(value)

            if save_images and image_value is not None:
                image = _as_pil_image(image_value)
                if image is not None:
                    image_filename = f"{split_name}_{idx:06d}.png"
                    image_path = images_dir / image_filename
                    image.save(image_path)
                    annotation["image_path"] = str(image_path)
                    annotation["image_size"] = list(image.size)

            annotations.append(annotation)

        metadata = {
            "dataset": spec.key,
            "source": spec.hf_id,
            "split": split_name,
            "num_samples": len(annotations),
            "columns": columns,
            "saved_images": save_images,
            "streaming": streaming,
        }

        with open(annotations_dir / f"{split_name}_annotations.json", "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2)
        with open(annotations_dir / f"{split_name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {len(annotations)} annotations to {annotations_dir}")


def download_os_atlas(root_dir: Path, subset: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub or upgrade datasets to download OS-Atlas files.") from exc

    if subset not in OS_ATLAS_PATTERNS:
        raise ValueError(f"Unknown OS-Atlas subset '{subset}'. Choose one of {list(OS_ATLAS_PATTERNS)}")

    out_dir = root_dir / "os_atlas_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading OS-Atlas metadata/annotations only by default.")
    print("Full OS-Atlas images are very large; use Hugging Face manually for image archives.")

    path = snapshot_download(
        repo_id="OS-Copilot/OS-Atlas-data",
        repo_type="dataset",
        local_dir=out_dir,
        allow_patterns=OS_ATLAS_PATTERNS[subset],
    )
    print(f"OS-Atlas files saved to: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NanoGUI training/evaluation datasets.")
    parser.add_argument(
        "dataset",
        choices=sorted(list(DATASETS.keys()) + ["os_atlas"]),
        help="Dataset key to download.",
    )
    parser.add_argument("--root-dir", default="./datasets", help="Directory where datasets are saved.")
    parser.add_argument("--split", action="append", help="Split to download. Repeat for multiple splits.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per split for smoke tests.")
    parser.add_argument("--no-images", action="store_true", help="Save annotations only.")
    parser.add_argument("--streaming", action="store_true", help="Stream from Hugging Face while writing local files.")
    parser.add_argument(
        "--os-atlas-subset",
        default="annotations",
        choices=sorted(OS_ATLAS_PATTERNS.keys()),
        help="OS-Atlas file pattern group to download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "os_atlas":
        download_os_atlas(root_dir, args.os_atlas_subset)
        return

    save_hf_dataset(
        spec=DATASETS[args.dataset],
        root_dir=root_dir,
        splits=args.split,
        max_samples=args.max_samples,
        save_images=not args.no_images,
        streaming=args.streaming,
    )


if __name__ == "__main__":
    main()
