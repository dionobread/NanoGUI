"""
Download planner-oriented GUI trajectory datasets for NanoGUI.

Planner data should teach:

    high-level task + optional screenshot/context -> ordered atomic sub-goals

This downloader saves raw-ish normalized records from trajectory datasets into:

    data/planner_sources/<dataset_key>/annotations/<split>_annotations.json

Use `build_planner_dataset.py` later if you want to convert these trajectories
into NanoGUI's exact `{"task": ..., "sub_goals": [...]}` planner SFT format.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class PlannerDatasetSpec:
    key: str
    hf_id: str
    task_fields: tuple[str, ...]
    trajectory_fields: tuple[str, ...]
    image_fields: tuple[str, ...] = ("image", "screenshot")
    notes: str = ""


PLANNER_DATASETS: dict[str, PlannerDatasetSpec] = {
    "mind2web": PlannerDatasetSpec(
        key="mind2web",
        hf_id="osunlp/Mind2Web",
        task_fields=("confirmed_task", "task", "intent", "annotation_id"),
        trajectory_fields=("actions", "action_reprs", "action_uid", "operation"),
        notes="Main web trajectory dataset for planner training.",
    ),
    "multimodal_mind2web": PlannerDatasetSpec(
        key="multimodal_mind2web",
        hf_id="osunlp/Multimodal-Mind2Web",
        task_fields=("confirmed_task", "task", "intent", "annotation_id"),
        trajectory_fields=("actions", "action_reprs", "action_uid", "operation"),
        notes="Mind2Web variant with screenshots where available.",
    ),
    "online_mind2web": PlannerDatasetSpec(
        key="online_mind2web",
        hf_id="osunlp/Online-Mind2Web",
        task_fields=("confirmed_task", "task", "intent", "annotation_id"),
        trajectory_fields=("actions", "action_reprs", "action_uid", "operation"),
        notes="Online benchmark-style Mind2Web data; may require accepting Hugging Face terms.",
    ),
    "aitw_single": PlannerDatasetSpec(
        key="aitw_single",
        hf_id="cjfcsjt/AITW_Single",
        task_fields=("goal", "instruction", "task"),
        trajectory_fields=("action", "actions", "action_type", "touch_point"),
        notes="Convenient Hugging Face mirror/subset of Android-in-the-Wild style data.",
    ),
}


def first_present(sample: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field_name in fields:
        if field_name in sample and sample[field_name] is not None:
            return sample[field_name]
    return None


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def as_pil_image(value: Any) -> Image.Image | None:
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


def available_splits(dataset: Any) -> list[str]:
    return list(dataset.keys())


def select_splits(dataset: Any, requested: list[str] | None) -> list[str]:
    if not requested:
        return available_splits(dataset)

    missing = [split for split in requested if split not in dataset]
    if missing:
        raise ValueError(f"Missing split(s) {missing}; available: {available_splits(dataset)}")
    return requested


def save_planner_dataset(
    spec: PlannerDatasetSpec,
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

    dataset_dir = root_dir / spec.key
    annotations_dir = dataset_dir / "annotations"
    images_dir = dataset_dir / "images"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading planner dataset: {spec.key} ({spec.hf_id})")
    if spec.notes:
        print(f"Note: {spec.notes}")

    dataset = load_dataset(spec.hf_id, streaming=streaming)
    selected_splits = select_splits(dataset, splits)

    for split_name in selected_splits:
        split_data = dataset[split_name]
        columns = getattr(split_data, "column_names", None)
        records: list[dict[str, Any]] = []
        print(f"Processing split '{split_name}'")

        for idx, sample in enumerate(split_data):
            if max_samples is not None and idx >= max_samples:
                break

            sample_dict = dict(sample)
            task = first_present(sample_dict, spec.task_fields)
            trajectory = first_present(sample_dict, spec.trajectory_fields)
            image_value = first_present(sample_dict, spec.image_fields)

            record: dict[str, Any] = {
                "id": f"{spec.key}_{split_name}_{idx:06d}",
                "dataset": spec.key,
                "source": spec.hf_id,
                "split": split_name,
                "task": task or "",
                "trajectory": json_safe(trajectory),
                "raw": {},
            }

            for key, value in sample_dict.items():
                if key in set(spec.image_fields):
                    continue
                record["raw"][key] = json_safe(value)

            if save_images and image_value is not None:
                image = as_pil_image(image_value)
                if image is not None:
                    image_path = images_dir / f"{split_name}_{idx:06d}.png"
                    image.save(image_path)
                    record["image_path"] = str(image_path)
                    record["image_size"] = list(image.size)

            records.append(record)

        with open(annotations_dir / f"{split_name}_annotations.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        with open(annotations_dir / f"{split_name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": spec.key,
                    "source": spec.hf_id,
                    "split": split_name,
                    "num_samples": len(records),
                    "columns": columns,
                    "saved_images": save_images,
                    "streaming": streaming,
                },
                f,
                indent=2,
            )

        print(f"Saved {len(records)} records to {annotations_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download planner trajectory datasets for NanoGUI.")
    parser.add_argument(
        "dataset",
        choices=sorted(list(PLANNER_DATASETS.keys()) + ["all"]),
        help="Planner dataset to download, or 'all'.",
    )
    parser.add_argument("--root-dir", default="./datasets/planner_sources")
    parser.add_argument("--split", action="append", help="Split to download. Repeat for multiple splits.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per split for smoke tests.")
    parser.add_argument("--no-images", action="store_true", help="Save annotations only.")
    parser.add_argument("--streaming", action="store_true", help="Stream from Hugging Face while writing local files.")
    parser.add_argument(
        "--skip-online",
        action="store_true",
        help="When dataset=all, skip Online-Mind2Web because it may require HF access approval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    keys = list(PLANNER_DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    if args.skip_online and "online_mind2web" in keys:
        keys.remove("online_mind2web")

    for key in keys:
        save_planner_dataset(
            spec=PLANNER_DATASETS[key],
            root_dir=root_dir,
            splits=args.split,
            max_samples=args.max_samples,
            save_images=not args.no_images,
            streaming=args.streaming,
        )


if __name__ == "__main__":
    main()
