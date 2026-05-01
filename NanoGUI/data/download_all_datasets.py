"""
Unified dataset downloader for NanoGUI.

Combines all dataset download scripts into one file with subcommands.

"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
            raise ValueError(
                f"Missing split(s) {missing}; available: {list(dataset.keys())}"
            )
        return requested
    return list(dataset.keys())


# ---------------------------------------------------------------------------
# GUI Datasets (grounding)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuiDatasetSpec:
    key: str
    hf_id: str
    output_dir: str
    instruction_fields: tuple[str, ...] = ("instruction", "task", "intent", "confirmed_task")
    bbox_fields: tuple[str, ...] = ("bbox", "box", "target_bbox")
    image_fields: tuple[str, ...] = ("image", "screenshot")
    default_split: str | None = None
    notes: str = ""
    load_kwargs: dict[str, Any] = field(default_factory=dict)


GUI_DATASETS: dict[str, GuiDatasetSpec] = {
    "screenspot": GuiDatasetSpec(
        key="screenspot",
        hf_id="rootsautomation/ScreenSpot",
        output_dir="screenspot",
        default_split="test",
        notes="Grounder benchmark. Keep as evaluation data when possible.",
    ),
    "screenspot_pro": GuiDatasetSpec(
        key="screenspot_pro",
        hf_id="Voxel51/ScreenSpot-Pro",
        output_dir="screenspot_pro",
        default_split="train",
        notes="Hard high-resolution GUI grounding benchmark.",
    ),
    "omniact": GuiDatasetSpec(
        key="omniact",
        hf_id="Writer/OmniAct",
        output_dir="omniact",
        instruction_fields=("task", "instruction"),
        bbox_fields=("box", "bbox"),
        notes="Grounding/action dataset for desktop and web UIs.",
    ),
    "salesforce_grounding": GuiDatasetSpec(
        key="salesforce_grounding",
        hf_id="Salesforce/grounding_dataset",
        output_dir="salesforce_grounding",
        instruction_fields=("instruction", "description"),
        bbox_fields=("bbox",),
        notes="Combined GUI grounding data from several sources.",
    ),
}


def download_gui_dataset(
    spec: GuiDatasetSpec,
    root_dir: Path,
    splits: list[str] | None,
    max_samples: int | None,
    save_images: bool,
    streaming: bool,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("pip install datasets") from exc

    out_dir = root_dir / spec.output_dir
    images_dir = out_dir / "images"
    annotations_dir = out_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {spec.key}: {spec.hf_id}")
    if spec.notes:
        print(f"  Note: {spec.notes}")

    dataset = load_dataset(spec.hf_id, streaming=streaming, **spec.load_kwargs)
    selected_splits = _select_splits(dataset, splits)

    for split_name in selected_splits:
        split_data = dataset[split_name]
        annotations: list[dict[str, Any]] = []
        columns = getattr(split_data, "column_names", None)
        print(f"  Processing split '{split_name}'")

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

        print(f"  Saved {len(annotations)} annotations")


# ---------------------------------------------------------------------------
# Planner Datasets (trajectories)
# ---------------------------------------------------------------------------

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
        notes="Online benchmark-style Mind2Web data; may require accepting HF terms.",
    ),
    "aitw_single": PlannerDatasetSpec(
        key="aitw_single",
        hf_id="cjfcsjt/AITW_Single",
        task_fields=("goal", "instruction", "task"),
        trajectory_fields=("action", "actions", "action_type", "touch_point"),
        notes="Convenient HF mirror/subset of Android-in-the-Wild style data.",
    ),
}


def download_planner_dataset(
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
        raise RuntimeError("pip install datasets") from exc

    dataset_dir = root_dir / spec.key
    annotations_dir = dataset_dir / "annotations"
    images_dir = dataset_dir / "images"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading planner dataset: {spec.key} ({spec.hf_id})")
    if spec.notes:
        print(f"  Note: {spec.notes}")

    dataset = load_dataset(spec.hf_id, streaming=streaming)
    selected_splits = _select_splits(dataset, splits)

    for split_name in selected_splits:
        split_data = dataset[split_name]
        columns = getattr(split_data, "column_names", None)
        records: list[dict[str, Any]] = []
        print(f"  Processing split '{split_name}'")

        for idx, sample in enumerate(split_data):
            if max_samples is not None and idx >= max_samples:
                break

            sample_dict = dict(sample)
            task = _first_present(sample_dict, spec.task_fields)
            trajectory = _first_present(sample_dict, spec.trajectory_fields)
            image_value = _first_present(sample_dict, spec.image_fields)

            record: dict[str, Any] = {
                "id": f"{spec.key}_{split_name}_{idx:06d}",
                "dataset": spec.key,
                "source": spec.hf_id,
                "split": split_name,
                "task": task or "",
                "trajectory": _json_safe(trajectory),
                "raw": {},
            }

            for key, value in sample_dict.items():
                if key in set(spec.image_fields):
                    continue
                record["raw"][key] = _json_safe(value)

            if save_images and image_value is not None:
                image = _as_pil_image(image_value)
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

        print(f"  Saved {len(records)} records")


# ---------------------------------------------------------------------------
# Legacy / backward-compat downloads
# ---------------------------------------------------------------------------

def download_screenspot_v2(root: Path, save_images: bool = True) -> None:
    """Download ScreenSpot-v2 (legacy entry point)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("pip install datasets") from exc

    out_dir = root / "screenspot_v2"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    anno_path = out_dir / "annotations.json"
    if anno_path.exists():
        print(f"ScreenSpot-v2 already exists at {anno_path}")
        return

    print("Downloading ScreenSpot-v2...")
    ds = load_dataset("HongxinLi/ScreenSpot_v2", split="test")

    annotations = []
    for i, item in enumerate(tqdm(ds, desc="ScreenSpot-v2")):
        img = item["image"]
        if img is None:
            continue

        img_filename = f"ssv2_{i:05d}.png"
        img_path = img_dir / img_filename
        if save_images:
            img.save(img_path)

        annotations.append({
            "id": f"ssv2_{i}",
            "image_path": str(img_path) if save_images else "",
            "bbox": item["bbox"],
            "instruction": item["instruction"],
            "data_type": item["data_type"],
            "data_source": item["data_source"],
        })

    with open(anno_path, "w") as f:
        json.dump(annotations, f, indent=2)

    sources = Counter(a["data_source"] for a in annotations)
    print(f"ScreenSpot-v2: {len(annotations)} samples, sources: {dict(sources)}")


def download_seeclick(root: Path, num_images: int = 500, save_images: bool = True) -> None:
    """Download SeeClick subset (legacy entry point)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("pip install datasets") from exc

    out_dir = root / "seeclick"
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

    for sample in tqdm(ds, desc="SeeClick", total=num_images):
        if img_count >= num_images:
            break

        img = sample["image"]
        if img is None:
            continue

        img_filename = f"seeclick_{img_count:05d}.png"
        img_path = img_dir / img_filename
        if save_images:
            img.save(img_path)

        for elem in sample.get("elements", []):
            bbox = elem.get("bbox")
            instruction = elem.get("instruction", "")
            data_type = elem.get("data_type", "unknown")

            if not bbox or not instruction:
                continue

            annotations.append({
                "image_path": str(img_path) if save_images else "",
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


def create_screenspot_splits(root: Path) -> None:
    """Create train/val/test splits from ScreenSpot test annotations."""
    anno_path = root / "screenspot" / "annotations" / "test_annotations.json"
    out_dir = root / "screenspot" / "annotations"

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
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]

    for name, split_data in [("train", train), ("val", val), ("test_split", test)]:
        path = out_dir / f"{name}_annotations.json"
        with open(path, "w") as f:
            json.dump(split_data, f, indent=2)

    print(f"ScreenSpot splits: train={len(train)}, val={len(val)}, test={len(test)}")


# ---------------------------------------------------------------------------
# OmniAct (custom downloader with analysis)
# ---------------------------------------------------------------------------

def download_omniact(root: Path, save_images: bool = True) -> None:
    """Download OmniAct with optional analysis output."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("pip install datasets") from exc

    out_dir = root / "omniact"
    images_dir = out_dir / "images"
    annotations_dir = out_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading OmniAct (Writer/OmniAct)...")
    dataset = load_dataset("Writer/OmniAct")

    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} samples")
        annotations = []

        for idx, sample in enumerate(split_data):
            annotation = {
                "id": f"{split_name}_{idx:05d}",
                "instruction": sample.get("task", ""),
                "data_type": sample.get("data_type", "unknown"),
            }
            if "box" in sample:
                annotation["bbox"] = sample["box"]

            img = sample.get("image")
            if save_images and img is not None:
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                img_path = images_dir / f"{split_name}_{idx:05d}.png"
                img.save(img_path)
                annotation["image_path"] = str(img_path)

            annotations.append(annotation)

        with open(annotations_dir / f"{split_name}_annotations.json", "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2)

        with open(annotations_dir / f"{split_name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {"split": split_name, "num_samples": len(annotations), "columns": split_data.column_names},
                f,
                indent=2,
            )

        print(f"  Saved {len(annotations)} annotations")


# ---------------------------------------------------------------------------
# OS-Atlas
# ---------------------------------------------------------------------------

OS_ATLAS_PATTERNS = {
    "annotations": ["*.json", "**/*.json", "prompts.json", "README.md"],
    "mobile_annotations": ["mobile_domain/*.json", "prompts.json", "README.md"],
    "desktop_annotations": ["desktop_domain/*.json", "prompts.json", "README.md"],
    "web_annotations": ["web_domain/*.json", "prompts.json", "README.md"],
}


def download_os_atlas(root: Path, subset: str = "annotations") -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("pip install huggingface_hub") from exc

    if subset not in OS_ATLAS_PATTERNS:
        raise ValueError(f"Unknown subset '{subset}'. Choose from {list(OS_ATLAS_PATTERNS)}")

    out_dir = root / "os_atlas"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading OS-Atlas metadata/annotations only.")
    print("  (Full images are very large; fetch manually if needed.)")

    path = snapshot_download(
        repo_id="OS-Copilot/OS-Atlas-data",
        repo_type="dataset",
        local_dir=out_dir,
        allow_patterns=OS_ATLAS_PATTERNS[subset],
    )
    print(f"  Saved to: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root-dir", default="./datasets", help="Output directory for datasets.")
    parser.add_argument("--split", action="append", help="Split to download (repeatable).")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit per split.")
    parser.add_argument("--no-images", action="store_true", help="Annotations only.")
    parser.add_argument("--streaming", action="store_true", help="Stream from Hugging Face.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified NanoGUI dataset downloader.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s gui screenspot --no-images
  %(prog)s gui all --no-images --max-samples 5000
  %(prog)s planner mind2web --no-images
  %(prog)s planner all --no-images --skip-online
  %(prog)s legacy seeclick
  %(prog)s legacy screenspot_v2
  %(prog)s omniact --no-images
  %(prog)s os-atlas --subset annotations
  %(prog)s all --no-images
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ gui
    gui_parser = subparsers.add_parser("gui", help="Download GUI grounding datasets.")
    gui_parser.add_argument(
        "dataset",
        choices=sorted(list(GUI_DATASETS.keys()) + ["all"]),
        help="Dataset key or 'all'.",
    )
    _add_common_args(gui_parser)

    # ------------------------------------------------------------------ planner
    planner_parser = subparsers.add_parser("planner", help="Download planner trajectory datasets.")
    planner_parser.add_argument(
        "dataset",
        choices=sorted(list(PLANNER_DATASETS.keys()) + ["all"]),
        help="Dataset key or 'all'.",
    )
    _add_common_args(planner_parser)
    planner_parser.add_argument(
        "--skip-online", action="store_true",
        help="Skip Online-Mind2Web when dataset=all (requires HF access approval).",
    )

    # ------------------------------------------------------------------ legacy
    legacy_parser = subparsers.add_parser("legacy", help="Download legacy datasets (SeeClick, ScreenSpot-v2, splits).")
    legacy_parser.add_argument(
        "dataset",
        choices=["seeclick", "screenspot_v2", "splits"],
        help="Legacy dataset to download.",
    )
    legacy_parser.add_argument("--root-dir", default="./datasets", help="Output directory.")
    legacy_parser.add_argument("--seeclick-images", type=int, default=500, help="Images for SeeClick.")
    legacy_parser.add_argument("--no-images", action="store_true", help="Skip saving images.")

    # ------------------------------------------------------------------ omniact
    omniact_parser = subparsers.add_parser("omniact", help="Download OmniAct dataset.")
    omniact_parser.add_argument("--root-dir", default="./datasets", help="Output directory.")
    omniact_parser.add_argument("--no-images", action="store_true", help="Skip saving images.")

    # ------------------------------------------------------------------ os-atlas
    os_atlas_parser = subparsers.add_parser("os-atlas", help="Download OS-Atlas annotations.")
    os_atlas_parser.add_argument(
        "--subset", default="annotations", choices=sorted(OS_ATLAS_PATTERNS.keys()),
        help="OS-Atlas file pattern group.",
    )
    os_atlas_parser.add_argument("--root-dir", default="./datasets", help="Output directory.")

    # ------------------------------------------------------------------ all
    all_parser = subparsers.add_parser("all", help="Download all GUI + planner datasets.")
    _add_common_args(all_parser)
    all_parser.add_argument(
        "--skip-online", action="store_true",
        help="Skip Online-Mind2Web.",
    )
    all_parser.add_argument("--skip-legacy", action="store_true", help="Skip SeeClick and ScreenSpot-v2.")
    all_parser.add_argument("--skip-os-atlas", action="store_true", help="Skip OS-Atlas.")
    all_parser.add_argument("--seeclick-images", type=int, default=500)

    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    save_images = not getattr(args, "no_images", False)

    # ------------------------------------------------------------------ gui
    if args.command == "gui":
        keys = list(GUI_DATASETS.keys()) if args.dataset == "all" else [args.dataset]
        for key in keys:
            download_gui_dataset(
                spec=GUI_DATASETS[key],
                root_dir=root_dir,
                splits=args.split,
                max_samples=args.max_samples,
                save_images=save_images,
                streaming=args.streaming,
            )

    # ------------------------------------------------------------------ planner
    elif args.command == "planner":
        keys = list(PLANNER_DATASETS.keys()) if args.dataset == "all" else [args.dataset]
        if args.skip_online and "online_mind2web" in keys:
            keys.remove("online_mind2web")
        for key in keys:
            download_planner_dataset(
                spec=PLANNER_DATASETS[key],
                root_dir=root_dir / "planner_sources",
                splits=args.split,
                max_samples=args.max_samples,
                save_images=save_images,
                streaming=args.streaming,
            )

    # ------------------------------------------------------------------ legacy
    elif args.command == "legacy":
        if args.dataset == "seeclick":
            download_seeclick(root_dir, num_images=args.seeclick_images, save_images=save_images)
        elif args.dataset == "screenspot_v2":
            download_screenspot_v2(root_dir, save_images=save_images)
        elif args.dataset == "splits":
            create_screenspot_splits(root_dir)

    # ------------------------------------------------------------------ omniact
    elif args.command == "omniact":
        download_omniact(root_dir, save_images=save_images)

    # ------------------------------------------------------------------ os-atlas
    elif args.command == "os-atlas":
        download_os_atlas(root_dir, subset=args.subset)

    # ------------------------------------------------------------------ all
    elif args.command == "all":
        for key in GUI_DATASETS:
            download_gui_dataset(
                spec=GUI_DATASETS[key],
                root_dir=root_dir,
                splits=args.split,
                max_samples=args.max_samples,
                save_images=save_images,
                streaming=args.streaming,
            )
        planner_keys = list(PLANNER_DATASETS.keys())
        if args.skip_online and "online_mind2web" in planner_keys:
            planner_keys.remove("online_mind2web")
        for key in planner_keys:
            download_planner_dataset(
                spec=PLANNER_DATASETS[key],
                root_dir=root_dir / "planner_sources",
                splits=args.split,
                max_samples=args.max_samples,
                save_images=save_images,
                streaming=args.streaming,
            )
        if not args.skip_legacy:
            download_screenspot_v2(root_dir, save_images=save_images)
            download_seeclick(root_dir, num_images=args.seeclick_images, save_images=save_images)
            create_screenspot_splits(root_dir)
        if not args.skip_os_atlas:
            download_os_atlas(root_dir, subset="annotations")

    print("\nDone.")


if __name__ == "__main__":
    main()
