"""
Build verifier training data from local GUI grounding annotations.

Verifier examples are generated from annotations that contain:
  - image_path
  - instruction
  - bbox in normalized [x1, y1, x2, y2] or pixel coordinates

For each grounding sample, this script creates:
  - one positive candidate point inside the target bbox
  - one or more negative candidate points outside the target bbox

It can optionally render marker-overlay images for a VLM verifier.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def load_annotations(data_dir: Path, split: str) -> list[dict[str, Any]]:
    path = data_dir / "annotations" / f"{split}_annotations.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing annotations file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got: {bbox}")

    x1, y1, x2, y2 = [float(v) for v in bbox]

    # Some datasets use [x, y, width, height]. If the third/fourth values look
    # like extents rather than bottom-right corners, convert them conservatively.
    if x2 < x1 or y2 < y1:
        x2 = x1 + abs(x2)
        y2 = y1 + abs(y2)

    # Pixel coordinates -> normalized coordinates.
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height

    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return [x1, y1, x2, y2]


def bbox_center(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def point_inside_bbox(point: list[float], bbox: list[float], margin: float = 0.0) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def random_negative_point(bbox: list[float], rng: random.Random) -> list[float]:
    for _ in range(200):
        point = [rng.random(), rng.random()]
        if not point_inside_bbox(point, bbox, margin=0.03):
            return point

    # Deterministic fallback if the bbox is unusually large.
    x1, y1, x2, y2 = bbox
    candidates = [
        [max(0.02, x1 - 0.08), (y1 + y2) / 2.0],
        [min(0.98, x2 + 0.08), (y1 + y2) / 2.0],
        [(x1 + x2) / 2.0, max(0.02, y1 - 0.08)],
        [(x1 + x2) / 2.0, min(0.98, y2 + 0.08)],
    ]
    for point in candidates:
        if not point_inside_bbox(point, bbox, margin=0.01):
            return point
    return [0.02, 0.02]


def render_marker(image_path: Path, point: list[float], out_path: Path, color: str) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x = int(point[0] * width)
    y = int(point[1] * height)
    radius = max(8, int(min(width, height) * 0.012))

    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        outline=color,
        width=max(3, radius // 3),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def build_examples(
    data_dir: Path,
    split: str,
    output_dir: Path,
    negatives_per_positive: int,
    render_overlays: bool,
    seed: int,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    annotations = load_annotations(data_dir, split)
    examples: list[dict[str, Any]] = []
    overlay_dir = output_dir / "images"

    for idx, sample in enumerate(annotations):
        if max_samples is not None and idx >= max_samples:
            break

        image_path_value = sample.get("image_path")
        bbox_value = sample.get("bbox")
        instruction = sample.get("instruction", "")
        if not image_path_value or not bbox_value:
            continue

        image_path = Path(image_path_value)
        if not image_path.exists():
            continue

        with Image.open(image_path) as image:
            width, height = image.size
        bbox = normalize_bbox(bbox_value, width, height)
        positive_point = bbox_center(bbox)

        candidates = [("success", positive_point, "The proposed point is inside the target element.")]
        for _ in range(negatives_per_positive):
            candidates.append(("failure", random_negative_point(bbox, rng), "The proposed point is outside the target element."))

        for candidate_idx, (status, point, reason) in enumerate(candidates):
            example_id = f"{split}_{idx:06d}_{candidate_idx}"
            example = {
                "id": example_id,
                "source_sample_id": sample.get("id", f"{split}_{idx}"),
                "instruction": instruction,
                "sub_goal": instruction,
                "action_description": f"Click at normalized coordinate ({point[0]:.4f}, {point[1]:.4f})",
                "candidate_point": point,
                "target_bbox": bbox,
                "status": status,
                "reason": reason,
                "image_path": str(image_path),
            }

            if render_overlays:
                overlay_path = overlay_dir / f"{example_id}.png"
                render_marker(image_path, point, overlay_path, "green" if status == "success" else "red")
                example["overlay_image_path"] = str(overlay_path)

            examples.append(example)

    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NanoGUI verifier data from grounding annotations.")
    parser.add_argument("--data-dir", default="./data/screenspot", help="Local dataset directory.")
    parser.add_argument("--split", default="test", help="Annotation split to convert.")
    parser.add_argument("--output-dir", default="./data/verifier", help="Directory for verifier annotations.")
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--render-overlays", action="store_true", help="Save screenshots with candidate click markers.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    examples = build_examples(
        data_dir=Path(args.data_dir),
        split=args.split,
        output_dir=output_dir,
        negatives_per_positive=args.negatives_per_positive,
        render_overlays=args.render_overlays,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    out_path = annotations_dir / f"{args.split}_annotations.json"
    metadata_path = annotations_dir / f"{args.split}_metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_data_dir": args.data_dir,
                "source_split": args.split,
                "num_examples": len(examples),
                "negatives_per_positive": args.negatives_per_positive,
                "render_overlays": args.render_overlays,
            },
            f,
            indent=2,
        )

    print(f"Saved {len(examples)} verifier examples to {out_path}")


if __name__ == "__main__":
    main()
