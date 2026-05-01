"""
Evaluate the Grounder model on ScreenSpot test set.

Uses GUI-Actor's native coordinate-free pointer inference for accurate results.

Usage:
    python scripts/eval_grounder.py --model GUI-Actor-3B-Qwen2.5-VL
    python scripts/eval_grounder.py --model GUI-Actor-2B-Qwen2-VL --max-samples 50
    python scripts/eval_grounder.py --download  # Download ScreenSpot first
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add NanoGUI to path for gui_actor package
sys.path.insert(0, str(Path(__file__).parent.parent / "NanoGUI"))


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def discover_models() -> List[str]:
    models_dir = get_project_root() / "models"
    if not models_dir.exists():
        return []
    return sorted([d.name for d in models_dir.iterdir() if d.is_dir() and (d / "config.json").exists()])


class GUIActorGrounder:
    """Wrapper for GUI-Actor model with native pointer-based inference."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoProcessor
        from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
        from gui_actor.inference import inference

        self.inference = inference
        self.model_name = model_name
        model_path = get_project_root() / "models" / model_name

        logger.info("Loading %s (bfloat16)...", model_name)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        logger.info("Model loaded on %s", next(self.model.parameters()).device)

        self.system_message = (
            "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
            "your task is to locate the screen element that corresponds to the instruction. "
            "You should output a PyAutoGUI action that performs a click on the correct position. "
            "To indicate the click location, we will use some special tokens, which is used to refer "
            "to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>)."
        )

    def predict(self, image: Image.Image, instruction: str) -> Tuple[float, float]:
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            },
        ]
        pred = self.inference(
            conversation, self.model, self.tokenizer, self.processor,
            use_placeholder=True, topk=1,
        )
        px, py = pred["topk_points"][0]
        return px, py

    def close(self):
        import gc
        import torch
        del self.model
        del self.processor
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("GUI-Actor model closed.")


class StandardGrounder:
    """Wrapper for standard VLM models that output text coordinates."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        self.model_name = model_name
        model_path = get_project_root() / "models" / model_name

        logger.info("Loading %s (FP16)...", model_name)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Detect model class from config
        import json
        with open(model_path / "config.json") as f:
            cfg = json.load(f)
        arch = (cfg.get("architectures") or [""])[0]
        mtype = cfg.get("model_type", "")

        if "Qwen2_5_VL" in arch or mtype == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration as cls
        elif "Qwen2VL" in arch or mtype == "qwen2_vl":
            from transformers import Qwen2VLForConditionalGeneration as cls
        else:
            cls = AutoModelForCausalLM

        self.model = cls.from_pretrained(
            model_path, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True,
        )
        logger.info("Model loaded on %s", next(self.model.parameters()).device)

    def predict(self, image: Image.Image, instruction: str) -> Tuple[float, float]:
        import torch
        import re

        messages = [
            {"role": "system", "content": "You are a GUI grounding assistant. Predict the normalized coordinates (x, y) of UI elements."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Task: {instruction}\nOutput the normalized coordinates (x, y) of the target element."}
            ]}
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        match = re.search(r"\(?(0\.\d+)[,\s]+(0\.\d+)\)?", response)
        if match:
            return float(match.group(1)), float(match.group(2))

        match = re.search(r"(\d+\.?\d*)[,\s]+(\d+\.?\d*)", response)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            if x > 1.0:
                x = x / image.width
            if y > 1.0:
                y = y / image.height
            return x, y

        logger.warning("Could not parse coordinates from: %r", response)
        return 0.5, 0.5

    def close(self):
        import gc
        import torch
        del self.model
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Standard grounder closed.")


def is_inside_bbox(
    x: float, y: float, bbox: List[float], img_w: int, img_h: int
) -> bool:
    """
    Check if normalized (x, y) falls inside bbox.

    Args:
        bbox: [x1, y1, x2, y2] in normalized coordinates (ScreenSpot format)
    """
    bx1, by1, bx2, by2 = bbox
    return bx1 <= x <= bx2 and by1 <= y <= by2


def load_screenspot_annotations(split: str = "test") -> List[Dict]:
    """Load ScreenSpot annotations for a given split."""
    anno_path = get_project_root() / "datasets" / "screenspot" / "annotations" / f"{split}_annotations.json"
    if not anno_path.exists():
        raise FileNotFoundError(
            f"ScreenSpot {split} annotations not found at {anno_path}\n"
            f"Download first: python scripts/download_datasets.py --screenspot"
        )

    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter valid samples
    valid = []
    for item in data:
        if "bbox" not in item or "image_path" not in item:
            continue
        img_path = Path(item["image_path"].replace("\\", "/"))
        if not img_path.exists():
            # Try relative to project root
            img_path = get_project_root() / img_path
            if not img_path.exists():
                # ScreenSpot annotations use 'data/screenspot/images' but repo uses 'datasets/screenspot/images'
                parts = img_path.parts
                if "data" in parts:
                    idx = parts.index("data")
                    new_parts = parts[:idx] + ("datasets",) + parts[idx + 1:]
                    img_path = Path(*new_parts)
                if not img_path.exists():
                    continue
            item["image_path"] = str(img_path)
        valid.append(item)

    return valid


def evaluate_grounder(
    model_name: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    batch_size: int = 1,
) -> Dict:
    """
    Evaluate grounder on ScreenSpot.

    Returns dict with accuracy and per-sample results.
    """
    logger.info("=" * 60)
    logger.info("Grounder Evaluation")
    logger.info("=" * 60)
    logger.info("Model: %s", model_name)
    logger.info("Split: %s", split)

    # Load data
    logger.info("Loading annotations...")
    samples = load_screenspot_annotations(split)
    if max_samples:
        samples = samples[:max_samples]
    logger.info("Samples: %d", len(samples))

    # Load model
    grounder = GUIActorGrounder(model_name)

    # Evaluate
    correct = 0
    total = 0
    results = []

    for i, sample in enumerate(samples):
        instruction = sample.get("instruction", "")
        bbox = sample["bbox"]  # [x1, y1, x2, y2] normalized
        img_path = sample["image_path"]

        # Load image
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # Predict
        t0 = time.time()
        pred_x, pred_y = grounder.predict(image, instruction)
        inference_time = time.time() - t0

        # Check accuracy
        is_correct = is_inside_bbox(pred_x, pred_y, bbox, img_w, img_h)
        if is_correct:
            correct += 1
        total += 1

        # Ground truth center
        gt_x = (bbox[0] + bbox[2]) / 2
        gt_y = (bbox[1] + bbox[3]) / 2

        # Distance
        dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

        results.append({
            "id": sample.get("id", i),
            "instruction": instruction,
            "predicted": (round(pred_x, 4), round(pred_y, 4)),
            "ground_truth": (round(gt_x, 4), round(gt_y, 4)),
            "distance": round(dist, 4),
            "correct": is_correct,
            "inference_time_ms": round(inference_time * 1000, 1),
        })

        if (i + 1) % 10 == 0:
            acc = correct / total * 100
            logger.info("Progress: %d/%d | Accuracy: %.1f%%", i + 1, len(samples), acc)

    grounder.close()

    accuracy = correct / total * 100 if total > 0 else 0
    avg_time = np.mean([r["inference_time_ms"] for r in results])
    avg_dist = np.mean([r["distance"] for r in results])

    summary = {
        "model": model_name,
        "split": split,
        "total_samples": total,
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "avg_inference_time_ms": round(avg_time, 1),
        "avg_distance": round(avg_dist, 4),
        "results": results,
    }

    return summary


def print_summary(summary: Dict):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model:    {summary['model']}")
    print(f"  Split:    {summary['split']}")
    print(f"  Samples:  {summary['total_samples']}")
    print(f"  Correct:  {summary['correct']}")
    print(f"  Accuracy: {summary['accuracy']:.1f}%")
    print(f"  Avg time: {summary['avg_inference_time_ms']:.1f} ms")
    print(f"  Avg dist: {summary['avg_distance']:.4f}")
    print("=" * 60)

    # Breakdown by data type if available
    by_type = {}
    for r in summary["results"]:
        dtype = "unknown"
        if "web" in r["instruction"].lower():
            dtype = "web"
        elif "desktop" in r["instruction"].lower():
            dtype = "desktop"
        elif "mobile" in r["instruction"].lower() or "android" in r["instruction"].lower() or "ios" in r["instruction"].lower():
            dtype = "mobile"

        if dtype not in by_type:
            by_type[dtype] = {"total": 0, "correct": 0}
        by_type[dtype]["total"] += 1
        if r["correct"]:
            by_type[dtype]["correct"] += 1

    if len(by_type) > 1:
        print("\n  By platform:")
        for dtype, stats in sorted(by_type.items()):
            acc = stats["correct"] / stats["total"] * 100
            print(f"    {dtype:10s}: {acc:.1f}% ({stats['correct']}/{stats['total']})")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Grounder on ScreenSpot using GUI-Actor native inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/eval_grounder.py --model GUI-Actor-3B-Qwen2.5-VL
  python scripts/eval_grounder.py --model GUI-Actor-2B-Qwen2-VL --max-samples 50
  python scripts/eval_grounder.py --model GUI-Actor-3B-Qwen2.5-VL --split train --max-samples 100
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model directory name (e.g. GUI-Actor-3B-Qwen2.5-VL)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="ScreenSpot split to evaluate (default: test)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (for quick tests)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Validate model exists
    if args.model not in discover_models():
        print(f"ERROR: Model '{args.model}' not found.")
        print(f"Available: {discover_models()}")
        raise SystemExit(1)

    # Run evaluation
    try:
        summary = evaluate_grounder(
            model_name=args.model,
            split=args.split,
            max_samples=args.max_samples,
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nTo download ScreenSpot:")
        print("  python scripts/download_datasets.py --screenspot")
        raise SystemExit(1)

    # Print results
    print_summary(summary)

    # Save if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
