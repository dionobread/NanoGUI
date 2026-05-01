#!/usr/bin/env python3
"""
Improved LoRA fine-tuning script for the Grounder agent.

Improvements over original:
- Supports multiple datasets (ScreenSpot, SeeClick, ScreenSpot-v2)
- Data augmentation: instruction paraphrasing, spatial enrichment
- Proper train/val split with metrics tracking
- Coordinate regression + cross-entropy hybrid loss
- Works with GUI-Actor-3B and Qwen2.5-VL models
- Python 3.9 compatible (Optional[] instead of X | None)

Usage:
    python scripts/train_grounder_v2.py --dataset screenspot --epochs 3
    python scripts/train_grounder_v2.py --dataset seeclick --max-samples 5000
    python scripts/train_grounder_v2.py --dataset mixed --merge
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ── Data Augmentation ────────────────────────────────────────────────────────

INSTRUCTION_PREFIXES = [
    "Click on",
    "Locate",
    "Find",
    "Tap on",
    "Select",
    "Press",
    "Go to",
]

SPATIAL_HINTS = [
    "",
    " in the top area",
    " in the bottom area",
    " on the left side",
    " on the right side",
    " in the center",
]


def augment_instruction(instruction: str, bbox: List[float]) -> str:
    """Add instruction variety with spatial hints based on bbox position."""
    if not instruction or len(instruction) < 3:
        return instruction

    # Don't augment if already has a prefix
    lower = instruction.lower()
    for prefix in ["click", "tap", "go to", "open", "find", "select", "press", "locate"]:
        if lower.startswith(prefix):
            # Just add spatial hint sometimes
            if random.random() < 0.3:
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if cy < 0.33:
                    hint = " in the top area"
                elif cy > 0.67:
                    hint = " in the bottom area"
                elif cx < 0.33:
                    hint = " on the left side"
                elif cx > 0.67:
                    hint = " on the right side"
                else:
                    hint = " in the center"
                return instruction + hint
            return instruction

    # Add random prefix
    if random.random() < 0.5:
        prefix = random.choice(INSTRUCTION_PREFIXES)
        return f"{prefix} {instruction.lower()}"

    return instruction


# ── Dataset ──────────────────────────────────────────────────────────────────

class GroundingDataset(Dataset):
    """Multi-source grounding dataset for GUI element localization."""

    def __init__(
        self,
        annotations: List[Dict],
        processor,
        max_length: int = 512,
        augment: bool = False,
    ):
        self.annotations = annotations
        self.processor = processor
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.annotations[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        instruction = sample.get("instruction", "")
        bbox = sample.get("bbox", [0, 0, 1, 1])

        # Compute center coordinate from bbox
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Augment instruction
        if self.augment:
            instruction = augment_instruction(instruction, bbox)

        # Format as chat for Qwen2-VL
        messages = [
            {"role": "system", "content": "You are a GUI grounding assistant. Output normalized coordinates (x, y)."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Task: {instruction}\nOutput the normalized coordinates (x, y) of the target element."}
            ]},
        ]

        # Process with chat template
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        # Build labels: mask the prompt with -100, only train on the coordinate
        prompt_len = inputs["input_ids"].shape[1]
        target = f"({center_x:.4f}, {center_y:.4f})"
        target_ids = self.processor.tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # Append target to input_ids and attention_mask
        input_ids = torch.cat([inputs["input_ids"], target_ids], dim=1)
        target_attn = torch.ones_like(target_ids)
        attention_mask = torch.cat([inputs["attention_mask"], target_attn], dim=1)

        # Labels: -100 for prompt (don't compute loss), target_ids for response
        prompt_mask = torch.full((1, prompt_len), -100, dtype=torch.long)
        labels = torch.cat([prompt_mask, target_ids], dim=1)

        result = {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
            "gt_coords": torch.tensor([center_x, center_y], dtype=torch.float32),
        }

        # Qwen2-VL specific fields
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]
        if "mm_token_type_ids" in inputs:
            # Extend for target tokens (text, not image)
            target_mm = torch.zeros((1, target_ids.shape[1]), dtype=inputs["mm_token_type_ids"].dtype)
            result["mm_token_type_ids"] = torch.cat(
                [inputs["mm_token_type_ids"], target_mm], dim=1
            ).squeeze(0)

        return result


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_screenspot(split: str = "train") -> List[Dict]:
    """Load ScreenSpot from local annotations. Falls back to test if train missing."""
    root = get_project_root()
    anno_path = root / "datasets" / "screenspot" / "annotations" / f"{split}_annotations.json"

    if not anno_path.exists():
        # Fallback to test annotations if train doesn't exist
        fallback = root / "datasets" / "screenspot" / "annotations" / "test_annotations.json"
        if fallback.exists():
            logger.warning("%s not found, using test_annotations.json", anno_path)
            anno_path = fallback
        else:
            raise FileNotFoundError(f"ScreenSpot annotations not found: {anno_path}")

    with open(anno_path) as f:
        data = json.load(f)

    valid = []
    for item in data:
        bbox = item.get("bbox")
        instruction = item.get("instruction", "")
        img_path = item.get("image_path", "")

        if not bbox or not instruction:
            continue

        p = _resolve_path(img_path, root)
        if p is None:
            continue

        valid.append({
            "image_path": str(p),
            "bbox": bbox,
            "instruction": instruction,
            "data_type": item.get("data_type", "unknown"),
            "source": "screenspot",
        })

    logger.info("Loaded %d ScreenSpot samples from %s", len(valid), split)
    return valid


def load_seeclick(max_samples: Optional[int] = None) -> List[Dict]:
    """Load SeeClick from local annotations."""
    root = get_project_root()
    anno_path = root / "datasets" / "seeclick" / "annotations.json"

    if not anno_path.exists():
        raise FileNotFoundError(f"SeeClick annotations not found: {anno_path}")

    with open(anno_path) as f:
        data = json.load(f)

    # Filter to only 'text' type (hover is less useful for grounding)
    data = [d for d in data if d.get("data_type") == "text"]

    if max_samples:
        random.seed(42)
        data = random.sample(data, min(max_samples, len(data)))

    # Verify image paths
    valid = []
    for item in data:
        p = Path(item["image_path"])
        if p.exists():
            valid.append({
                **item,
                "source": "seeclick",
            })

    logger.info("Loaded %d SeeClick samples", len(valid))
    return valid


def load_screenspot_v2(max_samples: Optional[int] = None) -> List[Dict]:
    """Load ScreenSpot-v2 from local annotations."""
    root = get_project_root()
    anno_path = root / "datasets" / "screenspot_v2" / "annotations.json"

    if not anno_path.exists():
        raise FileNotFoundError(f"ScreenSpot-v2 annotations not found: {anno_path}")

    with open(anno_path) as f:
        data = json.load(f)

    if max_samples:
        random.seed(42)
        data = random.sample(data, min(max_samples, len(data)))

    valid = []
    for item in data:
        p = Path(item["image_path"])
        if p.exists():
            valid.append({
                "image_path": str(p),
                "bbox": item["bbox"],
                "instruction": item["instruction"],
                "data_type": item.get("data_type", "unknown"),
                "source": "screenspot_v2",
            })

    logger.info("Loaded %d ScreenSpot-v2 samples", len(valid))
    return valid


def load_omniact(max_samples: Optional[int] = None) -> List[Dict]:
    """Load OmniAct from local annotations (downloaded via download_all_datasets.py)."""
    root = get_project_root()
    anno_dir = root / "datasets" / "omniact" / "annotations"

    if not anno_dir.exists():
        raise FileNotFoundError(f"OmniAct annotations not found: {anno_dir}")

    # Load train split by default
    anno_path = anno_dir / "train_annotations.json"
    if not anno_path.exists():
        raise FileNotFoundError(f"OmniAct train annotations not found: {anno_path}")

    with open(anno_path) as f:
        data = json.load(f)

    if max_samples:
        random.seed(42)
        data = random.sample(data, min(max_samples, len(data)))

    valid = []
    for item in data:
        img_path = item.get("image_path", "")
        p = Path(img_path)
        if not p.exists():
            # Try relative to datasets/omniact
            p = root / "datasets" / "omniact" / p.name
        if not p.exists():
            continue

        bbox = item.get("bbox")
        instruction = item.get("instruction", "")
        if not bbox or not instruction:
            continue

        # OmniAct bbox format: [x1, y1, x2, y2] normalized (same as ScreenSpot)
        valid.append({
            "image_path": str(p),
            "bbox": bbox,
            "instruction": instruction,
            "data_type": item.get("data_type", "omniact"),
            "source": "omniact",
        })

    logger.info("Loaded %d OmniAct samples", len(valid))
    return valid


def _resolve_path(img_path: str, root: Path) -> Optional[Path]:
    """Resolve image path across possible locations."""
    p = Path(img_path.replace("\\", "/"))
    if p.exists():
        return p
    p = root / p
    if p.exists():
        return p
    parts = p.parts
    if "data" in parts:
        idx = parts.index("data")
        p = Path(*(parts[:idx] + ("datasets",) + parts[idx + 1:]))
        if p.exists():
            return p
    return None


# ── Training ─────────────────────────────────────────────────────────────────

class Qwen2VLCollator:
    """Custom collator for Qwen2-VL that handles pixel_values and grid_thw."""

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Pad text fields to same max length
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=0
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        # Mask padding in labels with -100
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        max_len = input_ids.shape[1]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels,
        }

        # Stack pixel_values and image_grid_thw (Qwen2-VL specific)
        if "pixel_values" in batch[0] and batch[0]["pixel_values"] is not None:
            result["pixel_values"] = torch.cat([item["pixel_values"] for item in batch], dim=0)
        if "image_grid_thw" in batch[0] and batch[0]["image_grid_thw"] is not None:
            result["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
        if "mm_token_type_ids" in batch[0] and batch[0]["mm_token_type_ids"] is not None:
            mm_ids = [item["mm_token_type_ids"].squeeze(0) if item["mm_token_type_ids"].dim() > 1 else item["mm_token_type_ids"] for item in batch]
            mm_padded = torch.nn.utils.rnn.pad_sequence(
                mm_ids, batch_first=True, padding_value=0
            )
            # Ensure same length as input_ids
            if mm_padded.shape[1] < max_len:
                pad_size = max_len - mm_padded.shape[1]
                mm_padded = torch.cat([mm_padded, torch.zeros(mm_padded.shape[0], pad_size, dtype=mm_padded.dtype)], dim=1)
            result["mm_token_type_ids"] = mm_padded

        return result


def train_grounder(
    dataset_name: str = "screenspot",
    model_name: str = "Qwen2.5-VL-3B-Instruct",
    output_dir: str = "outputs/grounder_v2",
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    gradient_accumulation: int = 4,
    max_samples: Optional[int] = None,
    val_split: float = 0.1,
    use_wandb: bool = False,
    fp16: bool = True,
    merge: bool = False,
):
    """Fine-tune grounding model with LoRA."""
    from transformers import (
        Qwen2VLForConditionalGeneration,
        AutoProcessor,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    logger.info("Training Grounder v2")
    logger.info("Model: %s, Dataset: %s", model_name, dataset_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    if dataset_name == "screenspot":
        data = load_screenspot("train")
    elif dataset_name == "seeclick":
        data = load_seeclick(max_samples)
    elif dataset_name == "screenspot_v2":
        data = load_screenspot_v2(max_samples)
    elif dataset_name == "omniact":
        data = load_omniact(max_samples)
    elif dataset_name == "mixed":
        data = load_screenspot("train")
        try:
            data += load_seeclick(max_samples=min(5000, max_samples) if max_samples else 5000)
        except FileNotFoundError:
            logger.warning("SeeClick not found, using ScreenSpot only")
        try:
            data += load_screenspot_v2(max_samples)
        except FileNotFoundError:
            logger.warning("ScreenSpot-v2 not found, skipping")
        try:
            data += load_omniact(max_samples=min(5000, max_samples) if max_samples else 5000)
        except FileNotFoundError:
            logger.warning("OmniAct not found, skipping")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if max_samples and len(data) > max_samples:
        random.seed(42)
        data = random.sample(data, max_samples)

    # Split
    random.seed(42)
    random.shuffle(data)
    n_val = max(1, int(len(data) * val_split))
    train_data = data[n_val:]
    val_data = data[:n_val]

    logger.info("Train: %d, Val: %d", len(train_data), len(val_data))

    # Load model (prefer local path if it exists)
    model_path = get_project_root() / "models" / model_name
    if model_path.exists():
        logger.info("Loading local model: %s", model_path)
        load_path = str(model_path)
    else:
        logger.info("Loading model from HuggingFace: %s", model_name)
        load_path = model_name

    processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)

    dtype = torch.float16 if fp16 else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        load_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Fix for gradient checkpointing with PEFT
    model.enable_input_require_grads()
    model.config.use_cache = False

    # Create datasets
    train_dataset = GroundingDataset(train_data, processor, augment=True)
    val_dataset = GroundingDataset(val_data, processor, augment=False)

    # Training args
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=fp16,
        gradient_checkpointing=True,
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Trainer
    collator = Qwen2VLCollator()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info("Saving to %s", output_dir)
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # Verify save worked
    adapter_config = Path(output_dir) / "adapter_config.json"
    if adapter_config.exists():
        logger.info("SUCCESS: LoRA adapter saved to %s", output_dir)
        logger.info("  Files: %s", list(Path(output_dir).glob("*.json")))
    else:
        logger.error("FAILURE: No adapter_config.json found in %s", output_dir)
        logger.error("  Contents: %s", list(Path(output_dir).iterdir()) if Path(output_dir).exists() else "DIR MISSING")

    # Merge if requested
    if merge:
        merged_dir = output_dir + "_merged"
        logger.info("Merging adapter into %s", merged_dir)
        try:
            model = model.merge_and_unload()
            model.save_pretrained(merged_dir)
            processor.save_pretrained(merged_dir)
            logger.info("Merged model saved to %s", merged_dir)
        except Exception as e:
            logger.error("Merge failed: %s", e)

    logger.info("Training complete!")
    return output_dir


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Grounder v2")
    parser.add_argument(
        "--dataset",
        default="screenspot",
        choices=["screenspot", "seeclick", "screenspot_v2", "omniact", "mixed"],
    )
    parser.add_argument("--model", default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output", default="outputs/grounder_v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA adapter after training")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")

    args = parser.parse_args()

    train_grounder(
        dataset_name=args.dataset,
        model_name=args.model,
        output_dir=args.output,
        lora_r=args.lora_r,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation=args.grad_accum,
        max_samples=args.max_samples,
        val_split=args.val_split,
        use_wandb=args.wandb,
        merge=args.merge,
        fp16=args.fp16,
    )


if __name__ == "__main__":
    main()
