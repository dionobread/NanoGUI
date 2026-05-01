"""
LoRA fine-tuning script for the Grounder agent.

Trains Qwen2-VL-2B on ScreenSpot dataset for GUI element grounding.
Uses coordinate regression loss (MSE on normalized x, y).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


class ScreenSpotDataset(Dataset):
    """Dataset for GUI grounding with coordinate regression."""

    def __init__(
        self,
        annotations_path: str,
        processor: Qwen2VLProcessor,
        max_length: int = 512,
    ):
        self.annotations = self._load_annotations(annotations_path)
        self.processor = processor
        self.max_length = max_length

    def _load_annotations(self, path: str) -> List[Dict]:
        """Load and validate annotations."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Filter valid samples with bbox
        valid = [s for s in data if "bbox" in s and Path(s.get("image_path", "")).exists()]
        logger.info("Loaded %d valid samples from %s", len(valid), path)
        return valid

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.annotations[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")

        # Get bbox and convert to normalized center coordinate
        bbox = sample["bbox"]  # [x, y, w, h] or [x1, y1, x2, y2]
        img_w, img_h = image.size

        # ScreenSpot format: [x1, y1, x2, y2] normalized
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
        else:
            center_x, center_y = 0.5, 0.5  # fallback

        # Format as target string
        target = f"({center_x:.4f}, {center_y:.4f})"

        # Build prompt
        instruction = sample.get("instruction", "Click on the target element.")
        prompt = f"<image>\nTask: {instruction}\nOutput the normalized coordinates (x, y) of the target element."

        # Tokenize
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )

        # Tokenize target
        labels = self.processor.tokenizer(
            target,
            return_tensors="pt",
            max_length=32,
            truncation=True,
        )["input_ids"]

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"],
            "labels": labels.squeeze(0),
            # Store ground truth for coordinate loss
            "gt_coords": torch.tensor([center_x, center_y], dtype=torch.float32),
        }


class CoordinateLossTrainer(Trainer):
    """Custom trainer with coordinate regression loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute combined cross-entropy and coordinate MSE loss."""
        # Extract ground truth coords before passing to model
        gt_coords = inputs.pop("gt_coords", None)

        # Forward pass
        outputs = model(**inputs)
        ce_loss = outputs.loss

        # Add coordinate loss if we can parse predictions
        if gt_coords is not None:
            # Get generated tokens
            generated_ids = outputs.logits.argmax(dim=-1)
            # Decode to text
            pred_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Parse coordinates from predictions
            pred_coords = []
            for text in pred_texts:
                coords = self._parse_coords(text)
                pred_coords.append(coords)

            pred_coords = torch.tensor(pred_coords, dtype=torch.float32, device=gt_coords.device)
            coord_loss = nn.functional.mse_loss(pred_coords, gt_coords)

            # Combined loss
            loss = ce_loss + 0.1 * coord_loss
        else:
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _parse_coords(text: str) -> List[float]:
        """Parse (x, y) from model output."""
        import re
        match = re.search(r"\(?(0\.\d+)[,\s]+(0\.\d+)\)?", text)
        if match:
            return [float(match.group(1)), float(match.group(2))]
        return [0.5, 0.5]  # default to center


def train_lora_grounder(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    data_path: str = "./data/screenspot/annotations/train_annotations.json",
    output_dir: str = "./outputs/lora_grounder",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
    gradient_accumulation_steps: int = 4,
    fp16: bool = True,
    use_wandb: bool = False,
):
    """
    Fine-tune Qwen2-VL-2B with LoRA on ScreenSpot.

    Args:
        model_name: Base model from HuggingFace
        data_path: Path to annotations JSON
        output_dir: Where to save checkpoints
        lora_r: LoRA rank (keep small: 8-32)
        lora_alpha: LoRA scaling
        lora_dropout: Dropout for LoRA layers
        epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Peak learning rate
        warmup_steps: Warmup for scheduler
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        gradient_accumulation_steps: Effective batch size multiplier
        fp16: Use mixed precision
        use_wandb: Enable Weights & Biases logging
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting LoRA training for Grounder agent")
    logger.info("Model: %s", model_name)
    logger.info("Output: %s", output_dir)

    # Load processor and model
    logger.info("Loading model...")
    processor = Qwen2VLProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        device_map="auto",
    )

    # Configure LoRA
    logger.info("Configuring LoRA (r=%d, alpha=%d)...", lora_r, lora_alpha)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    logger.info("Loading dataset from %s", data_path)
    train_dataset = ScreenSpotDataset(data_path, processor)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        fp16=fp16,
        gradient_checkpointing=True,
        report_to="wandb" if use_wandb else None,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues with PIL
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        processor.tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info("Saving final model to %s", output_dir)
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    logger.info("Training complete!")
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Grounder with LoRA")
    parser.add_argument("--data", default="./data/screenspot/annotations/train_annotations.json")
    parser.add_argument("--output", default="./outputs/lora_grounder")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")

    args = parser.parse_args()

    train_lora_grounder(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.r,
        use_wandb=args.wandb,
    )
