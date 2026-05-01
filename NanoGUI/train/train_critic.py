"""
Training script for the Critic (Verifier) agent — v2 with fixes.

Fixes from original:
- Correct bbox format: ScreenSpot uses [x1,y1,x2,y2] normalized, not [x,y,w,h] pixels
- Fixed paths: ./datasets/ instead of ./data/
- Prediction-aware training: uses grounder outputs, not synthetic random bboxes
- Class-balanced loss to handle imbalanced positive/negative ratios

Option A: Lightweight ResNet-18 binary classifier.
Input: Screenshot with click-point overlay
Output: Accept/Reject probability
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default image size for ResNet
IMAGE_SIZE = (224, 224)

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class CriticSample:
    """Single training sample for the Critic."""
    image_path: str
    click_x: float  # normalized [0, 1]
    click_y: float  # normalized [0, 1]
    label: int  # 1 = accept (correct), 0 = reject (incorrect)
    instruction: str = ""


class CriticClassifier(nn.Module):
    """ResNet-18 based binary classifier for action verification.

    Input: Screenshot with click point rendered as a crosshair
    Output: Logit for binary classification (accept vs reject)
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class CriticDataset(Dataset):
    """Dataset for training the Critic with click-point overlays."""

    def __init__(
        self,
        samples: List[CriticSample],
        augment: bool = False,
    ):
        self.samples = samples
        self.augment = augment

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(0.1),
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]

        image = Image.open(sample.image_path).convert("RGB")
        img_w, img_h = image.size

        # Draw click point as crosshair overlay
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)

        px = int(sample.click_x * img_w)
        py = int(sample.click_y * img_h)

        color = (0, 255, 0) if sample.label == 1 else (255, 0, 0)
        cross_size = max(8, min(img_w, img_h) // 20)

        # Draw crosshair
        draw.line([(px - cross_size, py), (px + cross_size, py)], fill=color, width=3)
        draw.line([(px, py - cross_size), (px, py + cross_size)], fill=color, width=3)
        # Draw circle around click point
        draw.ellipse(
            [(px - cross_size // 2, py - cross_size // 2),
             (px + cross_size // 2, py + cross_size // 2)],
            outline=color, width=2,
        )

        tensor = self.transform(overlay)
        return tensor, sample.label


def create_prediction_based_samples(
    predictions: List[Dict],
) -> Tuple[List[CriticSample], List[CriticSample]]:
    """Create training samples from grounder predictions (prediction-aware)."""
    positive = []
    negative = []

    for pred in predictions:
        img_path = pred.get("image_path", "")
        instruction = pred.get("instruction", "")
        pred_x, pred_y = pred.get("predicted", [0.5, 0.5])
        inside = pred.get("inside_bbox", False)

        if not img_path:
            continue

        p = Path(img_path)
        if not p.exists():
            continue

        sample = CriticSample(
            image_path=str(p),
            click_x=pred_x,
            click_y=pred_y,
            label=1 if inside else 0,
            instruction=instruction,
        )

        if inside:
            positive.append(sample)
        else:
            negative.append(sample)

    logger.info("Prediction-based samples: %d positive, %d negative", len(positive), len(negative))
    return positive, negative


def create_screenspot_samples(
    annotations_path: str,
    val_split: float = 0.15,
) -> Tuple[List[CriticSample], List[CriticSample]]:
    """Load ScreenSpot annotations and create positive samples with GT centers."""
    root = Path(__file__).resolve().parent.parent.parent

    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    positive = []
    for item in data:
        bbox = item.get("bbox")
        instruction = item.get("instruction", "")
        img_path = item.get("image_path", "")

        if not bbox or not instruction:
            continue

        # Resolve path
        p = Path(img_path.replace("\\", "/"))
        if not p.exists():
            p = root / p
        if not p.exists():
            parts = p.parts
            if "data" in parts:
                idx = parts.index("data")
                p = Path(*(parts[:idx] + ("datasets",) + parts[idx + 1:]))
        if not p.exists():
            continue

        # ScreenSpot format: [x1, y1, x2, y2] normalized [0, 1]
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        positive.append(CriticSample(
            image_path=str(p),
            click_x=center_x,
            click_y=center_y,
            label=1,
            instruction=instruction,
        ))

    # Generate negatives: random points NOT inside the bbox
    negative = []
    for pos in positive:
        for _ in range(3):  # 3 negatives per positive
            for attempt in range(10):
                rx = random.random()
                ry = random.random()
                # Make sure not inside original bbox
                bbox_x1 = pos.click_x - 0.05
                bbox_y1 = pos.click_y - 0.05
                bbox_x2 = pos.click_x + 0.05
                bbox_y2 = pos.click_y + 0.05
                if not (bbox_x1 <= rx <= bbox_x2 and bbox_y1 <= ry <= bbox_y2):
                    break

            negative.append(CriticSample(
                image_path=pos.image_path,
                click_x=rx,
                click_y=ry,
                label=0,
                instruction=pos.instruction,
            ))

    # Split
    all_samples = positive + negative
    random.shuffle(all_samples)
    n_val = int(len(all_samples) * val_split)
    return all_samples[n_val:], all_samples[:n_val]


def train_critic(
    annotations_path: str = "datasets/screenspot/annotations/train_annotations.json",
    predictions_path: Optional[str] = None,
    output_dir: str = "outputs/critic_v2",
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train the Critic binary classifier."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Critic v2 training on %s", device)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load training data
    if predictions_path:
        logger.info("Loading predictions from %s", predictions_path)
        with open(predictions_path) as f:
            predictions = json.load(f)

        pos, neg = create_prediction_based_samples(predictions)

        if not pos or not neg:
            logger.warning("Not enough prediction data, falling back to ScreenSpot GT")
            train_samples, val_samples = create_screenspot_samples(annotations_path)
        else:
            # Balance classes
            min_count = min(len(pos), len(neg))
            random.shuffle(pos)
            random.shuffle(neg)
            balanced = pos[:min_count] + neg[:min_count]
            random.shuffle(balanced)

            n_val = max(1, int(0.15 * len(balanced)))
            train_samples = balanced[n_val:]
            val_samples = balanced[:n_val]
    else:
        train_samples, val_samples = create_screenspot_samples(annotations_path)

    logger.info("Dataset: %d train, %d val", len(train_samples), len(val_samples))

    # Count classes
    train_pos = sum(1 for s in train_samples if s.label == 1)
    train_neg = sum(1 for s in train_samples if s.label == 0)
    logger.info("Train: %d positive, %d negative", train_pos, train_neg)

    # Create datasets
    train_dataset = CriticDataset(train_samples, augment=True)
    val_dataset = CriticDataset(val_samples, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = CriticClassifier(pretrained=True).to(device)

    # Class-balanced loss
    pos_weight = torch.tensor([train_neg / max(1, train_pos)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.float().to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{train_correct/train_total:.3f}"})

        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device).unsqueeze(1)

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(1, val_total)
        train_acc = train_correct / max(1, train_total)

        logger.info(
            "Epoch %d: train_loss=%.4f, train_acc=%.4f, val_loss=%.4f, val_acc=%.4f",
            epoch + 1, train_loss / len(train_loader), train_acc,
            val_loss / max(1, len(val_loader)), val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save as safetensors if possible
            save_path = output_path / "best_critic"
            try:
                from safetensors.torch import save_file
                save_file(model.state_dict(), str(save_path) + ".safetensors")
            except ImportError:
                torch.save(model.state_dict(), str(save_path) + ".pt")
            logger.info("Saved best model (val_acc=%.4f)", val_acc)

    # Save final model
    final_path = output_path / "critic_final"
    try:
        from safetensors.torch import save_file
        save_file(model.state_dict(), str(final_path) + ".safetensors")
    except ImportError:
        torch.save(model.state_dict(), str(final_path) + ".pt")

    # Save metadata
    meta = {
        "val_accuracy": best_val_acc,
        "epochs": epochs,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
    }
    with open(output_path / "critic_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Training complete! Best val_acc: %.4f", best_val_acc)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Critic v2 classifier")
    parser.add_argument("--annotations", default="datasets/screenspot/annotations/train_annotations.json")
    parser.add_argument("--predictions", default=None, help="Pre-generated predictions JSON")
    parser.add_argument("--output", default="outputs/critic_v2")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    train_critic(
        annotations_path=args.annotations,
        predictions_path=args.predictions,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
