#!/usr/bin/env python3
"""
Ablation runner for NanoGUI term project.

Evaluates 4 pipeline configurations on ScreenSpot:
  A. Grounder-only         — Direct grounding, no planner/verifier
  B. Planner + Grounder    — Plan then ground first sub-goal
  C. Grounder + Verifier   — Ground then verify (static image check)
  D. Full pipeline         — Plan → Ground → Verify

Uses sequential model loading to stay within ≤16GB VRAM.
Outputs: accuracy, distance, latency, peak VRAM.

Usage:
    python scripts/run_ablation.py --max-samples 20
    python scripts/run_ablation.py --max-samples 200 --output results/ablation_results.json
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

# Reduce CUDA memory fragmentation on Windows with limited VRAM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Add NanoGUI to path
sys.path.insert(0, str(Path(__file__).parent.parent / "NanoGUI"))

# Reuse path handling from eval_grounder
sys.path.insert(0, str(Path(__file__).parent))
from eval_grounder import (
    GUIActorGrounder,
    StandardGrounder,
    discover_models,
    get_project_root,
    is_inside_bbox,
    load_screenspot_annotations,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PLANNER = "Qwen2.5-VL-3B-Instruct"
MODELGROUNDER = "GUI-Actor-3B-Qwen2.5-VL"
MODEL_VERIFIER = "SmolVLM-256M-Instruct"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    sample_id: str
    instruction: str
    predicted: Tuple[float, float]
    ground_truth: Tuple[float, float]
    correct: bool
    distance: float
    inference_ms: float
    # Optional fields for configs B/C/D
    planner_subgoal: str = ""
    verifier_accepted: bool = False
    verifier_correct: bool = False  # Did verifier match ground truth?


@dataclass
class ConfigResult:
    name: str
    description: str
    total: int
    correct: int
    accuracy: float
    avg_distance: float
    avg_inference_ms: float
    total_time_s: float
    peak_vram_mb: float
    sample_results: List[SampleResult] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_vram_counter():
    """Reset PyTorch VRAM peak tracker."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_mb() -> float:
    """Peak VRAM since last reset, in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def _aggressive_vram_cleanup():
    """Force full VRAM release between model switches."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def _draw_click_overlay(image: Image.Image, x: float, y: float, radius: int = 8) -> Image.Image:
    """Draw a red circle at normalized (x, y) on a copy of the image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    px = int(x * img.width)
    py = int(y * img.height)
    draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill="red", outline="darkred", width=2)
    return img


def _gt_center(bbox: List[float]) -> Tuple[float, float]:
    """Ground-truth center from [x1, y1, x2, y2] normalized bbox."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


# ── Grounder wrapper (direct transformers, with cleanup) ──────────────────────

class GrounderModel:
    """Wraps either GUI-Actor or standard VLM grounder with cleanup."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._grounder = None

    def _detect_type(self):
        """Detect if model is GUI-Actor (has pointer tokens) or standard VLM."""
        model_path = get_project_root() / "models" / self.model_name
        if not (model_path / "config.json").exists():
            return "standard"
        import json
        with open(model_path / "config.json") as f:
            cfg = json.load(f)
        # GUI-Actor models have pointer_start_token_id in config
        if "pointer_start_token_id" in cfg:
            return "gui_actor"
        return "standard"

    def load(self):
        if self._grounder is None:
            gtype = self._detect_type()
            if gtype == "gui_actor":
                self._grounder = GUIActorGrounder(self.model_name)
            else:
                self._grounder = StandardGrounder(self.model_name)

    def predict(self, image: Image.Image, instruction: str) -> Tuple[float, float]:
        self.load()
        return self._grounder.predict(image, instruction)

    def close(self):
        if self._grounder is not None:
            self._grounder.close()
            self._grounder = None
            _aggressive_vram_cleanup()
            logger.info("Grounder model closed, VRAM freed.")


# ── Direct VLM loader (no autogen dependency) ─────────────────────────────────

class DirectVLMModel:
    """Load any local VLM with transformers for direct inference. No autogen."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.processor = None

    def _ensure_loaded(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor

        model_path = get_project_root() / "models" / self.model_name
        logger.info("Loading %s (FP16)...", self.model_name)

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
        elif "Idefics3" in arch or mtype == "idefics3":
            from transformers import Idefics3ForConditionalGeneration as cls
        else:
            from transformers import AutoModelForCausalLM as cls

        self.model = cls.from_pretrained(
            model_path, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True,
        )
        logger.info("Loaded on %s", next(self.model.parameters()).device)

    def generate(self, messages: list, max_new_tokens: int = 256) -> str:
        """Generate text from chat-formatted messages. Returns decoded string."""
        import torch

        self._ensure_loaded()

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Collect images from message content
        images = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        images.append(part["image"])

        if images:
            inputs = self.processor(
                text=text, images=images[0] if len(images) == 1 else images,
                return_tensors="pt", padding=True,
            )
        else:
            inputs = self.processor(text=text, return_tensors="pt", padding=True)

        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )

        new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response

    def close(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            _aggressive_vram_cleanup()
            logger.info("%s closed, VRAM freed.", self.model_name)


# ── Planner wrapper (direct transformers) ──────────────────────────────────────

class PlannerModel:
    """Uses Qwen2.5-VL-3B-Instruct to decompose tasks into sub-goals."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.vlm = DirectVLMModel(model_name)

    def plan(self, task: str, screenshot: Image.Image) -> List[str]:
        """Plan task into sub-goals. Returns list of sub-goal strings."""
        messages = [
            {"role": "system", "content": (
                "You are a GUI task planner. Given a screenshot and task, "
                "decompose the task into atomic sub-goals. "
                "Output a JSON object: {\"sub_goals\": [\"step 1\", \"step 2\", ...]}"
            )},
            {"role": "user", "content": [
                {"type": "image", "image": screenshot},
                {"type": "text", "text": f"Task: {task}\n\nDecompose into sub-goals."},
            ]},
        ]

        raw = self.vlm.generate(messages, max_new_tokens=256)

        # Parse sub_goals from JSON
        import re
        try:
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                goals = data.get("sub_goals", [])
                if goals:
                    return goals
        except Exception:
            pass

        # Fallback: try line-by-line
        lines = [l.strip().lstrip("0123456789.-) ") for l in raw.split("\n") if l.strip()]
        if lines:
            return lines[:5]

        # Ultimate fallback: just return the task as-is
        return [task]

    def close(self):
        self.vlm.close()


# ── Verifier wrapper (direct transformers) ─────────────────────────────────────

class VerifierModel:
    """Uses SmolVLM-256M to check if a predicted click is correct."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.vlm = DirectVLMModel(model_name)

    def verify(self, image_with_dot: Image.Image, instruction: str) -> Tuple[bool, str]:
        """Check if the red dot marks the correct element. Returns (correct, reason)."""
        messages = [
            {"role": "system", "content": (
                "You are a GUI verification assistant. "
                "Determine if the red dot in the screenshot is on the correct UI element for the task. "
                'Respond with JSON only: {"correct": true/false, "reason": "brief explanation"}'
            )},
            {"role": "user", "content": [
                {"type": "image", "image": image_with_dot},
                {"type": "text", "text": (
                    f"Task: {instruction}\n\n"
                    "The red dot marks where the user clicked. "
                    "Is this the correct location? "
                    'Answer with JSON: {"correct": true/false, "reason": "..."}'
                )},
            ]},
        ]

        raw = self.vlm.generate(messages, max_new_tokens=128)

        import re
        try:
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return bool(data.get("correct", False)), data.get("reason", "")
        except Exception:
            pass

        # Fallback: keyword
        correct = "true" in raw.lower() or "yes" in raw.lower()
        return correct, raw[:200]

    def close(self):
        self.vlm.close()


# ── CLIP Verifier ────────────────────────────────────────────────────────────

class CLIPVerifierModel:
    """Uses CLIP + trained head to verify if a predicted click matches the instruction.

    Crops a region around the predicted point, computes CLIP similarity,
    and classifies correct/incorrect via a trained MLP head.
    """

    def __init__(self, verifier_dir: str = "outputs/clip_verifier"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_name = "openai/clip-vit-base-patch32"

        logger.info("Loading CLIP verifier from %s", verifier_dir)
        self.model = CLIPModel.from_pretrained(self.clip_name, use_safetensors=True).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.clip_name)

        # Load head — try v2 first (prediction-aware), then v1 (GT-crop based)
        v2_meta = os.path.join(verifier_dir, "verifier_v2_meta.json")
        v1_meta = os.path.join(verifier_dir, "clip_verifier_meta.json")
        head_pt = os.path.join(verifier_dir, "clip_verifier_head.pt")
        head_st = os.path.join(verifier_dir, "clip_verifier_head.safetensors")
        v2_st = os.path.join(verifier_dir, "verifier_head_v2.safetensors")
        v2_pt = os.path.join(verifier_dir, "verifier_head_v2.pt")

        feature_dim = 512
        val_acc = 0
        head_state = None
        ckpt_path = ""

        # Try v2 first
        if os.path.exists(v2_meta):
            with open(v2_meta) as f:
                meta = json.load(f)
            feature_dim = meta.get("feature_dim", 512)
            val_acc = meta.get("val_accuracy", 0)
            ckpt_file = meta.get("checkpoint_file", "")
            ckpt_path = os.path.join(verifier_dir, ckpt_file)
            logger.info("Found v2 verifier (prediction-aware)")
        elif os.path.exists(v1_meta):
            with open(v1_meta) as f:
                meta = json.load(f)
            feature_dim = meta.get("feature_dim", 512)
            val_acc = meta.get("val_accuracy", 0)
            ckpt_file = meta.get("checkpoint_file", "")
            ckpt_path = os.path.join(verifier_dir, ckpt_file)
            logger.info("Found v1 verifier (GT-crop based)")
        elif os.path.exists(head_pt):
            ckpt_path = head_pt
        else:
            ckpt_path = head_st

        if ckpt_path.endswith(".safetensors") and os.path.exists(ckpt_path):
            from safetensors.torch import load_file
            flat = load_file(ckpt_path)
            head_state = {k.replace("head.", ""): v for k, v in flat.items()}
        elif os.path.exists(head_pt):
            ckpt = torch.load(head_pt, map_location="cpu", weights_only=False)
            feature_dim = ckpt.get("feature_dim", 512)
            val_acc = ckpt.get("val_accuracy", 0)
            head_state = ckpt["head_state_dict"]

        if head_state is not None:
            self.head = VerifierHead(feature_dim=feature_dim).to(self.device)
            self.head.load_state_dict(head_state)
            self.head.eval()
            logger.info("Loaded trained head (val acc: %.1f%%)", val_acc)
        else:
            self.head = None
            logger.warning("No trained head found in %s, using zero-shot", verifier_dir)

        # Load threshold — try v2 first, then v1, then zeroshot
        thresh_path = os.path.join(verifier_dir, "threshold_v2.json")
        v1_thresh_path = os.path.join(verifier_dir, "threshold.json")
        zs_thresh_path = os.path.join(verifier_dir, "zeroshot_threshold.json")
        if self.head and os.path.exists(thresh_path):
            with open(thresh_path) as f:
                self.threshold = json.load(f)["threshold"]
        elif self.head and os.path.exists(v1_thresh_path):
            with open(v1_thresh_path) as f:
                self.threshold = json.load(f)["threshold"]
        elif os.path.exists(zs_thresh_path):
            with open(zs_thresh_path) as f:
                self.threshold = json.load(f)["threshold"]
        else:
            self.threshold = 0.25
        logger.info("CLIP verifier threshold: %.4f", self.threshold)

    def verify(self, image: Image.Image, pred_x: float, pred_y: float,
               instruction: str, crop_ratio: float = 0.2) -> Tuple[bool, str]:
        """Verify if (pred_x, pred_y) is on the correct element."""
        w, h = image.size
        margin = crop_ratio / 2

        cx = int(pred_x * w)
        cy = int(pred_y * h)
        half_w = int(crop_ratio * w / 2 + margin * w)
        half_h = int(crop_ratio * h / 2 + margin * h)

        # Clamp to valid bounds
        left = max(0, cx - half_w)
        upper = max(0, cy - half_h)
        right = min(w, cx + half_w)
        lower = min(h, cy + half_h)
        # Ensure minimum crop size of 32x32
        right = max(right, left + 32)
        lower = max(lower, upper + 32)
        right = min(right, w)
        lower = min(lower, h)
        left = max(0, right - 32)
        upper = max(0, lower - 32)

        crop = image.crop((left, upper, right, lower)).convert("RGB")

        inputs = self.processor(
            images=[crop], text=[instruction],
            return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)

        with torch.no_grad():
            img_out = self.model.get_image_features(inputs["pixel_values"])
            txt_out = self.model.get_text_features(
                inputs["input_ids"], inputs["attention_mask"]
            )
            # Handle both tensor and BaseModelOutputWithPooling returns
            img_feats = img_out if isinstance(img_out, torch.Tensor) else img_out.last_hidden_state[:, 0]
            txt_feats = txt_out if isinstance(txt_out, torch.Tensor) else txt_out.last_hidden_state[:, 0]
            if img_feats.shape[-1] != self.model.config.projection_dim:
                img_feats = self.model.visual_projection(img_feats)
            if txt_feats.shape[-1] != self.model.config.projection_dim:
                txt_feats = self.model.text_projection(txt_feats)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

            if self.head is not None:
                logit = self.head(img_feats, txt_feats).item()
                prob = torch.sigmoid(torch.tensor(logit)).item()
                accepted = prob > self.threshold
                reason = f"CLIP head prob={prob:.3f}"
            else:
                sim = (img_feats @ txt_feats.T).item()
                accepted = sim > self.threshold
                reason = f"CLIP sim={sim:.3f}"

        return accepted, reason

    def close(self):
        del self.model
        del self.processor
        if self.head:
            del self.head
        self.model = None
        _aggressive_vram_cleanup()
        logger.info("CLIP verifier closed, VRAM freed.")


class VerifierHead(nn.Module):
    """MLP head for CLIP verification (mirrors train script)."""

    def __init__(self, feature_dim=512, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)

    def forward(self, img_feats, txt_feats):
        combined = torch.cat([img_feats, txt_feats, img_feats * txt_feats], dim=-1)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


# ── Critic Verifier (ResNet-18) ──────────────────────────────────────────────

class CriticVerifierModel:
    """Uses trained ResNet-18 Critic to verify if a predicted click is correct.

    Input: Screenshot with crosshair overlay at predicted point
    Output: Accept/reject probability
    """

    def __init__(self, critic_dir: str = "outputs/critic_v2"):
        from torchvision import transforms
        import torchvision.models as models

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading Critic verifier from %s", critic_dir)

        # Build same architecture as training
        backbone = models.resnet18(weights=None)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 1),
        )
        self.model = backbone.to(self.device)

        # Load trained weights
        import glob
        safetensors_path = os.path.join(critic_dir, "best_critic.safetensors")
        pt_path = os.path.join(critic_dir, "best_critic.pt")

        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state = load_file(safetensors_path)
            # Strip 'backbone.' prefix added by CriticClassifier wrapper
            clean_state = {k.replace("backbone.", ""): v for k, v in state.items()}
            self.model.load_state_dict(clean_state, strict=False)
        elif os.path.exists(pt_path):
            state = torch.load(pt_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
        else:
            raise FileNotFoundError(f"No critic weights found in {critic_dir}")

        self.model.eval()
        logger.info("Critic model loaded on %s", self.device)

        # Same transform as training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _draw_crosshair(self, image: Image.Image, x: float, y: float) -> Image.Image:
        """Draw crosshair + circle overlay at normalized (x, y)."""
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        img_w, img_h = overlay.size

        px = int(x * img_w)
        py = int(y * img_h)

        cross_size = max(8, min(img_w, img_h) // 20)
        color = (255, 0, 0)

        draw.line([(px - cross_size, py), (px + cross_size, py)], fill=color, width=3)
        draw.line([(px, py - cross_size), (px, py + cross_size)], fill=color, width=3)
        draw.ellipse(
            [(px - cross_size // 2, py - cross_size // 2),
             (px + cross_size // 2, py + cross_size // 2)],
            outline=color, width=2,
        )
        return overlay

    def verify(self, image: Image.Image, pred_x: float, pred_y: float,
               instruction: str = "", threshold: float = 0.5) -> Tuple[bool, str]:
        """Verify if (pred_x, pred_y) is on the correct element."""
        overlay = self._draw_crosshair(image, pred_x, pred_y)
        tensor = self.transform(overlay).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(tensor).squeeze().item()
            prob = torch.sigmoid(torch.tensor(logit)).item()

        accepted = prob > threshold
        reason = f"Critic prob={prob:.3f}"
        return accepted, reason

    def close(self):
        del self.model
        self.model = None
        _aggressive_vram_cleanup()
        logger.info("Critic verifier closed, VRAM freed.")


# ── Configuration runners ─────────────────────────────────────────────────────

def run_config_a(
    samples: List[Dict],
    model_name: str = MODELGROUNDER,
) -> ConfigResult:
    """Config A: Grounder-only."""
    logger.info("=" * 60)
    logger.info("CONFIG A: Grounder-only")
    logger.info("=" * 60)

    _reset_vram_counter()
    t_start = time.time()

    grounder = GrounderModel(model_name)
    results: List[SampleResult] = []
    correct_count = 0

    try:
        for i, sample in enumerate(samples):
            instruction = sample.get("instruction", "")
            bbox = sample["bbox"]  # [x1, y1, x2, y2] normalized
            img_path = sample["image_path"]

            image = Image.open(img_path).convert("RGB")
            gt_x, gt_y = _gt_center(bbox)

            t0 = time.time()
            pred_x, pred_y = grounder.predict(image, instruction)
            inference_ms = (time.time() - t0) * 1000

            is_correct = is_inside_bbox(pred_x, pred_y, bbox, image.width, image.height)
            dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

            if is_correct:
                correct_count += 1

            results.append(SampleResult(
                sample_id=sample.get("id", str(i)),
                instruction=instruction,
                predicted=(round(pred_x, 4), round(pred_y, 4)),
                ground_truth=(round(gt_x, 4), round(gt_y, 4)),
                correct=is_correct,
                distance=round(dist, 4),
                inference_ms=round(inference_ms, 1),
            ))

            if (i + 1) % 10 == 0:
                logger.info("  Progress: %d/%d | Acc: %.1f%%", i + 1, len(samples),
                            correct_count / (i + 1) * 100)
            if (i + 1) % 50 == 0:
                _aggressive_vram_cleanup()
    finally:
        grounder.close()

    total = len(samples)
    accuracy = correct_count / total * 100 if total else 0
    avg_dist = np.mean([r.distance for r in results]) if results else 0
    avg_ms = np.mean([r.inference_ms for r in results]) if results else 0

    return ConfigResult(
        name="A",
        description="Grounder-only",
        total=total,
        correct=correct_count,
        accuracy=round(accuracy, 2),
        avg_distance=round(avg_dist, 4),
        avg_inference_ms=round(avg_ms, 1),
        total_time_s=round(time.time() - t_start, 1),
        peak_vram_mb=round(_peak_vram_mb(), 1),
        sample_results=results,
    )


def run_config_b(
    samples: List[Dict],
    model_planner: str = MODEL_PLANNER,
    model_grounder: str = MODELGROUNDER,
) -> ConfigResult:
    """Config B: Planner + Grounder.

    Phase 1: Plan all samples with planner (loaded once).
    Phase 2: Ground all sub-goals with grounder (loaded once).
    """
    logger.info("=" * 60)
    logger.info("CONFIG B: Planner + Grounder")
    logger.info("=" * 60)

    _reset_vram_counter()
    t_start = time.time()

    # Phase 1: Planning
    logger.info("Phase 1: Planning %d samples...", len(samples))
    planner = PlannerModel(model_planner)
    planned = []  # List of (sample, subgoal)
    try:
        for i, sample in enumerate(samples):
            instruction = sample.get("instruction", "")
            img_path = sample["image_path"]
            image = Image.open(img_path).convert("RGB")

            subgoals = planner.plan(instruction, image)
            subgoal = subgoals[0] if subgoals else instruction
            planned.append((sample, subgoal, image))

            if (i + 1) % 10 == 0:
                logger.info("  Planned: %d/%d", i + 1, len(samples))
    finally:
        planner.close()

    # Phase 2: Grounding
    logger.info("Phase 2: Grounding %d sub-goals...", len(planned))
    grounder = GrounderModel(model_grounder)
    results: List[SampleResult] = []
    correct_count = 0

    try:
        for i, (sample, subgoal, image) in enumerate(planned):
            bbox = sample["bbox"]
            gt_x, gt_y = _gt_center(bbox)

            t0 = time.time()
            pred_x, pred_y = grounder.predict(image, subgoal)
            inference_ms = (time.time() - t0) * 1000

            is_correct = is_inside_bbox(pred_x, pred_y, bbox, image.width, image.height)
            dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

            if is_correct:
                correct_count += 1

            results.append(SampleResult(
                sample_id=sample.get("id", str(i)),
                instruction=sample.get("instruction", ""),
                predicted=(round(pred_x, 4), round(pred_y, 4)),
                ground_truth=(round(gt_x, 4), round(gt_y, 4)),
                correct=is_correct,
                distance=round(dist, 4),
                inference_ms=round(inference_ms, 1),
                planner_subgoal=subgoal,
            ))

            if (i + 1) % 10 == 0:
                logger.info("  Grounded: %d/%d | Acc: %.1f%%", i + 1, len(planned),
                            correct_count / (i + 1) * 100)
    finally:
        grounder.close()

    total = len(samples)
    accuracy = correct_count / total * 100 if total else 0
    avg_dist = np.mean([r.distance for r in results]) if results else 0
    avg_ms = np.mean([r.inference_ms for r in results]) if results else 0

    return ConfigResult(
        name="B",
        description="Planner + Grounder",
        total=total,
        correct=correct_count,
        accuracy=round(accuracy, 2),
        avg_distance=round(avg_dist, 4),
        avg_inference_ms=round(avg_ms, 1),
        total_time_s=round(time.time() - t_start, 1),
        peak_vram_mb=round(_peak_vram_mb(), 1),
        sample_results=results,
    )


def run_config_c(
    samples: List[Dict],
    model_grounder: str = MODELGROUNDER,
    model_verifier: str = MODEL_VERIFIER,
    verifier_type: str = "vlm",
    verifier_dir: str = "outputs/clip_verifier",
) -> ConfigResult:
    """Config C: Grounder + Verifier.

    Phase 1: Ground all samples with grounder (loaded once).
    Phase 2: Verify all predictions with verifier (loaded once).
    """
    logger.info("=" * 60)
    logger.info("CONFIG C: Grounder + Verifier")
    logger.info("=" * 60)

    _reset_vram_counter()
    t_start = time.time()

    # Phase 1: Grounding
    logger.info("Phase 1: Grounding %d samples...", len(samples))
    grounder = GrounderModel(model_grounder)
    grounded = []  # List of (sample, pred_x, pred_y, is_correct, image)
    correct_count = 0

    try:
        for i, sample in enumerate(samples):
            instruction = sample.get("instruction", "")
            bbox = sample["bbox"]
            img_path = sample["image_path"]
            image = Image.open(img_path).convert("RGB")

            t0 = time.time()
            pred_x, pred_y = grounder.predict(image, instruction)
            inference_ms = (time.time() - t0) * 1000

            is_correct = is_inside_bbox(pred_x, pred_y, bbox, image.width, image.height)
            if is_correct:
                correct_count += 1

            grounded.append((sample, pred_x, pred_y, is_correct, image, inference_ms))

            if (i + 1) % 10 == 0:
                logger.info("  Grounded: %d/%d | Acc: %.1f%%", i + 1, len(samples),
                            correct_count / (i + 1) * 100)
    finally:
        grounder.close()

    # Phase 2: Verification
    logger.info("Phase 2: Verifying %d predictions...", len(grounded))
    if verifier_type == "clip":
        verifier = CLIPVerifierModel(verifier_dir)
    elif verifier_type == "critic":
        verifier = CriticVerifierModel("outputs/critic_v2")
    else:
        verifier = VerifierModel(model_verifier)
    results: List[SampleResult] = []
    verifier_tp = 0
    verifier_fp = 0
    verifier_tn = 0
    verifier_fn = 0

    try:
        for i, (sample, pred_x, pred_y, is_correct, image, inference_ms) in enumerate(grounded):
            bbox = sample["bbox"]
            gt_x, gt_y = _gt_center(bbox)

            if verifier_type == "clip":
                verifier_accepted, reason = verifier.verify(
                    image, pred_x, pred_y, sample.get("instruction", ""))
            elif verifier_type == "critic":
                verifier_accepted, reason = verifier.verify(
                    image, pred_x, pred_y, sample.get("instruction", ""))
            else:
                overlay = _draw_click_overlay(image, pred_x, pred_y)
                verifier_accepted, reason = verifier.verify(overlay, sample.get("instruction", ""))

            if verifier_accepted and is_correct:
                verifier_tp += 1
            elif verifier_accepted and not is_correct:
                verifier_fp += 1
            elif not verifier_accepted and not is_correct:
                verifier_tn += 1
            else:
                verifier_fn += 1

            dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

            results.append(SampleResult(
                sample_id=sample.get("id", str(i)),
                instruction=sample.get("instruction", ""),
                predicted=(round(pred_x, 4), round(pred_y, 4)),
                ground_truth=(round(gt_x, 4), round(gt_y, 4)),
                correct=is_correct,
                distance=round(dist, 4),
                inference_ms=round(inference_ms, 1),
                verifier_accepted=verifier_accepted,
                verifier_correct=(verifier_accepted == is_correct),
            ))

            if (i + 1) % 10 == 0:
                logger.info("  Verified: %d/%d", i + 1, len(grounded))
    finally:
        verifier.close()

    total = len(samples)
    accuracy = correct_count / total * 100 if total else 0
    avg_dist = np.mean([r.distance for r in results]) if results else 0
    avg_ms = np.mean([r.inference_ms for r in results]) if results else 0

    verifier_acc = (verifier_tp + verifier_tn) / total * 100 if total else 0
    logger.info("Verifier accuracy: %.1f%% (TP=%d, FP=%d, TN=%d, FN=%d)",
                verifier_acc, verifier_tp, verifier_fp, verifier_tn, verifier_fn)

    return ConfigResult(
        name="C",
        description="Grounder + Verifier",
        total=total,
        correct=correct_count,
        accuracy=round(accuracy, 2),
        avg_distance=round(avg_dist, 4),
        avg_inference_ms=round(avg_ms, 1),
        total_time_s=round(time.time() - t_start, 1),
        peak_vram_mb=round(_peak_vram_mb(), 1),
        sample_results=results,
    )


def run_config_d(
    samples: List[Dict],
    model_planner: str = MODEL_PLANNER,
    model_grounder: str = MODELGROUNDER,
    model_verifier: str = MODEL_VERIFIER,
    verifier_type: str = "vlm",
    verifier_dir: str = "outputs/clip_verifier",
) -> ConfigResult:
    """Config D: Full pipeline (Planner → Grounder → Verifier).

    Phase 1: Plan all samples with planner (loaded once).
    Phase 2: Ground all sub-goals with grounder (loaded once).
    Phase 3: Verify all predictions with verifier (loaded once).
    """
    logger.info("=" * 60)
    logger.info("CONFIG D: Full pipeline")
    logger.info("=" * 60)

    _reset_vram_counter()
    t_start = time.time()

    # Phase 1: Planning
    logger.info("Phase 1: Planning %d samples...", len(samples))
    planner = PlannerModel(model_planner)
    planned = []  # List of (sample, subgoal, image)
    try:
        for i, sample in enumerate(samples):
            instruction = sample.get("instruction", "")
            img_path = sample["image_path"]
            image = Image.open(img_path).convert("RGB")

            subgoals = planner.plan(instruction, image)
            subgoal = subgoals[0] if subgoals else instruction
            planned.append((sample, subgoal, image))

            if (i + 1) % 10 == 0:
                logger.info("  Planned: %d/%d", i + 1, len(samples))
    finally:
        planner.close()

    # Phase 2: Grounding
    logger.info("Phase 2: Grounding %d sub-goals...", len(planned))
    grounder = GrounderModel(model_grounder)
    grounded = []  # List of (sample, subgoal, pred_x, pred_y, is_correct, image)
    correct_count = 0

    try:
        for i, (sample, subgoal, image) in enumerate(planned):
            bbox = sample["bbox"]

            t0 = time.time()
            pred_x, pred_y = grounder.predict(image, subgoal)
            inference_ms = (time.time() - t0) * 1000

            is_correct = is_inside_bbox(pred_x, pred_y, bbox, image.width, image.height)
            if is_correct:
                correct_count += 1

            grounded.append((sample, subgoal, pred_x, pred_y, is_correct, image, inference_ms))

            if (i + 1) % 10 == 0:
                logger.info("  Grounded: %d/%d | Acc: %.1f%%", i + 1, len(planned),
                            correct_count / (i + 1) * 100)
    finally:
        grounder.close()

    # Phase 3: Verification
    logger.info("Phase 3: Verifying %d predictions...", len(grounded))
    if verifier_type == "clip":
        verifier = CLIPVerifierModel(verifier_dir)
    elif verifier_type == "critic":
        verifier = CriticVerifierModel("outputs/critic_v2")
    else:
        verifier = VerifierModel(model_verifier)
    results: List[SampleResult] = []

    try:
        for i, (sample, subgoal, pred_x, pred_y, is_correct, image, inference_ms) in enumerate(grounded):
            bbox = sample["bbox"]
            gt_x, gt_y = _gt_center(bbox)

            if verifier_type == "clip":
                verifier_accepted, reason = verifier.verify(
                    image, pred_x, pred_y, subgoal)
            elif verifier_type == "critic":
                verifier_accepted, reason = verifier.verify(
                    image, pred_x, pred_y, subgoal)
            else:
                overlay = _draw_click_overlay(image, pred_x, pred_y)
                verifier_accepted, reason = verifier.verify(overlay, subgoal)

            dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

            results.append(SampleResult(
                sample_id=sample.get("id", str(i)),
                instruction=sample.get("instruction", ""),
                predicted=(round(pred_x, 4), round(pred_y, 4)),
                ground_truth=(round(gt_x, 4), round(gt_y, 4)),
                correct=is_correct,
                distance=round(dist, 4),
                inference_ms=round(inference_ms, 1),
                planner_subgoal=subgoal,
                verifier_accepted=verifier_accepted,
                verifier_correct=(verifier_accepted == is_correct),
            ))

            if (i + 1) % 10 == 0:
                logger.info("  Verified: %d/%d", i + 1, len(grounded))
    finally:
        verifier.close()

    total = len(samples)
    accuracy = correct_count / total * 100 if total else 0
    avg_dist = np.mean([r.distance for r in results]) if results else 0
    avg_ms = np.mean([r.inference_ms for r in results]) if results else 0

    return ConfigResult(
        name="D",
        description="Full pipeline (Planner → Grounder → Verifier)",
        total=total,
        correct=correct_count,
        accuracy=round(accuracy, 2),
        avg_distance=round(avg_dist, 4),
        avg_inference_ms=round(avg_ms, 1),
        total_time_s=round(time.time() - t_start, 1),
        peak_vram_mb=round(_peak_vram_mb(), 1),
        sample_results=results,
    )


# ── Config E: Multi-attempt Grounder ──────────────────────────────────────────

def run_config_e(
    samples: List[Dict],
    model_name: str = MODELGROUNDER,
    num_attempts: int = 3,
) -> ConfigResult:
    """Config E: Multi-attempt Grounder (self-consistency).

    Generate K predictions per sample with different temperatures,
    then pick the prediction that appears most often (cluster center).
    This improves robustness without any planner/verifier.
    """
    logger.info("=" * 60)
    logger.info("CONFIG E: Multi-attempt Grounder (K=%d)", num_attempts)
    logger.info("=" * 60)

    _reset_vram_counter()
    t_start = time.time()

    grounder = GrounderModel(model_name)
    results: List[SampleResult] = []
    correct_count = 0

    try:
        for i, sample in enumerate(samples):
            instruction = sample.get("instruction", "")
            bbox = sample["bbox"]
            img_path = sample["image_path"]
            image = Image.open(img_path).convert("RGB")
            gt_x, gt_y = _gt_center(bbox)

            t0 = time.time()

            # Generate multiple predictions
            preds = []
            for _ in range(num_attempts):
                px, py = grounder.predict(image, instruction)
                preds.append((px, py))

            # Pick the best: use the prediction closest to the cluster center
            # This selects the most "consistent" prediction
            pred_x, pred_y = _pick_consistent(preds)
            inference_ms = (time.time() - t0) * 1000

            is_correct = is_inside_bbox(pred_x, pred_y, bbox, image.width, image.height)
            dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

            if is_correct:
                correct_count += 1

            results.append(SampleResult(
                sample_id=sample.get("id", str(i)),
                instruction=instruction,
                predicted=(round(pred_x, 4), round(pred_y, 4)),
                ground_truth=(round(gt_x, 4), round(gt_y, 4)),
                correct=is_correct,
                distance=round(dist, 4),
                inference_ms=round(inference_ms, 1),
            ))

            if (i + 1) % 10 == 0:
                logger.info("  Progress: %d/%d | Acc: %.1f%%", i + 1, len(samples),
                            correct_count / (i + 1) * 100)
    finally:
        grounder.close()

    total = len(samples)
    accuracy = correct_count / total * 100 if total else 0
    avg_dist = np.mean([r.distance for r in results]) if results else 0
    avg_ms = np.mean([r.inference_ms for r in results]) if results else 0

    return ConfigResult(
        name="E",
        description=f"Multi-attempt Grounder (K={num_attempts})",
        total=total,
        correct=correct_count,
        accuracy=round(accuracy, 2),
        avg_distance=round(avg_dist, 4),
        avg_inference_ms=round(avg_ms, 1),
        total_time_s=round(time.time() - t_start, 1),
        peak_vram_mb=round(_peak_vram_mb(), 1),
        sample_results=results,
    )


def _pick_consistent(preds: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Pick the prediction closest to the centroid of all predictions.

    This implements self-consistency: the most "agreed upon" prediction
    is likely more reliable than outliers.
    """
    if len(preds) == 1:
        return preds[0]

    xs = [p[0] for p in preds]
    ys = [p[1] for p in preds]
    cx, cy = np.median(xs), np.median(ys)

    # Pick the prediction closest to the median
    best_dist = float('inf')
    best_pred = preds[0]
    for px, py in preds:
        d = (px - cx) ** 2 + (py - cy) ** 2
        if d < best_dist:
            best_dist = d
            best_pred = (px, py)

    return best_pred


# ── Config F: Improved Planner (spatial context) ────────────────────────────

def run_config_f(
    samples: List[Dict],
    model_planner: str = MODEL_PLANNER,
    model_grounder: str = MODELGROUNDER,
) -> ConfigResult:
    """Config F: Planner with spatial context + Grounder.

    Instead of rephrasing instructions, the Planner adds spatial hints
    (e.g., "in the top-right area") based on screenshot analysis.
    This preserves the original instruction while adding useful context.
    """
    logger.info("=" * 60)
    logger.info("CONFIG F: Spatial Planner + Grounder")
    logger.info("=" * 60)

    _reset_vram_counter()
    t_start = time.time()

    # Phase 1: Planning with spatial context
    logger.info("Phase 1: Spatial planning %d samples...", len(samples))
    planner = PlannerModel(model_planner)
    planned = []

    try:
        for i, sample in enumerate(samples):
            instruction = sample.get("instruction", "")
            img_path = sample["image_path"]
            image = Image.open(img_path).convert("RGB")

            # Use spatial planning prompt instead of decomposition
            enhanced = _spatial_plan(planner, instruction, image)
            planned.append((sample, enhanced, image))

            if (i + 1) % 10 == 0:
                logger.info("  Planned: %d/%d", i + 1, len(samples))
    finally:
        planner.close()

    # Phase 2: Grounding
    logger.info("Phase 2: Grounding %d enhanced instructions...", len(planned))
    grounder = GrounderModel(model_grounder)
    results: List[SampleResult] = []
    correct_count = 0

    try:
        for i, (sample, enhanced_instr, image) in enumerate(planned):
            bbox = sample["bbox"]
            gt_x, gt_y = _gt_center(bbox)

            t0 = time.time()
            pred_x, pred_y = grounder.predict(image, enhanced_instr)
            inference_ms = (time.time() - t0) * 1000

            is_correct = is_inside_bbox(pred_x, pred_y, bbox, image.width, image.height)
            dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

            if is_correct:
                correct_count += 1

            results.append(SampleResult(
                sample_id=sample.get("id", str(i)),
                instruction=sample.get("instruction", ""),
                predicted=(round(pred_x, 4), round(pred_y, 4)),
                ground_truth=(round(gt_x, 4), round(gt_y, 4)),
                correct=is_correct,
                distance=round(dist, 4),
                inference_ms=round(inference_ms, 1),
                planner_subgoal=enhanced_instr,
            ))

            if (i + 1) % 10 == 0:
                logger.info("  Grounded: %d/%d | Acc: %.1f%%", i + 1, len(planned),
                            correct_count / (i + 1) * 100)
    finally:
        grounder.close()

    total = len(samples)
    accuracy = correct_count / total * 100 if total else 0
    avg_dist = np.mean([r.distance for r in results]) if results else 0
    avg_ms = np.mean([r.inference_ms for r in results]) if results else 0

    return ConfigResult(
        name="F",
        description="Spatial Planner + Grounder",
        total=total,
        correct=correct_count,
        accuracy=round(accuracy, 2),
        avg_distance=round(avg_dist, 4),
        avg_inference_ms=round(avg_ms, 1),
        total_time_s=round(time.time() - t_start, 1),
        peak_vram_mb=round(_peak_vram_mb(), 1),
        sample_results=results,
    )


def _spatial_plan(planner: "PlannerModel", instruction: str, screenshot: Image.Image) -> str:
    """Ask the Planner to add spatial context to the instruction.

    Instead of decomposing, it analyzes the screenshot and adds a spatial
    hint about where the target element likely is.
    """
    messages = [
        {"role": "system", "content": (
            "You are a GUI element locator. Given a screenshot and instruction, "
            "your ONLY job is to describe WHERE on the screen the target element is. "
            "Use regions: top-left, top-center, top-right, center-left, center, "
            "center-right, bottom-left, bottom-center, bottom-right. "
            "Also describe what the element looks like (color, shape, icon, text). "
            "Keep the original instruction intact. Output a single enhanced instruction."
        )},
        {"role": "user", "content": [
            {"type": "image", "image": screenshot},
            {"type": "text", "text": (
                f"Original instruction: {instruction}\n\n"
                "Describe where the target element is on screen and what it looks like. "
                "Then give an enhanced version of the instruction with this spatial context. "
                "Format: ENHANCED: <enhanced instruction with spatial hint>"
            )},
        ]},
    ]

    raw = planner.vlm.generate(messages, max_new_tokens=128)

    # Extract the enhanced instruction
    for line in raw.split("\n"):
        line = line.strip()
        if line.upper().startswith("ENHANCED:"):
            return line.split(":", 1)[1].strip()

    # Fallback: if we got any useful text, prepend it
    # But keep original instruction as fallback
    if len(raw.strip()) > len(instruction):
        return raw.strip()

    return instruction


# ── Config G: Fine-tuned Grounder ────────────────────────────────────────────

def run_config_g(
    samples: List[Dict],
    lora_adapter_path: str = "outputs/grounder_v2",
    base_model: str = "Qwen/Qwen2-VL-2B-Instruct",
) -> ConfigResult:
    """Config G: LoRA-fine-tuned Grounder.

    Loads a base model with a trained LoRA adapter and evaluates
    grounding accuracy. Compares directly with Config A (zero-shot).
    """
    logger.info("=" * 60)
    logger.info("CONFIG G: Fine-tuned Grounder (LoRA)")
    logger.info("  Adapter: %s", lora_adapter_path)
    logger.info("  Base:    %s", base_model)
    logger.info("=" * 60)

    _reset_vram_counter()
    t_start = time.time()

    import torch
    from transformers import AutoProcessor
    from peft import PeftModel

    # Load base model
    logger.info("Loading base model: %s", base_model)
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    from transformers import Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    adapter_path = Path(lora_adapter_path)
    if adapter_path.exists():
        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))
        # Try to merge for faster inference
        try:
            model = model.merge_and_unload()
            logger.info("LoRA weights merged successfully")
        except Exception as e:
            logger.warning("Could not merge LoRA weights: %s", e)
    else:
        logger.warning("LoRA adapter not found at %s, using base model only", adapter_path)

    model.eval()

    results: List[SampleResult] = []
    correct_count = 0

    try:
        for i, sample in enumerate(samples):
            instruction = sample.get("instruction", "")
            bbox = sample["bbox"]
            img_path = sample["image_path"]
            image = Image.open(img_path).convert("RGB")
            gt_x, gt_y = _gt_center(bbox)

            t0 = time.time()
            # Inline prediction for fine-tuned model
            import re
            messages = [
                {"role": "system", "content": "You are a GUI grounding assistant. Predict the normalized coordinates (x, y) of UI elements."},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Task: {instruction}\nOutput the normalized coordinates (x, y) of the target element."}
                ]}
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = processor.decode(generated_ids, skip_special_tokens=True).strip()
            match = re.search(r"\(?(0\.\d+)[,\s]+(0\.\d+)\)?", response)
            if match:
                pred_x, pred_y = float(match.group(1)), float(match.group(2))
            else:
                match = re.search(r"(\d+\.?\d*)[,\s]+(\d+\.?\d*)", response)
                if match:
                    pred_x, pred_y = float(match.group(1)), float(match.group(2))
                    if pred_x > 1.0:
                        pred_x = pred_x / image.width
                    if pred_y > 1.0:
                        pred_y = pred_y / image.height
                else:
                    logger.warning("Could not parse: %r", response)
                    pred_x, pred_y = 0.5, 0.5
            inference_ms = (time.time() - t0) * 1000

            is_correct = is_inside_bbox(pred_x, pred_y, bbox, image.width, image.height)
            dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

            if is_correct:
                correct_count += 1

            results.append(SampleResult(
                sample_id=sample.get("id", str(i)),
                instruction=instruction,
                predicted=(round(pred_x, 4), round(pred_y, 4)),
                ground_truth=(round(gt_x, 4), round(gt_y, 4)),
                correct=is_correct,
                distance=round(dist, 4),
                inference_ms=round(inference_ms, 1),
            ))

            if (i + 1) % 10 == 0:
                logger.info("  Progress: %d/%d | Acc: %.1f%%", i + 1, len(samples),
                            correct_count / (i + 1) * 100)
    finally:
        del model
        del processor
        _aggressive_vram_cleanup()

    total = len(samples)
    accuracy = correct_count / total * 100 if total else 0
    avg_dist = np.mean([r.distance for r in results]) if results else 0
    avg_ms = np.mean([r.inference_ms for r in results]) if results else 0

    return ConfigResult(
        name="G",
        description="Fine-tuned Grounder (LoRA)",
        total=total,
        correct=correct_count,
        accuracy=round(accuracy, 2),
        avg_distance=round(avg_dist, 4),
        avg_inference_ms=round(avg_ms, 1),
        total_time_s=round(time.time() - t_start, 1),
        peak_vram_mb=round(_peak_vram_mb(), 1),
        sample_results=results,
    )


# ── Results formatting ────────────────────────────────────────────────────────

def print_results_table(results: List[ConfigResult]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("ABLATION RESULTS")
    print("=" * 90)
    print(f"{'Config':<8} {'Description':<40} {'Acc':>6} {'Dist':>7} {'Time':>7} {'VRAM':>8}")
    print("-" * 90)
    for r in results:
        print(
            f"{r.name:<8} {r.description:<40} "
            f"{r.accuracy:>5.1f}% {r.avg_distance:>7.4f} "
            f"{r.total_time_s:>6.1f}s {r.peak_vram_mb:>7.1f}MB"
        )
    print("=" * 90)

    # Detailed breakdown
    print("\nDetailed Metrics:")
    print("-" * 90)
    for r in results:
        print(f"\n  Config {r.name}: {r.description}")
        print(f"    Samples:     {r.total}")
        print(f"    Correct:     {r.correct}/{r.total}")
        print(f"    Accuracy:    {r.accuracy:.1f}%")
        print(f"    Avg dist:    {r.avg_distance:.4f}")
        print(f"    Avg time:    {r.avg_inference_ms:.1f} ms/sample")
        print(f"    Total time:  {r.total_time_s:.1f} s")
        print(f"    Peak VRAM:   {r.peak_vram_mb:.1f} MB")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run NanoGUI ablation on ScreenSpot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ablation.py --max-samples 20
  python scripts/run_ablation.py --max-samples 200 --output results/ablation_results.json
  python scripts/run_ablation.py --configs A C --max-samples 50
        """,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (default: all 1272)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=["A", "B", "C", "D", "E", "F", "G", "all"],
        default=["all"],
        help="Which configs to run (default: all)",
    )
    parser.add_argument(
        "--planner-model",
        type=str,
        default=None,
        help="Override planner model directory (e.g. Qwen2.5-VL-3B-Instruct-Planner)",
    )
    parser.add_argument(
        "--verifier-type",
        choices=["vlm", "clip", "critic"],
        default="vlm",
        help="Verifier type: vlm (SmolVLM), clip (CLIP classifier), or critic (ResNet-18) (default: vlm)",
    )
    parser.add_argument(
        "--verifier-dir",
        type=str,
        default="outputs/clip_verifier",
        help="Directory with trained CLIP verifier head (for --verifier-type clip)",
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        default="outputs/grounder_v2",
        help="Path to LoRA adapter for Config G (default: outputs/grounder_v2)",
    )
    parser.add_argument(
        "--num-attempts",
        type=int,
        default=3,
        help="Number of attempts for multi-attempt grounding in Config E (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Override planner model if specified
    if args.planner_model:
        global MODEL_PLANNER
        MODEL_PLANNER = args.planner_model
        logger.info("Using planner model override: %s", MODEL_PLANNER)

    # Validate models exist
    available = discover_models()
    if args.verifier_type == "vlm":
        needed = {MODEL_PLANNER, MODELGROUNDER, MODEL_VERIFIER}
    else:
        needed = {MODEL_PLANNER, MODELGROUNDER}  # CLIP doesn't need a local model dir
    missing = needed - set(available)
    if missing:
        print(f"ERROR: Missing models: {missing}")
        print(f"Available: {available}")
        raise SystemExit(1)

    # Load samples
    logger.info("Loading ScreenSpot annotations...")
    samples = load_screenspot_annotations("test")
    if args.max_samples:
        samples = samples[:args.max_samples]
    logger.info("Samples: %d", len(samples))

    # Determine which configs to run
    configs_to_run = set(args.configs)
    if "all" in configs_to_run:
        configs_to_run = {"A", "B", "C", "D", "E", "F", "G"}

    all_results: List[ConfigResult] = []

    # Run each config
    if "A" in configs_to_run:
        all_results.append(run_config_a(samples))
    if "B" in configs_to_run:
        _aggressive_vram_cleanup()
        all_results.append(run_config_b(samples))
    if "C" in configs_to_run:
        _aggressive_vram_cleanup()
        all_results.append(run_config_c(
            samples,
            verifier_type=args.verifier_type, verifier_dir=args.verifier_dir))
    if "D" in configs_to_run:
        _aggressive_vram_cleanup()
        all_results.append(run_config_d(
            samples,
            verifier_type=args.verifier_type, verifier_dir=args.verifier_dir))
    if "E" in configs_to_run:
        _aggressive_vram_cleanup()
        all_results.append(run_config_e(
            samples,
            num_attempts=args.num_attempts))
    if "F" in configs_to_run:
        _aggressive_vram_cleanup()
        all_results.append(run_config_f(samples))
    if "G" in configs_to_run:
        _aggressive_vram_cleanup()
        all_results.append(run_config_g(
            samples,
            lora_adapter_path=args.lora_adapter))

    # Print results
    print_results_table(all_results)

    # Save if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable dict
        output = {
            "configs": [
                {
                    "name": r.name,
                    "description": r.description,
                    "total": r.total,
                    "correct": r.correct,
                    "accuracy": r.accuracy,
                    "avg_distance": r.avg_distance,
                    "avg_inference_ms": r.avg_inference_ms,
                    "total_time_s": r.total_time_s,
                    "peak_vram_mb": r.peak_vram_mb,
                    "samples": [
                        {
                            "sample_id": s.sample_id,
                            "instruction": s.instruction,
                            "predicted": s.predicted,
                            "ground_truth": s.ground_truth,
                            "correct": s.correct,
                            "distance": s.distance,
                            "inference_ms": s.inference_ms,
                            "planner_subgoal": s.planner_subgoal,
                            "verifier_accepted": s.verifier_accepted,
                            "verifier_correct": s.verifier_correct,
                        }
                        for s in r.sample_results
                    ],
                }
                for r in all_results
            ]
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
