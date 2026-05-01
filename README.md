# Divide and Ground: A Lightweight Multi-Agent Framework for GUI Task Execution with Small Vision-Language Models

**Course:** COMP646 Deep Learning for Vision and Language  
**Date:** May 2026

---

## Overview

This project implements a three-agent pipeline for GUI automation using small vision-language models (VLMs):

1. **Planner** (Qwen2.5-VL-3B-Instruct) — Decomposes high-level tasks into atomic sub-goals
2. **Grounder** (GUI-Actor-3B-Qwen2.5-VL) — Locates UI elements via coordinate-free pointer tokens
3. **Verifier** (ResNet-18 Critic or SmolVLM-256M) — Validates proposed actions via screenshot analysis

The system is designed to fit within a single GPU with 16–44 GB VRAM using sequential model loading.

---

## System Architecture

```
User Task (e.g., "Open Chrome and search for cats")
        |
        v
+------------------+     +------------------+     +------------------+
|   PLANNER AGENT  | --> |  GROUNDER AGENT  | --> |  VERIFIER AGENT  |
| Qwen2.5-VL-3B    |     | GUI-Actor-3B     |     | ResNet-18 Critic |
| (task decomposition)   | (pointer-based     |     | or SmolVLM-256M  |
|                        |  grounding)        |     | (accept/reject)  |
+------------------+     +------------------+     +------------------+
        |                         |                         |
        v                         v                         v
   Sub-goal list          (x, y) click point        Binary decision
                                                      (retry if reject)
```

### Key Design Decisions

- **Sequential loading:** Only one large model is loaded in GPU memory at a time. Each agent loads its model on demand, runs inference, then unloads before the next agent starts.
- **GUI-Actor native inference:** The Grounder uses Microsoft's GUI-Actor model with its custom pointer-token architecture (not text coordinate output), achieving ~91% accuracy on ScreenSpot-v2.
- **No 4-bit quantization:** All models run at full FP16/bfloat16 precision for maximum accuracy.
- **LoRA fine-tuning:** An optional fine-tuned Grounder (Config G) uses LoRA (r=16, alpha=32) on Qwen2.5-VL-3B trained on mixed GUI datasets.

---

## Models Used

| Agent | Model | Size | Precision | Source |
|-------|-------|------|-----------|--------|
| Planner | Qwen2.5-VL-3B-Instruct | 3B | bfloat16 | Local |
| Grounder | GUI-Actor-3B-Qwen2.5-VL | 3B | bfloat16 | Local (Microsoft) |
| Grounder (alt) | Qwen2.5-VL-3B-Instruct + LoRA | 3B + 20M | float16 | Fine-tuned |
| Verifier (VLM) | SmolVLM-256M-Instruct | 256M | float16 | Local |
| Verifier (Critic) | ResNet-18 + custom head | 11M | float32 | Trained |

All models are stored under `models/` and loaded locally (no Hugging Face download required at runtime).

---

## Datasets

| Dataset | Samples | Split | Usage |
|---------|---------|-------|-------|
| ScreenSpot | 1,272 | test | Primary evaluation |
| ScreenSpot | ~400 | train | LoRA training + Critic training |
| SeeClick | ~500 images, ~8.5K elements | — | LoRA training augmentation |
| ScreenSpot-v2 | 1,272 | test | Secondary evaluation |
| OmniAct | 6,750+ | train/test | Optional training data |

ScreenSpot annotations use normalized bounding box format `[x1, y1, x2, y2]`.

**Download additional datasets:** See `NanoGUI/data/README.md` for unified downloaders supporting ScreenSpot-Pro, Salesforce Grounding, OS-Atlas, Mind2Web, AITW, and more.

---

## Ablation Study (7 Configurations)

The main experimental result is an ablation study comparing seven pipeline configurations on ScreenSpot:

| Config | Description | Components Used |
|--------|-------------|-----------------|
| **A** | Grounder-only | GUI-Actor-3B |
| **B** | Planner + Grounder | Planner → Grounder |
| **C** | Grounder + Verifier | Grounder → Critic |
| **D** | Full pipeline | Planner → Grounder → Critic |
| **E** | Multi-attempt Grounder | GUI-Actor (K=3 predictions, self-consistency) |
| **F** | Spatial Planner + Grounder | Planner with spatial hints → Grounder |
| **G** | Fine-tuned Grounder | Qwen2.5-VL-3B + LoRA adapter |

### Metrics Collected

- **Accuracy:** % of predictions inside ground-truth bounding box
- **Avg Distance:** Euclidean distance to bbox center
- **Avg Inference Time:** ms per sample
- **Peak VRAM:** Maximum GPU memory allocated
- **Total Wall Time:** Per-configuration runtime

---

## Key Technical Fixes (vs. Initial Implementation)

1. **GUI-Actor pointer inference:** The initial implementation loaded GUI-Actor as a standard text-generation model and tried to parse `(x, y)` coordinates from its output. This produced ~7% accuracy because GUI-Actor uses special pointer tokens, not text. The fix uses Microsoft's native `gui_actor.inference()` API, which extracts coordinates from the model's custom pointer head.

2. **Removed 4-bit quantization:** All 4-bit / `bitsandbytes` code was removed. Models now load at full FP16/bfloat16 precision.

3. **Fixed LoRA training:** The original training script had a bug where `mm_token_type_ids` was overwritten instead of extended, causing data corruption. Training also failed to save output files. The fix corrects the collator and adds save verification.

4. **Critic verifier:** A ResNet-18 binary classifier trained on click-point overlays achieves 97.4% validation accuracy on accept/reject decisions.

---

## Running Experiments

### Quick Test (20 samples, ~10 min)

```bash
python scripts/eval_grounder.py \
    --model GUI-Actor-3B-Qwen2.5-VL \
    --max-samples 20
```

### Full Ablation (all configs, 1,272 samples)

```bash
python scripts/run_ablation.py \
    --configs A B C D E F G \
    --output results/ablation_full.json
```

### Train Fine-tuned Grounder (Config G)

```bash
python scripts/train_grounder_v2.py \
    --dataset mixed \
    --model Qwen2.5-VL-3B-Instruct \
    --output outputs/grounder_v2 \
    --epochs 5 \
    --batch-size 1 \
    --lr 5e-5 \
    --lora-r 16 \
    --grad-accum 8 \
    --merge \
    --fp16
```

### Train Critic Verifier

```bash
python NanoGUI/train/train_critic.py \
    --annotations datasets/screenspot/annotations/train_annotations.json \
    --output outputs/critic_final \
    --epochs 10 \
    --batch-size 64 \
    --lr 1e-4
```

### NOTS Cluster (Full Pipeline)

```bash
sbatch scripts/nots_all_experiments.slurm
```

This runs all phases automatically:
1. Train Grounder (LoRA)
2. Train Critic (ResNet-18)
3. Ablation: 50 samples
4. Ablation: 200 samples
5. Ablation: full 1,272 samples
6. Generate summary table

Monitor with: `squeue -u $USER`

---

## Project Structure

```
NanoGUI/
├── agents/                    # Planner, Grounder, Verifier agents
│   ├── base.py               # Base classes, configs, data structures
│   ├── Planner.py
│   ├── Grounder.py
│   └── Verifier.py
├── core/                      # Screen capture, action executor
│   ├── action_executor.py
│   └── screen_capture.py
├── data/                      # Data loaders and download scripts
│   ├── test_data_loader.py
│   ├── train_data_loader.py
│   └── download_all_datasets.py
├── gui_actor/                 # Patched GUI-Actor source (pointer inference)
│   ├── inference.py
│   ├── modeling_qwen25vl.py
│   └── constants.py
├── models/                    # Local model directories
│   ├── Qwen2.5-VL-3B-Instruct/
│   ├── GUI-Actor-3B-Qwen2.5-VL/
│   ├── GUI-Actor-Verifier-2B/
│   └── SmolVLM-256M-Instruct/
├── train/                     # Training scripts
│   ├── train_grounder.py     # Legacy coordinate-regression trainer
│   └── train_critic.py       # ResNet-18 Critic trainer
├── config.py                  # Centralized configuration
├── config.yaml                # User-editable config overrides
└── pipeline.py                # Full pipeline orchestration

scripts/
├── eval_grounder.py           # ScreenSpot evaluation (GUI-Actor native)
├── run_ablation.py            # Ablation study runner (7 configs)
├── run_all.py                 # One-click experiment pipeline
├── train_grounder_v2.py       # Improved LoRA fine-tuning
├── download_all_datasets.py   # Unified dataset downloader (GUI, planner, legacy)
├── test_pipeline_e2e.py       # End-to-end pipeline test
├── nots_all_experiments.slurm # NOTS cluster SLURM script
└── nots_setup.sh              # NOTS one-time setup

datasets/
├── screenspot/                # ScreenSpot annotations + images
├── seeclick/                  # SeeClick subset
├── screenspot_v2/             # ScreenSpot-v2 annotations + images
└── omniact/                   # OmniAct annotations + images

outputs/                       # Training outputs
├── grounder_v2/              # LoRA adapter
├── grounder_v2_merged/       # Merged model
└── critic_final/             # Trained Critic weights

results/                       # Ablation JSON results
├── ablation_all_50.json
├── ablation_all_200.json
└── ablation_final_full.json
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch>=2.2.0` (with CUDA)
- `transformers>=4.40.0`
- `peft>=0.8.0` (for LoRA)
- `qwen-vl-utils>=0.0.1`
- `datasets>=2.16.0`

**Note:** `bitsandbytes` is **not** required. All models run at full precision.

---

## References

[1] Cheng, K., et al. *SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents.* ACL 2024.

[2] Wu, Q., et al. *GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents.* arXiv 2025. https://github.com/microsoft/GUI-Actor

[3] Hu, E., et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.

[4] Wang, P., et al. *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution.* arXiv 2024.

[5] Bai, J., et al. *Qwen2.5-VL Technical Report.* arXiv 2025.

[6] Deng, X., et al. *Mind2Web: Towards a Generalist Agent for the Web.* NeurIPS 2023.

---

## Notes for Report Writing

### What to highlight

1. **The pointer-token innovation:** GUI-Actor's coordinate-free grounding (using special tokens to reference visual patches) is the key technical advance that makes small models competitive with 72B-parameter baselines on GUI tasks.

2. **Ablation design:** The 7 configurations (A–G) systematically isolate each component's contribution. Config A establishes the strong single-model baseline (~90%); Configs B–D show whether adding Planner/Verifier helps or hurts; Config E tests self-consistency; Config F tests spatial prompting; Config G tests whether fine-tuning improves over zero-shot.

3. **Resource efficiency:** The entire pipeline fits on a single GPU (tested on RTX 4070 8GB local, A100 40GB on NOTS) by loading models sequentially rather than concurrently.

4. **Training data mix:** The fine-tuned model uses ScreenSpot + SeeClick + ScreenSpot-v2 + OmniAct for broader domain coverage.

5. **Critic as a lightweight classifier:** Unlike prior work that uses a full VLM as verifier, our ResNet-18 Critic is 100x smaller (11M vs. 256M–3B parameters) and runs in ~2ms per image.

### Result interpretation

- If Config A ≈ Config D: the Planner and Verifier add overhead without benefit (the Grounder is already accurate).
- If Config E > Config A: self-consistency improves robustness.
- If Config G > Config A: LoRA fine-tuning successfully adapts to the target domain.
- If Critic precision is high but recall is low: the Verifier is conservative (few false accepts, many false rejects).
