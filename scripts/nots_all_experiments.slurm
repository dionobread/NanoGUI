#!/bin/bash
#SBATCH --job-name=nanogui_all
#SBATCH --partition=commons
#SBATCH --account=commons
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Note: We intentionally do NOT use set -e here
# so that fallback commands can run if a phase fails

echo "============================================================"
echo "NanoGUI COMPLETE Experiment Pipeline"
echo "Started at $(date)"
echo "Running on: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================================"
nvidia-smi

source /scratch/gz27/venvs/nanogui/bin/activate
cd /scratch/gz27/nanogui
mkdir -p outputs logs results datasets

export HF_HOME=/scratch/gz27/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/scratch/gz27/nanogui/NanoGUI:$PYTHONPATH
mkdir -p $HF_HOME

# ── PHASE 0: Verify datasets exist ───────────────────────────────────────────
echo ""
echo "============================================================"
echo "PHASE 0: Checking datasets"
echo "============================================================"

for ds in screenspot seeclick screenspot_v2; do
    if [ ! -d "datasets/$ds" ]; then
        echo "[MISSING] datasets/$ds - attempting download..."
        python -m NanoGUI.data.download_all_datasets legacy $ds --no-images 2>/dev/null || echo "Download failed for $ds"
    else
        echo "[OK] datasets/$ds"
    fi
done

# ── PHASE 1: Train Grounder (LoRA on Qwen2.5-VL-3B) ─────────────────────────
echo ""
echo "============================================================"
echo "PHASE 1: Train Grounder on Mixed Data"
echo "  Model: Qwen2.5-VL-3B-Instruct (local)"
echo "  Data: ScreenSpot + SeeClick + ScreenSpot-v2"
echo "  Epochs: 5 | Batch: 1 | GradAccum: 8 | LR: 5e-5"
echo "============================================================"

# Skip training if adapter already exists
if [ -f "outputs/grounder_v2/adapter_config.json" ]; then
    echo "LoRA adapter already exists, skipping training"
else
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

    if [ $? -ne 0 ]; then
        echo "ERROR: Grounder training failed!"
        echo "Trying with screenspot only..."
        python scripts/train_grounder_v2.py \
            --dataset screenspot \
            --model Qwen2.5-VL-3B-Instruct \
            --output outputs/grounder_v2 \
            --epochs 3 \
            --batch-size 1 \
            --lr 5e-5 \
            --lora-r 16 \
            --grad-accum 8 \
            --merge \
            --fp16
    fi
fi

echo "Grounder training complete at $(date)"
ls -la outputs/grounder_v2/ 2>/dev/null || echo "No LoRA output"
ls -la outputs/grounder_v2_merged/ 2>/dev/null || echo "No merged output"

# ── PHASE 2: Train Critic (ResNet-18) ───────────────────────────────────────
echo ""
echo "============================================================"
echo "PHASE 2: Train Critic Verifier (ResNet-18)"
echo "  Epochs: 10 | Batch: 64 | LR: 1e-4"
echo "============================================================"

if [ -f "outputs/critic_final/best_critic.safetensors" ] || [ -f "outputs/critic_final/best_critic.pt" ]; then
    echo "Critic model already exists, skipping training"
else
    python NanoGUI/train/train_critic.py \
        --annotations datasets/screenspot/annotations/train_annotations.json \
        --output outputs/critic_final \
        --epochs 10 \
        --batch-size 64 \
        --lr 1e-4

    if [ $? -ne 0 ]; then
        echo "WARNING: Critic training failed, continuing without it"
    fi
fi

echo "Critic training complete at $(date)"
ls -la outputs/critic_final/ 2>/dev/null || echo "No critic output"

# ── PHASE 3: Ablation 50 samples (all configs) ──────────────────────────────
echo ""
echo "============================================================"
echo "PHASE 3: Ablation Study (50 samples, all configs)"
echo "  A: Grounder-only"
echo "  B: Planner + Grounder"
echo "  C: Grounder + Critic Verifier"
echo "  D: Full pipeline"
echo "  E: Multi-attempt Grounder (K=3)"
echo "  F: Spatial Planner + Grounder"
echo "  G: Fine-tuned Grounder (LoRA)"
echo "============================================================"

# Determine available models and verifier
LORA_PATH="outputs/grounder_v2"
if [ ! -f "$LORA_PATH/adapter_config.json" ]; then
    LORA_PATH="outputs/grounder_v2_merged"
fi

CRITIC_DIR="outputs/critic_final"
if [ ! -f "$CRITIC_DIR/best_critic.safetensors" ] && [ ! -f "$CRITIC_DIR/best_critic.pt" ]; then
    echo "Critic not available, using VLM verifier for Config C"
    VERIFIER_TYPE="vlm"
    VERIFIER_DIR="outputs/clip_verifier"
else
    echo "Using Critic verifier for Config C"
    VERIFIER_TYPE="critic"
    VERIFIER_DIR="$CRITIC_DIR"
fi

# Build config list based on what's available
CONFIGS="A B C D E F"
if [ -f "$LORA_PATH/adapter_config.json" ] || [ -f "$LORA_PATH/config.json" ]; then
    CONFIGS="A B C D E F G"
    echo "Fine-tuned model available, including Config G"
else
    echo "Fine-tuned model NOT available, skipping Config G"
fi

python scripts/run_ablation.py \
    --configs $CONFIGS \
    --max-samples 50 \
    --num-attempts 3 \
    --verifier-type $VERIFIER_TYPE \
    --verifier-dir $VERIFIER_DIR \
    --lora-adapter "$LORA_PATH" \
    --output results/ablation_all_50.json

if [ $? -ne 0 ]; then
    echo "WARNING: Full ablation failed, trying reduced configs"
    python scripts/run_ablation.py \
        --configs A B C D E F \
        --max-samples 50 \
        --num-attempts 3 \
        --verifier-type vlm \
        --output results/ablation_all_50.json
fi

echo "Ablation (50) complete at $(date)"

# ── PHASE 4: Ablation 200 samples ───────────────────────────────────────────
echo ""
echo "============================================================"
echo "PHASE 4: Ablation Study (200 samples)"
echo "============================================================"

python scripts/run_ablation.py \
    --configs $CONFIGS \
    --max-samples 200 \
    --num-attempts 3 \
    --verifier-type $VERIFIER_TYPE \
    --verifier-dir $VERIFIER_DIR \
    --lora-adapter "$LORA_PATH" \
    --output results/ablation_all_200.json

if [ $? -ne 0 ]; then
    echo "WARNING: 200-sample ablation failed, trying 100"
    python scripts/run_ablation.py \
        --configs A B C D E F \
        --max-samples 100 \
        --num-attempts 3 \
        --verifier-type vlm \
        --output results/ablation_all_200.json
fi

echo "Ablation (200) complete at $(date)"

# ── PHASE 5: Full evaluation (all 1272 test samples) ────────────────────────
echo ""
echo "============================================================"
echo "PHASE 5: Full Evaluation (all test samples)"
echo "============================================================"

python scripts/run_ablation.py \
    --configs A E F G \
    --num-attempts 3 \
    --verifier-type $VERIFIER_TYPE \
    --verifier-dir $VERIFIER_DIR \
    --lora-adapter "$LORA_PATH" \
    --output results/ablation_final_full.json

if [ $? -ne 0 ]; then
    echo "WARNING: Full evaluation failed"
fi

# ── PHASE 6: Generate summary ───────────────────────────────────────────────
echo ""
echo "============================================================"
echo "PHASE 6: Results Summary"
echo "============================================================"

python -c "
import json, glob, sys
print('\n=== ALL RESULTS ===\n')
for path in sorted(glob.glob('results/ablation_*.json')):
    try:
        with open(path) as f:
            data = json.load(f)
        configs = data.get('configs', data)
        if isinstance(configs, dict):
            configs = configs.values()
        print(f'File: {path}')
        print(f'{'Config':<8} {'Description':<40} {'Acc':>7} {'Dist':>7} {'Time':>8}')
        print('-' * 75)
        for c in configs:
            if isinstance(c, dict):
                print(f\"{c.get('name', '?'):<8} {c.get('description', ''):<40} {c.get('accuracy', 0):>6.1f}% {c.get('avg_distance', 0):>7.4f} {c.get('total_time_s', 0):>7.1f}s\")
        print()
    except Exception as e:
        print(f'Error reading {path}: {e}')
"

echo ""
echo "============================================================"
echo "COMPLETE at $(date)"
echo "============================================================"
echo ""
echo "Outputs:"
echo "  Grounder LoRA:      outputs/grounder_v2/"
echo "  Grounder merged:    outputs/grounder_v2_merged/"
echo "  Critic:             outputs/critic_final/"
echo "  Ablation 50:        results/ablation_all_50.json"
echo "  Ablation 200:       results/ablation_all_200.json"
echo "  Ablation full:      results/ablation_final_full.json"
echo ""
echo "To download results:"
echo "  scp -r nots:/scratch/gz27/nanogui/results/ ./results/"
echo "  scp -r nots:/scratch/gz27/nanogui/outputs/ ./outputs/"
