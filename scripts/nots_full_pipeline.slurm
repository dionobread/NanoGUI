#!/bin/bash
#SBATCH --job-name=nanogui_full
#SBATCH --output=/scratch/gz27/nanogui/logs/%j_full.out
#SBATCH --error=/scratch/gz27/nanogui/logs/%j_full.err
#SBATCH --partition=commons
#SBATCH --account=commons
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00

echo "============================================================"
echo "NanoGUI Full Experiment Pipeline"
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

# ── PHASE 0: Fix datasets ───────────────────────────────────────────────
echo ""
echo "PHASE 0: Fixing datasets"
echo "============================================================"

# Fix Salesforce Grounding bbox normalization
if [ -d "datasets/salesforce_grounding" ]; then
    echo "Fixing Salesforce Grounding bboxes..."
    python scripts/fix_salesforce_bboxes.py
fi

# ── PHASE 1: Verify GUI-Actor (quick smoke test) ────────────────────────
echo ""
echo "PHASE 1: Smoke test GUI-Actor"
echo "============================================================"
python scripts/eval_grounder.py --model GUI-Actor-3B-Qwen2.5-VL --max-samples 20 --output results/smoke_test.json

# Check if smoke test passed (>50% accuracy)
python3 -c "
import json, sys
with open('results/smoke_test.json') as f:
    data = json.load(f)
acc = data.get('accuracy', 0)
print(f'Smoke test accuracy: {acc}%')
if acc < 50:
    print('ERROR: GUI-Actor accuracy too low. Aborting.')
    sys.exit(1)
print('Smoke test PASSED.')
" || exit 1

# ── PHASE 2: Train LoRA Grounder (Config G) ─────────────────────────────
echo ""
echo "PHASE 2: Training LoRA Grounder (Config G)"
echo "============================================================"

# Only train if adapter doesn't exist
if [ ! -f "outputs/grounder_v2/adapter_config.json" ]; then
    python scripts/train_grounder_v2.py \
        --dataset mixed \
        --output outputs/grounder_v2 \
        --epochs 5 \
        --batch-size 1 \
        --lr 5e-5 \
        --lora-r 16 \
        --grad-accum 8 \
        --merge \
        --fp16 \
        --max-samples 10000
    echo "LoRA training complete."
else
    echo "LoRA adapter already exists, skipping training."
fi

# ── PHASE 3: Full ablation (all 7 configs) ──────────────────────────────
echo ""
echo "PHASE 3: Full ablation (1272 samples, 7 configs)"
echo "============================================================"

python scripts/run_ablation.py \
    --configs A B C D E F G \
    --output results/ablation_final_v2.json \
    --lora-adapter outputs/grounder_v2

# ── PHASE 4: Summary ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Results Summary"
echo "============================================================"

python3 -c "
import json
from pathlib import Path

p = Path('results/ablation_final_v2.json')
if p.exists():
    with open(p) as f:
        data = json.load(f)
    print('Config | Description                  | Accuracy | Avg Dist | Time')
    print('-'*75)
    for c in data.get('configs', []):
        print(f\"{c['name']:<6} | {c['description']:<28} | {c['accuracy']:>7.2f}% | {c['avg_distance']:>8.4f} | {c['total_time_s']:>7.1f}s\")
else:
    print('No results file found.')
"

echo ""
echo "All done at $(date)"
echo "============================================================"
