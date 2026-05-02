#!/bin/bash
#SBATCH --job-name=test_gui_actor
#SBATCH --output=/scratch/gz27/nanogui/logs/%j_test_gui_actor.out
#SBATCH --error=/scratch/gz27/nanogui/logs/%j_test_gui_actor.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -e
cd /scratch/gz27/nanogui
source /scratch/gz27/venvs/nanogui/bin/activate
export PYTHONPATH=/scratch/gz27/nanogui/NanoGUI:$PYTHONPATH
export HF_HOME=/scratch/gz27/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Testing GUI-Actor native pointer inference (20 samples)..."
python scripts/eval_grounder.py --model GUI-Actor-3B-Qwen2.5-VL --max-samples 20
