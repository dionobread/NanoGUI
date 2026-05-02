#!/bin/bash
#SBATCH --job-name=nanogui_quick_bcef
#SBATCH --partition=debug
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gz27/nanogui/logs/quick_bcef_%j.out
#SBATCH --error=/scratch/gz27/nanogui/logs/quick_bcef_%j.err

set -e
source /scratch/gz27/venvs/nanogui/bin/activate
export HF_HOME=/scratch/gz27/hf_cache
export PYTHONUSERBASE=/scratch/gz27/python_pkgs
export PYTHONPATH=/scratch/gz27/nanogui/NanoGUI:$PYTHONPATH

cd /scratch/gz27/nanogui

echo "=== Quick Ablation B+C+E+F (Debug) ==="
echo "Start: $(date)"

python scripts/run_ablation.py --configs B C E F --max-samples 30 --output results/quick_bcef.json

echo "End: $(date)"
