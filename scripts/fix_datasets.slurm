#!/bin/bash
#SBATCH --job-name=nanogui_fix_ds
#SBATCH --partition=debug
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --output=/scratch/gz27/nanogui/logs/fix_datasets_%j.out
#SBATCH --error=/scratch/gz27/nanogui/logs/fix_datasets_%j.err

set -e
source /scratch/gz27/venvs/nanogui/bin/activate
export HF_HOME=/scratch/gz27/hf_cache
export PYTHONUSERBASE=/scratch/gz27/python_pkgs
export PYTHONPATH=/scratch/gz27/nanogui:$PYTHONPATH

cd /scratch/gz27/nanogui

echo "=== Dataset Fix Job ==="
echo "Start: $(date)"

# Step 1: Re-download OmniAct with images
python /scratch/gz27/nanogui/scripts/_fix_omniact.py

# Step 2: Re-download Salesforce with images
python /scratch/gz27/nanogui/scripts/_fix_salesforce.py

# Step 3: Fix Salesforce bbox normalization
python /scratch/gz27/nanogui/scripts/_fix_bboxes.py

echo "End: $(date)"
