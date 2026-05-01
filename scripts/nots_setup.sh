#!/bin/bash
# Setup script to run ONCE on NOTS
# SSH into NOTS first, then run this script

set -e

NETID=$USER  # Uses your logged-in username
PROJECT_DIR=/scratch/$NETID/nanogui
VENV_DIR=/scratch/$NETID/venvs/nanogui

echo "Setting up NanoGUI on NOTS for user: $NETID"

# Create directories
mkdir -p $PROJECT_DIR
mkdir -p /scratch/$NETID/venvs
mkdir -p $PROJECT_DIR/logs
mkdir -p $PROJECT_DIR/results

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
echo "Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate
pip install Pillow numpy
pip install autogen-agentchat autogen-ext
pip install peft trl datasets pyyaml safetensors

# Verify GPU
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "Setup complete!"
echo "Next steps:"
echo "  1. Upload your project: scp -r ./NanoGUI $NETID@nots.rice.edu:$PROJECT_DIR"
echo "  2. Submit job: cd $PROJECT_DIR && sbatch scripts/nots_all_experiments.slurm"
echo "  3. Monitor: squeue -u $NETID"
