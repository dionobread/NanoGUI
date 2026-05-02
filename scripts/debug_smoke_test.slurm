#!/bin/bash
#SBATCH --job-name=nanogui_debug
#SBATCH --partition=debug
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gz27/nanogui/logs/debug_smoke_%j.out
#SBATCH --error=/scratch/gz27/nanogui/logs/debug_smoke_%j.err

set -e
source /scratch/gz27/venvs/nanogui/bin/activate
export HF_HOME=/scratch/gz27/hf_cache
export PYTHONUSERBASE=/scratch/gz27/python_pkgs

cd /scratch/gz27/nanogui

echo "=== Debug Smoke Test ==="
echo "Start: $(date)"

python << 'PYEOF'
import sys
sys.path.insert(0, '/scratch/gz27/nanogui/NanoGUI')

from PIL import Image
import torch

MODEL_DIR = "/scratch/gz27/nanogui/models"

print("Test 1: GUI-Actor...")
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.inference import inference
from transformers import AutoProcessor

model_path = f"{MODEL_DIR}/GUI-Actor-3B-Qwen2.5-VL"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
tokenizer = processor.tokenizer
model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
).eval()
print("GUI-Actor loaded successfully!")

img = Image.new('RGB', (800, 600), color='white')
conversation = [
    {"role": "system", "content": [{"type": "text", "text": "You are a GUI agent."}]},
    {"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Click the button"},
    ]},
]
pred = inference(conversation, model, tokenizer, processor, use_placeholder=True, topk=1)
px, py = pred["topk_points"][0]
print(f"Inference OK: pred=({px:.3f}, {py:.3f})")

del model
del processor
del tokenizer
import gc
gc.collect()
torch.cuda.empty_cache()

print("\nTest 2: Planner (Qwen2.5-VL)...")
from transformers import Qwen2_5_VLForConditionalGeneration
model_path = f"{MODEL_DIR}/Qwen2.5-VL-3B-Instruct"
m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map="cuda",
    trust_remote_code=True,
)
print("Planner model loaded successfully!")
del m
gc.collect()
torch.cuda.empty_cache()

print("\nTest 3: Verifier (SmolVLM)...")
from transformers import AutoModelForVision2Seq
model_path = f"{MODEL_DIR}/SmolVLM-256M-Instruct"
m = AutoModelForVision2Seq.from_pretrained(
    model_path,
    device_map="cuda",
    trust_remote_code=True,
)
print("Verifier model loaded successfully!")
del m
gc.collect()
torch.cuda.empty_cache()

print("\n=== ALL SMOKE TESTS PASSED ===")
PYEOF

echo "End: $(date)"
