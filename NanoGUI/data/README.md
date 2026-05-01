# NanoGUI Data Scripts

Dataset downloaders and preprocessing utilities.

## Quick Start

Install dependencies:

```bash
pip install datasets huggingface_hub
```

## Downloaders

### Unified GUI Dataset Downloader

`download_gui_datasets.py` downloads grounding datasets from Hugging Face:

```bash
# ScreenSpot (evaluation benchmark)
python NanoGUI/data/download_gui_datasets.py screenspot

# ScreenSpot-Pro (harder benchmark)
python NanoGUI/data/download_gui_datasets.py screenspot_pro --split train

# OmniAct (desktop + web grounding)
python NanoGUI/data/download_gui_datasets.py omniact --split train --split val

# Salesforce Grounding (combined sources)
python NanoGUI/data/download_gui_datasets.py salesforce_grounding --split train

# OS-Atlas annotations only (JSON first, images are huge)
python NanoGUI/data/download_gui_datasets.py os_atlas --os-atlas-subset annotations
```

### Planner Trajectory Downloader

`download_planner_datasets.py` downloads trajectory data for planner training:

```bash
# Individual datasets
python NanoGUI/data/download_planner_datasets.py mind2web --no-images
python NanoGUI/data/download_planner_datasets.py multimodal_mind2web
python NanoGUI/data/download_planner_datasets.py aitw_single --max-samples 5000

# All at once
python NanoGUI/data/download_planner_datasets.py all --no-images --skip-online
```

### Verifier Dataset Builder

`build_verifier_dataset.py` creates accept/reject training examples from grounding annotations:

```bash
python NanoGUI/data/build_verifier_dataset.py \
  --data-dir ./datasets/screenspot \
  --split test \
  --negatives-per-positive 2 \
  --render-overlays
```

This generates positive examples (click inside target bbox) and negative examples (click outside) for training the Critic verifier.

## Output Format

All datasets are normalized to:

```text
datasets/<dataset_name>/
  images/
  annotations/
    <split>_annotations.json
    <split>_metadata.json
```

Grounding annotations contain:

```json
{
  "instruction": "Click the search bar",
  "bbox": [0.1, 0.2, 0.4, 0.3],
  "image_path": "datasets/screenspot/images/test_000000.png"
}
```

## Legacy Scripts

- `scripts/download_datasets.py` — Simple ScreenSpot/SeeClick/ScreenSpot-v2 downloader (for backward compatibility)
- `NanoGUI/data/download_omniact.py` — Standalone OmniAct downloader
