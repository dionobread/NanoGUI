# NanoGUI Data Scripts

Use `download_gui_datasets.py` for dataset downloads and `build_verifier_dataset.py`
to create verifier examples from local grounding annotations.

Install the dataset dependency first:

```bash
pip install datasets huggingface_hub
```

## Quick Smoke Tests

Download a tiny ScreenSpot sample:

```bash
python NanoGUI/data/download_gui_datasets.py screenspot --max-samples 10
```

Build verifier examples from that sample:

```bash
python NanoGUI/data/build_verifier_dataset.py \
  --data-dir ./data/screenspot \
  --split test \
  --max-samples 10 \
  --render-overlays
```

## Recommended Downloads

Grounder training/evaluation:

```bash
python NanoGUI/data/download_gui_datasets.py salesforce_grounding --split train
python NanoGUI/data/download_gui_datasets.py omniact --split train --split val
python NanoGUI/data/download_gui_datasets.py screenspot
python NanoGUI/data/download_gui_datasets.py screenspot_pro
```

Planner trajectory data:

```bash
python NanoGUI/data/download_planner_datasets.py mind2web --no-images
python NanoGUI/data/download_planner_datasets.py multimodal_mind2web
python NanoGUI/data/download_planner_datasets.py online_mind2web --no-images
python NanoGUI/data/download_planner_datasets.py aitw_single --max-samples 5000
```

Or download the main planner sources together:

```bash
python NanoGUI/data/download_planner_datasets.py all --no-images --skip-online
```

Large OS-Atlas annotations:

```bash
python NanoGUI/data/download_gui_datasets.py os_atlas --os-atlas-subset annotations
```

`OS-Copilot/OS-Atlas-data` is very large if you include image archives. The
script intentionally downloads JSON/metadata patterns first so you can inspect
the files before pulling hundreds of GB of images.

## Local Output Format

Most datasets are normalized into:

```text
data/<dataset>/
  images/
  annotations/
    <split>_annotations.json
    <split>_metadata.json
```

Grounding annotations use these common fields when available:

```json
{
  "instruction": "Click the search bar",
  "bbox": [0.1, 0.2, 0.4, 0.3],
  "image_path": "data/screenspot/images/test_000000.png"
}
```

Verifier annotations use:

```json
{
  "sub_goal": "Click the search bar",
  "action_description": "Click at normalized coordinate (0.2500, 0.2500)",
  "candidate_point": [0.25, 0.25],
  "target_bbox": [0.1, 0.2, 0.4, 0.3],
  "status": "success"
}
```
