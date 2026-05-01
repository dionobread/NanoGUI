#!/usr/bin/env python3
"""
One-click experiment runner for NanoGUI term project.

Runs the complete experiment pipeline:
  1. Download datasets (if not already present)
  2. Train grounder on mixed data (ScreenSpot + SeeClick + ScreenSpot-v2)
  3. Run ablation with all configs (A-G)
  4. Save combined results

Usage:
    python scripts/run_all.py --quick          # Fast test with 20 samples
    python scripts/run_all.py --samples 200    # Medium run
    python scripts/run_all.py                  # Full run (all 1272 samples)
    python scripts/run_all.py --skip-train     # Skip training, just run ablation
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_root() -> Path:
    return Path(__file__).resolve().parent.parent


def step(name: str):
    """Print a step header."""
    print(f"\n{'=' * 70}")
    print(f"  STEP: {name}")
    print(f"{'=' * 70}\n")


def run_cmd(cmd: list, desc: str) -> bool:
    """Run a command and report status."""
    logger.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True)
        logger.info("[OK] %s completed", desc)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("[FAIL] %s failed with exit code %d", desc, e.returncode)
        return False
    except FileNotFoundError:
        logger.error("[FAIL] Command not found: %s", cmd[0])
        return False


def check_datasets(root: Path) -> dict:
    """Check which datasets are available."""
    datasets = {
        "screenspot": (root / "datasets" / "screenspot" / "annotations" / "train_annotations.json").exists(),
        "seeclick": (root / "datasets" / "seeclick" / "annotations.json").exists(),
        "screenspot_v2": (root / "datasets" / "screenspot_v2" / "annotations.json").exists(),
        "omniact": (root / "datasets" / "omniact" / "annotations" / "train_annotations.json").exists(),
    }
    return datasets


def count_samples(root: Path) -> dict:
    """Count available training samples per dataset."""
    counts = {}
    ss_train = root / "datasets" / "screenspot" / "annotations" / "train_annotations.json"
    if ss_train.exists():
        with open(ss_train) as f:
            counts["screenspot"] = len(json.load(f))

    sc = root / "datasets" / "seeclick" / "annotations.json"
    if sc.exists():
        with open(sc) as f:
            counts["seeclick"] = len(json.load(f))

    v2 = root / "datasets" / "screenspot_v2" / "annotations.json"
    if v2.exists():
        with open(v2) as f:
            counts["screenspot_v2"] = len(json.load(f))

    counts["total"] = sum(v for v in counts.values())
    return counts


def main():
    parser = argparse.ArgumentParser(description="Run all NanoGUI experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test (20 samples)")
    parser.add_argument("--samples", type=int, default=None, help="Number of test samples")
    parser.add_argument("--skip-train", action="store_true", help="Skip grounder training")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset downloads")
    parser.add_argument("--dataset", default="mixed", choices=[
        "screenspot", "seeclick", "screenspot_v2", "omniact", "mixed"
    ], help="Training dataset (default: mixed)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--output", default="results/full_ablation.json", help="Output file")
    parser.add_argument("--configs", nargs="+", default=["all"],
                        choices=["A", "B", "C", "D", "E", "F", "G", "all"])

    args = parser.parse_args()
    root = get_root()
    max_samples = 20 if args.quick else args.samples

    print("=" * 70)
    print("NanoGUI Full Experiment Pipeline")
    print("=" * 70)

    # ── Step 0: Check datasets ──────────────────────────────────────────────
    step("Checking datasets")
    datasets = check_datasets(root)
    sample_counts = count_samples(root)

    print("Dataset availability:")
    for name, available in datasets.items():
        count = sample_counts.get(name, "?")
        status = "[OK]" if available else "[MISSING]"
        print(f"  {status} {name}: {count} samples")

    print(f"\n  Total training data: {sample_counts.get('total', 0)} samples")

    # ── Step 1: Download missing datasets ───────────────────────────────────
    if not args.skip_download:
        missing = [name for name, avail in datasets.items() if not avail]
        if missing:
            step(f"Downloading missing datasets: {missing}")
            # Use existing download scripts
            if not datasets.get("screenspot"):
                run_cmd(
                    [sys.executable, "-m", "NanoGUI.data.download_all_datasets", "gui", "screenspot", "--no-images"],
                    "ScreenSpot download"
                )
            if not datasets.get("seeclick"):
                run_cmd(
                    [sys.executable, "-m", "NanoGUI.data.download_all_datasets", "legacy", "seeclick", "--no-images"],
                    "SeeClick download"
                )
            if not datasets.get("omniact"):
                logger.info("OmniAct download requires HF access. Run manually:")
                logger.info("  python -m NanoGUI.data.download_all_datasets omniact --no-images")

            # Re-check
            datasets = check_datasets(root)
        else:
            print("  All datasets present, skipping downloads.")

    # ── Step 2: Train grounder ──────────────────────────────────────────────
    lora_path = root / "outputs" / "grounder_v2"
    if not args.skip_train and not (lora_path / "adapter_config.json").exists():
        step(f"Training Grounder on {args.dataset}")

        # Pick the best available dataset
        if args.dataset == "mixed":
            # Use whatever we have
            if datasets.get("screenspot"):
                train_dataset = "mixed"
            elif datasets.get("seeclick"):
                train_dataset = "seeclick"
            else:
                train_dataset = "screenspot"
        else:
            train_dataset = args.dataset

        cmd = [
            sys.executable, str(root / "scripts" / "train_grounder_v2.py"),
            "--dataset", train_dataset,
            "--epochs", str(args.epochs),
            "--output", str(lora_path),
            "--merge",
        ]
        if max_samples:
            cmd += ["--max-samples", str(max_samples * 10)]  # More data for training

        success = run_cmd(cmd, f"Grounder training ({train_dataset})")
        if success:
            print(f"  LoRA adapter saved to: {lora_path}")
    elif (lora_path / "adapter_config.json").exists():
        print(f"\n  [OK] Trained grounder already exists at {lora_path}")
    else:
        print("\n  [SKIP] Grounder training skipped (--skip-train)")

    # ── Step 3: Run ablation ────────────────────────────────────────────────
    step("Running ablation")

    cmd = [
        sys.executable, str(root / "scripts" / "run_ablation.py"),
        "--configs", *args.configs,
        "--output", str(root / args.output),
    ]
    if max_samples:
        cmd += ["--max-samples", str(max_samples)]

    # Add LoRA path for Config G if adapter exists
    if (lora_path / "adapter_config.json").exists():
        cmd += ["--lora-adapter", str(lora_path)]

    run_cmd(cmd, "Ablation study")

    # ── Step 4: Print summary ───────────────────────────────────────────────
    step("Results summary")

    output_path = root / args.output
    if output_path.exists():
        with open(output_path) as f:
            data = json.load(f)

        configs = data.get("configs", data.get("configs", []))
        if isinstance(configs, dict):
            configs_list = configs.values()
        elif isinstance(configs, list):
            configs_list = configs
        else:
            configs_list = []

        print(f"\n{'Config':<8} {'Description':<40} {'Acc':>7} {'Dist':>7} {'Time':>8}")
        print("-" * 75)
        for c in configs_list:
            if isinstance(c, dict):
                print(
                    f"{c.get('name', '?'):<8} "
                    f"{c.get('description', ''):<40} "
                    f"{c.get('accuracy', 0):>6.1f}% "
                    f"{c.get('avg_distance', 0):>7.4f} "
                    f"{c.get('total_time_s', 0):>7.1f}s"
                )

        print(f"\nFull results saved to: {output_path}")
    else:
        print("  No results file found.")

    print("\n" + "=" * 70)
    print("Experiment pipeline complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
