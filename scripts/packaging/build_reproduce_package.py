#!/usr/bin/env python3
"""
Build the compact reproducibility package distributed with the repository.
"""

import os
import json
import shutil
import yaml
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
PACKAGE_DIR = "reproduce_package"


def create_package_structure():
    """Create the reproducibility-package directory structure"""
    dirs = [
        f"{PACKAGE_DIR}/scripts/preprocessing",
        f"{PACKAGE_DIR}/scripts/inference",
        f"{PACKAGE_DIR}/scripts/training",
        f"{PACKAGE_DIR}/scripts/analysis",
        f"{PACKAGE_DIR}/scripts/reporting",
        f"{PACKAGE_DIR}/scripts/paper",
        f"{PACKAGE_DIR}/config",
        f"{PACKAGE_DIR}/outputs",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def copy_scripts():
    """Copy all scripts"""
    script_mappings = {
        # preprocessing
        "scripts/preprocessing/freeze_config.py": "scripts/preprocessing/",
        "scripts/preprocessing/build_grid_block_split.py": "scripts/preprocessing/",
        "scripts/preprocessing/build_country_holdout_split.py": "scripts/preprocessing/",
        "scripts/preprocessing/train_xgboost_out_of_fold.py": "scripts/preprocessing/",
        "scripts/preprocessing/build_neighbor_index.py": "scripts/preprocessing/",
        "scripts/preprocessing/make_prompts.py": "scripts/preprocessing/",
        # inference
        "scripts/inference/stage1_vllm.py": "scripts/inference/",
        "scripts/inference/baseline_uncertainty.py": "scripts/inference/",
        "scripts/inference/risk_coverage.py": "scripts/inference/",
        # training
        "scripts/training/train_json_auditor.py": "scripts/training/",
        "scripts/training/evaluate_json_auditor.py": "scripts/training/",
        # analysis
        "scripts/analysis/generate_stage2_audits.py": "scripts/analysis/",
        "scripts/analysis/oracle_ablation.py": "scripts/analysis/",
        # reporting
        "scripts/reporting/make_figures.py": "scripts/reporting/",
        "scripts/reporting/make_tables.py": "scripts/reporting/",
        # paper
        "scripts/paper/write_paper_assets.py": "scripts/paper/",
    }

    copied = 0
    for src, dst_dir in script_mappings.items():
        src_path = src
        dst_path = os.path.join(PACKAGE_DIR, dst_dir, os.path.basename(src))
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied += 1
        else:
            print(f"  Missing: {src_path}")

    print(f"  Copied {copied} scripts")


def copy_config():
    """Copy configuration files"""
    configs = [
        "config/run_config.yaml",
        "execution_log.json",
    ]
    for cfg in configs:
        if os.path.exists(cfg):
            shutil.copy2(cfg, f"{PACKAGE_DIR}/config/")
            print(f"  Copied: {cfg}")


def create_readme():
    """Create the package README."""
    readme = r"""# GeoAuditor reproducibility package

This directory provides a compact, package-local version of the GeoAuditor pipeline. It is intended for artifact review, controlled reruns, and manuscript-oriented output generation.

## Included components

- `config/run_config.yaml`: package-local experiment configuration
- `scripts/preprocessing/`: configuration freezing, split generation, XGBoost baselines, and prompt generation
- `scripts/inference/`: stage-1 inference and uncertainty estimation
- `scripts/training/`: optional structured-auditor training
- `scripts/analysis/`: oracle and ablation analysis
- `scripts/reporting/`: figures and tables
- `scripts/paper/`: paper-facing text assets
- `run_reproduction.sh`: single entry point for the compact pipeline

## Environment

Install the main repository requirements first:

```bash
pip install -r ../requirements.txt
```

Additional environment variables:

- `QWEN_MODEL_DIR` for local model-based training scripts
- `LLM_API_KEY` for scripts that call a remote completion endpoint

## Expected inputs

The package assumes that large data and output files are prepared outside version control. In particular, the pipeline expects:

- processed tabular data under `data/processed/`
- split and retrieval artifacts produced during preprocessing
- an available Qwen checkpoint for training or vLLM serving

## Execution order

### 1. Preprocessing and baseline

```bash
python scripts/preprocessing/freeze_config.py
python scripts/preprocessing/build_grid_block_split.py
python scripts/preprocessing/build_country_holdout_split.py
python scripts/preprocessing/train_xgboost_out_of_fold.py
python scripts/preprocessing/build_neighbor_index.py
python scripts/preprocessing/make_prompts.py
```

### 2. Inference

Start the vLLM server before inference:

```bash
python -m vllm.entrypoints.openai.api_server --model <path_to_qwen3_8b> --port 8000
python scripts/inference/baseline_uncertainty.py
python scripts/inference/stage1_vllm.py
python scripts/inference/risk_coverage.py
```

### 3. Optional auditor training

```bash
python scripts/training/train_json_auditor.py
python scripts/training/evaluate_json_auditor.py
```

### 4. Analysis and ablations

```bash
python scripts/analysis/oracle_ablation.py
python scripts/analysis/generate_stage2_audits.py
```

### 5. Reporting assets

```bash
python scripts/reporting/make_figures.py
python scripts/reporting/make_tables.py
python scripts/paper/write_paper_assets.py
```

## Runner script

You can also execute the compact pipeline with:

```bash
bash run_reproduction.sh
```

## Configuration notes

- Grid cell size: `0.1` degrees
- Neighbor count: `5`
- Delta multiplier: `1.5`
- Default seed: `42`
- Oracle prompts are included for analysis only

## Anti-leakage rules

1. Held-out countries are excluded from both training and retrieval
2. Grid blocking prevents within-country train/test overlap
3. Out-of-fold predictions use `GroupKFold(groups=grid_id)`
4. Oracle variants are isolated from deployable evaluation settings
"""

    with open(f"{PACKAGE_DIR}/README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    print("  Created README.md")


def create_run_reproduction():
    """Create the one-command runner script"""
    run_all = """#!/bin/bash
# GeoAuditor - Full Pipeline
set -e

echo "=== Preprocessing and baseline ==="
python scripts/preprocessing/freeze_config.py
python scripts/preprocessing/build_grid_block_split.py
python scripts/preprocessing/build_country_holdout_split.py
python scripts/preprocessing/train_xgboost_out_of_fold.py
python scripts/preprocessing/build_neighbor_index.py
python scripts/preprocessing/make_prompts.py

echo "=== Inference ==="
echo "Start the vLLM server before running inference."
python scripts/inference/baseline_uncertainty.py
python scripts/inference/stage1_vllm.py
python scripts/inference/risk_coverage.py

echo "=== Optional auditor training ==="
python scripts/training/train_json_auditor.py

echo "=== Analysis and ablations ==="
python scripts/analysis/oracle_ablation.py
python scripts/analysis/generate_stage2_audits.py

echo "=== Figures and tables ==="
python scripts/reporting/make_figures.py
python scripts/reporting/make_tables.py

echo "=== Paper assets ==="
python scripts/paper/write_paper_assets.py

echo "=== Done! ==="
"""
    with open(f"{PACKAGE_DIR}/run_reproduction.sh", "w") as f:
        f.write(run_all)
    os.chmod(f"{PACKAGE_DIR}/run_reproduction.sh", 0o755)
    print("  Created run_reproduction.sh")


def main():
    print("=" * 60)
    print("Reproducibility package builder")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    print("\nCreating package structure...")
    create_package_structure()

    print("\nCopying scripts...")
    copy_scripts()

    print("\nCopying config...")
    copy_config()

    print("\nCreating README...")
    create_readme()

    print("\nCreating run_reproduction.sh...")
    create_run_reproduction()

    total_files = 0
    for root, dirs, files in os.walk(PACKAGE_DIR):
        total_files += len(files)

    print(f"\n{'=' * 60}")
    print(f"Package created: {PACKAGE_DIR}/")
    print(f"Total files: {total_files}")
    print("=" * 60)


if __name__ == "__main__":
    main()
