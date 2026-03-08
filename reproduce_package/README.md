# GeoAuditor reproducibility package

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
