# GeoAuditor

GeoAuditor is the research codebase behind our spatially aware auditing pipeline for selective poverty prediction across 30 African countries. The repository contains data preparation, leakage-controlled evaluation splits, baseline models, LLM inference, structured stage-2 auditors, analysis utilities, and a compact reproducibility package for artifact review.

## What this repository covers

- Cross-country evaluation with leave-one-country-out country holdouts
- Within-country evaluation with grid-blocked splits to reduce spatial leakage
- Neighbor-aware prompting and stage-1 LLM prediction pipelines
- Selective prediction metrics, including risk-coverage, AURC, and AUGRC
- Oracle-context, conflict, governance, and case-study analysis scripts
- Figure, table, and paper-asset generation for the main manuscript

## Repository structure

- `config/`: shared experiment configuration, including `run_config.yaml`
- `scripts/preprocessing/`: split construction, baseline training, prompt generation, and retrieval index building
- `scripts/inference/`: stage-1 inference and uncertainty estimation
- `scripts/training/`: LoRA and structured-auditor training scripts
- `scripts/analysis/`: ablations and hybrid-gating analysis
- `scripts/metrics/`: generalized risk-coverage evaluation
- `scripts/oracle_context/`: context-variant construction, inference, and analysis
- `scripts/conflict_analysis/`: neighborhood-conflict analysis utilities
- `scripts/governance/`: governance queue generation and audit triage helpers
- `scripts/case_study/`: narrative sample selection, stage-2 audits, and qualitative plots
- `scripts/reporting/`: figure and table generation
- `scripts/paper/`: paper-facing text assets
- `scripts/prompting/`: few-shot and self-consistency prompting experiments
- `reproduce_package/`: compact artifact-oriented subset of the pipeline
- `figures/`: static figures generated during exploratory analysis

## Environment setup

Create a clean Python environment and install the repository requirements:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Additional runtime dependencies:

- `vllm` for the main stage-1 inference path
- `geopandas` and `shapely` for map and spatial plotting scripts
- `bitsandbytes` if you want to run QLoRA-style training

Environment variables used by the codebase:

- `LLM_API_KEY`: required by scripts that call a remote completion endpoint
- `QWEN_MODEL_DIR`: local path to the Qwen checkpoint used by training scripts

## Expected data layout

Large datasets, checkpoints, and generated outputs are not stored in version control. The main scripts assume the following layout:

- `data/processed/DHS_africa_30_1121.csv` for the multi-country dataset
- `data/processed/NG_covariates_1107.csv` for the Nigeria semantic-training subset
- `data/splits/` for generated country-holdout and grid-block split files
- `data/indices/` for neighbor retrieval indices
- `data/predictions/` for baseline and stage-1 prediction outputs
- `outputs/` for analysis products, prompting experiments, and paper assets

## Reproducing the main pipeline

### 1. Preprocessing and baselines

```bash
python scripts/preprocessing/freeze_config.py
python scripts/preprocessing/build_grid_block_split.py
python scripts/preprocessing/build_country_holdout_split.py
python scripts/preprocessing/train_xgboost_out_of_fold.py
python scripts/preprocessing/build_neighbor_index.py
python scripts/preprocessing/make_prompts.py
```

### 2. Stage-1 inference

Start the vLLM server before running the local inference path:

```bash
python -m vllm.entrypoints.openai.api_server --model <path_to_qwen3_8b> --port 8000
python scripts/inference/baseline_uncertainty.py
python scripts/inference/stage1_vllm.py
python scripts/inference/risk_coverage.py
```

### 3. Optional stage-2 auditor training

```bash
python scripts/training/train_json_auditor.py
python scripts/training/evaluate_json_auditor.py
```

Optional LoRA training variants are grouped under `scripts/training/`, including `train_qwen_lora_multi_country.py`, `train_qwen_lora_nigeria_semantic.py`, and `train_qwen_lora_local.py`.

### 4. Analysis and stress tests

```bash
python scripts/analysis/oracle_ablation.py
python scripts/analysis/evaluate_hybrid_gating.py
python scripts/analysis/generate_stage2_audits.py
python scripts/oracle_context/run_context_variant_inference.py
python scripts/oracle_context/analyze_context_variants.py
python scripts/conflict_analysis/analyze_neighborhood_conflict.py
```

### 5. Reporting and paper assets

```bash
python scripts/reporting/make_figures.py
python scripts/reporting/make_tables.py
python scripts/paper/write_paper_assets.py
```

## Artifact-oriented reproduction path

If you only need the compact artifact workflow, use `reproduce_package/`. It contains a trimmed script set, package-local documentation, and a single runner:

```bash
bash reproduce_package/run_reproduction.sh
```

See `reproduce_package/README.md` for the package-local execution order and assumptions.

## Notes

- Oracle variants are included for analysis only and should not be treated as deployable settings
- API keys are read from environment variables and are not stored in the repository
- GPU hardware is required for vLLM serving and LoRA training; preprocessing and reporting scripts run on CPU

## Citation

If you use this repository, please cite the accompanying paper and reference this repository in any artifact or software appendix.

## License

See `LICENSE`.
