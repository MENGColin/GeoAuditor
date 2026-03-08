#!/bin/bash
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
