#!/usr/bin/env python3
"""
Evaluate hybrid gating policies that combine XGBoost and LLM predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy import stats

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / 'outputs' / 'hybrid'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load predictions
print('Loading prediction data...')
xgb_df = pd.read_parquet(BASE_DIR / 'outputs/inference/baseline_xgb_predictions.parquet')
llm_df = pd.read_parquet(BASE_DIR / 'outputs/inference/stage1_llm_zeroshot_predictions.parquet')

print(f'XGB samples: {len(xgb_df)}, LLM samples: {len(llm_df)}')
print(f'XGB columns: {list(xgb_df.columns)}')
print(f'LLM columns: {list(llm_df.columns)}')

# Merge on common key
merge_keys = ['row_id'] if 'row_id' in xgb_df.columns else ['sample_id']
if 'row_id' not in xgb_df.columns and 'sample_id' not in xgb_df.columns:
    # Try index-based merge
    xgb_df = xgb_df.reset_index()
    llm_df = llm_df.reset_index()
    merge_keys = ['index']

# Check available columns for merging
print(f'Attempting merge on: {merge_keys}')
