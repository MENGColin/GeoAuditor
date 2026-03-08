# -*- coding: utf-8 -*-
"""
Freeze the shared experiment configuration and feature schema.
"""

import os
import sys
import yaml
import hashlib
import numpy as np
import pandas as pd
import random
import torch
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data/processed/DHS_africa_30_1121.csv"
CONFIG_PATH = PROJECT_ROOT / "config/run_config.yaml"

SEED = 42
TASKS = ["is_water_poor", "is_electr_poor", "is_facility_poor", "is_tele_poor", "is_u5mr_poor"]

EXCLUDE_COLS = {
    "is_water_poor", "is_electr_poor", "is_facility_poor", "is_tele_poor", "is_u5mr_poor",
    "water_quintile", "electr_quintile", "facility_quintile", "tele_quintile", "u5mr_quintile",
    "water_time_gt30_rate", "electr_poor_rate", "facility_delivery_rate", "telephone", "u5mr_synth",
    "system:index", ".geo", "cluster_id", "country", "lat", "lon", "households_n",
}

MISSING_POLICY = {
    "numeric": "median",
    "categorical": "most_frequent",
    "invalid_value": -9999,
}

SPATIAL_PARAMS = {
    "cell_size": 0.1,
    "cell_sizes_sensitivity": [0.05, 0.1, 0.2],
    "K_neighbors": 5,
    "delta_dynamic_multiplier": 1.5,
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "eval_metric": "logloss",
}

def set_all_seeds(seed: int):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_hash(data) -> str:
    """Compute the SHA256 hash of the data"""
    if isinstance(data, pd.DataFrame):
        data_str = data.to_csv(index=False)
    elif isinstance(data, list):
        data_str = str(sorted(data))
    else:
        data_str = str(data)
    return hashlib.sha256(data_str.encode()).hexdigest()[:16]

def identify_feature_columns(df: pd.DataFrame, exclude_cols: set) -> list:
    """Identify feature columns while excluding labels and metadata"""
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if col.startswith("Unnamed"):
            continue
        feature_cols.append(col)
    return sorted(feature_cols)

def analyze_data(df: pd.DataFrame, tasks: list, feature_cols: list):
    """Summarize basic dataset statistics"""
    print("\n" + "="*60)
    print("Dataset summary report")
    print("="*60)
    
    print(f"\nSample count: {len(df)}")
    print(f"Country count: {df['country'].nunique()}")
    print(f"Feature count: {len(feature_cols)}")
    
    print("\nPositive rate by task:")
    for task in tasks:
        if task in df.columns:
            pos_rate = df[task].mean()
            n_valid = df[task].notna().sum()
            print(f"  {task}: {pos_rate:.3f} (n={n_valid})")
    
    print("\nSample count by country:")
    country_counts = df["country"].value_counts()
    for country, count in country_counts.head(10).items():
        print(f"  {country}: {count}")
    if len(country_counts) > 10:
        print(f"  ... {len(country_counts)} countries in total")
    
    print("\nMissing values by feature:")
    missing_info = []
    for col in feature_cols[:10]:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            n_invalid = (df[col] == MISSING_POLICY["invalid_value"]).sum() if df[col].dtype in [np.float64, np.int64] else 0
            if n_missing > 0 or n_invalid > 0:
                missing_info.append((col, n_missing, n_invalid))
    
    for col, n_missing, n_invalid in missing_info[:5]:
        print(f"  {col}: NA={n_missing}, Invalid={n_invalid}")

def main():
    print("Freeze configuration and feature schema")
    print("="*60)
    
    set_all_seeds(SEED)
    print(f"Random seed fixed: {SEED}")
    
    print(f"\nLoading data: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Raw data: {len(df)} samples, {len(df.columns)} columns")
    
    feature_cols = identify_feature_columns(df, EXCLUDE_COLS)
    print(f"Identified {len(feature_cols)} feature columns")
    
    analyze_data(df, TASKS, feature_cols)
    
    data_hash = compute_hash(df)
    feature_hash = compute_hash(feature_cols)
    missing_policy_hash = compute_hash(str(MISSING_POLICY))
    
    print(f"\nData hash: {data_hash}")
    print(f"Feature-column hash: {feature_hash}")
    print(f"Missing-policy hash: {missing_policy_hash}")
    
    config = {
        "meta": {
            "created_date": datetime.now().isoformat(),
            "seed": SEED,
            "data_file": str(INPUT_CSV),
            "n_samples": len(df),
            "n_countries": int(df["country"].nunique()),
            "n_features": len(feature_cols),
        },
        "hashes": {
            "data": data_hash,
            "features": feature_hash,
            "missing_policy": missing_policy_hash,
        },
        "tasks": TASKS,
        "feature_cols": feature_cols,
        "exclude_cols": sorted(list(EXCLUDE_COLS)),
        "missing_policy": MISSING_POLICY,
        "spatial_params": SPATIAL_PARAMS,
        "xgb_params": XGB_PARAMS,
        "task_stats": {
            task: {
                "positive_rate": float(df[task].mean()),
                "n_valid": int(df[task].notna().sum()),
            }
            for task in TASKS if task in df.columns
        },
        "country_list": sorted(df["country"].unique().tolist()),
    }
    
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"\nConfiguration saved to: {CONFIG_PATH}")
    
    print("\n" + "="*60)
    print("Configuration summary")
    print("="*60)
    print(f"Sample count: {config['meta']['n_samples']}")
    print(f"Country count: {config['meta']['n_countries']}")
    print(f"Feature count: {config['meta']['n_features']}")
    print(f"Task count: {len(TASKS)}")
    print(f"Grid cell size: {SPATIAL_PARAMS['cell_size']} deg")
    print(f"Neighbor countK: {SPATIAL_PARAMS['K_neighbors']}")
    
    print("\nPositive rate by task:")
    for task in TASKS:
        if task in config["task_stats"]:
            stats = config["task_stats"][task]
            print(f"  {task}: {stats['positive_rate']:.3f}")

if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
