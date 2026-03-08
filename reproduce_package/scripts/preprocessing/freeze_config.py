#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Freeze the shared experiment configuration and feature schema.
"""

import os
import json
import yaml
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
def calculate_hash(data):
    """Calculate SHA256 hash of data"""
    if isinstance(data, pd.DataFrame):
        data_str = data.to_csv(index=False).encode('utf-8')
    elif isinstance(data, list):
        data_str = json.dumps(sorted(data) if all(isinstance(x, str) for x in data) else data, sort_keys=True).encode('utf-8')
    else:
        data_str = str(data).encode('utf-8')
    return hashlib.sha256(data_str).hexdigest()[:16]

def main():
    print("="*60)
    print("Freeze configuration and feature schema")
    print(f"Time: {datetime.now()}")
    print("="*60)
    
    seed = 42
    np.random.seed(seed)
    
    tasks = ["is_water_poor", "is_electr_poor", "is_facility_poor", "is_tele_poor", "is_u5mr_poor"]
    
    data_path = "data/processed/DHS_data/DHS_africa_30_1121.csv"
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    if 'row_id' not in df.columns:
        df['row_id'] = range(len(df))
    
    label_cols = tasks + ['electr_quintile', 'water_quintile', 'facility_quintile', 'tele_quintile', 'u5mr_quintile']
    
    leak_cols = [
        'electr_poor_rate', 'facility_delivery_rate', 'water_time_gt30_rate',
        'telephone', 'u5mr_synth', 'households_n'
    ]
    
    meta_cols = ['system:index', 'row_id', 'country', 'lat', 'lon', 'cluster_id', '.geo']
    
    exclude_cols = set(label_cols + leak_cols + meta_cols)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\nFeature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols):
        print(f"  {i+1}. {col}")
    
    missing_policy = {
        "numeric": "median",
        "categorical": "most_frequent"
    }
    
    print(f"\nMissing values in features:")
    missing_counts = df[feature_cols].isnull().sum()
    for col in feature_cols:
        if missing_counts[col] > 0:
            print(f"  {col}: {missing_counts[col]} ({missing_counts[col]/len(df)*100:.1f}%)")
    
    data_hash = calculate_hash(df)
    feature_hash = calculate_hash(feature_cols)
    missing_policy_hash = calculate_hash(missing_policy)
    
    model_params = {
        "xgb": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "eval_metric": "auc"
        }
    }
    
    spatial_params = {
        "cell_size": 0.1,
        "K": 5,
        "delta_dynamic_multiplier": 1.5,
    }
    
    config = {
        "meta": {
            "project_name": "KDD-GeoAuditor",
            "version": "1.0",
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": seed,
            "data_hash": data_hash,
            "feature_hash": feature_hash,
            "missing_policy_hash": missing_policy_hash
        },
        "data": {
            "path": data_path,
            "tasks": tasks,
            "feature_cols": feature_cols,
            "label_cols": label_cols,
            "leak_cols": leak_cols,
            "meta_cols": meta_cols,
            "missing_policy": missing_policy,
            "sample_count": len(df),
            "country_count": df["country"].nunique()
        },
        "spatial": spatial_params,
        "model": model_params
    }
    
    os.makedirs("config", exist_ok=True)
    config_path = "config/run_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    output_data_path = "data/DHS_africa_30_processed.csv"
    df.to_csv(output_data_path, index=False)
    print(f"\nProcessed data saved to: {output_data_path}")
    
    print("\n" + "="*60)
    print("=== Data Freeze Summary ===")
    print("="*60)
    print(f"Sample count: {len(df)}")
    print(f"Country count: {df['country'].nunique()}")
    print(f"Countries: {sorted(df['country'].unique())}")
    print(f"Feature count: {len(feature_cols)}")
    print()
    print("Task positive rates:")
    for task in tasks:
        pos_rate = df[task].mean() * 100
        count = df[task].sum()
        print(f"  {task}: {pos_rate:.1f}% ({count}/{len(df)})")
    print()
    print(f"Configuration saved to: {config_path}")
    print(f"Data hash: {data_hash}")
    print(f"Feature hash: {feature_hash}")
    print("="*60)

if __name__ == "__main__":
    main()
