# -*- coding: utf-8 -*-
"""
Create the protocol B grid-based split used in within-country evaluation.
"""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data/processed/DHS_africa_30_1121.csv"
CONFIG_PATH = PROJECT_ROOT / "config/run_config.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data/splits/split_protocol_B_grid_block.csv"

N_FOLDS = 5

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def create_grid_id(lat: float, lon: float, cell_size: float) -> str:
    """Compute the grid ID from latitude and longitude"""
    grid_x = int(np.floor(lon / cell_size))
    grid_y = int(np.floor(lat / cell_size))
    return f"{grid_x}_{grid_y}"

def make_grid_split(df: pd.DataFrame, cell_size: float, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Build blocked k-fold splits separately for each country
    Ensure that all samples with the same grid_id stay in the same fold
    """
    np.random.seed(seed)
    
    df = df.copy()
    df["grid_id"] = df.apply(lambda row: create_grid_id(row["lat"], row["lon"], cell_size), axis=1)
    
    df["fold"] = -1
    
    countries = df["country"].unique()
    
    for country in countries:
        country_mask = df["country"] == country
        country_df = df[country_mask]
        
        unique_grids = country_df["grid_id"].unique()
        n_grids = len(unique_grids)
        
        if n_grids < n_folds:
            grid_to_fold = {grid: i % n_folds for i, grid in enumerate(unique_grids)}
        else:
            shuffled_grids = np.random.permutation(unique_grids)
            grid_to_fold = {}
            for i, grid in enumerate(shuffled_grids):
                grid_to_fold[grid] = i % n_folds
        
        for idx in country_df.index:
            grid = df.loc[idx, "grid_id"]
            df.loc[idx, "fold"] = grid_to_fold[grid]
    
    return df

def validate_split(df: pd.DataFrame, n_folds: int):
    """Validate the split by checking that a grid_id never appears in multiple folds within one country"""
    print("\nValidating the split...")
    errors = []
    
    for country in df["country"].unique():
        country_df = df[df["country"] == country]
        
        for grid_id in country_df["grid_id"].unique():
            grid_df = country_df[country_df["grid_id"] == grid_id]
            folds_in_grid = grid_df["fold"].unique()
            
            if len(folds_in_grid) > 1:
                errors.append(f"country {country}, grid_id {grid_id} appears in multiple fold: {folds_in_grid}")
    
    if errors:
        print(f"Found {len(errors)} errors:")
        for err in errors[:5]:
            print(f"  {err}")
        raise ValueError("Split validation failed!")
    else:
        print("[OK] All checks passed: same grid_id appears in only one fold")
    
    print("\nSample distribution by fold:")
    for fold in range(n_folds):
        fold_df = df[df["fold"] == fold]
        print(f"  Fold {fold}: {len(fold_df)} samples ({len(fold_df)/len(df)*100:.1f}%)")
    
    print("\nPer-country fold distribution (first 5 countries):")
    for country in sorted(df["country"].unique())[:5]:
        country_df = df[df["country"] == country]
        fold_counts = country_df["fold"].value_counts().sort_index()
        print(f"  {country}: {fold_counts.to_dict()}")

def main():
    print("Create protocol B grid split")
    print("="*60)
    
    config = load_config()
    cell_size = config["spatial_params"]["cell_size"]
    seed = config["meta"]["seed"]
    
    print(f"Grid cell size: {cell_size} deg")
    print(f"random seed: {seed}")
    print(f"Number of folds: {N_FOLDS}")
    
    print(f"\nLoading data: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Sample count: {len(df)}")
    
    print("\nCreategridsplit...")
    df_split = make_grid_split(df, cell_size, N_FOLDS, seed)
    
    validate_split(df_split, N_FOLDS)
    
    output_df = df_split[["system:index", "country", "lat", "lon", "grid_id", "fold"]].copy()
    output_df["cell_size"] = cell_size
    output_df = output_df.rename(columns={"system:index": "row_id"})
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSplit results saved to: {OUTPUT_PATH}")
    
    print("\nGrid statistics:")
    print(f"  Total grid count: {df_split['grid_id'].nunique()}")
    grid_sizes = df_split.groupby(["country", "grid_id"]).size()
    print(f"  Average samples per grid: {grid_sizes.mean():.2f}")
    print(f"  Maximum samples per grid: {grid_sizes.max()}")
    print(f"  Minimum samples per grid: {grid_sizes.min()}")

if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
