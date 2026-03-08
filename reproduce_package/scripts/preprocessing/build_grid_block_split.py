#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create the protocol B grid-based split used in within-country evaluation.
"""

import os
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
def calculate_grid_id(lat, lon, cell_size):
    """Calculate grid ID based on latitude, longitude and cell size"""
    grid_x = int(np.floor(lon / cell_size))
    grid_y = int(np.floor(lat / cell_size))
    return f"{grid_x}_{grid_y}"

def main():
    print("="*60)
    print("Create protocol B grid split")
    print(f"Time: {datetime.now()}")
    print("="*60)
    
    config_path = "config/run_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    data_path = config['data']['path']
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    if 'row_id' not in df.columns:
        df['row_id'] = range(len(df))
    
    cell_size = config['spatial']['cell_size']
    print(f"Cell size: {cell_size} deg (~{cell_size * 111:.1f} km at the equator)")
    
    df['grid_id'] = df.apply(lambda row: calculate_grid_id(row['lat'], row['lon'], cell_size), axis=1)
    
    results = []
    
    print(f"\nProcessing {df['country'].nunique()} countries...")
    
    for country in sorted(df['country'].unique()):
        country_df = df[df['country'] == country].copy()
        country_df = country_df.reset_index(drop=True)
        
        n_grids = country_df['grid_id'].nunique()
        n_samples = len(country_df)
        
        n_splits = min(5, n_grids)
        
        if n_splits < 2:
            country_df['fold'] = 0
            print(f"  {country}: {n_samples} samples, {n_grids} grids -> 1 fold (too few grids)")
        else:
            gkf = GroupKFold(n_splits=n_splits)
            country_df['fold'] = -1
            
            for fold, (_, test_idx) in enumerate(gkf.split(country_df, groups=country_df['grid_id'])):
                country_df.loc[country_df.index[test_idx], 'fold'] = fold
            
            print(f"  {country}: {n_samples} samples, {n_grids} grids -> {n_splits} folds")
        
        country_df['cell_size'] = cell_size
        results.append(country_df[['row_id', 'country', 'grid_id', 'fold', 'cell_size']])
    
    result_df = pd.concat(results, ignore_index=True)
    
    print("\n" + "="*40)
    print("Self-check: Grid overlap between folds")
    print("="*40)
    overlap_found = False
    
    for country in sorted(result_df['country'].unique()):
        country_df = result_df[result_df['country'] == country]
        folds = sorted(country_df['fold'].unique())
        
        for i, fold in enumerate(folds):
            fold_grids = set(country_df[country_df['fold'] == fold]['grid_id'])
            
            for other_fold in folds[i+1:]:
                other_fold_grids = set(country_df[country_df['fold'] == other_fold]['grid_id'])
                overlap = fold_grids.intersection(other_fold_grids)
                
                if overlap:
                    print(f"  WARNING: {country} - fold {fold} and {other_fold} share grids: {list(overlap)[:3]}...")
                    overlap_found = True
    
    if not overlap_found:
        print("  [OK] No grid overlap found between folds (within countries)")
    
    output_path = "data/split_protocol_B_grid_block.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("=== Grid Split Summary ===")
    print("="*60)
    print(f"Total samples: {len(result_df)}")
    print(f"Countries: {result_df['country'].nunique()}")
    print(f"Total grid cells: {result_df['grid_id'].nunique()}")
    print(f"Cell size: {cell_size} deg")
    print(f"\nFold distribution:")
    print(result_df['fold'].value_counts().sort_index())
    print(f"\nOutput saved to: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
