#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create the leave-one-country-out split used for cross-country evaluation.
"""

import os
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
def main():
    print("="*60)
    print("Create protocol A country split")
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
    
    countries = sorted(df['country'].unique())
    print(f"Processing {len(countries)} countries for Protocol A split")
    print(f"Countries: {countries}")
    
    results = []
    
    print("\nGenerating splits for each held-out country:")
    for heldout_country in countries:
        train_count = len(df[df['country'] != heldout_country])
        test_count = len(df[df['country'] == heldout_country])
        
        print(f"  {heldout_country}: train={train_count}, test={test_count}")
        
        country_split = df[['row_id', 'country']].copy()
        country_split['heldout_country'] = heldout_country
        
        country_split['is_test'] = (country_split['country'] == heldout_country).astype(int)
        
        results.append(country_split)
    
    result_df = pd.concat(results, ignore_index=True)
    
    output_path = "data/split_protocol_A_country_holdout.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("=== Country OOD Split Summary ===")
    print("="*60)
    print(f"Total rows in output: {len(result_df)}")
    print(f"Original samples: {len(df)}")
    print(f"Countries: {len(countries)}")
    print(f"Splits generated: {len(countries)} (one per held-out country)")
    
    print("\nTest samples per held-out country:")
    test_counts = result_df[result_df['is_test'] == 1].groupby('heldout_country').size()
    for country in countries:
        print(f"  {country}: {test_counts[country]}")
    
    print(f"\nOutput saved to: {output_path}")
    print("\nNote: For each heldout_country, use only samples with is_test=0 for training/retrieval/OOF")
    print("="*60)

if __name__ == "__main__":
    main()
