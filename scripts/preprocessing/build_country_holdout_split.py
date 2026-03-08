# -*- coding: utf-8 -*-
"""
Create the leave-one-country-out split used for cross-country evaluation.
"""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data/processed/DHS_africa_30_1121.csv"
CONFIG_PATH = PROJECT_ROOT / "config/run_config.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data/splits/split_protocol_A_country_holdout.csv"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_country_ood_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the leave-one-country-out split
    Generate one is_test mask for each country
    """
    countries = sorted(df["country"].unique())
    
    results = []
    
    for heldout_country in countries:
        is_test = (df["country"] == heldout_country).astype(int)
        
        temp_df = pd.DataFrame({
            "row_id": df["system:index"],
            "country": df["country"],
            "heldout_country": heldout_country,
            "is_test": is_test,
        })
        results.append(temp_df)
    
    return pd.concat(results, ignore_index=True)

def validate_split(df_split: pd.DataFrame, df_orig: pd.DataFrame):
    """Validate the split"""
    print("\nValidating the split...")
    
    countries = df_orig["country"].unique()
    n_countries = len(countries)
    n_samples = len(df_orig)
    
    for country in countries:
        country_split = df_split[df_split["heldout_country"] == country]
        assert len(country_split) == n_samples, f"heldout_country={country} record count is incorrect"
    
    errors = []
    for country in countries:
        country_split = df_split[df_split["heldout_country"] == country]
        
        test_samples = country_split[country_split["is_test"] == 1]
        test_countries = test_samples["country"].unique()
        
        if len(test_countries) != 1 or test_countries[0] != country:
            errors.append(f"heldout_country={country}: test-sample country is incorrect")
        
        train_samples = country_split[country_split["is_test"] == 0]
        if country in train_samples["country"].values:
            errors.append(f"heldout_country={country}: training split still contains samples from the held-out country")
    
    if errors:
        for err in errors:
            print(f"  Error: {err}")
        raise ValueError("Split validation failed!")
    else:
        print("[OK] All checks passed")
    
    print(f"\nSplit summary:")
    print(f"  Total country count: {n_countries}")
    print(f"  Total sample count: {n_samples}")
    print(f"  Total records: {len(df_split)} (= {n_countries} x {n_samples})")
    
    print("\nTest-set size by country:")
    for country in sorted(countries)[:10]:
        country_split = df_split[df_split["heldout_country"] == country]
        n_test = country_split["is_test"].sum()
        print(f"  {country}: {n_test} Test samples, {n_samples - n_test} Training samples")
    if n_countries > 10:
        print(f"  ... {n_countries} countries in total")

def main():
    print("Create protocol A country split")
    print("="*60)
    
    config = load_config()
    
    print(f"Loading data: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Sample count: {len(df)}")
    print(f"Country count: {df['country'].nunique()}")
    
    print("\nCreate the leave-one-country-out split...")
    df_split = make_country_ood_split(df)
    
    validate_split(df_split, df)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_split.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSplit results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
