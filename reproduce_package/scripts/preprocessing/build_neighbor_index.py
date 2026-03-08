#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the spatial retrieval index used for neighbor-aware prompting.
"""

import os
import numpy as np
import pandas as pd
import pickle
import yaml
from sklearn.neighbors import BallTree
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
def calculate_delta_dynamic(train_df, multiplier=1.5):
    """Compute a dynamic distance threshold for each country"""
    delta_dict = {}
    
    for country in train_df["country"].unique():
        country_df = train_df[train_df["country"] == country]
        coords = country_df[["lat", "lon"]].values
        
        if len(coords) > 1:
            coords_rad = np.radians(coords)
            tree = BallTree(coords_rad, metric="haversine")
            distances_rad, _ = tree.query(coords_rad, k=2)
            avg_nn_dist = np.mean(distances_rad[:, 1]) * 6371.0
            delta_dynamic = multiplier * avg_nn_dist
            delta_dict[country] = delta_dynamic
        else:
            delta_dict[country] = 10.0
    
    return delta_dict

def build_retrieval_index(train_df, oof_pred_df, output_path):
    """Build the retrieval index"""
    coords = train_df[["lat", "lon"]].values
    coords_rad = np.radians(coords)
    tree = BallTree(coords_rad, metric="haversine")
    
    index_data = {
        "tree": tree,
        "coords": coords,
        "row_ids": train_df["row_id"].values,
        "countries": train_df["country"].values,
        "lat": train_df["lat"].values,
        "lon": train_df["lon"].values,
        "oof_probs": oof_pred_df
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(index_data, f)
    
    print(f"  Index saved to: {output_path}")
    return index_data

def main():
    print("="*60)
    print("Build retrieval index")
    print(f"Time: {datetime.now()}")
    print("="*60)
    
    with open("config/run_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    data_path = config["data"]["path"]
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    if "row_id" not in df.columns:
        df["row_id"] = range(len(df))
    
    oof_path = "data/predictions/neighbor_xgb_oof_train_pred.parquet"
    if os.path.exists(oof_path):
        print(f"Loading OOF predictions from: {oof_path}")
        oof_df = pd.read_parquet(oof_path)
    else:
        print(f"Warning: OOF predictions not found at {oof_path}")
        print("Please run train_xgboost_out_of_fold.py first")
        oof_df = None
    
    protocol_a_df = pd.read_csv("data/split_protocol_A_country_holdout.csv")
    
    multiplier = config["spatial"]["delta_dynamic_multiplier"]
    tasks = config["data"]["tasks"]
    
    os.makedirs("data/indices", exist_ok=True)
    
    heldout_countries = sorted(protocol_a_df["heldout_country"].unique())
    
    for heldout_country in heldout_countries:
        print(f"\nBuilding index for held-out country: {heldout_country}")
        
        scene_split = protocol_a_df[protocol_a_df["heldout_country"] == heldout_country]
        merged_df = df.merge(scene_split[["row_id", "is_test"]], on="row_id", how="left")
        train_df = merged_df[merged_df["is_test"] == 0].copy().reset_index(drop=True)
        
        print(f"  Train samples for index: {len(train_df)}")
        
        oof_probs = {}
        if oof_df is not None:
            scene_oof = oof_df[oof_df["heldout_country"] == heldout_country]
            for task in tasks:
                task_oof = scene_oof[scene_oof["task"] == task][["row_id", "p_hat_oof"]]
                oof_probs[task] = dict(zip(task_oof["row_id"], task_oof["p_hat_oof"]))
        
        delta_dict = calculate_delta_dynamic(train_df, multiplier)
        avg_delta = np.mean(list(delta_dict.values()))
        print(f"  Average delta_dynamic: {avg_delta:.2f} km")
        
        output_path = f"data/indices/neighbor_index_{heldout_country}.pkl"
        index_data = build_retrieval_index(train_df, oof_probs, output_path)
        index_data["delta_dict"] = delta_dict
        
        with open(output_path, "wb") as f:
            pickle.dump(index_data, f)
    
    print("\n" + "="*60)
    print("=== Retrieval Index Summary ===")
    print("="*60)
    print(f"Built {len(heldout_countries)} indices")
    print(f"Index location: data/indices/neighbor_index_{{country}}.pkl")

if __name__ == "__main__":
    main()
