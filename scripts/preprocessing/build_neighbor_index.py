# -*- coding: utf-8 -*-
"""
Build the spatial retrieval index used for neighbor-aware prompting.
"""

import os
import yaml
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.neighbors import BallTree
from typing import Tuple, List, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data/processed/DHS_africa_30_1121.csv"
CONFIG_PATH = PROJECT_ROOT / "config/run_config.yaml"
OUTPUT_INDEX_PATH = PROJECT_ROOT / "data/indices/neighbor_index_info.pkl"

EARTH_RADIUS_KM = 6371.0

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_avg_nn_distance(coords: np.ndarray, sample_size: int = 1000) -> float:
    """Compute the average nearest-neighbor distance"""
    if len(coords) <= 1:
        return 0.0
    
    coords_rad = np.radians(coords)
    tree = BallTree(coords_rad, metric="haversine")
    
    n_samples = min(sample_size, len(coords))
    sample_idx = np.random.choice(len(coords), n_samples, replace=False)
    sample_coords = coords_rad[sample_idx]
    
    distances, _ = tree.query(sample_coords, k=2)
    nn_distances = distances[:, 1] * EARTH_RADIUS_KM
    
    return np.mean(nn_distances)

def build_neighbor_index_for_country(df_train: pd.DataFrame, heldout_country: str) -> Dict:
    """Build the neighbor index for a specific held-out country"""
    coords = df_train[["lat", "lon"]].values
    coords_rad = np.radians(coords)
    
    tree = BallTree(coords_rad, metric="haversine")
    
    country_avg_nn_dist = {}
    for country in df_train["country"].unique():
        country_mask = df_train["country"] == country
        country_coords = coords[country_mask]
        if len(country_coords) > 1:
            avg_dist = compute_avg_nn_distance(country_coords)
            country_avg_nn_dist[country] = avg_dist
        else:
            country_avg_nn_dist[country] = 0.0
    
    return {
        "tree": tree,
        "coords": coords,
        "coords_rad": coords_rad,
        "row_ids": df_train["system:index"].values,
        "countries": df_train["country"].values,
        "country_avg_nn_dist": country_avg_nn_dist,
        "heldout_country": heldout_country,
    }

def get_neighbors(index_info: Dict, lat: float, lon: float, K: int, 
                   delta_multiplier: float, target_country: str = None) -> Tuple[List[str], List[float]]:
    """Query neighbors"""
    tree = index_info["tree"]
    coords = index_info["coords"]
    row_ids = index_info["row_ids"]
    country_avg_nn_dist = index_info["country_avg_nn_dist"]
    
    if target_country and target_country in country_avg_nn_dist:
        delta_dynamic = delta_multiplier * country_avg_nn_dist[target_country]
    else:
        delta_dynamic = delta_multiplier * np.mean(list(country_avg_nn_dist.values()))
    
    query_rad = np.radians([[lat, lon]])
    
    k_query = min(K * 3, len(coords))
    distances_rad, indices = tree.query(query_rad, k=k_query)
    
    distances_km = distances_rad[0] * EARTH_RADIUS_KM
    indices = indices[0]
    
    neighbor_row_ids = []
    neighbor_distances = []
    
    for dist, idx in zip(distances_km, indices):
        if dist > delta_dynamic and len(neighbor_row_ids) < K:
            neighbor_row_ids.append(row_ids[idx])
            neighbor_distances.append(dist)
    
    return neighbor_row_ids, neighbor_distances

def main():
    print("Build retrieval index")
    print("="*60)
    
    config = load_config()
    spatial_params = config["spatial_params"]
    K = spatial_params["K_neighbors"]
    delta_multiplier = spatial_params["delta_dynamic_multiplier"]
    
    print(f"Neighbor count K: {K}")
    print(f"Distance threshold multiplier: {delta_multiplier}")
    
    print(f"\nLoading data...")
    df = pd.read_csv(INPUT_CSV)
    
    countries = sorted(df["country"].unique())
    print(f"Country count: {len(countries)}")
    
    all_indices = {}
    country_stats = {}
    
    for heldout_idx, heldout_country in enumerate(countries):
        print(f"\nBuilding the index - heldout_country: {heldout_country} ({heldout_idx+1}/{len(countries)})")
        
        train_mask = df["country"] != heldout_country
        df_train = df[train_mask].copy().reset_index(drop=True)
        
        print(f"  Train setSample count: {len(df_train)}")
        
        index_info = build_neighbor_index_for_country(df_train, heldout_country)
        
        all_indices[heldout_country] = {
            "coords": index_info["coords"],
            "row_ids": index_info["row_ids"],
            "countries": index_info["countries"],
            "country_avg_nn_dist": index_info["country_avg_nn_dist"],
        }
        
        avg_nn_dist = np.mean(list(index_info["country_avg_nn_dist"].values()))
        country_stats[heldout_country] = {
            "n_train": len(df_train),
            "avg_nn_dist_km": avg_nn_dist,
            "delta_dynamic_km": delta_multiplier * avg_nn_dist,
        }
        
        print(f"  Average nearest-neighbor distance: {avg_nn_dist:.2f} km")
    
    OUTPUT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        "indices": all_indices,
        "country_stats": country_stats,
        "K": K,
        "delta_multiplier": delta_multiplier,
    }
    
    with open(OUTPUT_INDEX_PATH, "wb") as f:
        pickle.dump(save_data, f)
    
    print(f"\nIndex metadata saved to: {OUTPUT_INDEX_PATH}")
    
    print("\n" + "="*60)
    print("Summary statistics")
    print("="*60)
    
    all_avg_nn = [stats["avg_nn_dist_km"] for stats in country_stats.values()]
    print(f"Average nearest-neighbor distance: {np.mean(all_avg_nn):.2f} km")

if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
