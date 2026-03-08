#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate JSONL prompts for realistic and oracle evaluation settings.
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
import yaml
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
def load_index(heldout_country):
    """Load the retrieval index for the specified scenario"""
    index_path = f"data/indices/neighbor_index_{heldout_country}.pkl"
    if not os.path.exists(index_path):
        return None
    with open(index_path, "rb") as f:
        return pickle.load(f)

def get_neighbors(target_lat, target_lon, target_country, index_data, K=5, delta_min=0):
    """Retrieve the K neighbors for the target location"""
    if index_data is None:
        return []

    tree = index_data["tree"]
    coords = index_data["coords"]
    row_ids = index_data["row_ids"]
    countries = index_data["countries"]
    delta_dict = index_data.get("delta_dict", {})
    delta_dynamic = delta_dict.get(target_country, delta_min)

    target_coords_rad = np.radians([[target_lat, target_lon]])
    n_query = min(K * 3 + 10, len(coords))
    distances_rad, indices = tree.query(target_coords_rad, k=n_query)
    distances_km = distances_rad[0] * 6371.0

    neighbors = []
    for dist_km, idx in zip(distances_km, indices[0]):
        if dist_km > delta_dynamic and len(neighbors) < K:
            neighbors.append({
                "row_id": row_ids[idx],
                "distance_km": dist_km,
                "country": countries[idx],
                "lat": coords[idx][0],
                "lon": coords[idx][1]
            })
    return neighbors

def calculate_neighbor_stats_realistic(neighbors, index_data, task):
    """Compute neighbor statistics in realistic mode using OOF probabilities"""
    if not neighbors or index_data is None:
        return {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5, "topK_mean": 0.5}

    oof_probs = index_data.get("oof_probs", {}).get(task, {})
    probs = [oof_probs.get(n["row_id"], 0.5) for n in neighbors]

    if not probs:
        return {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5, "topK_mean": 0.5}

    probs = np.array(probs)
    return {
        "mean": float(np.mean(probs)),
        "std": float(np.std(probs)),
        "min": float(np.min(probs)),
        "max": float(np.max(probs)),
        "topK_mean": float(np.mean(sorted(probs, reverse=True)[:min(3, len(probs))]))
    }

def calculate_neighbor_stats_oracle(neighbors, df, task):
    """Compute neighbor statistics in oracle mode using true labels"""
    if not neighbors:
        return {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5, "topK_mean": 0.5, "label_rate": 0.5}

    neighbor_row_ids = [n["row_id"] for n in neighbors]
    labels = df[df["row_id"].isin(neighbor_row_ids)][task].values

    if len(labels) == 0:
        return {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5, "topK_mean": 0.5, "label_rate": 0.5}

    labels = labels.astype(float)
    return {
        "mean": float(np.mean(labels)),
        "std": float(np.std(labels)),
        "min": float(np.min(labels)),
        "max": float(np.max(labels)),
        "topK_mean": float(np.mean(sorted(labels, reverse=True)[:min(3, len(labels))])),
        "label_rate": float(np.mean(labels))
    }

def get_level(value):
    """Convert numeric values into bucketed text labels"""
    if value < 0.25:
        return "Very Low"
    elif value < 0.5:
        return "Low"
    elif value < 0.75:
        return "Medium"
    else:
        return "High"

def generate_prompt_stage1(sample, task, neighbor_stats, mode="realistic"):
    """Generate the Stage 1 prompt"""
    task_readable = task.replace("_", " ").replace("is ", "")

    prompt = f"""You are a poverty prediction expert. Predict whether this area is {task_readable} based on the following information:

Area information:
- Country: {sample["country"]}
- Latitude: {sample["lat"]:.4f}
- Longitude: {sample["lon"]:.4f}

Neighbor context ({mode}):
- Mean probability: {neighbor_stats["mean"]:.4f} ({get_level(neighbor_stats["mean"])})
- Std deviation: {neighbor_stats["std"]:.4f}
- Min probability: {neighbor_stats["min"]:.4f} ({get_level(neighbor_stats["min"])})
- Max probability: {neighbor_stats["max"]:.4f} ({get_level(neighbor_stats["max"])})
- Top 3 mean: {neighbor_stats["topK_mean"]:.4f} ({get_level(neighbor_stats["topK_mean"])})

Answer with 0 or 1 only."""

    return prompt

def generate_prompt_stage2(sample, task, neighbor_stats, mode="realistic"):
    """Generate the Stage 2 prompt"""
    task_readable = task.replace("_", " ").replace("is ", "")

    prompt = f"""You are a poverty prediction auditor. Analyze the following area for {task_readable}:

Area information:
- Country: {sample["country"]}
- Latitude: {sample["lat"]:.4f}
- Longitude: {sample["lon"]:.4f}

Neighbor context ({mode}):
- Mean probability: {neighbor_stats["mean"]:.4f}
- Std deviation: {neighbor_stats["std"]:.4f}
- Min probability: {neighbor_stats["min"]:.4f}
- Max probability: {neighbor_stats["max"]:.4f}
- Top 3 mean: {neighbor_stats["topK_mean"]:.4f}

Return a JSON object with keys: environmental_assessment, conflict_check, key_factors (list), audit_available (bool)."""

    return prompt

def main():
    print("="*60)
    print("Generate prompts")
    print(f"Time: {datetime.now()}")
    print("="*60)

    with open("config/run_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["path"]
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)

    if "row_id" not in df.columns:
        df["row_id"] = range(len(df))

    protocol_a_df = pd.read_csv("data/split_protocol_A_country_holdout.csv")
    tasks = config["data"]["tasks"]
    K = config["spatial"]["K"]

    print(f"Tasks: {tasks}")
    print(f"K neighbors: {K}")

    heldout_countries = sorted(protocol_a_df["heldout_country"].unique())

    for mode in ["realistic", "oracle"]:
        print(f"\n{'='*40}")
        print(f"Generating {mode} prompts")
        print(f"{'='*40}")

        train_prompts, val_prompts, test_prompts = [], [], []

        for heldout_country in tqdm(heldout_countries, desc=f"{mode}"):
            index_data = load_index(heldout_country)
            if index_data is None:
                print(f"  Warning: No index for {heldout_country}")
                continue

            scene_split = protocol_a_df[protocol_a_df["heldout_country"] == heldout_country]
            merged_df = df.merge(scene_split[["row_id", "is_test"]], on="row_id", how="left")

            train_df = merged_df[merged_df["is_test"] == 0].copy()
            test_df = merged_df[merged_df["is_test"] == 1].copy()

            np.random.seed(42)
            val_indices = np.random.choice(train_df.index, size=int(len(train_df) * 0.2), replace=False)
            val_df = train_df.loc[val_indices]
            train_df = train_df.drop(val_indices)

            for split_name, split_df, prompts_list in [
                ("train", train_df, train_prompts),
                ("val", val_df, val_prompts),
                ("test", test_df, test_prompts)
            ]:
                for _, row in split_df.iterrows():
                    for task in tasks:
                        neighbors = get_neighbors(row["lat"], row["lon"], row["country"], index_data, K)

                        if mode == "realistic":
                            stats = calculate_neighbor_stats_realistic(neighbors, index_data, task)
                        else:
                            stats = calculate_neighbor_stats_oracle(neighbors, df, task)

                        prompt_obj = {
                            "row_id": int(row["row_id"]),
                            "heldout_country": heldout_country,
                            "split": split_name,
                            "task": task,
                            "prompt_stage1": generate_prompt_stage1(row, task, stats, mode),
                            "prompt_stage2": generate_prompt_stage2(row, task, stats, mode),
                            "label": int(row[task])
                        }
                        prompts_list.append(prompt_obj)

        output_dir = f"data/prompts_{mode}"
        os.makedirs(output_dir, exist_ok=True)

        for split_name, prompts in [("train", train_prompts), ("val", val_prompts), ("test", test_prompts)]:
            output_path = os.path.join(output_dir, f"{split_name}.jsonl")
            with open(output_path, "w", encoding="utf-8") as f:
                for prompt in prompts:
                    json.dump(prompt, f, ensure_ascii=False)
                    f.write("\n")
            print(f"  Saved {len(prompts)} {split_name} prompts to {output_path}")

    print("\n" + "="*60)
    print("=== JSONL Prompts Summary ===")
    print("="*60)
    print("Realistic prompts: data/prompts_realistic/")
    print("Oracle prompts: data/prompts_oracle/")

if __name__ == "__main__":
    main()
