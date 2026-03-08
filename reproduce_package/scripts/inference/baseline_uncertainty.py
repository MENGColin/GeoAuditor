#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute uncertainty estimates for the baseline model outputs.
"""

import os
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
def load_config():
    with open("config/run_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_entropy(p):
    """Compute binary entropy"""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def calculate_margin(p):
    """Compute the margin from 0.5"""
    return np.abs(p - 0.5)


def main():
    print("=" * 60)
    print("Compute baseline uncertainty")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    config = load_config()
    tasks = config["data"]["tasks"]

    test_pred_path = "data/predictions/neighbor_xgb_test_pred.parquet"
    print(f"\nLoading test predictions from: {test_pred_path}")
    test_df = pd.read_parquet(test_pred_path)
    print(f"Loaded {len(test_df)} predictions")

    print("\nCalculating uncertainty metrics...")

    test_df["p1"] = test_df["p_hat_test"]
    test_df["p0"] = 1 - test_df["p1"]

    test_df["entropy"] = calculate_entropy(test_df["p1"])
    test_df["margin"] = calculate_margin(test_df["p1"])

    test_df["pred"] = (test_df["p1"] >= 0.5).astype(int)

    test_df["method"] = "XGB"

    output_cols = [
        "row_id", "task", "heldout_country", "method",
        "p0", "p1", "entropy", "margin", "pred", "label"
    ]
    baseline_df = test_df[output_cols].copy()

    output_dir = "outputs/inference"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "baseline_xgb_predictions.parquet")
    baseline_df.to_parquet(output_path, index=False)
    print(f"\nSaved baseline predictions to {output_path}")

    print(f"\n{'=' * 40}")
    print("Summary Statistics:")
    print(f"{'=' * 40}")
    print(f"Total samples: {len(baseline_df)}")

    print("\nPer-task statistics:")
    for task in tasks:
        task_df = baseline_df[baseline_df["task"] == task]
        accuracy = (task_df["pred"] == task_df["label"]).mean()
        avg_entropy = task_df["entropy"].mean()
        avg_margin = task_df["margin"].mean()
        print(f"  {task}:")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    Avg Entropy: {avg_entropy:.4f}")
        print(f"    Avg Margin: {avg_margin:.4f}")

    print("\nPer-country Accuracy (averaged across tasks):")
    country_acc = baseline_df.groupby("heldout_country").apply(
        lambda x: (x["pred"] == x["label"]).mean()
    ).sort_values()

    print(f"  Worst: {country_acc.index[0]} ({country_acc.iloc[0]:.4f})")
    print(f"  Best: {country_acc.index[-1]} ({country_acc.iloc[-1]:.4f})")
    print(f"  Mean: {country_acc.mean():.4f}")
    print(f"  Std: {country_acc.std():.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
