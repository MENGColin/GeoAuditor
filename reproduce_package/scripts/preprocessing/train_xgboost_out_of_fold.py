#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the XGBoost baseline and export out-of-fold predictions.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
def calculate_grid_id(lat, lon, cell_size):
    grid_x = int(np.floor(lon / cell_size))
    grid_y = int(np.floor(lat / cell_size))
    return f"{grid_x}_{grid_y}"

def main():
    print("="*60)
    print("Train XGBoost OOF predictions")
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
    feature_cols = config["data"]["feature_cols"]
    cell_size = config["spatial"]["cell_size"]

    print(f"Tasks: {tasks}")
    print(f"Features: {len(feature_cols)}")

    categorical_cols = []
    for col in feature_cols:
        if df[col].dtype == "object":
            categorical_cols.append(col)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"  Encoded categorical column: {col}")

    df["grid_id"] = df.apply(lambda row: calculate_grid_id(row["lat"], row["lon"], cell_size), axis=1)

    imputer = SimpleImputer(strategy="median")

    oof_results = []
    test_results = []
    auc_summary = []

    heldout_countries = sorted(protocol_a_df["heldout_country"].unique())

    for heldout_country in heldout_countries:
        print(f"\nProcessing held-out country: {heldout_country}")

        scene_split = protocol_a_df[protocol_a_df["heldout_country"] == heldout_country]
        merged_df = df.merge(scene_split[["row_id", "is_test"]], on="row_id", how="left")

        train_df = merged_df[merged_df["is_test"] == 0].copy().reset_index(drop=True)
        test_df = merged_df[merged_df["is_test"] == 1].copy().reset_index(drop=True)

        print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

        X_train_raw = train_df[feature_cols].values.astype(float)
        X_test_raw = test_df[feature_cols].values.astype(float)

        X_train = imputer.fit_transform(X_train_raw)
        X_test = imputer.transform(X_test_raw)

        for task in tasks:
            y_train = train_df[task].values
            y_test = test_df[task].values

            valid_train_mask = ~np.isnan(y_train)

            gkf = GroupKFold(n_splits=5)
            oof_pred = np.zeros(len(train_df))
            oof_fold = np.zeros(len(train_df), dtype=int)

            for fold, (train_idx, val_idx) in enumerate(gkf.split(
                X_train[valid_train_mask],
                y_train[valid_train_mask],
                groups=train_df.loc[valid_train_mask, "grid_id"]
            )):
                actual_train_idx = np.where(valid_train_mask)[0][train_idx]
                actual_val_idx = np.where(valid_train_mask)[0][val_idx]

                X_train_fold = X_train[actual_train_idx]
                X_val_fold = X_train[actual_val_idx]
                y_train_fold = y_train[actual_train_idx]

                model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    eval_metric="auc", use_label_encoder=False, verbosity=0
                )
                model.fit(X_train_fold, y_train_fold)
                oof_pred[actual_val_idx] = model.predict_proba(X_val_fold)[:, 1]
                oof_fold[actual_val_idx] = fold

            valid_mask = valid_train_mask & (oof_pred > 0)
            if valid_mask.sum() > 0:
                oof_auc = roc_auc_score(y_train[valid_mask], oof_pred[valid_mask])
                print(f"    {task}: OOF AUC = {oof_auc:.4f}")
                auc_summary.append({
                    "heldout_country": heldout_country, "task": task,
                    "oof_auc": oof_auc, "train_samples": len(train_df)
                })

            model.fit(X_train[valid_train_mask], y_train[valid_train_mask])
            test_pred = model.predict_proba(X_test)[:, 1]

            oof_df = train_df[["row_id", "country", "lat", "lon", "grid_id"]].copy()
            oof_df["task"] = task
            oof_df["p_hat_oof"] = oof_pred
            oof_df["fold_id"] = oof_fold
            oof_df["heldout_country"] = heldout_country
            oof_df["label"] = y_train
            oof_results.append(oof_df)

            test_df_result = test_df[["row_id", "country", "lat", "lon", "grid_id"]].copy()
            test_df_result["task"] = task
            test_df_result["p_hat_test"] = test_pred
            test_df_result["heldout_country"] = heldout_country
            test_df_result["label"] = y_test
            test_results.append(test_df_result)

    print("\nSaving results...")
    oof_result_df = pd.concat(oof_results, ignore_index=True)
    test_result_df = pd.concat(test_results, ignore_index=True)

    os.makedirs("data/predictions", exist_ok=True)
    oof_result_df.to_parquet("data/predictions/neighbor_xgb_oof_train_pred.parquet", index=False)
    test_result_df.to_parquet("data/predictions/neighbor_xgb_test_pred.parquet", index=False)

    auc_df = pd.DataFrame(auc_summary)
    auc_df.to_csv("data/predictions/xgb_auc_summary.csv", index=False)

    print(f"\nTotal OOF samples: {len(oof_result_df)}")
    print(f"Total test samples: {len(test_result_df)}")
    print("\nAverage OOF AUC per task:")
    for task in tasks:
        task_auc = auc_df[auc_df["task"] == task]["oof_auc"].mean()
        print(f"  {task}: {task_auc:.4f}")

if __name__ == "__main__":
    main()
