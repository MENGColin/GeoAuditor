# -*- coding: utf-8 -*-
"""
Train the XGBoost baseline and export out-of-fold predictions.
"""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data/processed/DHS_africa_30_1121.csv"
CONFIG_PATH = PROJECT_ROOT / "config/run_config.yaml"
SPLIT_A_PATH = PROJECT_ROOT / "data/splits/split_protocol_A_country_holdout.csv"
SPLIT_B_PATH = PROJECT_ROOT / "data/splits/split_protocol_B_grid_block.csv"
OUTPUT_OOF_PATH = PROJECT_ROOT / "data/predictions/neighbor_xgb_oof_train_pred.parquet"
OUTPUT_TEST_PATH = PROJECT_ROOT / "data/predictions/neighbor_xgb_test_pred.parquet"

N_FOLDS = 5

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def prepare_features(df: pd.DataFrame, feature_cols: list, missing_policy: dict) -> pd.DataFrame:
    """Prepare features by handling missing and invalid values"""
    X = df[feature_cols].copy()
    
    invalid_value = missing_policy.get("invalid_value", -9999)
    X = X.replace(invalid_value, np.nan)
    
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64, float, int]:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
    
    return X

def train_xgb_oof(X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series, 
                   xgb_params: dict, n_folds: int = 5) -> np.ndarray:
    """
    Train XGBoost with GroupKFold and generate out-of-fold predictions
    The groups argument ensures that samples from the same grid_id never cross folds
    """
    oof_preds = np.zeros(len(X_train))
    
    gkf = GroupKFold(n_splits=n_folds)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    
    return oof_preds

def main():
    print("Train XGBoost OOF predictions")
    print("="*60)
    
    config = load_config()
    feature_cols = config["feature_cols"]
    tasks = config["tasks"]
    xgb_params = config["xgb_params"]
    missing_policy = config["missing_policy"]
    
    print(f"Feature count: {len(feature_cols)}")
    print(f"Task count: {len(tasks)}")
    
    print(f"\nLoading data...")
    df = pd.read_csv(INPUT_CSV)
    df_split_A = pd.read_csv(SPLIT_A_PATH)
    df_split_B = pd.read_csv(SPLIT_B_PATH)
    
    grid_info = df_split_B[["row_id", "grid_id"]].drop_duplicates()
    df = df.merge(grid_info, left_on="system:index", right_on="row_id", how="left")
    
    print("Preparing features...")
    X_all = prepare_features(df, feature_cols, missing_policy)
    
    countries = sorted(df["country"].unique())
    print(f"Country count: {len(countries)}")
    
    oof_results = []
    test_results = []
    
    for heldout_idx, heldout_country in enumerate(countries):
        print(f"\nProcessing heldout_country: {heldout_country} ({heldout_idx+1}/{len(countries)})")
        
        train_mask = df["country"] != heldout_country
        test_mask = df["country"] == heldout_country
        
        X_train = X_all[train_mask].reset_index(drop=True)
        X_test = X_all[test_mask].reset_index(drop=True)
        
        groups_train = df[train_mask]["grid_id"].reset_index(drop=True)
        
        train_row_ids = df[train_mask]["system:index"].values
        test_row_ids = df[test_mask]["system:index"].values
        
        print(f"  Train set: {len(X_train)}, Test set: {len(X_test)}")
        
        for task in tasks:
            y_train = df[train_mask][task].reset_index(drop=True)
            y_test = df[test_mask][task].reset_index(drop=True)
            
            oof_preds = train_xgb_oof(X_train, y_train, groups_train, xgb_params, N_FOLDS)
            
            try:
                oof_auc = roc_auc_score(y_train, oof_preds)
            except:
                oof_auc = np.nan
            
            model_full = XGBClassifier(**xgb_params)
            model_full.fit(X_train, y_train, verbose=False)
            test_preds = model_full.predict_proba(X_test)[:, 1]
            
            try:
                test_auc = roc_auc_score(y_test, test_preds)
            except:
                test_auc = np.nan
            
            print(f"    {task}: OOF_AUC={oof_auc:.4f}, Test_AUC={test_auc:.4f}")
            
            for i, row_id in enumerate(train_row_ids):
                oof_results.append({
                    "row_id": row_id,
                    "task": task,
                    "heldout_country": heldout_country,
                    "p_hat_oof": oof_preds[i],
                    "country": df[train_mask].iloc[i]["country"],
                })
            
            for i, row_id in enumerate(test_row_ids):
                test_results.append({
                    "row_id": row_id,
                    "task": task,
                    "heldout_country": heldout_country,
                    "p_hat_test": test_preds[i],
                })
    
    print("\nSaving outputs...")
    
    df_oof = pd.DataFrame(oof_results)
    df_test = pd.DataFrame(test_results)
    
    OUTPUT_OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_oof.to_parquet(OUTPUT_OOF_PATH, index=False)
    df_test.to_parquet(OUTPUT_TEST_PATH, index=False)
    
    print(f"OOF predictions saved to: {OUTPUT_OOF_PATH}")
    print(f"Test predictions saved to: {OUTPUT_TEST_PATH}")
    
    print("\n" + "="*60)
    print("Summary statistics")
    print("="*60)
    print(f"OOF record count: {len(df_oof)}")
    print(f"Test record count: {len(df_test)}")
    
    print("\nAverage performance by task:")
    for task in tasks:
        task_oof = df_oof[df_oof["task"] == task]
        task_test = df_test[df_test["task"] == task]
        
        print(f"  {task}: Average p_hat={task_oof['p_hat_oof'].mean():.4f}")

if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
