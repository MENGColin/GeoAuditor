#!/usr/bin/env python3
"""
Select case-study samples for downstream qualitative analysis.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = "outputs/case_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_predictions():
    """Load all prediction outputs"""
    print("Loading prediction results...")

    # LLM Realistic
    llm_df = pd.read_parquet("outputs/inference/stage1_llm_zeroshot_predictions.parquet")
    print(f"  [OK] LLM Realistic: {len(llm_df)} samples")

    # XGB
    xgb_df = pd.read_parquet("outputs/inference/baseline_xgb_predictions.parquet")
    print(f"  [OK] XGB: {len(xgb_df)} samples")

    # Merge on row_id
    merged_df = llm_df.merge(
        xgb_df[["row_id", "pred", "p1"]],
        on="row_id",
        suffixes=("_llm", "_xgb")
    )

    print(f"  [OK] Merged: {len(merged_df)} samples")

    return merged_df


def select_stratum_a(df: pd.DataFrame, target_n=300, max_per_country_task=30) -> pd.DataFrame:
    """
    Stratum A: XGB wrong + LLM correct + low entropy

    Show cases where the LLM is more accurate than XGB
    """
    print("\n" + "="*60)
    print("Selecting Stratum A: LLM adds value")
    print("="*60)

    candidates = df[
        (df["pred_xgb"] != df["label"]) &
        (df["pred_llm"] == df["label"]) &
        (df["entropy"].notna())
    ].copy()

    entropy_threshold = candidates["entropy"].quantile(0.30)
    candidates = candidates[candidates["entropy"] <= entropy_threshold]

    print(f"Candidates: {len(candidates)}")
    print(f"  Entropy threshold (30%): {entropy_threshold:.4f}")

    # Diversity sampling: cap per (country, task)
    selected = []
    for (country, task), group in candidates.groupby(["heldout_country", "task"]):
        n_to_sample = min(len(group), max_per_country_task)
        sampled = group.sample(n=n_to_sample, random_state=RANDOM_SEED)
        selected.append(sampled)

    selected_df = pd.concat(selected, ignore_index=True)

    if len(selected_df) > target_n:
        selected_df = selected_df.sample(n=target_n, random_state=RANDOM_SEED)

    print(f"Selected: {len(selected_df)} samples")
    print(f"  Task distribution: {selected_df['task'].value_counts().to_dict()}")
    print(f"  Country coverage: {selected_df['heldout_country'].nunique()} countries")

    selected_df["stratum"] = "A_llm_adds_value"

    return selected_df


def select_stratum_b(df: pd.DataFrame, target_n=300, max_per_country_task=30) -> pd.DataFrame:
    """
    Stratum B: low-confidence XGB(|p-0.5|<0.1)

    Show uncertain cases that warrant manual auditing
    """
    print("\n" + "="*60)
    print("Selecting Stratum B: XGB uncertain")
    print("="*60)

    candidates = df[
        (df["p1_xgb"] >= 0.4) &
        (df["p1_xgb"] <= 0.6)
    ].copy()

    print(f"Candidates: {len(candidates)}")
    print(f"  XGB prob range: [0.4, 0.6]")

    # Diversity sampling
    selected = []
    for (country, task), group in candidates.groupby(["heldout_country", "task"]):
        n_to_sample = min(len(group), max_per_country_task)
        sampled = group.sample(n=n_to_sample, random_state=RANDOM_SEED)
        selected.append(sampled)

    selected_df = pd.concat(selected, ignore_index=True)

    if len(selected_df) > target_n:
        selected_df = selected_df.sample(n=target_n, random_state=RANDOM_SEED)

    print(f"Selected: {len(selected_df)} samples")
    print(f"  Task distribution: {selected_df['task'].value_counts().to_dict()}")

    selected_df["stratum"] = "B_xgb_uncertain"

    return selected_df


def select_stratum_c(df: pd.DataFrame, target_n=300, max_per_country_task=30) -> pd.DataFrame:
    """
    Stratum C: high entropy(top 20%)

    Show cases that should be deferred
    """
    print("\n" + "="*60)
    print("Selecting Stratum C: High entropy / should defer")
    print("="*60)

    entropy_threshold = df["entropy"].quantile(0.80)
    candidates = df[df["entropy"] >= entropy_threshold].copy()

    print(f"Candidates: {len(candidates)}")
    print(f"  Entropy threshold (80%): {entropy_threshold:.4f}")

    # Diversity sampling
    selected = []
    for (country, task), group in candidates.groupby(["heldout_country", "task"]):
        n_to_sample = min(len(group), max_per_country_task)
        sampled = group.sample(n=n_to_sample, random_state=RANDOM_SEED)
        selected.append(sampled)

    selected_df = pd.concat(selected, ignore_index=True)

    if len(selected_df) > target_n:
        selected_df = selected_df.sample(n=target_n, random_state=RANDOM_SEED)

    print(f"Selected: {len(selected_df)} samples")
    print(f"  Task distribution: {selected_df['task'].value_counts().to_dict()}")

    selected_df["stratum"] = "C_high_entropy_defer"

    return selected_df


def create_sample_manifest(case_study_df: pd.DataFrame):
    """Create a sample manifest for reproducibility"""
    manifest = {
        "created_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "total_samples": len(case_study_df),
        "strata": {},
        "sample_ids_by_stratum": {}
    }

    for stratum in ["A_llm_adds_value", "B_xgb_uncertain", "C_high_entropy_defer"]:
        stratum_df = case_study_df[case_study_df["stratum"] == stratum]
        manifest["strata"][stratum] = {
            "count": len(stratum_df),
            "task_distribution": stratum_df["task"].value_counts().to_dict(),
            "country_coverage": int(stratum_df["heldout_country"].nunique())
        }
        manifest["sample_ids_by_stratum"][stratum] = stratum_df["row_id"].tolist()

    import json
    with open(f"{OUTPUT_DIR}/case_study_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved manifest to {OUTPUT_DIR}/case_study_manifest.json")


def main():
    print("=" * 60)
    print("Case Study Sample Selection")
    print("CRITICAL: analysis script")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    df = load_predictions()

    stratum_a = select_stratum_a(df, target_n=300, max_per_country_task=30)
    stratum_b = select_stratum_b(df, target_n=300, max_per_country_task=30)
    stratum_c = select_stratum_c(df, target_n=300, max_per_country_task=30)

    case_study_df = pd.concat([stratum_a, stratum_b, stratum_c], ignore_index=True)

    print("\n" + "="*60)
    print("Case Study Sample Summary")
    print("="*60)
    print(f"Total samples: {len(case_study_df)}")
    print(f"  Stratum A (LLM adds value): {len(stratum_a)}")
    print(f"  Stratum B (XGB uncertain): {len(stratum_b)}")
    print(f"  Stratum C (High entropy): {len(stratum_c)}")
    print(f"\nTask distribution:")
    print(case_study_df["task"].value_counts())
    print(f"\nCountry coverage: {case_study_df['heldout_country'].nunique()} / 30")

    output_path = f"{OUTPUT_DIR}/case_study_sample.parquet"
    case_study_df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")

    create_sample_manifest(case_study_df)

    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)

    assert case_study_df["row_id"].nunique() == len(case_study_df), \
        "Duplicate row_ids found!"
    print("Row IDs are unique")

    assert case_study_df["stratum"].isin([
        "A_llm_adds_value", "B_xgb_uncertain", "C_high_entropy_defer"
    ]).all(), "Invalid stratum label!"
    print("Stratum labels valid")

    required_cols = ["row_id", "task", "heldout_country", "pred_llm", "pred_xgb",
                     "label", "entropy", "p1_llm", "p1_xgb", "stratum"]
    missing_cols = set(required_cols) - set(case_study_df.columns)
    assert len(missing_cols) == 0, f"Missing columns: {missing_cols}"
    print("All required columns present")

    import hashlib
    ids_str = ",".join(map(str, sorted(case_study_df["row_id"].tolist())))
    checksum = hashlib.md5(ids_str.encode()).hexdigest()
    print(f"Reproducibility checksum: {checksum}")

    with open(f"{OUTPUT_DIR}/reproducibility_checksum.txt", "w") as f:
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write(f"Total samples: {len(case_study_df)}\n")
        f.write(f"Row IDs MD5: {checksum}\n")

    print("\n" + "="*60)
    print("CASE STUDY SAMPLE SELECTION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Build few-shot exemplars")
    print("2. Run Stage 2 audits (zero-shot + few-shot)")
    print("3. Select narrative cases")
    print("4. Generate figure panels")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "="*60)
        print("Error")
        print("="*60)
        print(str(e))
        import traceback
        traceback.print_exc()
        exit(1)
