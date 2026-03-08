#!/usr/bin/env python3
import os, numpy as np, pandas as pd
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
OUTPUT_DIR = "outputs/case_study"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_predictions():
    llm_df = pd.read_parquet("outputs/inference/stage1_llm_zeroshot_predictions.parquet")
    xgb_df = pd.read_parquet("outputs/inference/baseline_xgb_predictions.parquet")
    # FIX: merge on (row_id, task)
    merged_df = llm_df.merge(
        xgb_df[["row_id", "task", "pred", "p1"]],
        on=["row_id", "task"],
        suffixes=("_llm", "_xgb")
    )
    print(f"Loaded: LLM {len(llm_df)}, XGB {len(xgb_df)}, Merged {len(merged_df)}")
    return merged_df

df = load_predictions()

# Stratum A
candidates_a = df[
    (df["pred_xgb"] != df["label"]) &
    (df["pred_llm"] == df["label"]) &
    (df["entropy"].notna())
].copy()
entropy_th = candidates_a["entropy"].quantile(0.30)
candidates_a = candidates_a[candidates_a["entropy"] <= entropy_th]
selected_a = []
for (c, t), g in candidates_a.groupby(["heldout_country", "task"]):
    selected_a.append(g.sample(n=min(len(g), 30), random_state=RANDOM_SEED))
stratum_a = pd.concat(selected_a, ignore_index=True)
if len(stratum_a) > 300:
    stratum_a = stratum_a.sample(n=300, random_state=RANDOM_SEED)
stratum_a["stratum"] = "A_llm_adds_value"
print(f"Stratum A: {len(stratum_a)}")

# Stratum B
candidates_b = df[(df["p1_xgb"] >= 0.4) & (df["p1_xgb"] <= 0.6)].copy()
selected_b = []
for (c, t), g in candidates_b.groupby(["heldout_country", "task"]):
    selected_b.append(g.sample(n=min(len(g), 30), random_state=RANDOM_SEED))
stratum_b = pd.concat(selected_b, ignore_index=True)
if len(stratum_b) > 300:
    stratum_b = stratum_b.sample(n=300, random_state=RANDOM_SEED)
stratum_b["stratum"] = "B_xgb_uncertain"
print(f"Stratum B: {len(stratum_b)}")

# Stratum C
entropy_th_c = df["entropy"].quantile(0.80)
candidates_c = df[df["entropy"] >= entropy_th_c].copy()
selected_c = []
for (c, t), g in candidates_c.groupby(["heldout_country", "task"]):
    selected_c.append(g.sample(n=min(len(g), 30), random_state=RANDOM_SEED))
stratum_c = pd.concat(selected_c, ignore_index=True)
if len(stratum_c) > 300:
    stratum_c = stratum_c.sample(n=300, random_state=RANDOM_SEED)
stratum_c["stratum"] = "C_high_entropy_defer"
print(f"Stratum C: {len(stratum_c)}")

# Merge
case_study_df = pd.concat([stratum_a, stratum_b, stratum_c], ignore_index=True)
print(f"\nTotal: {len(case_study_df)}")

# Validation
assert case_study_df[["row_id", "task"]].drop_duplicates().shape[0] == len(case_study_df), \
    "Duplicate (row_id, task) pairs!"
print("Validation passed")

# Save
case_study_df.to_parquet(f"{OUTPUT_DIR}/case_study_sample.parquet", index=False)
print(f"Saved to {OUTPUT_DIR}/case_study_sample.parquet")

import json, hashlib
from pathlib import Path
manifest = {
    "created_at": datetime.now().isoformat(),
    "random_seed": RANDOM_SEED,
    "total_samples": len(case_study_df),
    "strata": {
        s: {
            "count": len(case_study_df[case_study_df["stratum"] == s]),
            "task_dist": case_study_df[case_study_df["stratum"] == s]["task"].value_counts().to_dict()
        }
        for s in ["A_llm_adds_value", "B_xgb_uncertain", "C_high_entropy_defer"]
    }
}
with open(f"{OUTPUT_DIR}/case_study_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

ids_str = ",".join(map(str, sorted(case_study_df["row_id"].tolist())))
checksum = hashlib.md5(ids_str.encode()).hexdigest()
print(f"Reproducibility checksum: {checksum}")
with open(f"{OUTPUT_DIR}/reproducibility_checksum.txt", "w") as f:
    f.write(f"Random seed: {RANDOM_SEED}\nChecksum: {checksum}\n")
print("COMPLETE")
