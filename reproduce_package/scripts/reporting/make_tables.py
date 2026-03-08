#!/usr/bin/env python3
"""
Generate manuscript tables from the processed experiment outputs.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
OUTPUT_DIR = "outputs/reporting"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METHOD_NAMES = {
    "XGB": "XGBoost",
    "NeighborXGB": "Neighbor-XGB",
    "realistic": "LLM-Auditor (Realistic)",
    "oracle": "LLM-Auditor (Oracle UB)"
}

TASK_NAMES = {
    "is_water_poor": "Water",
    "is_electr_poor": "Electricity",
    "is_facility_poor": "Facility",
    "is_tele_poor": "Telecom",
    "is_u5mr_poor": "U5MR"
}


def compute_aurc(df):
    """Compute AURC"""
    results = []
    for task in df["task"].unique():
        for country in df["heldout_country"].unique():
            subset = df[
                (df["task"] == task) &
                (df["heldout_country"] == country) &
                (df["pred"].notna())
            ].copy()
            if len(subset) == 0:
                continue
            subset = subset.sort_values("entropy")
            n = len(subset)
            coverages = np.arange(0.5, 1.01, 0.01)
            risks = []
            for cov in coverages:
                k = max(1, int(cov * n))
                top_k = subset.iloc[:k]
                risks.append((top_k["pred"] != top_k["label"]).mean())
            risks = np.array(risks)
            aurc = np.trapz(risks, coverages)
            full_risk = (subset["pred"] != subset["label"]).mean()
            e_aurc = aurc - full_risk * (coverages[-1] - coverages[0])
            results.append({
                "task": task,
                "heldout_country": country,
                "AURC": aurc,
                "E-AURC": e_aurc,
                "full_risk": full_risk,
                "n_samples": len(subset)
            })
    return pd.DataFrame(results)


def generate_main_table():
    """Generate the main table: Table 1 with AURC/E-AURC by task and method"""
    print("\n--- Generating Main Table (Table 1) ---")

    all_metrics = []

    # Realistic LLM
    realistic_path = "outputs/inference/stage1_llm_zeroshot_predictions.parquet"
    if os.path.exists(realistic_path):
        df = pd.read_parquet(realistic_path)
        metrics = compute_aurc(df)
        metrics["method"] = "realistic"
        all_metrics.append(metrics)

    # Oracle LLM
    oracle_path = "outputs/analysis/stage1_oracle_predictions.parquet"
    if os.path.exists(oracle_path):
        df = pd.read_parquet(oracle_path)
        metrics = compute_aurc(df)
        metrics["method"] = "oracle"
        all_metrics.append(metrics)

    # XGB Baseline
    xgb_path = "outputs/inference/baseline_xgb_predictions.parquet"
    if os.path.exists(xgb_path):
        df = pd.read_parquet(xgb_path)
        if "entropy" in df.columns:
            metrics = compute_aurc(df)
            metrics["method"] = "NeighborXGB"
            all_metrics.append(metrics)

    if not all_metrics:
        print("No metrics data available")
        return None

    all_df = pd.concat(all_metrics, ignore_index=True)

    summary_rows = []
    for method in all_df["method"].unique():
        method_df = all_df[all_df["method"] == method]
        for task in sorted(all_df["task"].unique()):
            task_df = method_df[method_df["task"] == task]
            if len(task_df) > 0:
                summary_rows.append({
                    "Method": METHOD_NAMES.get(method, method),
                    "Task": TASK_NAMES.get(task, task),
                    "Mean AURC": task_df["AURC"].mean(),
                    "Mean E-AURC": task_df["E-AURC"].mean(),
                    "Worst AURC": task_df["AURC"].max(),
                    "Std AURC": task_df["AURC"].std(),
                    "Avg Risk@Full": task_df["full_risk"].mean()
                })

    summary_df = pd.DataFrame(summary_rows)

    pivot_aurc = summary_df.pivot_table(
        index="Method", columns="Task", values="Mean AURC", aggfunc="first"
    ).round(4)

    pivot_eaurc = summary_df.pivot_table(
        index="Method", columns="Task", values="Mean E-AURC", aggfunc="first"
    ).round(4)

    summary_df.to_csv(f"{OUTPUT_DIR}/table_main_results.csv", index=False)
    pivot_aurc.to_csv(f"{OUTPUT_DIR}/table1_aurc_pivot.csv")

    latex = generate_table1_latex(summary_df)
    with open(f"{OUTPUT_DIR}/table_main_results.tex", "w") as f:
        f.write(latex)

    print(f"Saved: table_main_results.csv, table_main_results.tex")
    print(summary_df.round(4).to_string(index=False))

    return summary_df


def generate_ablation_table():
    """Generate the ablation and upper-bound comparison table: Table 2"""
    print("\n--- Generating Ablation Table (Table 2) ---")

    rows = []

    for setup in ["realistic", "oracle"]:
        path_map = {
            "realistic": "outputs/inference/stage1_llm_zeroshot_predictions.parquet",
            "oracle": "outputs/analysis/stage1_oracle_predictions.parquet"
        }
        path = path_map[setup]
        if os.path.exists(path):
            df = pd.read_parquet(path)
            metrics = compute_aurc(df)
            for task in metrics["task"].unique():
                task_m = metrics[metrics["task"] == task]
                rows.append({
                    "Setting": METHOD_NAMES.get(setup, setup),
                    "Task": TASK_NAMES.get(task, task),
                    "Mean AURC": task_m["AURC"].mean(),
                    "Mean E-AURC": task_m["E-AURC"].mean(),
                    "Worst AURC": task_m["AURC"].max(),
                    "Variance": task_m["AURC"].var()
                })

    ablation_path = "outputs/analysis/ablation_metrics.csv"
    if os.path.exists(ablation_path):
        ablation_df = pd.read_csv(ablation_path)
        for _, row in ablation_df.iterrows():
            rows.append({
                "Setting": f"{row.get('setting_name', 'unknown')}={row.get('setting_value', '?')}",
                "Task": TASK_NAMES.get(row.get("task", ""), row.get("task", "")),
                "Mean AURC": row.get("AURC", ""),
                "Mean E-AURC": row.get("E-AURC", ""),
                "Worst AURC": "",
                "Variance": ""
            })

    if not rows:
        print("No ablation data available")
        return None

    ablation_df = pd.DataFrame(rows)

    ablation_df.to_csv(f"{OUTPUT_DIR}/table_ablation_oracle.csv", index=False)

    # LaTeX
    latex = generate_table2_latex(ablation_df)
    with open(f"{OUTPUT_DIR}/table_ablation_oracle.tex", "w") as f:
        f.write(latex)

    print(f"Saved: table_ablation_oracle.csv, table_ablation_oracle.tex")
    print(ablation_df.round(4).to_string(index=False))

    return ablation_df


def generate_table1_latex(df):
    """Generate the LaTeX for Table 1"""
    tasks = sorted(df["Task"].unique())
    methods = df["Method"].unique()

    cols = " & ".join(["Method"] + tasks + ["Avg"])
    header = f"\\begin{{tabular}}{{l{'c' * (len(tasks) + 1)}}}\n"
    header += "\\toprule\n"
    header += cols + " \\\\\n"
    header += "\\midrule\n"

    body = ""
    for method in methods:
        method_df = df[df["Method"] == method]
        vals = []
        all_aurcs = []
        for task in tasks:
            task_val = method_df[method_df["Task"] == task]["Mean AURC"].values
            if len(task_val) > 0:
                vals.append(f"{task_val[0]:.4f}")
                all_aurcs.append(task_val[0])
            else:
                vals.append("--")
        avg = f"{np.mean(all_aurcs):.4f}" if all_aurcs else "--"
        vals.append(avg)
        body += f"{method} & " + " & ".join(vals) + " \\\\\n"

    footer = "\\bottomrule\n\\end{tabular}\n"
    caption = "\\caption{Main Results: AURC (lower is better) across tasks and methods.}\n"

    return header + body + footer + caption


def generate_table2_latex(df):
    """Generate the LaTeX for Table 2"""
    cols = ["Setting", "Task", "Mean AURC", "Mean E-AURC", "Worst AURC", "Variance"]

    header = f"\\begin{{tabular}}{{{'l' * 2}{'c' * 4}}}\n"
    header += "\\toprule\n"
    header += " & ".join(cols) + " \\\\\n"
    header += "\\midrule\n"

    body = ""
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            v = row[col]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        body += " & ".join(vals) + " \\\\\n"

    footer = "\\bottomrule\n\\end{tabular}\n"
    caption = "\\caption{Oracle Upper Bound \\\\& Ablation Results.}\n"

    return header + body + footer + caption


def main():
    print("=" * 60)
    print("Generate tables")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    main_table = generate_main_table()
    ablation_table = generate_ablation_table()

    print("\n" + "=" * 60)
    print("All tables generated!")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
