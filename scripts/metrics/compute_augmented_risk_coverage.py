#!/usr/bin/env python3
"""
Compute AUGRC summaries from prediction outputs.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
OUTPUT_DIR = "outputs/tables"
CACHE_DIR = "outputs/metrics_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def compute_augrc(
    df: pd.DataFrame,
    metric_fn="accuracy"
) -> pd.DataFrame:
    """
    ComputeAUGRC (Area Under Generalized Risk-Coverage)

    Args:
        df: DataFrame containing pred, label, entropy, task, and heldout_country
        metric_fn: one of 'accuracy', 'balanced_accuracy', or 'f1'

    Returns:
        DataFrame with AUGRC, E-AUGRC, full_risk per (task, country)
    """
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

            coverages = np.arange(1.0, 0.49, -0.01)
            risks = []

            for cov in coverages:
                k = max(1, int(cov * n))
                top_k = subset.iloc[:k]

                if metric_fn == "accuracy":
                    metric_val = (top_k["pred"] == top_k["label"]).mean()
                elif metric_fn == "balanced_accuracy":
                    # Balanced accuracy: average of recalls per class
                    y_true = top_k["label"].values
                    y_pred = top_k["pred"].values
                    recalls = []
                    for cls in [0, 1]:
                        mask = y_true == cls
                        if mask.sum() > 0:
                            recalls.append((y_pred[mask] == cls).mean())
                    metric_val = np.mean(recalls) if recalls else 0.0
                elif metric_fn == "f1":
                    # F1 score (binary)
                    y_true = top_k["label"].values
                    y_pred = top_k["pred"].values
                    tp = ((y_pred == 1) & (y_true == 1)).sum()
                    fp = ((y_pred == 1) & (y_true == 0)).sum()
                    fn = ((y_pred == 0) & (y_true == 1)).sum()
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    metric_val = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    raise ValueError(f"Unknown metric: {metric_fn}")

                risk = 1 - metric_val
                risks.append(risk)

            risks = np.array(risks)

            # Trapezoid integration: AUGRC
            augrc = np.trapz(risks, coverages)

            # Full risk (risk at 100% coverage)
            full_risk = risks[0]  # coverage=1.0

            # E-AUGRC: normalized relative to full-coverage risk
            # E-AUGRC = AUGRC - full_risk * (cov_max - cov_min)
            e_augrc = augrc - full_risk * (coverages[0] - coverages[-1])

            results.append({
                "task": task,
                "heldout_country": country,
                "AUGRC": augrc,
                "E-AUGRC": e_augrc,
                "full_coverage_risk": full_risk,
                "n_samples": len(subset),
                "metric": metric_fn
            })

    return pd.DataFrame(results)


def bootstrap_country_ci(
    df: pd.DataFrame,
    metric_fn="accuracy",
    n_bootstrap=500,
    random_seed=42
) -> pd.DataFrame:
    """
    Compute confidence intervals with country-level bootstrap resampling

    Args:
        df: raw prediction data
        metric_fn: metric function
        n_bootstrap: number of bootstrap draws
        random_seed: random seed

    Returns:
        DataFrame with mean, CI_lower, CI_upper per task
    """
    np.random.seed(random_seed)
    countries = df["heldout_country"].unique()
    tasks = df["task"].unique()

    bootstrap_results = {task: [] for task in tasks}

    for b in range(n_bootstrap):
        # Resample countries with replacement
        sampled_countries = np.random.choice(countries, size=len(countries), replace=True)

        # Build resampled dataset
        resampled_df = pd.concat([
            df[df["heldout_country"] == c]
            for c in sampled_countries
        ], ignore_index=True)

        # Compute AUGRC
        metrics_df = compute_augrc(resampled_df, metric_fn=metric_fn)

        # Aggregate by task (mean AUGRC across resampled countries)
        for task in tasks:
            task_augrc = metrics_df[metrics_df["task"] == task]["AUGRC"].mean()
            bootstrap_results[task].append(task_augrc)

    # Compute CI
    ci_results = []
    for task in tasks:
        boot_values = np.array(bootstrap_results[task])
        ci_results.append({
            "task": task,
            "mean_AUGRC": boot_values.mean(),
            "CI_lower": np.percentile(boot_values, 2.5),
            "CI_upper": np.percentile(boot_values, 97.5),
            "std": boot_values.std(),
            "metric": metric_fn
        })

    return pd.DataFrame(ci_results)


def generate_main_table(
    realistic_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    xgb_df: pd.DataFrame,
    metric_fn="accuracy"
) -> pd.DataFrame:
    """Generate the main AUGRC results table"""

    all_metrics = []

    # Realistic LLM
    if realistic_df is not None:
        metrics = compute_augrc(realistic_df, metric_fn=metric_fn)
        metrics["method"] = "LLM-Auditor (Realistic)"
        all_metrics.append(metrics)

    # Oracle LLM
    if oracle_df is not None:
        metrics = compute_augrc(oracle_df, metric_fn=metric_fn)
        metrics["method"] = "LLM-Auditor (Oracle UB)"
        all_metrics.append(metrics)

    # XGB Baseline
    if xgb_df is not None and "entropy" in xgb_df.columns:
        metrics = compute_augrc(xgb_df, metric_fn=metric_fn)
        metrics["method"] = "Neighbor-XGB"
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
                    "Method": method,
                    "Task": task.replace("is_", "").replace("_poor", ""),
                    "Mean AUGRC": task_df["AUGRC"].mean(),
                    "Mean E-AUGRC": task_df["E-AUGRC"].mean(),
                    "Worst AUGRC": task_df["AUGRC"].max(),
                    "Std AUGRC": task_df["AUGRC"].std(),
                    "Avg Risk@Full": task_df["full_coverage_risk"].mean()
                })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def generate_latex_table(summary_df: pd.DataFrame) -> str:
    """Generate the table in LaTeX format"""
    tasks = summary_df["Task"].unique()
    methods = summary_df["Method"].unique()

    # Header
    cols = " & ".join(["Method"] + list(tasks) + ["Avg"])
    header = f"\\begin{{tabular}}{{l{'c' * (len(tasks) + 1)}}}\n"
    header += "\\toprule\n"
    header += cols + " \\\\\n"
    header += "\\midrule\n"

    # Body
    body = ""
    for method in methods:
        method_df = summary_df[summary_df["Method"] == method]
        vals = []
        all_aurcs = []
        for task in tasks:
            task_val = method_df[method_df["Task"] == task]["Mean AUGRC"].values
            if len(task_val) > 0:
                vals.append(f"{task_val[0]:.4f}")
                all_aurcs.append(task_val[0])
            else:
                vals.append("--")
        avg = f"{np.mean(all_aurcs):.4f}" if all_aurcs else "--"
        vals.append(avg)
        body += f"{method} & " + " & ".join(vals) + " \\\\\n"

    # Footer
    footer = "\\bottomrule\n\\end{tabular}\n"
    caption = "\\caption{Main Results: AUGRC (Area Under Generalized Risk-Coverage, lower is better) across tasks and methods.}\n"

    return header + body + footer + caption


def main():
    print("=" * 60)
    print("AUGRC/E-AUGRC Metrics Computation")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    metric_fn = "accuracy"

    print("\nLoading prediction results...")
    realistic_df = None
    oracle_df = None
    xgb_df = None

    realistic_path = "outputs/inference/stage1_llm_zeroshot_predictions.parquet"
    if os.path.exists(realistic_path):
        realistic_df = pd.read_parquet(realistic_path)
        print(f"  Loaded Realistic: {len(realistic_df)} samples")

    oracle_path = "outputs/analysis/stage1_oracle_predictions.parquet"
    if os.path.exists(oracle_path):
        oracle_df = pd.read_parquet(oracle_path)
        print(f"  Loaded Oracle: {len(oracle_df)} samples")

    xgb_path = "outputs/inference/baseline_xgb_predictions.parquet"
    if os.path.exists(xgb_path):
        xgb_df = pd.read_parquet(xgb_path)
        print(f"  Loaded XGB: {len(xgb_df)} samples")

    print("\nGenerating main AUGRC table...")
    summary_df = generate_main_table(realistic_df, oracle_df, xgb_df, metric_fn=metric_fn)

    if summary_df is not None:
        summary_df.to_csv(f"{OUTPUT_DIR}/table_aug_rc_main.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR}/table_aug_rc_main.csv")

        latex = generate_latex_table(summary_df)
        with open(f"{OUTPUT_DIR}/table_aug_rc_main.tex", "w") as f:
            f.write(latex)
        print(f"  Saved: {OUTPUT_DIR}/table_aug_rc_main.tex")

        print("\n" + summary_df.round(4).to_string(index=False))

    # 2. Bootstrap CI
    print("\nComputing Bootstrap CI (500 resamples, this may take a few minutes)...")
    ci_results = []

    if realistic_df is not None:
        print("  Bootstrapping Realistic...")
        ci_realistic = bootstrap_country_ci(realistic_df, metric_fn=metric_fn, n_bootstrap=500)
        ci_realistic["method"] = "LLM-Auditor (Realistic)"
        ci_results.append(ci_realistic)

    if oracle_df is not None:
        print("  Bootstrapping Oracle...")
        ci_oracle = bootstrap_country_ci(oracle_df, metric_fn=metric_fn, n_bootstrap=500)
        ci_oracle["method"] = "LLM-Auditor (Oracle UB)"
        ci_results.append(ci_oracle)

    if xgb_df is not None and "entropy" in xgb_df.columns:
        print("  Bootstrapping XGB...")
        ci_xgb = bootstrap_country_ci(xgb_df, metric_fn=metric_fn, n_bootstrap=500)
        ci_xgb["method"] = "Neighbor-XGB"
        ci_results.append(ci_xgb)

    if ci_results:
        ci_df = pd.concat(ci_results, ignore_index=True)
        ci_df.to_csv(f"{OUTPUT_DIR}/table_aug_rc_bootstrap_ci.csv", index=False)
        print(f"\n  Saved: {OUTPUT_DIR}/table_aug_rc_bootstrap_ci.csv")
        print("\n" + ci_df.round(4).to_string(index=False))

    print("\nCaching per-country metrics for plotting...")
    all_detailed = []

    for method_name, df in [
        ("LLM-Auditor (Realistic)", realistic_df),
        ("LLM-Auditor (Oracle UB)", oracle_df),
        ("Neighbor-XGB", xgb_df)
    ]:
        if df is not None and (method_name != "Neighbor-XGB" or "entropy" in df.columns):
            metrics = compute_augrc(df, metric_fn=metric_fn)
            metrics["method"] = method_name
            all_detailed.append(metrics)

    if all_detailed:
        detailed_df = pd.concat(all_detailed, ignore_index=True)
        detailed_df.to_parquet(f"{CACHE_DIR}/augrc_per_country.parquet", index=False)
        print(f"  Saved: {CACHE_DIR}/augrc_per_country.parquet")

    print("\n" + "=" * 60)
    print("T01 AUGRC Metrics Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
