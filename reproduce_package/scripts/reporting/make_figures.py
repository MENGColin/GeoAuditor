#!/usr/bin/env python3
"""
Generate manuscript figures from the processed experiment outputs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
OUTPUT_DIR = "outputs/reporting"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_predictions(path):
    """Load prediction outputs"""
    if os.path.exists(path):
        return pd.read_parquet(path)
    print(f"Warning: {path} not found")
    return None


def compute_risk_coverage(df, method_name="LLM"):
    """Compute Risk-Coverage curve data"""
    curves = []

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

            for cov in coverages:
                k = max(1, int(cov * n))
                top_k = subset.iloc[:k]
                risk = (top_k["pred"] != top_k["label"]).mean()
                curves.append({
                    "method": method_name,
                    "task": task,
                    "heldout_country": country,
                    "coverage": cov,
                    "risk": risk,
                    "accuracy": 1 - risk
                })

    return pd.DataFrame(curves)


def plot_risk_coverage(curves_df, output_path="fig2_risk_coverage.pdf"):
    """Plot Risk-Coverage curves by task"""
    tasks = sorted(curves_df["task"].unique())
    methods = curves_df["method"].unique()

    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 4), sharey=True)
    if n_tasks == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        task_df = curves_df[curves_df["task"] == task]

        for method in methods:
            method_df = task_df[task_df["method"] == method]
            mean_curve = method_df.groupby("coverage")["risk"].mean().reset_index()
            std_curve = method_df.groupby("coverage")["risk"].std().reset_index()

            ax.plot(mean_curve["coverage"], mean_curve["risk"], label=method, linewidth=2)
            ax.fill_between(
                mean_curve["coverage"],
                mean_curve["risk"] - std_curve["risk"].fillna(0),
                mean_curve["risk"] + std_curve["risk"].fillna(0),
                alpha=0.15
            )

        ax.set_title(task.replace("is_", "").replace("_poor", ""), fontsize=11)
        ax.set_xlabel("Coverage")
        ax.set_xlim(0.5, 1.0)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Risk (Error Rate)")
    axes[0].legend(loc="upper left", fontsize=9)

    fig.suptitle("Figure 2: Risk-Coverage Curves (Lower is Better)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_worst_country(metrics_df, output_path="fig3_worst_country.pdf"):
    """Plot worst-country and variance analyses"""
    tasks = sorted(metrics_df["task"].unique())
    methods = metrics_df["method"].unique() if "method" in metrics_df.columns else ["LLM"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    task_data = []
    for task in tasks:
        for method in methods:
            if "method" in metrics_df.columns:
                task_metrics = metrics_df[(metrics_df["task"] == task) & (metrics_df["method"] == method)]
            else:
                task_metrics = metrics_df[metrics_df["task"] == task]
            task_data.append({
                "task": task.replace("is_", "").replace("_poor", ""),
                "method": method,
                "AURC": task_metrics["AURC"].values
            })

    x = np.arange(len(tasks))
    width = 0.35

    mean_aurcs = []
    worst_aurcs = []
    for task in tasks:
        if "method" in metrics_df.columns:
            task_m = metrics_df[metrics_df["task"] == task]
        else:
            task_m = metrics_df[metrics_df["task"] == task]
        mean_aurcs.append(task_m["AURC"].mean())
        worst_aurcs.append(task_m["AURC"].max())  # worst = highest AURC

    ax.bar(x - width/2, mean_aurcs, width, label="Mean AURC", color="steelblue")
    ax.bar(x + width/2, worst_aurcs, width, label="Worst Country", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("is_", "").replace("_poor", "") for t in tasks], rotation=15)
    ax.set_ylabel("AURC (Lower is Better)")
    ax.set_title("Mean vs Worst-Country AURC")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Right: Variance across countries
    ax = axes[1]
    variances = []
    for task in tasks:
        if "method" in metrics_df.columns:
            task_m = metrics_df[metrics_df["task"] == task]
        else:
            task_m = metrics_df[metrics_df["task"] == task]
        variances.append(task_m["AURC"].std())

    bars = ax.bar(x, variances, color="mediumseagreen")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("is_", "").replace("_poor", "") for t in tasks], rotation=15)
    ax.set_ylabel("Std Dev of AURC across Countries")
    ax.set_title("Country-Level Variance")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Figure 3: Worst-Country & Variance Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_sensitivity(ablation_df, output_path="fig4_sensitivity.pdf"):
    """Plot hyperparameter sensitivity curves"""
    if ablation_df is None or len(ablation_df) == 0:
        print("No ablation data available for sensitivity plot")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Ablation data pending\n(Run oracle_ablation.py first)",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_title("Figure 4: Sensitivity Analysis (Placeholder)")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.savefig(output_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        print(f"Saved placeholder: {output_path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    settings = ablation_df["setting_name"].unique()
    for i, setting in enumerate(settings[:2]):
        ax = axes[i] if i < 2 else axes[1]
        setting_df = ablation_df[ablation_df["setting_name"] == setting]

        if "AURC" in setting_df.columns:
            for task in setting_df["task"].unique():
                task_df = setting_df[setting_df["task"] == task]
                ax.plot(
                    task_df["setting_value"].astype(float),
                    task_df["AURC"],
                    marker="o",
                    label=task.replace("is_", "").replace("_poor", "")
                )
            ax.set_xlabel(setting)
            ax.set_ylabel("AURC")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Sensitivity: {setting}")

    fig.suptitle("Figure 4: Hyperparameter Sensitivity", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Generate figures")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    print("\nLoading data...")
    realistic_df = load_predictions("outputs/inference/stage1_llm_zeroshot_predictions.parquet")
    xgb_baseline = load_predictions("outputs/inference/baseline_xgb_predictions.parquet")
    oracle_df = load_predictions("outputs/analysis/stage1_oracle_predictions.parquet")

    print("\nComputing Risk-Coverage curves...")
    all_curves = []

    if realistic_df is not None:
        curves = compute_risk_coverage(realistic_df, "LLM-Auditor(Realistic)")
        all_curves.append(curves)

    if oracle_df is not None:
        curves = compute_risk_coverage(oracle_df, "LLM-Auditor(Oracle)")
        all_curves.append(curves)

    if xgb_baseline is not None and "entropy" in xgb_baseline.columns:
        curves = compute_risk_coverage(xgb_baseline, "Neighbor-XGB")
        all_curves.append(curves)

    if all_curves:
        curves_df = pd.concat(all_curves, ignore_index=True)
        curves_df.to_parquet(f"{OUTPUT_DIR}/risk_coverage_curves.parquet", index=False)

        # Figure 2: Risk-Coverage
        plot_risk_coverage(curves_df, f"{OUTPUT_DIR}/fig2_risk_coverage.pdf")

    print("\nGenerating worst-country figure...")
    metrics_path = "outputs/analysis/metrics_main.csv"
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        plot_worst_country(metrics_df, f"{OUTPUT_DIR}/fig3_worst_country.pdf")
    elif realistic_df is not None:
        from oracle_ablation import compute_aurc
        metrics_df = compute_aurc(realistic_df)
        plot_worst_country(metrics_df, f"{OUTPUT_DIR}/fig3_worst_country.pdf")

    # Figure 4: Sensitivity
    print("\nGenerating sensitivity figure...")
    ablation_path = "outputs/analysis/ablation_metrics.csv"
    ablation_df = pd.read_csv(ablation_path) if os.path.exists(ablation_path) else None
    plot_sensitivity(ablation_df, f"{OUTPUT_DIR}/fig4_sensitivity.pdf")

    print("\n" + "=" * 60)
    print("All figures generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
