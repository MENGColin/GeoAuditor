#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute risk-coverage curves and summary metrics for selective prediction.
"""

import os
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_auc_score
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
def load_config():
    with open("config/run_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_risk_coverage_curve(labels, predictions, uncertainties, n_points=100):
    """
    Compute the risk-coverage curve

    Args:
        labels: true label
        predictions: predicted label
        uncertainties: uncertainty score(lower means more certain)
        n_points: number of curve points

    Returns:
        coverages, risks, accuracies
    """
    n_samples = len(labels)

    sorted_indices = np.argsort(uncertainties)
    sorted_labels = np.array(labels)[sorted_indices]
    sorted_preds = np.array(predictions)[sorted_indices]

    coverages = []
    risks = []
    accuracies = []

    for coverage in np.linspace(0.1, 1.0, n_points):
        n_selected = int(coverage * n_samples)
        if n_selected == 0:
            continue

        selected_labels = sorted_labels[:n_selected]
        selected_preds = sorted_preds[:n_selected]

        errors = (selected_labels != selected_preds).sum()
        risk = errors / n_selected
        accuracy = 1 - risk

        coverages.append(coverage)
        risks.append(risk)
        accuracies.append(accuracy)

    return np.array(coverages), np.array(risks), np.array(accuracies)


def compute_aurc(coverages, risks):
    """
    ComputeAURC (Area Under Risk-Coverage Curve)
    lower is better
    """
    aurc = np.trapz(risks, coverages)
    return aurc


def compute_eaurc(coverages, risks, full_coverage_risk):
    """
    ComputeE-AURC (Excess AURC)
    excess risk relative to an ideal selective predictor
    """
    aurc = compute_aurc(coverages, risks)

    optimal_aurc = full_coverage_risk * (1 - full_coverage_risk) / 2

    eaurc = aurc - optimal_aurc
    return eaurc


def plot_risk_coverage(results_df, output_path, title="Risk-Coverage Curves"):
    """Plot the risk-coverage curve"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    tasks = results_df["task"].unique()
    methods = results_df["method"].unique()

    colors = {"XGB": "blue", "LLM": "red", "LLM-SFT": "green"}

    for i, task in enumerate(tasks):
        if i >= len(axes):
            break
        ax = axes[i]

        task_df = results_df[results_df["task"] == task]

        for method in methods:
            method_df = task_df[task_df["method"] == method]
            if len(method_df) == 0:
                continue

            coverages, risks, _ = compute_risk_coverage_curve(
                method_df["label"].values,
                method_df["pred"].values,
                method_df["entropy"].values
            )

            aurc = compute_aurc(coverages, risks)
            color = colors.get(method, "gray")
            ax.plot(coverages, risks, label=f"{method} (AURC={aurc:.4f})", color=color)

        ax.set_xlabel("Coverage")
        ax.set_ylabel("Risk (Error Rate)")
        ax.set_title(task)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 0.6])

    for i in range(len(tasks), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {output_path}")


def main():
    print("=" * 60)
    print("Compute risk-coverage metrics")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    config = load_config()
    tasks = config["data"]["tasks"]

    output_dir = "outputs/inference"
    os.makedirs(output_dir, exist_ok=True)

    baseline_path = os.path.join(output_dir, "baseline_xgb_predictions.parquet")
    print(f"\nLoading baseline predictions from: {baseline_path}")

    if not os.path.exists(baseline_path):
        print("Error: Baseline predictions not found. Run baseline_uncertainty.py first.")
        return

    baseline_df = pd.read_parquet(baseline_path)
    print(f"Loaded {len(baseline_df)} baseline predictions")

    llm_path = os.path.join(output_dir, "stage1_llm_zeroshot_predictions.parquet")
    if os.path.exists(llm_path):
        print(f"Loading LLM predictions from: {llm_path}")
        llm_df = pd.read_parquet(llm_path)
        llm_df["method"] = "LLM"
        print(f"Loaded {len(llm_df)} LLM predictions")

        all_results = pd.concat([baseline_df, llm_df], ignore_index=True)
    else:
        print("LLM predictions not found, using baseline only")
        all_results = baseline_df

    print("\nComputing risk-coverage metrics...")

    metrics_list = []
    curve_data = []

    for method in all_results["method"].unique():
        method_df = all_results[all_results["method"] == method]

        for task in tasks:
            task_df = method_df[method_df["task"] == task]

            if len(task_df) == 0:
                continue

            valid_mask = task_df["pred"].notna()
            if valid_mask.sum() == 0:
                continue

            valid_df = task_df[valid_mask]

            coverages, risks, accuracies = compute_risk_coverage_curve(
                valid_df["label"].values,
                valid_df["pred"].values,
                valid_df["entropy"].values
            )

            full_risk = (valid_df["pred"] != valid_df["label"]).mean()
            aurc = compute_aurc(coverages, risks)
            eaurc = compute_eaurc(coverages, risks, full_risk)

            for cov, risk, acc in zip(coverages, risks, accuracies):
                curve_data.append({
                    "method": method,
                    "task": task,
                    "heldout_country": "all",
                    "coverage": cov,
                    "risk": risk,
                    "accuracy": acc
                })

            country_aurc = []
            for country in valid_df["heldout_country"].unique():
                country_df = valid_df[valid_df["heldout_country"] == country]
                if len(country_df) < 10:
                    continue

                c_coverages, c_risks, _ = compute_risk_coverage_curve(
                    country_df["label"].values,
                    country_df["pred"].values,
                    country_df["entropy"].values
                )
                c_aurc = compute_aurc(c_coverages, c_risks)
                country_aurc.append(c_aurc)

            worst_country_aurc = max(country_aurc) if country_aurc else aurc
            aurc_variance = np.var(country_aurc) if country_aurc else 0

            metrics_list.append({
                "method": method,
                "task": task,
                "samples": len(valid_df),
                "full_coverage_accuracy": 1 - full_risk,
                "AURC": aurc,
                "E-AURC": eaurc,
                "worst_country_AURC": worst_country_aurc,
                "AURC_variance": aurc_variance
            })

            print(f"  {method} - {task}: AURC={aurc:.4f}, E-AURC={eaurc:.4f}")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(output_dir, "metrics_stage1.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")

    curve_df = pd.DataFrame(curve_data)
    curve_path = os.path.join(output_dir, "risk_coverage_curves.parquet")
    curve_df.to_parquet(curve_path, index=False)
    print(f"Saved curve data to {curve_path}")

    print("\nGenerating figures...")
    fig_path = os.path.join(output_dir, "fig2_risk_coverage_draft.png")
    plot_risk_coverage(all_results, fig_path)

    print(f"\n{'=' * 60}")
    print("Summary Table:")
    print(f"{'=' * 60}")
    print(metrics_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
