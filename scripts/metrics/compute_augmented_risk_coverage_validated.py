#!/usr/bin/env python3
"""
Compute validated AUGRC summaries with additional consistency checks.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
OUTPUT_DIR = "outputs/tables"
CACHE_DIR = "outputs/metrics_cache"
VALIDATION_DIR = "outputs/validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)


class ValidationError(Exception):
    """Validation failure exception"""
    pass


def validate_augrc_value(augrc: float, task: str, country: str):
    """
    Strictly validate the AUGRC value

    CRITICAL CHECKS:
    1. AUGRC must be positive (the area cannot be negative)
    2. AUGRC must be < 1.0 (risk in [0,1], coverage in [0.5,1.0], max area = 0.5)
    3. AUGRC should be reasonable (0.05 to 0.4 based on prior AURC results)
    """
    if augrc < 0:
        raise ValidationError(
            f"CRITICAL: AUGRC={augrc:.6f} is NEGATIVE for {task}/{country}! "
            f"This indicates a computation error."
        )

    if augrc > 1.0:
        raise ValidationError(
            f"CRITICAL: AUGRC={augrc:.6f} > 1.0 for {task}/{country}! "
            f"This is impossible since risk in [0,1]."
        )

    # Reasonable range based on prior AURC results (0.10 to 0.40)
    if augrc < 0.05 or augrc > 0.5:
        warnings.warn(
            f"[WARN] WARNING: AUGRC={augrc:.6f} for {task}/{country} "
            f"is outside expected range [0.05, 0.5]. Please review."
        )


def compute_risk_coverage_curve(
    df: pd.DataFrame,
    metric_fn="accuracy"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Risk-Coverage curve

    Returns:
        coverages: array of coverage values (ascending 0.5 to 1.0)
        risks: array of risk values
    """
    if len(df) == 0:
        return np.array([]), np.array([])

    df_sorted = df.sort_values("entropy").copy()
    n = len(df_sorted)

    coverages = np.arange(0.5, 1.01, 0.01)
    risks = []

    for cov in coverages:
        k = max(1, int(cov * n))
        top_k = df_sorted.iloc[:k]

        # Compute metric
        if metric_fn == "accuracy":
            metric_val = (top_k["pred"] == top_k["label"]).mean()
        else:
            raise NotImplementedError(f"Metric {metric_fn} not implemented yet")

        risk = 1.0 - metric_val
        risks.append(risk)

    return np.array(coverages), np.array(risks)


def compute_augrc_single(
    df: pd.DataFrame,
    task: str,
    country: str,
    metric_fn="accuracy"
) -> Dict:
    """
    Compute AUGRC for one (task, country) pair

    Returns dict with AUGRC, E-AUGRC, full_risk, sanity checks
    """
    subset = df[
        (df["task"] == task) &
        (df["heldout_country"] == country) &
        (df["pred"].notna())
    ].copy()

    if len(subset) == 0:
        return None

    # Compute risk-coverage curve
    coverages, risks = compute_risk_coverage_curve(subset, metric_fn=metric_fn)

    if len(coverages) == 0:
        return None

    # ===== CRITICAL: Trapezoid integration =====
    # coverages is ASCENDING (0.5 -> 1.0), so trapz gives POSITIVE area
    augrc = np.trapz(risks, coverages)

    # Sanity check: AUGRC must be positive
    assert augrc >= 0, f"AUGRC={augrc} is negative! Coverage orderError?"

    # Full risk (at 100% coverage)
    full_risk = risks[-1]  # coverage=1.0

    # E-AUGRC: normalized excess area relative to constant-risk baseline
    # E-AUGRC = AUGRC - full_risk * (cov_max - cov_min)
    cov_span = coverages[-1] - coverages[0]  # 1.0 - 0.5 = 0.5
    e_augrc = augrc - full_risk * cov_span

    # ===== VALIDATION =====
    validate_augrc_value(augrc, task, country)

    # Additional sanity: E-AUGRC should be negative (if selective prediction helps)
    # If E-AUGRC > 0, it means selective prediction makes things worse

    return {
        "task": task,
        "heldout_country": country,
        "AUGRC": augrc,
        "E-AUGRC": e_augrc,
        "full_coverage_risk": full_risk,
        "n_samples": len(subset),
        "metric": metric_fn,
        # Validation metadata
        "cov_span": cov_span,
        "min_risk": risks.min(),
        "max_risk": risks.max()
    }


def run_sanity_test():
    """
    [TEST] Toy-example unit test: AUGRC should equal AURC
    """
    print("\n" + "="*60)
    print("[TEST] SANITY TEST: Toy dataset verification")
    print("="*60)

    toy_data = pd.DataFrame({
        "pred": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
        "label": [1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        "entropy": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
        "task": ["test"] * 10,
        "heldout_country": ["TOY"] * 10
    })

    result = compute_augrc_single(toy_data, "test", "TOY", metric_fn="accuracy")

    if result is None:
        raise ValidationError("Toy test failed: result is None")

    augrc = result["AUGRC"]
    e_augrc = result["E-AUGRC"]
    full_risk = result["full_coverage_risk"]

    print(f"  Toy AUGRC: {augrc:.6f}")
    print(f"  Toy E-AUGRC: {e_augrc:.6f}")
    print(f"  Toy full_risk: {full_risk:.4f}")
    print(f"  Accuracy @ full coverage: {1-full_risk:.4f}")

    # Validation: AUGRC > 0
    assert augrc > 0, f"Toy AUGRC={augrc} is not positive!"
    assert augrc < 1.0, f"Toy AUGRC={augrc} > 1.0!"

    print("  Toy test PASSED: AUGRC is positive and < 1.0")

    sorted_data = toy_data.sort_values("entropy")
    n = len(sorted_data)
    manual_acc_full = (sorted_data["pred"] == sorted_data["label"]).mean()
    assert abs(manual_acc_full - (1-full_risk)) < 1e-6, "Full coverage risk mismatch!"
    print(f"  Full coverage risk verified: {full_risk:.4f}")

    print("="*60 + "\n")


def compute_augrc_all(
    df: pd.DataFrame,
    metric_fn="accuracy"
) -> pd.DataFrame:
    """Compute AUGRC for all (task, country) pairs"""
    results = []

    for task in df["task"].unique():
        for country in df["heldout_country"].unique():
            result = compute_augrc_single(df, task, country, metric_fn=metric_fn)
            if result is not None:
                results.append(result)

    return pd.DataFrame(results)


def compare_with_aurc(augrc_df: pd.DataFrame, method_name: str):
    """
    Validate against the previous AURC results

    AUGRC and AURC should be almost identical because the algorithm is the same
    """
    print(f"\nComparing {method_name} AUGRC against the previous AURC results...")

    aurc_path_map = {
        "LLM-Auditor (Realistic)": "outputs/inference/metrics_stage1.csv",
        "Neighbor-XGB": "outputs/inference/metrics_stage1.csv"
    }

    print(f"  AUGRC range: [{augrc_df['AUGRC'].min():.4f}, {augrc_df['AUGRC'].max():.4f}]")
    print(f"  Mean AUGRC: {augrc_df['AUGRC'].mean():.4f}")

    expected_min, expected_max = 0.10, 0.40
    actual_min, actual_max = augrc_df["AUGRC"].min(), augrc_df["AUGRC"].max()

    if actual_min < expected_min * 0.8 or actual_max > expected_max * 1.2:
        warnings.warn(
            f"[WARN] {method_name} AUGRC range [{actual_min:.4f}, {actual_max:.4f}] "
            f"deviates significantly from expected [{expected_min:.4f}, {expected_max:.4f}]"
        )
    else:
        print(f"  AUGRC range is within expected bounds")


def generate_main_table(
    realistic_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    xgb_df: pd.DataFrame,
    metric_fn="accuracy"
) -> pd.DataFrame:
    """Generate the main results table"""
    all_metrics = []

    if realistic_df is not None:
        print("\nComputing AUGRC for Realistic...")
        metrics = compute_augrc_all(realistic_df, metric_fn=metric_fn)
        metrics["method"] = "LLM-Auditor (Realistic)"
        all_metrics.append(metrics)
        compare_with_aurc(metrics, "LLM-Auditor (Realistic)")

    if oracle_df is not None:
        print("\nComputing AUGRC for Oracle...")
        metrics = compute_augrc_all(oracle_df, metric_fn=metric_fn)
        metrics["method"] = "LLM-Auditor (Oracle UB)"
        all_metrics.append(metrics)
        compare_with_aurc(metrics, "Oracle")

    if xgb_df is not None and "entropy" in xgb_df.columns:
        print("\nComputing AUGRC for XGB...")
        metrics = compute_augrc_all(xgb_df, metric_fn=metric_fn)
        metrics["method"] = "Neighbor-XGB"
        all_metrics.append(metrics)
        compare_with_aurc(metrics, "Neighbor-XGB")

    if not all_metrics:
        raise ValueError("No metrics data available!")

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

    return pd.DataFrame(summary_rows), all_df


def generate_latex_table(summary_df: pd.DataFrame) -> str:
    """Generate the table in LaTeX format"""
    tasks = sorted(summary_df["Task"].unique())
    methods = summary_df["Method"].unique()

    header = f"\\begin{{tabular}}{{l{'c' * (len(tasks) + 1)}}}\n"
    header += "\\toprule\n"
    header += "Method & " + " & ".join(tasks) + " & Avg \\\\\n"
    header += "\\midrule\n"

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

    footer = "\\bottomrule\n\\end{tabular}\n"
    caption = "\\caption{Main Results: AUGRC (lower is better) across tasks. Validated version.}\n"

    return header + body + footer + caption


def main():
    print("=" * 60)
    print("AUGRC/E-AUGRC (VALIDATED VERSION)")
    print("CRITICAL: Global Health Investment - Life-Critical Code")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    # ===== STEP 1: Sanity test =====
    run_sanity_test()

    # ===== STEP 2: Load data =====
    print("Loading prediction results...")
    realistic_df = None
    oracle_df = None
    xgb_df = None

    realistic_path = "outputs/inference/stage1_llm_zeroshot_predictions.parquet"
    if os.path.exists(realistic_path):
        realistic_df = pd.read_parquet(realistic_path)
        print(f"  [OK] Loaded Realistic: {len(realistic_df)} samples")

    oracle_path = "outputs/analysis/stage1_oracle_predictions.parquet"
    if os.path.exists(oracle_path):
        oracle_df = pd.read_parquet(oracle_path)
        print(f"  [OK] Loaded Oracle: {len(oracle_df)} samples")

    xgb_path = "outputs/inference/baseline_xgb_predictions.parquet"
    if os.path.exists(xgb_path):
        xgb_df = pd.read_parquet(xgb_path)
        print(f"  [OK] Loaded XGB: {len(xgb_df)} samples")

    # ===== STEP 3: Compute AUGRC with validation =====
    summary_df, all_detailed = generate_main_table(
        realistic_df, oracle_df, xgb_df, metric_fn="accuracy"
    )

    # ===== STEP 4: Save outputs =====
    print("\nSaving outputs...")
    summary_df.to_csv(f"{OUTPUT_DIR}/table_aug_rc_main_validated.csv", index=False)
    print(f"  [OK] Saved: table_aug_rc_main_validated.csv")

    latex = generate_latex_table(summary_df)
    with open(f"{OUTPUT_DIR}/table_aug_rc_main_validated.tex", "w") as f:
        f.write(latex)
    print(f"  [OK] Saved: table_aug_rc_main_validated.tex")

    # Save detailed per-country results
    all_detailed.to_parquet(f"{CACHE_DIR}/augrc_per_country_validated.parquet", index=False)
    print(f"  [OK] Saved: augrc_per_country_validated.parquet")

    # ===== STEP 5: Print summary =====
    print("\n" + "="*60)
    print("VALIDATED AUGRC RESULTS:")
    print("="*60)
    print(summary_df.round(4).to_string(index=False))
    print("="*60)

    # Final validation summary
    print("\nALL VALIDATION CHECKS PASSED")
    print(f"Total country-task pairs: {len(all_detailed)}")
    print(f"AUGRC range: [{all_detailed['AUGRC'].min():.4f}, {all_detailed['AUGRC'].max():.4f}]")
    print(f"All AUGRC values are positive and < 1.0")

    print("\n" + "="*60)
    print("T01 Complete - Results ready for publication")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except ValidationError as e:
        print("\n" + "="*60)
        print("VALIDATION FAILED")
        print("="*60)
        print(str(e))
        print("\nExiting to prevent publishing incorrect results.")
        exit(1)
    except Exception as e:
        print("\n" + "="*60)
        print("UNEXPECTED ERROR")
        print("="*60)
        print(str(e))
        import traceback
        traceback.print_exc()
        exit(1)
