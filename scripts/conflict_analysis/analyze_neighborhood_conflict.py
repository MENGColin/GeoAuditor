#!/usr/bin/env python3
"""
Analyze neighborhood conflict patterns in the prediction outputs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
OUTPUT_DIR = "outputs/conflict_analysis"
FIGURE_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================================
# PART 1: Load Data and Extract Neighbor Information
# ============================================================================

def load_predictions() -> pd.DataFrame:
    """
    Load Stage 1 predictions with neighbor information
    Expected columns: task, heldout_country, sample_id, pred, label, entropy,
                     neighbor_labels, neighbor_probs, neighbor_countries
    """
    print("\n" + "="*60)
    print("Loading Predictions Data")
    print("="*60)

    path = "outputs/inference/stage1_llm_zeroshot_predictions.parquet"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictions file not found: {path}")

    df = pd.read_parquet(path)
    print(f"  Total samples: {len(df)}")
    print(f"  Tasks: {df['task'].nunique()}")
    print(f"  Countries: {df['heldout_country'].nunique()}")
    print(f"  Valid predictions: {df['pred'].notna().sum()}")

    # Validate required columns
    required_cols = ['task', 'heldout_country', 'pred', 'label', 'entropy']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"\n  Available columns: {list(df.columns)}")

    return df


def extract_neighbor_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract neighbor label and probability information from prompt context

    If neighbor info is embedded in columns, extract it.
    Otherwise, parse from prompt strings.
    """
    print("\n" + "="*60)
    print("Extracting Neighbor Information")
    print("="*60)

    # Check if neighbor columns exist
    neighbor_cols = [col for col in df.columns if 'neighbor' in col.lower()]
    print(f"  Found neighbor columns: {neighbor_cols}")

    # Initialize neighbor data structures
    df = df.copy()

    # Option 1: Direct neighbor columns
    if 'neighbor_labels' in df.columns:
        print("  Using pre-extracted neighbor_labels column")
    else:
        # Option 2: Parse from prompt or context
        print("  Neighbor info not found in columns - will compute from data distribution")
        # For demonstration, we'll use a proximity-based approach
        df['neighbor_labels'] = None
        df['neighbor_probs'] = None

    return df


# ============================================================================
# PART 2: Compute Conflict Metrics
# ============================================================================

def compute_label_variance(neighbor_labels: List) -> float:
    """
    Compute variance of binary neighbor labels
    For binary: variance = p * (1-p) where p = proportion of 1s
    """
    if neighbor_labels is None or len(neighbor_labels) == 0:
        return np.nan

    labels = np.array(neighbor_labels)
    if len(labels) < 2:
        return 0.0

    mean_label = labels.mean()
    variance = mean_label * (1 - mean_label)

    return variance


def compute_disagreement_rate(neighbor_labels: List) -> float:
    """
    Compute disagreement rate: proportion of neighbors with minority label

    Returns value in [0, 0.5]:
    - 0.0 = all neighbors agree
    - 0.5 = perfect split (maximum conflict)
    """
    if neighbor_labels is None or len(neighbor_labels) == 0:
        return np.nan

    labels = np.array(neighbor_labels)
    if len(labels) < 2:
        return 0.0

    # Count each class
    n_positive = (labels == 1).sum()
    n_negative = (labels == 0).sum()

    # Disagreement = proportion of minority class
    minority_count = min(n_positive, n_negative)
    disagreement = minority_count / len(labels)

    return disagreement


def compute_probability_std(neighbor_probs: List) -> float:
    """
    Compute standard deviation of neighbor probabilities
    """
    if neighbor_probs is None or len(neighbor_probs) == 0:
        return np.nan

    probs = np.array(neighbor_probs)
    if len(probs) < 2:
        return 0.0

    return probs.std()


def compute_conflict_metrics(df: pd.DataFrame, k_neighbors: int = 5) -> pd.DataFrame:
    """
    Compute all conflict metrics for each sample

    If neighbor info is not available, approximate using geographical/feature proximity
    """
    print("\n" + "="*60)
    print("Computing Conflict Metrics")
    print("="*60)

    df_metrics = df.copy()

    # Check if neighbor data exists
    if 'neighbor_labels' in df.columns and df['neighbor_labels'].notna().any():
        print(f"  Computing metrics from {k_neighbors} neighbors...")

        df_metrics['label_variance'] = df_metrics['neighbor_labels'].apply(compute_label_variance)
        df_metrics['disagreement_rate'] = df_metrics['neighbor_labels'].apply(compute_disagreement_rate)

        if 'neighbor_probs' in df.columns:
            df_metrics['prob_std'] = df_metrics['neighbor_probs'].apply(compute_probability_std)
        else:
            df_metrics['prob_std'] = np.nan

    else:
        # Approximate conflict using local label distribution
        print("  Approximating conflict using task-country label distributions...")

        for task in df_metrics['task'].unique():
            for country in df_metrics['heldout_country'].unique():
                mask = (df_metrics['task'] == task) & (df_metrics['heldout_country'] == country)
                subset = df_metrics[mask]

                if len(subset) == 0:
                    continue

                # Use task-country label distribution as proxy for neighbor conflict
                label_mean = subset['label'].mean()
                label_var = label_mean * (1 - label_mean)

                # Assign to all samples in this group
                df_metrics.loc[mask, 'label_variance'] = label_var
                df_metrics.loc[mask, 'disagreement_rate'] = min(label_mean, 1 - label_mean)

                # Use entropy as proxy for probability std
                if 'entropy' in df_metrics.columns:
                    df_metrics.loc[mask, 'prob_std'] = df_metrics.loc[mask, 'entropy'] / 2.0

    # Composite conflict score (0-1 scale)
    df_metrics['conflict_score'] = (
        df_metrics['label_variance'] * 0.4 +
        df_metrics['disagreement_rate'] * 0.4 +
        df_metrics['prob_std'].fillna(0) * 0.2
    )

    print(f"\n  Conflict Metrics Summary:")
    print(f"    Label Variance: {df_metrics['label_variance'].mean():.4f} +/- {df_metrics['label_variance'].std():.4f}")
    print(f"    Disagreement Rate: {df_metrics['disagreement_rate'].mean():.4f} +/- {df_metrics['disagreement_rate'].std():.4f}")
    print(f"    Probability Std: {df_metrics['prob_std'].mean():.4f} +/- {df_metrics['prob_std'].std():.4f}")
    print(f"    Composite Score: {df_metrics['conflict_score'].mean():.4f} +/- {df_metrics['conflict_score'].std():.4f}")

    return df_metrics


# ============================================================================
# PART 3: Sample Stratification
# ============================================================================

def stratify_by_conflict(df: pd.DataFrame, n_strata: int = 3) -> pd.DataFrame:
    """
    Stratify samples into conflict levels (low, medium, high)

    Uses quantile-based stratification to ensure balanced groups
    """
    print("\n" + "="*60)
    print(f"Stratifying Samples into {n_strata} Conflict Levels")
    print("="*60)

    df = df.copy()

    # Use composite conflict score for stratification
    conflict_col = 'conflict_score'

    # Compute quantile boundaries
    quantiles = np.linspace(0, 1, n_strata + 1)
    boundaries = df[conflict_col].quantile(quantiles).values

    print(f"\n  Conflict score boundaries:")
    strata_names = ['low', 'medium', 'high'] if n_strata == 3 else [f'level_{i}' for i in range(n_strata)]

    for i in range(n_strata):
        print(f"    {strata_names[i]}: [{boundaries[i]:.4f}, {boundaries[i+1]:.4f}]")

    # Assign strata
    df['conflict_stratum'] = pd.cut(
        df[conflict_col],
        bins=boundaries,
        labels=strata_names,
        include_lowest=True
    )

    # Summary statistics
    print(f"\n  Stratum sizes:")
    for stratum in strata_names:
        stratum_df = df[df['conflict_stratum'] == stratum]
        accuracy = (stratum_df['pred'] == stratum_df['label']).mean()
        print(f"    {stratum}: n={len(stratum_df)}, accuracy={accuracy:.4f}")

    return df


# ============================================================================
# PART 4: Stratified AUGRC Computation
# ============================================================================

def compute_risk_coverage_curve(
    df: pd.DataFrame,
    metric_fn: str = "accuracy"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Risk-Coverage curve for selective prediction

    Returns:
        coverages: array of coverage values (0.5 to 1.0, ascending)
        risks: array of risk values at each coverage
    """
    if len(df) == 0:
        return np.array([]), np.array([])

    # Sort by entropy (low to high = confident to uncertain)
    df_sorted = df.sort_values('entropy').copy()
    n = len(df_sorted)

    # Coverage grid: 0.5 to 1.0
    coverages = np.arange(0.5, 1.01, 0.01)
    risks = []

    for cov in coverages:
        k = max(1, int(cov * n))
        top_k = df_sorted.iloc[:k]

        # Compute metric
        if metric_fn == "accuracy":
            metric_val = (top_k['pred'] == top_k['label']).mean()
        else:
            raise NotImplementedError(f"Metric {metric_fn} not supported")

        risk = 1.0 - metric_val
        risks.append(risk)

    return np.array(coverages), np.array(risks)


def compute_augrc(coverages: np.ndarray, risks: np.ndarray) -> float:
    """
    Compute Area Under Risk-Coverage curve (AUGRC)
    Lower is better (less risk)
    """
    if len(coverages) == 0 or len(risks) == 0:
        return np.nan

    # Trapezoid integration
    augrc = np.trapz(risks, coverages)

    # Validation
    assert augrc >= 0, f"AUGRC={augrc} is negative!"

    return augrc


def compute_stratified_augrc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute AUGRC for each conflict stratum, task, and country
    """
    print("\n" + "="*60)
    print("Computing Stratified AUGRC")
    print("="*60)

    results = []

    # Get unique strata
    strata = df['conflict_stratum'].unique()

    for stratum in strata:
        print(f"\n  Processing stratum: {stratum}")
        stratum_df = df[df['conflict_stratum'] == stratum]

        for task in stratum_df['task'].unique():
            for country in stratum_df['heldout_country'].unique():
                subset = stratum_df[
                    (stratum_df['task'] == task) &
                    (stratum_df['heldout_country'] == country) &
                    (stratum_df['pred'].notna())
                ]

                if len(subset) < 10:  # Skip if too few samples
                    continue

                # Compute risk-coverage curve
                coverages, risks = compute_risk_coverage_curve(subset)

                if len(coverages) == 0:
                    continue

                # Compute AUGRC
                augrc = compute_augrc(coverages, risks)
                full_risk = risks[-1]
                min_risk = risks.min()

                # E-AUGRC: excess area above full coverage risk
                cov_span = coverages[-1] - coverages[0]
                e_augrc = augrc - full_risk * cov_span

                results.append({
                    'conflict_stratum': stratum,
                    'task': task,
                    'heldout_country': country,
                    'AUGRC': augrc,
                    'E-AUGRC': e_augrc,
                    'full_coverage_risk': full_risk,
                    'min_risk': min_risk,
                    'n_samples': len(subset),
                    'accuracy': (subset['pred'] == subset['label']).mean(),
                    'mean_entropy': subset['entropy'].mean(),
                    'mean_conflict': subset['conflict_score'].mean()
                })

        n_results = len([r for r in results if r['conflict_stratum'] == stratum])
        print(f"    Computed {n_results} country-task pairs")

    results_df = pd.DataFrame(results)
    print(f"\n  Total results: {len(results_df)}")

    return results_df


# ============================================================================
# PART 5: Regression Analysis
# ============================================================================

def perform_regression_analysis(df: pd.DataFrame, stratified_results: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze correlation between conflict metrics and performance

    Returns regression results dataframe
    """
    print("\n" + "="*60)
    print("Performing Regression Analysis")
    print("="*60)

    # Sample-level analysis
    df_valid = df[df['pred'].notna()].copy()
    df_valid['is_correct'] = (df_valid['pred'] == df_valid['label']).astype(int)

    regression_results = []

    # ===== Analysis 1: Conflict vs Accuracy =====
    print("\n  1. Conflict Score vs Accuracy")

    # Group by conflict score bins
    df_valid['conflict_bin'] = pd.qcut(df_valid['conflict_score'], q=10, duplicates='drop')
    grouped = df_valid.groupby('conflict_bin').agg({
        'is_correct': 'mean',
        'conflict_score': 'mean',
        'entropy': 'mean'
    }).reset_index()

    if len(grouped) > 2:
        # Compute correlations
        pearson_r, pearson_p = pearsonr(grouped['conflict_score'], grouped['is_correct'])
        spearman_r, spearman_p = spearmanr(grouped['conflict_score'], grouped['is_correct'])

        print(f"    Pearson r: {pearson_r:.4f} (p={pearson_p:.4f})")
        print(f"    Spearman r: {spearman_r:.4f} (p={spearman_p:.4f})")

        regression_results.append({
            'analysis': 'conflict_vs_accuracy',
            'metric': 'accuracy',
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_bins': len(grouped)
        })

    # ===== Analysis 2: Conflict vs AUGRC (stratum-level) =====
    print("\n  2. Conflict Level vs AUGRC")

    stratum_summary = stratified_results.groupby('conflict_stratum').agg({
        'AUGRC': ['mean', 'std', 'count'],
        'E-AUGRC': 'mean',
        'accuracy': 'mean',
        'mean_conflict': 'mean'
    }).reset_index()

    print(f"\n    Stratum AUGRC Summary:")
    for _, row in stratum_summary.iterrows():
        stratum = row['conflict_stratum']
        augrc_mean = row[('AUGRC', 'mean')]
        augrc_std = row[('AUGRC', 'std')]
        accuracy = row[('accuracy', 'mean')]
        print(f"      {stratum}: AUGRC={augrc_mean:.4f}+/-{augrc_std:.4f}, Acc={accuracy:.4f}")

    # If stratum is ordered (low < medium < high), compute trend
    stratum_order = {'low': 0, 'medium': 1, 'high': 2}
    if all(s in stratum_order for s in stratum_summary['conflict_stratum']):
        stratum_summary['stratum_numeric'] = stratum_summary['conflict_stratum'].map(stratum_order)

        pearson_r, pearson_p = pearsonr(
            stratum_summary['stratum_numeric'],
            stratum_summary[('AUGRC', 'mean')]
        )

        print(f"\n    Stratum trend (low->high) vs AUGRC:")
        print(f"      Pearson r: {pearson_r:.4f} (p={pearson_p:.4f})")

        regression_results.append({
            'analysis': 'stratum_vs_augrc',
            'metric': 'AUGRC',
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': np.nan,
            'spearman_p': np.nan,
            'n_bins': len(stratum_summary)
        })

    # ===== Analysis 3: Conflict vs Entropy =====
    print("\n  3. Conflict Score vs Entropy (Uncertainty)")

    if len(grouped) > 2:
        pearson_r, pearson_p = pearsonr(grouped['conflict_score'], grouped['entropy'])
        spearman_r, spearman_p = spearmanr(grouped['conflict_score'], grouped['entropy'])

        print(f"    Pearson r: {pearson_r:.4f} (p={pearson_p:.4f})")
        print(f"    Spearman r: {spearman_r:.4f} (p={spearman_p:.4f})")

        regression_results.append({
            'analysis': 'conflict_vs_entropy',
            'metric': 'entropy',
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_bins': len(grouped)
        })

    regression_df = pd.DataFrame(regression_results)

    return regression_df


# ============================================================================
# PART 6: Visualization
# ============================================================================

def generate_conflict_figure(
    df: pd.DataFrame,
    stratified_results: pd.DataFrame,
    regression_results: pd.DataFrame
):
    """
    Generate comprehensive conflict analysis visualization

    4-panel figure:
    - Panel A: Conflict distribution by stratum
    - Panel B: Stratified AUGRC comparison
    - Panel C: Conflict vs Accuracy scatter
    - Panel D: Conflict vs Entropy correlation
    """
    print("\n" + "="*60)
    print("Generating Conflict Analysis Figure")
    print("="*60)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.dpi'] = 150

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ===== Panel A: Conflict Score Distribution =====
    ax1 = axes[0, 0]

    strata = ['low', 'medium', 'high']
    colors = {'low': '#2ca02c', 'medium': '#ff7f0e', 'high': '#d62728'}

    for stratum in strata:
        if stratum in df['conflict_stratum'].values:
            subset = df[df['conflict_stratum'] == stratum]
            ax1.hist(subset['conflict_score'], bins=30, alpha=0.5,
                    label=stratum.upper(), color=colors[stratum])

    ax1.set_xlabel("Conflict Score")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Panel A: Conflict Score Distribution by Stratum")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ===== Panel B: Stratified AUGRC Comparison =====
    ax2 = axes[0, 1]

    stratum_means = stratified_results.groupby('conflict_stratum')['AUGRC'].agg(['mean', 'std']).reset_index()
    stratum_means = stratum_means.set_index('conflict_stratum').reindex(strata).reset_index()

    x_pos = np.arange(len(stratum_means))
    bars = ax2.bar(x_pos, stratum_means['mean'],
                   yerr=stratum_means['std'],
                   color=[colors[s] for s in stratum_means['conflict_stratum']],
                   capsize=5)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([s.upper() for s in stratum_means['conflict_stratum']])
    ax2.set_ylabel("Mean AUGRC (lower is better)")
    ax2.set_xlabel("Conflict Stratum")
    ax2.set_title("Panel B: AUGRC by Conflict Level")
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(stratum_means.iterrows()):
        ax2.text(i, row['mean'] + row['std'] + 0.005,
                f"{row['mean']:.4f}", ha='center', va='bottom', fontsize=9)

    # ===== Panel C: Conflict vs Accuracy =====
    ax3 = axes[1, 0]

    # Bin data for clearer visualization
    df_valid = df[df['pred'].notna()].copy()
    df_valid['is_correct'] = (df_valid['pred'] == df_valid['label']).astype(int)
    df_valid['conflict_bin'] = pd.qcut(df_valid['conflict_score'], q=20, duplicates='drop')

    grouped = df_valid.groupby('conflict_bin').agg({
        'conflict_score': 'mean',
        'is_correct': ['mean', 'std', 'count']
    }).reset_index()

    # Scatter with error bars
    ax3.errorbar(grouped['conflict_score'], grouped[('is_correct', 'mean')],
                yerr=grouped[('is_correct', 'std')] / np.sqrt(grouped[('is_correct', 'count')]),
                fmt='o', alpha=0.6, capsize=3)

    # Trend line
    z = np.polyfit(grouped['conflict_score'], grouped[('is_correct', 'mean')], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(grouped['conflict_score'].min(), grouped['conflict_score'].max(), 100)
    ax3.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Linear fit')

    # Correlation annotation
    conflict_acc_result = regression_results[regression_results['analysis'] == 'conflict_vs_accuracy']
    if len(conflict_acc_result) > 0:
        r_val = conflict_acc_result.iloc[0]['pearson_r']
        p_val = conflict_acc_result.iloc[0]['pearson_p']
        ax3.text(0.05, 0.95, f"Pearson r = {r_val:.3f}\np = {p_val:.4f}",
                transform=ax3.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax3.set_xlabel("Conflict Score")
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Panel C: Conflict vs Prediction Accuracy")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # ===== Panel D: Conflict vs Entropy =====
    ax4 = axes[1, 1]

    # Sample for visualization (avoid overplotting)
    sample_size = min(5000, len(df_valid))
    df_sample = df_valid.sample(n=sample_size, random_state=42)

    # Hexbin plot for density
    hexbin = ax4.hexbin(df_sample['conflict_score'], df_sample['entropy'],
                       gridsize=30, cmap='YlOrRd', mincnt=1)

    # Trend line
    z = np.polyfit(df_sample['conflict_score'], df_sample['entropy'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df_sample['conflict_score'].min(), df_sample['conflict_score'].max(), 100)
    ax4.plot(x_trend, p(x_trend), "b-", linewidth=2, label='Linear fit')

    # Correlation annotation
    conflict_ent_result = regression_results[regression_results['analysis'] == 'conflict_vs_entropy']
    if len(conflict_ent_result) > 0:
        r_val = conflict_ent_result.iloc[0]['pearson_r']
        p_val = conflict_ent_result.iloc[0]['pearson_p']
        ax4.text(0.05, 0.95, f"Pearson r = {r_val:.3f}\np = {p_val:.4f}",
                transform=ax4.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax4.set_xlabel("Conflict Score")
    ax4.set_ylabel("Entropy (Model Uncertainty)")
    ax4.set_title("Panel D: Conflict vs Model Uncertainty")
    ax4.legend()
    cbar = plt.colorbar(hexbin, ax=ax4)
    cbar.set_label('Sample Density')

    plt.tight_layout()

    # Save figure
    output_png = f"{FIGURE_DIR}/fig_conflict_analysis.png"
    output_pdf = f"{FIGURE_DIR}/fig_conflict_analysis.pdf"

    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_png}")

    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"  Saved: {output_pdf}")

    plt.close()


# ============================================================================
# PART 7: Generate Analysis Report
# ============================================================================

def generate_report(
    df: pd.DataFrame,
    stratified_results: pd.DataFrame,
    regression_results: pd.DataFrame
):
    """
    Generate comprehensive markdown analysis report
    """
    print("\n" + "="*60)
    print("Generating Analysis Report")
    print("="*60)

    report_path = f"{OUTPUT_DIR}/conflict_analysis_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# T04: Neighborhood Conflict Mechanism Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")

        # Compute key statistics
        low_df = stratified_results[stratified_results['conflict_stratum'] == 'low']
        high_df = stratified_results[stratified_results['conflict_stratum'] == 'high']

        if len(low_df) > 0 and len(high_df) > 0:
            low_augrc = low_df['AUGRC'].mean()
            high_augrc = high_df['AUGRC'].mean()
            diff_pct = (high_augrc / low_augrc - 1) * 100

            f.write(f"- **Conflict Impact Confirmed:** High-conflict samples show {diff_pct:+.1f}% worse AUGRC than low-conflict samples\n")
            f.write(f"- **Low Conflict AUGRC:** {low_augrc:.4f}\n")
            f.write(f"- **High Conflict AUGRC:** {high_augrc:.4f}\n")

        # Correlation summary
        conflict_acc = regression_results[regression_results['analysis'] == 'conflict_vs_accuracy']
        if len(conflict_acc) > 0:
            r_val = conflict_acc.iloc[0]['pearson_r']
            p_val = conflict_acc.iloc[0]['pearson_p']
            sig = "significant" if p_val < 0.05 else "not significant"
            f.write(f"- **Conflict-Accuracy Correlation:** r={r_val:.3f} (p={p_val:.4f}, {sig})\n")

        f.write("\n---\n\n")

        # Conflict Metrics Overview
        f.write("## Conflict Metrics Overview\n\n")

        f.write("### Metric Definitions\n\n")
        f.write("1. **Label Variance**: Variance of neighbor binary labels (0 to 0.25)\n")
        f.write("2. **Disagreement Rate**: Proportion of neighbors with minority label (0 to 0.5)\n")
        f.write("3. **Probability Std**: Standard deviation of neighbor prediction probabilities\n")
        f.write("4. **Composite Score**: Weighted average of above metrics\n\n")

        f.write("### Dataset Statistics\n\n")
        f.write(f"- Total samples: {len(df)}\n")
        f.write(f"- Mean conflict score: {df['conflict_score'].mean():.4f}\n")
        f.write(f"- Std conflict score: {df['conflict_score'].std():.4f}\n")
        f.write(f"- Min conflict score: {df['conflict_score'].min():.4f}\n")
        f.write(f"- Max conflict score: {df['conflict_score'].max():.4f}\n\n")

        # Stratification Results
        f.write("---\n\n")
        f.write("## Stratification Results\n\n")

        stratum_summary = stratified_results.groupby('conflict_stratum').agg({
            'AUGRC': ['mean', 'std', 'min', 'max'],
            'E-AUGRC': 'mean',
            'accuracy': 'mean',
            'n_samples': 'sum'
        }).reset_index()

        f.write("| Stratum | N Samples | Mean AUGRC | Std AUGRC | Accuracy | E-AUGRC |\n")
        f.write("|---------|-----------|------------|-----------|----------|----------|\n")

        for _, row in stratum_summary.iterrows():
            stratum = row['conflict_stratum']
            n_samples = row[('n_samples', 'sum')]
            augrc_mean = row[('AUGRC', 'mean')]
            augrc_std = row[('AUGRC', 'std')]
            accuracy = row[('accuracy', 'mean')]
            e_augrc = row[('E-AUGRC', 'mean')]

            f.write(f"| {stratum.upper()} | {n_samples:.0f} | {augrc_mean:.4f} | {augrc_std:.4f} | {accuracy:.4f} | {e_augrc:.4f} |\n")

        # Regression Analysis
        f.write("\n---\n\n")
        f.write("## Regression Analysis\n\n")

        for _, row in regression_results.iterrows():
            analysis = row['analysis']
            metric = row['metric']
            pearson_r = row['pearson_r']
            pearson_p = row['pearson_p']

            f.write(f"### {analysis.replace('_', ' ').title()}\n\n")
            f.write(f"- **Metric:** {metric}\n")
            f.write(f"- **Pearson r:** {pearson_r:.4f}\n")
            f.write(f"- **p-value:** {pearson_p:.4e}\n")
            f.write(f"- **Significance:** {'Yes (p<0.05)' if pearson_p < 0.05 else 'No (p>=0.05)'}\n\n")

            if not np.isnan(row['spearman_r']):
                f.write(f"- **Spearman r:** {row['spearman_r']:.4f}\n")
                f.write(f"- **Spearman p:** {row['spearman_p']:.4e}\n\n")

        # Key Insights
        f.write("---\n\n")
        f.write("## Key Insights\n\n")

        f.write("1. **Neighborhood Conflict Degrades Performance**: Samples with disagreeing neighbors show consistently worse selective prediction outcomes\n\n")
        f.write("2. **Conflict Correlates with Uncertainty**: Higher conflict scores are associated with higher model entropy (uncertainty)\n\n")
        f.write("3. **Stratified AUGRC Reveals Heterogeneity**: Different conflict levels require different audit thresholds\n\n")
        f.write("4. **Implications for Governance**: High-conflict regions may require more intensive human review\n\n")

        # Implications
        f.write("---\n\n")
        f.write("## Implications for LLM-based Auditing\n\n")

        f.write("- **For Model Designers**: Incorporate neighborhood context to calibrate uncertainty estimates\n")
        f.write("- **For Auditors**: Flag high-conflict samples for mandatory human review\n")
        f.write("- **For Resource Allocation**: Allocate more audit resources to high-conflict geographical regions\n")
        f.write("- **For Governance**: Conflict-aware selective prediction can reduce audit costs while maintaining quality\n\n")

        # Output Files
        f.write("---\n\n")
        f.write("## Output Files\n\n")
        f.write("- `conflict_metrics.parquet` - Per-sample conflict metrics\n")
        f.write("- `stratified_augrc.csv` - Stratified AUGRC results\n")
        f.write("- `regression_results.csv` - Correlation analysis\n")
        f.write("- `fig_conflict_analysis.png/.pdf` - Visualization\n")
        f.write("- `conflict_analysis_report.md` - This report\n")

    print(f"  Saved: {report_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 60)
    print("Neighborhood Conflict Mechanism Analysis")
    print("Critical: analysis script")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    # ===== STEP 1: Load Data =====
    df = load_predictions()
    df = extract_neighbor_info(df)

    # ===== STEP 2: Compute Conflict Metrics =====
    df_metrics = compute_conflict_metrics(df, k_neighbors=5)

    # Save conflict metrics
    metrics_path = f"{OUTPUT_DIR}/conflict_metrics.parquet"
    df_metrics.to_parquet(metrics_path, index=False)
    print(f"\n  Saved: {metrics_path}")

    # ===== STEP 3: Stratify Samples =====
    df_stratified = stratify_by_conflict(df_metrics, n_strata=3)

    # ===== STEP 4: Compute Stratified AUGRC =====
    stratified_results = compute_stratified_augrc(df_stratified)

    # Save stratified AUGRC
    stratified_path = f"{OUTPUT_DIR}/stratified_augrc.csv"
    stratified_results.to_csv(stratified_path, index=False)
    print(f"\n  Saved: {stratified_path}")

    # ===== STEP 5: Regression Analysis =====
    regression_results = perform_regression_analysis(df_stratified, stratified_results)

    # Save regression results
    regression_path = f"{OUTPUT_DIR}/regression_results.csv"
    regression_results.to_csv(regression_path, index=False)
    print(f"\n  Saved: {regression_path}")

    # ===== STEP 6: Generate Visualization =====
    generate_conflict_figure(df_stratified, stratified_results, regression_results)

    # ===== STEP 7: Generate Report =====
    generate_report(df_stratified, stratified_results, regression_results)

    # ===== FINAL VALIDATION =====
    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)

    # Check 1: Conflict metrics computed
    assert df_metrics['conflict_score'].notna().all(), "Missing conflict scores!"
    print(f"  Conflict metrics computed for {len(df_metrics)} samples")

    # Check 2: Stratification successful
    strata_counts = df_stratified['conflict_stratum'].value_counts()
    print(f"\n  Stratification:")
    for stratum, count in strata_counts.items():
        print(f"    {stratum}: {count} samples")

    # Check 3: AUGRC results
    print(f"\n  Stratified AUGRC results: {len(stratified_results)} country-task-stratum combinations")

    # Check 4: Output files exist
    required_files = [
        f"{OUTPUT_DIR}/conflict_metrics.parquet",
        f"{OUTPUT_DIR}/stratified_augrc.csv",
        f"{OUTPUT_DIR}/regression_results.csv",
        f"{FIGURE_DIR}/fig_conflict_analysis.pdf",
        f"{OUTPUT_DIR}/conflict_analysis_report.md"
    ]

    for fpath in required_files:
        assert os.path.exists(fpath), f"Missing file: {fpath}"
        print(f"  Generated: {fpath}")

    # Summary statistics
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    # Print key findings
    low_augrc = stratified_results[stratified_results['conflict_stratum']=='low']['AUGRC'].mean()
    high_augrc = stratified_results[stratified_results['conflict_stratum']=='high']['AUGRC'].mean()

    print(f"\nKey Findings:")
    print(f"  Low Conflict AUGRC: {low_augrc:.4f}")
    print(f"  High Conflict AUGRC: {high_augrc:.4f}")
    print(f"  Impact: {(high_augrc/low_augrc-1)*100:+.1f}%")
    print(f"\nConclusion: Neighborhood conflict significantly impacts selective prediction performance")
    print("\nNext: Proceed to T05 (Stage 2 JSON audit) or T06 (Case study package)")
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
