#!/usr/bin/env python3
"""
Generate the multi-panel case-study figure used in reporting.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# ===== Configuration =====
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
INPUT_DIR = "outputs/case_study"
OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 12

# Color palette
STRATUM_COLORS = {
    'A_llm_adds_value': '#2E86AB',      # Blue - LLM adds value
    'B_xgb_uncertain': '#A23B72',       # Purple - XGB uncertain
    'C_high_entropy_defer': '#F18F01',  # Orange - High entropy
    'A': '#2E86AB',                     # Blue - LLM adds value (simplified)
    'B': '#A23B72',                     # Purple - XGB uncertain (simplified)
    'C': '#F18F01'                      # Orange - High entropy (simplified)
}

STRATUM_LABELS = {
    'A_llm_adds_value': 'Stratum A\n(LLM adds value)',
    'B_xgb_uncertain': 'Stratum B\n(XGB uncertain)',
    'C_high_entropy_defer': 'Stratum C\n(High entropy)',
    'A': 'Stratum A\n(LLM adds value)',
    'B': 'Stratum B\n(XGB uncertain)',
    'C': 'Stratum C\n(High entropy)'
}


# ===== Data Loading =====

def load_case_study_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load case-study data"""
    print("Loading case study data...")

    sample_path = f"{INPUT_DIR}/case_study_sample.parquet"
    if not os.path.exists(sample_path):
        print(f"ERROR: {sample_path} not found!")
        sys.exit(1)

    samples_df = pd.read_parquet(sample_path)
    print(f"  [OK] Samples: {len(samples_df)}")

    # 2. Audit results
    audit_path = f"{INPUT_DIR}/stage2_zeroshot_audits.parquet"
    if os.path.exists(audit_path):
        audits_df = pd.read_parquet(audit_path)
        print(f"  [OK] Audits: {len(audits_df)}")
    else:
        print(f"  Warning: {audit_path} not found, using samples only")
        audits_df = samples_df.copy()
        # Add dummy audit columns
        audits_df['audit_prediction'] = audits_df['pred_llm']
        audits_df['audit_recommendation'] = 'Accept LLM'
        audits_df['audit_confidence'] = 'Medium'

    # 3. Narrative cases
    narrative_path = f"{INPUT_DIR}/narrative_cases.parquet"
    if os.path.exists(narrative_path):
        narrative_df = pd.read_parquet(narrative_path)
        print(f"  [OK] Narrative cases: {len(narrative_df)}")
    else:
        print(f"  Warning: {narrative_path} not found, selecting random cases")
        narrative_df = samples_df.sample(n=min(5, len(samples_df)), random_state=42)

    return samples_df, audits_df, narrative_df


# ===== Panel A: Risk-Coverage Curves by Stratum =====

def compute_risk_coverage_by_stratum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Risk-Coverage curves for each stratum

    Args:
        df: DataFrame with columns [row_id, stratum, entropy, pred_llm, label]

    Returns:
        DataFrame with columns [stratum, coverage, risk, accuracy]
    """
    print("\nComputing Risk-Coverage curves by stratum...")

    results = []
    coverages = np.arange(0.5, 1.01, 0.02)  # 0.5 to 1.0, step 0.02

    for stratum in df['stratum'].unique():
        stratum_df = df[df['stratum'] == stratum].copy()

        # Sort by entropy (low to high = high to low confidence)
        stratum_df = stratum_df.sort_values('entropy')
        n = len(stratum_df)

        for cov in coverages:
            # Select top k samples with lowest entropy
            k = max(1, int(cov * n))
            top_k = stratum_df.iloc[:k]

            # Compute risk
            risk = (top_k['pred_llm'] != top_k['label']).mean()

            results.append({
                'stratum': stratum,
                'coverage': cov,
                'risk': risk,
                'accuracy': 1 - risk,
                'n_samples': k
            })

    rc_df = pd.DataFrame(results)

    print("  Risk-Coverage summary:")
    for stratum in rc_df['stratum'].unique():
        stratum_rc = rc_df[rc_df['stratum'] == stratum]
        idx_80 = (stratum_rc['coverage'] - 0.80).abs().idxmin()
        cov_80 = stratum_rc.loc[idx_80]
        print(f"    {stratum}: Risk@80%={cov_80['risk']:.3f}, Acc@80%={cov_80['accuracy']:.3f}")

    return rc_df


def plot_risk_coverage_panel(ax, rc_df: pd.DataFrame):
    """
    Panel A: Risk-Coverage curves
    Show selective-prediction performance across the three strata
    """
    print("\nPlotting Panel A: Risk-Coverage curves...")

    for stratum in sorted(rc_df['stratum'].unique()):
        stratum_data = rc_df[rc_df['stratum'] == stratum]

        ax.plot(
            stratum_data['coverage'],
            stratum_data['risk'],
            label=STRATUM_LABELS[stratum],
            color=STRATUM_COLORS[stratum],
            linewidth=2.5,
            alpha=0.9
        )

    # Styling
    ax.set_xlabel('Coverage', fontweight='bold')
    ax.set_ylabel('Risk (Error Rate)', fontweight='bold')
    ax.set_title('A. Risk-Coverage by Stratum', fontweight='bold', pad=10)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, max(rc_df['risk'].max() * 1.1, 0.5))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.95)

    # Annotation: "Lower is better"
    ax.text(0.52, ax.get_ylim()[1] * 0.95, 'Lower is better ->',
            fontsize=7, style='italic', alpha=0.6)


# ===== Panel B: Audit Agreement Heatmap =====

def compute_audit_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the LLM-vs-XGB audit agreement matrix

    Returns:
        Heatmap data: rows=LLM pred, cols=XGB pred, values=audit recommendation counts
    """
    print("\nComputing audit agreement matrix...")

    # Create agreement matrix
    agreement_data = []

    for llm_pred in [0, 1]:
        for xgb_pred in [0, 1]:
            subset = df[(df['pred_llm'] == llm_pred) & (df['pred_xgb'] == xgb_pred)]

            if len(subset) > 0:
                # Count audit recommendations
                recommendations = subset['audit_recommendation'].value_counts()
                total = len(subset)

                # Primary recommendation
                if 'Accept LLM' in recommendations:
                    primary = 'Accept LLM'
                    count = recommendations['Accept LLM']
                elif 'Accept XGB' in recommendations:
                    primary = 'Accept XGB'
                    count = recommendations['Accept XGB']
                elif 'Defer' in recommendations:
                    primary = 'Defer'
                    count = recommendations['Defer']
                else:
                    primary = 'Unknown'
                    count = 0

                agreement_data.append({
                    'llm_pred': llm_pred,
                    'xgb_pred': xgb_pred,
                    'recommendation': primary,
                    'count': count,
                    'total': total,
                    'percentage': count / total if total > 0 else 0
                })
            else:
                agreement_data.append({
                    'llm_pred': llm_pred,
                    'xgb_pred': xgb_pred,
                    'recommendation': 'N/A',
                    'count': 0,
                    'total': 0,
                    'percentage': 0
                })

    agreement_df = pd.DataFrame(agreement_data)

    print("  Agreement matrix:")
    print(agreement_df.pivot_table(index='llm_pred', columns='xgb_pred', values='total', fill_value=0))

    return agreement_df


def plot_audit_agreement_panel(ax, agreement_df: pd.DataFrame):
    """
    Panel B: Audit agreement heatmap
    Show audit agreement between LLM and XGB predictions
    """
    print("\nPlotting Panel B: Audit agreement heatmap...")

    # Create pivot table for heatmap
    # Use total counts
    heatmap_data = agreement_df.pivot_table(
        index='llm_pred',
        columns='xgb_pred',
        values='total',
        fill_value=0
    )

    # Reorder to match prediction values
    heatmap_data = heatmap_data.reindex([1, 0], axis=0)  # rows: 1 (Poor), 0 (Not Poor)
    heatmap_data = heatmap_data.reindex([0, 1], axis=1)  # cols: 0 (Not Poor), 1 (Poor)

    # Plot heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='g',
        cmap='YlOrRd',
        cbar_kws={'label': 'Sample Count'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray',
        square=True
    )

    # Styling
    ax.set_xlabel('XGB Prediction', fontweight='bold')
    ax.set_ylabel('LLM Prediction', fontweight='bold')
    ax.set_title('B. Audit Agreement Matrix', fontweight='bold', pad=10)
    ax.set_xticklabels(['Not Poor', 'Poor'], rotation=0)
    ax.set_yticklabels(['Poor', 'Not Poor'], rotation=0)

    # Add agreement annotation
    total_agree = heatmap_data.iloc[0, 1] + heatmap_data.iloc[1, 0]  # Diagonal
    total_disagree = heatmap_data.iloc[0, 0] + heatmap_data.iloc[1, 1]  # Off-diagonal
    total = total_agree + total_disagree
    if total > 0:
        agree_pct = total_agree / total * 100
        ax.text(1.0, -0.15, f'Agreement: {agree_pct:.1f}%',
                transform=ax.transAxes, ha='right', fontsize=7, style='italic')


# ===== Panel C: Confidence Calibration Scatter =====

def compute_calibration_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute calibration data

    Confidence = 1 - entropy (normalized)
    Binned accuracy vs predicted confidence
    """
    print("\nComputing calibration data...")

    # Normalize entropy to [0, 1] confidence
    max_entropy = np.log(2)  # Binary classification
    df = df.copy()
    df['confidence'] = 1 - (df['entropy'] / max_entropy)
    df['correct'] = (df['pred_llm'] == df['label']).astype(int)

    # Bin by confidence
    bins = np.arange(0, 1.05, 0.05)
    df['confidence_bin'] = pd.cut(df['confidence'], bins=bins, include_lowest=True)

    # Compute actual accuracy per bin and stratum
    calibration_data = []

    for stratum in df['stratum'].unique():
        stratum_df = df[df['stratum'] == stratum]

        for bin_interval in stratum_df['confidence_bin'].dropna().unique():
            bin_df = stratum_df[stratum_df['confidence_bin'] == bin_interval]

            if len(bin_df) > 0:
                calibration_data.append({
                    'stratum': stratum,
                    'confidence': bin_interval.mid,
                    'actual_accuracy': bin_df['correct'].mean(),
                    'n_samples': len(bin_df)
                })

    cal_df = pd.DataFrame(calibration_data)

    print(f"  Calibration points: {len(cal_df)}")

    return cal_df


def plot_calibration_panel(ax, cal_df: pd.DataFrame):
    """
    Panel C: Confidence calibration scatter
    Plot prediction confidence vs. empirical accuracy, colored by stratum
    """
    print("\nPlotting Panel C: Confidence calibration...")

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.4, label='Perfect calibration')

    # Scatter points by stratum
    for stratum in sorted(cal_df['stratum'].unique()):
        stratum_data = cal_df[cal_df['stratum'] == stratum]

        ax.scatter(
            stratum_data['confidence'],
            stratum_data['actual_accuracy'],
            label=STRATUM_LABELS[stratum].replace('\n', ' '),
            color=STRATUM_COLORS[stratum],
            alpha=0.7,
            s=stratum_data['n_samples'] * 0.5,  # Size by sample count
            edgecolors='white',
            linewidth=0.5
        )

    # Styling
    ax.set_xlabel('Predicted Confidence (1 - entropy)', fontweight='bold')
    ax.set_ylabel('Actual Accuracy', fontweight='bold')
    ax.set_title('C. Confidence Calibration', fontweight='bold', pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.95, fontsize=7)
    ax.set_aspect('equal', adjustable='box')

    # Annotation
    ax.text(0.05, 0.95, 'Well calibrated ->',
            fontsize=7, style='italic', alpha=0.6)


# ===== Panel D: Narrative Case Illustrations =====

def prepare_narrative_features(narrative_df: pd.DataFrame, samples_df: pd.DataFrame) -> Dict:
    """
    Prepare narrative-case feature values for radar plots

    Extract key features: 
    - Entropy
    - LLM confidence (1 - entropy)
    - XGB confidence (|p - 0.5| * 2)
    - Agreement (LLM == XGB)
    - Correctness (pred == label)
    """
    print("\nPreparing narrative case features...")
    # Merge with samples_df to get probability columns
    narrative_merged = narrative_df.merge(samples_df[["row_id", "p1_xgb", "p0", "p1_llm"]], on="row_id", how="left")

    cases_data = {}

    for idx, row in narrative_merged.iterrows():
        row_id = row['row_id']

        # Compute features
        max_entropy = np.log(2)
        llm_confidence = 1 - (row['entropy'] / max_entropy)
        xgb_confidence = abs(row['p1_xgb'] - 0.5) * 2
        agreement = 1 if row['pred_llm'] == row['pred_xgb'] else 0
        llm_correct = 1 if row['pred_llm'] == row['label'] else 0
        xgb_correct = 1 if row['pred_xgb'] == row['label'] else 0

        cases_data[row_id] = {
            'type': row.get('narrative_type', 'unknown'),
            'stratum': row['stratum'],
            'features': {
                'LLM Conf.': llm_confidence,
                'XGB Conf.': xgb_confidence,
                'LLM Correct': llm_correct,
                'XGB Correct': xgb_correct,
                'Agreement': agreement
            },
            'label': row['label'],
            'pred_llm': row['pred_llm'],
            'pred_xgb': row['pred_xgb']
        }

    print(f"  Prepared {len(cases_data)} narrative cases")

    return cases_data


def plot_radar_chart(ax, features: Dict, title: str, color: str):
    """
    Plot the radar chart for one case

    Args:
        ax: matplotlib axis
        features: dict of {feature_name: value}
        title: case title
        color: stratum color
    """
    # Feature names and values
    categories = list(features.keys())
    values = list(features.values())

    # Number of variables
    N = len(categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Close the plot
    values += values[:1]
    angles += angles[:1]

    # Plot
    ax.plot(angles, values, 'o-', linewidth=2, color=color, alpha=0.8)
    ax.fill(angles, values, alpha=0.25, color=color)

    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=6)

    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=5)

    # Title
    ax.set_title(title, fontsize=8, fontweight='bold', pad=10)

    # Grid
    ax.grid(True, linestyle='--', alpha=0.3)


def plot_narrative_panel(fig, gs, cases_data: Dict):
    """
    Panel D: Narrative case illustrations
    Plot radar charts for five cases (success and failure)
    """
    print("\nPlotting Panel D: Narrative case illustrations...")

    # Select up to 5 cases (prioritize 4 success + 1 failure)
    success_cases = {k: v for k, v in cases_data.items() if v['type'] == 'success'}
    failure_cases = {k: v for k, v in cases_data.items() if v['type'] == 'failure'}

    selected_cases = []

    # Add up to 4 success cases
    for case_id in list(success_cases.keys())[:4]:
        selected_cases.append((case_id, cases_data[case_id]))

    # Add up to 1 failure case
    for case_id in list(failure_cases.keys())[:1]:
        selected_cases.append((case_id, cases_data[case_id]))

    # If fewer than 5, just use what we have
    if len(selected_cases) < 5:
        remaining = [k for k in cases_data.keys() if k not in [c[0] for c in selected_cases]]
        for case_id in remaining[:5 - len(selected_cases)]:
            selected_cases.append((case_id, cases_data[case_id]))

    n_cases = len(selected_cases)
    print(f"  Plotting {n_cases} narrative cases")

    # Create subplot grid for Panel D (2 rows x 3 cols, use 5 plots)
    for i, (case_id, case_data) in enumerate(selected_cases[:5]):
        # Calculate position in subplot grid
        row = i // 3
        col = i % 3

        # Create subplot with polar projection
        ax = fig.add_subplot(gs[row, col], projection='polar')

        # Get stratum color
        stratum = case_data['stratum']
        color = STRATUM_COLORS[stratum]

        # Case title
        case_type = case_data['type'].capitalize()
        pred_status = "[OK]" if case_data['pred_llm'] == case_data['label'] else "FAIL"
        title = f"Case {i+1}: {case_type} {pred_status}"

        # Plot radar
        plot_radar_chart(ax, case_data['features'], title, color)

    # Add panel title
    panel_title = fig.text(
        0.525, 0.48,
        'D. Narrative Case Illustrations (Feature Profiles)',
        fontsize=11,
        fontweight='bold',
        ha='center'
    )


# ===== Main Figure Generation =====

def generate_4panel_figure(
    samples_df: pd.DataFrame,
    audits_df: pd.DataFrame,
    narrative_df: pd.DataFrame
):
    """
    Generate4-panel case study figure

    Layout:
    +-----------------+-----------------+
    |   Panel A       |   Panel B       |  Row 1
    | (Risk-Coverage) | (Agreement)     |
    +-----------------+-----------------+
    |   Panel C       |   Panel D       |  Row 2
    | (Calibration)   | (Narratives)    |
    +-----------------+-----------------+
    """
    print("\n" + "="*60)
    print("Generating 4-Panel Case Study Figure")
    print("="*60)

    # Create figure with custom gridspec
    fig = plt.figure(figsize=(16, 12))

    # Main grid: 2 rows x 2 cols
    main_gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                               left=0.08, right=0.95, top=0.93, bottom=0.08)

    # ===== Panel A: Risk-Coverage =====
    ax_a = fig.add_subplot(main_gs[0, 0])
    rc_df = compute_risk_coverage_by_stratum(samples_df)
    plot_risk_coverage_panel(ax_a, rc_df)

    # ===== Panel B: Audit Agreement =====
    ax_b = fig.add_subplot(main_gs[0, 1])
    agreement_df = compute_audit_agreement(audits_df)
    plot_audit_agreement_panel(ax_b, agreement_df)

    # ===== Panel C: Calibration =====
    ax_c = fig.add_subplot(main_gs[1, 0])
    cal_df = compute_calibration_data(samples_df)
    plot_calibration_panel(ax_c, cal_df)

    # ===== Panel D: Narratives =====
    # Create sub-gridspec for radar charts (2 rows x 3 cols)
    panel_d_gs = main_gs[1, 1].subgridspec(2, 3, hspace=0.4, wspace=0.3)
    cases_data = prepare_narrative_features(narrative_df, samples_df)
    plot_narrative_panel(fig, panel_d_gs, cases_data)

    # ===== Figure title =====
    fig.suptitle(
        'Case Study Analysis: Risk-Coverage, Agreement, Calibration, and Narrative Cases',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    return fig


def save_figure(fig, basename="fig_case_study_4panels"):
    """Savepublication-ready figure"""
    print("\n" + "="*60)
    print("Saving figures...")
    print("="*60)

    # PNG (300 DPI)
    png_path = f"{OUTPUT_DIR}/{basename}.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [OK] PNG: {png_path}")

    # PDF (vector)
    pdf_path = f"{OUTPUT_DIR}/{basename}.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"  [OK] PDF: {pdf_path}")

    plt.close(fig)

    return png_path, pdf_path


def validate_outputs(png_path: str, pdf_path: str):
    """Validate the expected output files."""
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)

    # Check file existence
    assert os.path.exists(png_path), f"PNG not created: {png_path}"
    assert os.path.exists(pdf_path), f"PDF not created: {pdf_path}"
    print("Output files exist")

    # Check file sizes
    png_size = os.path.getsize(png_path) / (1024 * 1024)  # MB
    pdf_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB

    print(f"PNG size: {png_size:.2f} MB")
    print(f"PDF size: {pdf_size:.2f} MB")

    assert png_size > 0.1, "PNG file too small (likely corrupt)"
    assert pdf_size > 0.01, "PDF file too small (likely corrupt)"

    print("File sizes are reasonable")


# ===== Main Execution =====

def main():
    print("=" * 60)
    print("Generate Case Study 4-Panel Figure")
    print("CRITICAL: analysis script")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    # Load data
    samples_df, audits_df, narrative_df = load_case_study_data()

    # Generate figure
    fig = generate_4panel_figure(samples_df, audits_df, narrative_df)

    # Save outputs
    png_path, pdf_path = save_figure(fig)

    # Validate
    validate_outputs(png_path, pdf_path)

    # Summary
    print("\n" + "="*60)
    print("CASE STUDY FIGURE GENERATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - {png_path} (300 DPI raster)")
    print(f"  - {pdf_path} (vector graphics)")
    print("\nFigure panels:")
    print("  A. Risk-Coverage curves by stratum")
    print("  B. Audit agreement heatmap (LLM vs XGB)")
    print("  C. Confidence calibration scatter")
    print("  D. Narrative case illustrations (radar charts)")
    print("\nNext steps:")
    print("  1. Review figures for publication quality")
    print("  2. Include in governance report")
    print("  3. Prepare manuscript figures section")
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
        sys.exit(1)
