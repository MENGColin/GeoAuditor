"""
Generate governance-oriented priority queues and country maps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
import json
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent  # scripts/governance -> scripts -> project_root
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"

# Input files
LLM_PREDICTIONS = OUTPUTS_DIR / "inference" / "stage1_llm_zeroshot_predictions.parquet"
XGB_PREDICTIONS = OUTPUTS_DIR / "inference" / "baseline_xgb_predictions.parquet"

# Output directories
GOVERNANCE_DIR = OUTPUTS_DIR / "governance"
MAPS_DIR = GOVERNANCE_DIR / "country_maps"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Top-N configurations
TOP_N_VALUES = [50, 100, 200]

# 30 African countries (based on DHS survey availability)
AFRICAN_COUNTRIES = [
    'Angola', 'Benin', 'Burkina Faso', 'Burundi', 'Cameroon',
    'Chad', 'Comoros', 'Congo Democratic Republic', 'Congo',
    'Cote d\'Ivoire', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana',
    'Guinea', 'Kenya', 'Lesotho', 'Liberia', 'Madagascar',
    'Malawi', 'Mali', 'Mozambique', 'Namibia', 'Niger',
    'Nigeria', 'Rwanda', 'Senegal', 'Sierra Leone', 'Tanzania',
    'Togo', 'Uganda', 'Zambia', 'Zimbabwe'
]

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Setup logging configuration."""
    GOVERNANCE_DIR.mkdir(parents=True, exist_ok=True)

    log_file = GOVERNANCE_DIR / f"governance_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================

def validate_input_files() -> bool:
    """Validate that all required input files exist."""
    logger.info("Validating input files...")

    missing_files = []

    if not LLM_PREDICTIONS.exists():
        missing_files.append(str(LLM_PREDICTIONS))

    if not XGB_PREDICTIONS.exists():
        missing_files.append(str(XGB_PREDICTIONS))

    if missing_files:
        logger.error(f"Missing input files: {missing_files}")
        return False

    logger.info("All input files validated successfully")
    return True


def load_predictions() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load LLM and XGB predictions."""
    logger.info("Loading prediction data...")

    try:
        llm_df = pd.read_parquet(LLM_PREDICTIONS)
        logger.info(f"Loaded LLM predictions: {len(llm_df)} records")

        xgb_df = pd.read_parquet(XGB_PREDICTIONS)
        logger.info(f"Loaded XGB predictions: {len(xgb_df)} records")

        # Validate required columns
        required_cols = ['country', 'latitude', 'longitude', 'wealth_index_predicted']

        for df_name, df in [("LLM", llm_df), ("XGB", xgb_df)]:
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"{df_name} predictions missing columns: {missing_cols}")

        return llm_df, xgb_df

    except Exception as e:
        logger.error(f"Error loading predictions: {e}")
        raise


# =============================================================================
# GRID-LEVEL AGGREGATION
# =============================================================================

def aggregate_to_grid(df: pd.DataFrame, grid_size: float = 0.1) -> pd.DataFrame:
    """
    Aggregate predictions to grid cells.

    Args:
        df: DataFrame with predictions
        grid_size: Grid cell size in degrees (default: 0.1 = ~11km at equator)

    Returns:
        DataFrame with grid-level aggregated predictions
    """
    logger.info(f"Aggregating to {grid_size}-degree grid cells...")

    df = df.copy()

    # Create grid cell IDs
    df['grid_lat'] = (df['latitude'] / grid_size).round() * grid_size
    df['grid_lon'] = (df['longitude'] / grid_size).round() * grid_size
    df['grid_id'] = df['country'] + '_' + \
                    df['grid_lat'].astype(str) + '_' + \
                    df['grid_lon'].astype(str)

    # Aggregate by grid cell
    grid_df = df.groupby(['country', 'grid_id', 'grid_lat', 'grid_lon']).agg({
        'wealth_index_predicted': 'mean',
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()

    # Rename for clarity
    grid_df.rename(columns={
        'latitude': 'center_lat',
        'longitude': 'center_lon',
        'wealth_index_predicted': 'poverty_score'
    }, inplace=True)

    logger.info(f"Created {len(grid_df)} grid cells from {len(df)} predictions")

    return grid_df


# =============================================================================
# PRIORITY QUEUE GENERATION
# =============================================================================

def generate_priority_queues(
    llm_grid: pd.DataFrame,
    xgb_grid: pd.DataFrame,
    top_n_values: List[int] = TOP_N_VALUES
) -> Dict[int, pd.DataFrame]:
    """
    Generate Top-N priority queues for each country.

    Args:
        llm_grid: Grid-aggregated LLM predictions
        xgb_grid: Grid-aggregated XGB predictions
        top_n_values: List of N values for Top-N queues

    Returns:
        Dictionary mapping N to priority queue DataFrames
    """
    logger.info(f"Generating priority queues for Top-N: {top_n_values}...")

    # Merge LLM and XGB scores
    merged = llm_grid.merge(
        xgb_grid[['grid_id', 'poverty_score']],
        on='grid_id',
        suffixes=('_llm', '_xgb'),
        how='outer'
    )

    # Fill missing values with median (for areas only in one dataset)
    merged['poverty_score_llm'].fillna(merged['poverty_score_llm'].median(), inplace=True)
    merged['poverty_score_xgb'].fillna(merged['poverty_score_xgb'].median(), inplace=True)

    priority_queues = {}

    for n in top_n_values:
        logger.info(f"Generating Top-{n} priority queues...")

        country_queues = []

        for country in merged['country'].unique():
            country_data = merged[merged['country'] == country].copy()

            # Skip if country has fewer grid cells than N
            if len(country_data) < n:
                logger.warning(f"{country} has only {len(country_data)} grid cells (< {n}), using all available")
                top_n_country = country_data.copy()
            else:
                # Sort by poverty score (lower = poorer = higher priority)
                # Using LLM scores as primary ranking
                top_n_country = country_data.nsmallest(n, 'poverty_score_llm')

            # Add ranking
            top_n_country = top_n_country.copy()
            top_n_country['priority_rank'] = range(1, len(top_n_country) + 1)

            country_queues.append(top_n_country)

        priority_queue_df = pd.concat(country_queues, ignore_index=True)

        # Select and order columns
        columns_order = [
            'country', 'priority_rank', 'grid_id',
            'grid_lat', 'grid_lon', 'center_lat', 'center_lon',
            'poverty_score_llm', 'poverty_score_xgb'
        ]

        priority_queue_df = priority_queue_df[columns_order]

        priority_queues[n] = priority_queue_df

        logger.info(f"Generated Top-{n} queue with {len(priority_queue_df)} total grid cells")

    return priority_queues


# =============================================================================
# OVERLAP ANALYSIS
# =============================================================================

def calculate_jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union > 0 else 0.0


def analyze_overlap(priority_queues: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Analyze overlap between LLM and XGB priority rankings.

    Args:
        priority_queues: Dictionary of priority queues by Top-N

    Returns:
        DataFrame with overlap analysis results
    """
    logger.info("Analyzing LLM vs XGB overlap...")

    overlap_results = []

    for n, queue_df in priority_queues.items():
        logger.info(f"Analyzing overlap for Top-{n}...")

        for country in queue_df['country'].unique():
            country_data = queue_df[queue_df['country'] == country]

            # Get Top-N grid cells by LLM and XGB scores
            llm_top_n = set(country_data.nsmallest(n, 'poverty_score_llm')['grid_id'])
            xgb_top_n = set(country_data.nsmallest(n, 'poverty_score_xgb')['grid_id'])

            # Calculate Jaccard similarity
            jaccard = calculate_jaccard_similarity(llm_top_n, xgb_top_n)

            # Calculate Spearman correlation on ranks
            # Create ranking for both methods
            country_data_sorted = country_data.copy()
            country_data_sorted['rank_llm'] = country_data_sorted['poverty_score_llm'].rank()
            country_data_sorted['rank_xgb'] = country_data_sorted['poverty_score_xgb'].rank()

            spearman_corr, spearman_pval = spearmanr(
                country_data_sorted['rank_llm'],
                country_data_sorted['rank_xgb']
            )

            overlap_results.append({
                'country': country,
                'top_n': n,
                'total_grids': len(country_data),
                'llm_top_n_count': len(llm_top_n),
                'xgb_top_n_count': len(xgb_top_n),
                'overlap_count': len(llm_top_n.intersection(xgb_top_n)),
                'jaccard_similarity': jaccard,
                'spearman_correlation': spearman_corr,
                'spearman_pvalue': spearman_pval
            })

    overlap_df = pd.DataFrame(overlap_results)

    # Calculate summary statistics
    for n in priority_queues.keys():
        subset = overlap_df[overlap_df['top_n'] == n]
        logger.info(f"\nTop-{n} Overlap Summary:")
        logger.info(f"  Mean Jaccard: {subset['jaccard_similarity'].mean():.3f}")
        logger.info(f"  Mean Spearman: {subset['spearman_correlation'].mean():.3f}")
        logger.info(f"  Median Jaccard: {subset['jaccard_similarity'].median():.3f}")
        logger.info(f"  Median Spearman: {subset['spearman_correlation'].median():.3f}")

    return overlap_df


# =============================================================================
# MAP VISUALIZATION
# =============================================================================

def create_country_priority_map(
    country: str,
    priority_queue: pd.DataFrame,
    output_path: Path,
    top_n: int = 50
):
    """
    Create priority map for a single country.

    Args:
        country: Country name
        priority_queue: Priority queue DataFrame
        output_path: Path to save the map
        top_n: Number of top priority areas to highlight
    """
    country_data = priority_queue[priority_queue['country'] == country].copy()

    if len(country_data) == 0:
        logger.warning(f"No data for {country}, skipping map")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # LLM Priority Map
    scatter1 = ax1.scatter(
        country_data['center_lon'],
        country_data['center_lat'],
        c=country_data['poverty_score_llm'],
        cmap='RdYlGn_r',  # Red (poor) to Green (wealthy), reversed
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

    # Highlight top N priority areas
    top_n_llm = country_data.nsmallest(min(top_n, len(country_data)), 'poverty_score_llm')
    ax1.scatter(
        top_n_llm['center_lon'],
        top_n_llm['center_lat'],
        s=200,
        facecolors='none',
        edgecolors='red',
        linewidth=2,
        label=f'Top-{top_n} Priority'
    )

    ax1.set_title(f'{country} - LLM Priority Map', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Poverty Score (lower = poorer)')

    # XGB Priority Map
    scatter2 = ax2.scatter(
        country_data['center_lon'],
        country_data['center_lat'],
        c=country_data['poverty_score_xgb'],
        cmap='RdYlGn_r',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

    # Highlight top N priority areas
    top_n_xgb = country_data.nsmallest(min(top_n, len(country_data)), 'poverty_score_xgb')
    ax2.scatter(
        top_n_xgb['center_lon'],
        top_n_xgb['center_lat'],
        s=200,
        facecolors='none',
        edgecolors='red',
        linewidth=2,
        label=f'Top-{top_n} Priority'
    )

    ax2.set_title(f'{country} - XGB Priority Map', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude', fontsize=12)
    ax2.set_ylabel('Latitude', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Poverty Score (lower = poorer)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved map for {country} to {output_path}")


def generate_all_country_maps(priority_queues: Dict[int, pd.DataFrame]):
    """Generate maps for all countries."""
    logger.info("Generating country priority maps...")

    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    # Use Top-50 queue for visualization
    priority_queue = priority_queues[50]

    countries = priority_queue['country'].unique()

    for i, country in enumerate(countries, 1):
        logger.info(f"Generating map {i}/{len(countries)}: {country}")

        # Clean country name for filename
        safe_country_name = country.replace(' ', '_').replace('\'', '')
        output_path = MAPS_DIR / f"{safe_country_name}_priority_map.png"

        try:
            create_country_priority_map(
                country,
                priority_queue,
                output_path,
                top_n=50
            )
        except Exception as e:
            logger.error(f"Error generating map for {country}: {e}")


def create_publication_figure(priority_queues: Dict[int, pd.DataFrame]):
    """Create publication-quality figure showing selected country maps."""
    logger.info("Creating publication figure...")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Select 6 representative countries for publication
    selected_countries = ['Nigeria', 'Kenya', 'Ethiopia', 'Ghana', 'Tanzania', 'Uganda']

    priority_queue = priority_queues[50]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, country in enumerate(selected_countries):
        ax = axes[idx]

        country_data = priority_queue[priority_queue['country'] == country]

        if len(country_data) == 0:
            ax.text(0.5, 0.5, f'No data for {country}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(country, fontsize=12, fontweight='bold')
            continue

        # Plot LLM priority scores
        scatter = ax.scatter(
            country_data['center_lon'],
            country_data['center_lat'],
            c=country_data['poverty_score_llm'],
            cmap='RdYlGn_r',
            s=50,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.3
        )

        # Highlight top 50 priority areas
        top_50 = country_data.nsmallest(50, 'poverty_score_llm')
        ax.scatter(
            top_50['center_lon'],
            top_50['center_lat'],
            s=100,
            facecolors='none',
            edgecolors='red',
            linewidth=1.5
        )

        ax.set_title(country, fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax, label='Poverty Score')

    plt.suptitle('Priority Areas for Poverty Intervention (Top-50 Grid Cells)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    output_path = FIGURES_DIR / "fig_governance_maps.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved publication figure to {output_path}")


# =============================================================================
# MANIFEST GENERATION
# =============================================================================

def generate_manifest(priority_queues: Dict[int, pd.DataFrame], overlap_df: pd.DataFrame):
    """Generate manifest file documenting all outputs."""
    logger.info("Generating manifest file...")

    manifest = {
        'generated_at': datetime.now().isoformat(),
        'script_version': '1.0',
        'random_seed': RANDOM_SEED,
        'configuration': {
            'top_n_values': TOP_N_VALUES,
            'grid_size_degrees': 0.1,
            'input_files': {
                'llm_predictions': str(LLM_PREDICTIONS),
                'xgb_predictions': str(XGB_PREDICTIONS)
            }
        },
        'outputs': {
            'priority_queues': {},
            'overlap_analysis': str(GOVERNANCE_DIR / "overlap_analysis.csv"),
            'country_maps': str(MAPS_DIR),
            'publication_figure': str(FIGURES_DIR / "fig_governance_maps.pdf")
        },
        'statistics': {
            'total_countries': len(priority_queues[50]['country'].unique()),
            'overlap_summary': {}
        }
    }

    # Add priority queue info
    for n, queue_df in priority_queues.items():
        manifest['outputs']['priority_queues'][f'top_{n}'] = {
            'file': str(GOVERNANCE_DIR / f"priority_queues_top{n}.csv"),
            'total_grids': len(queue_df),
            'countries': len(queue_df['country'].unique())
        }

    # Add overlap statistics
    for n in TOP_N_VALUES:
        subset = overlap_df[overlap_df['top_n'] == n]
        manifest['statistics']['overlap_summary'][f'top_{n}'] = {
            'mean_jaccard': float(subset['jaccard_similarity'].mean()),
            'median_jaccard': float(subset['jaccard_similarity'].median()),
            'mean_spearman': float(subset['spearman_correlation'].mean()),
            'median_spearman': float(subset['spearman_correlation'].median())
        }

    manifest_path = GOVERNANCE_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved manifest to {manifest_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("T02: GOVERNANCE PRIORITY QUEUE AND MAP GENERATION")
    logger.info("=" * 80)

    start_time = datetime.now()

    try:
        # Step 1: Validate inputs
        if not validate_input_files():
            raise FileNotFoundError("Required input files are missing")

        # Step 2: Load predictions
        llm_df, xgb_df = load_predictions()

        # Step 3: Grid-level aggregation
        llm_grid = aggregate_to_grid(llm_df, grid_size=0.1)
        xgb_grid = aggregate_to_grid(xgb_df, grid_size=0.1)

        # Step 4: Generate priority queues
        priority_queues = generate_priority_queues(llm_grid, xgb_grid, TOP_N_VALUES)

        # Step 5: Save priority queues
        logger.info("Saving priority queues...")
        for n, queue_df in priority_queues.items():
            output_path = GOVERNANCE_DIR / f"priority_queues_top{n}.csv"
            queue_df.to_csv(output_path, index=False)
            logger.info(f"Saved Top-{n} queue to {output_path}")

        # Step 6: Overlap analysis
        overlap_df = analyze_overlap(priority_queues)
        overlap_path = GOVERNANCE_DIR / "overlap_analysis.csv"
        overlap_df.to_csv(overlap_path, index=False)
        logger.info(f"Saved overlap analysis to {overlap_path}")

        # Step 7: Generate maps
        generate_all_country_maps(priority_queues)

        # Step 8: Create publication figure
        create_publication_figure(priority_queues)

        # Step 9: Generate manifest
        generate_manifest(priority_queues, overlap_df)

        # Final summary
        elapsed_time = datetime.now() - start_time
        logger.info("=" * 80)
        logger.info("EXECUTION COMPLETED SUCCESSFULLY")
        logger.info(f"Total execution time: {elapsed_time}")
        logger.info(f"Total countries processed: {len(priority_queues[50]['country'].unique())}")
        logger.info(f"Priority queues generated: {len(priority_queues)}")
        logger.info(f"Maps generated: {len(list(MAPS_DIR.glob('*.png')))}")
        logger.info(f"All outputs saved to: {GOVERNANCE_DIR}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
