#!/usr/bin/env python3
"""
Visualize mining performance across different miner types (CPU/GPU/QPU).

Reads structured CSV data from process_mining_comparison.py and generates
analysis charts (8 PNG files total):

1. Threshold Probabilities - Probability that a random nonce meets various
   difficulty thresholds for each miner type
2. Nonces by Threshold - Cumulative nonces meeting each threshold (normalized)
3. Proportion by Threshold - Percentage of nonces meeting each threshold
4. Win Rate Charts (3 files):
   - Actual device counts
   - Standard projection (CPU 2^20, GPU 2^10)
   - Extreme projection (CPU 2^40, GPU 2^20)
5. Expected Time by Threshold - Expected time to mine a block at each threshold
6. Nonces per Block - Distribution of nonces required to mine a block

Usage:
    python tools/visualize_mining_performance.py mining_data.csv --output-dir charts/
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Standard color scheme for miner types
MINER_COLORS = {
    'CPU': '#e74c3c',
    'GPU': '#3498db',
    'QPU': '#2ecc71',
}


def _add_difficulty_annotations(
    ax: plt.Axes,
    num_nodes: Optional[int],
    num_edges: Optional[int]
) -> None:
    """Add difficulty annotations to x-axis tick labels.

    Converts energy values to difficulty scores and appends them to tick labels.
    Silently skips if energy_utils module is not available.
    """
    if not num_nodes or not num_edges:
        return

    try:
        from shared.energy_utils import energy_to_difficulty
        xticks = ax.get_xticks()
        new_labels = []
        for energy in xticks:
            difficulty = energy_to_difficulty(energy, num_nodes, num_edges)
            new_labels.append(f'{energy:.0f}\n({difficulty:.2f})')
        ax.set_xticks(xticks)
        ax.set_xticklabels(new_labels)
    except ImportError:
        if not getattr(_add_difficulty_annotations, '_import_warned', False):
            print("Note: Difficulty annotations disabled (shared.energy_utils not available)",
                  file=sys.stderr)
            _add_difficulty_annotations._import_warned = True


def load_mining_csv(filepath: str) -> pd.DataFrame:
    """Load mining data CSV into DataFrame.

    Required columns: miner_type, energy, valid, miner_machine, process,
                      threshold, start_time, end_time, time_to_solution
    """
    df = pd.read_csv(filepath)

    # Validate required columns
    required_columns = {
        'miner_type', 'energy', 'valid', 'miner_machine', 'process',
        'threshold', 'start_time', 'end_time', 'time_to_solution'
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")

    # Parse date columns if present
    for col in ['start_time', 'end_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


def normalize_miner_type(miner_type: str) -> str:
    """Normalize miner type to CPU/GPU/QPU for display."""
    miner_type = miner_type.lower()
    if miner_type == 'cuda':
        return 'GPU'
    return miner_type.upper()


def get_miner_groups(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Pre-group DataFrame by miner_type for efficiency.

    Returns a dict mapping raw miner_type strings to their DataFrames.
    """
    return {miner_type: group for miner_type, group in df.groupby('miner_type')}


def calculate_threshold_probabilities(
    energies: np.ndarray,
    thresholds: List[float]
) -> List[float]:
    """
    Calculate probability that a random nonce meets each threshold.

    Args:
        energies: Array of all mining attempt energies
        thresholds: List of difficulty thresholds to check

    Returns:
        List of probabilities (one per threshold)
    """
    if len(energies) == 0:
        return [0.0] * len(thresholds)

    total_attempts = len(energies)
    probabilities = []

    for threshold in thresholds:
        # Count how many attempts meet or exceed threshold (more negative = better)
        meets_threshold = np.sum(energies <= threshold)
        probability = meets_threshold / total_attempts
        probabilities.append(probability)

    return probabilities


def plot_threshold_probabilities(
    df: pd.DataFrame,
    thresholds: List[float],
    topology_info: str,
    num_nodes: Optional[int] = None,
    num_edges: Optional[int] = None,
    output_file: str = 'threshold_probabilities.png'
):
    """Plot probability of nonce meeting threshold for each miner type."""
    plt.figure(figsize=(12, 7))

    for miner_type in df['miner_type'].unique():
        display_type = normalize_miner_type(miner_type)
        miner_df = df[df['miner_type'] == miner_type]
        energies = miner_df['energy'].values

        probabilities = calculate_threshold_probabilities(energies, thresholds)

        color = MINER_COLORS.get(display_type, '#95a5a6')
        n_attempts = len(energies)

        plt.plot(
            thresholds,
            probabilities,
            marker='o',
            linewidth=2,
            markersize=8,
            label=f'{display_type} (n={n_attempts:,})',
            color=color
        )

    plt.xlabel('Difficulty Threshold (Energy)', fontsize=12, fontweight='bold')
    plt.ylabel('Probability of Meeting Threshold', fontsize=12, fontweight='bold')
    plt.title(f'Probability of Nonce Meeting Difficulty Threshold\n{topology_info}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()

    _add_difficulty_annotations(plt.gca(), num_nodes, num_edges)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved threshold probability chart to {output_file}")


def plot_nonces_by_threshold(
    df: pd.DataFrame,
    thresholds: List[float],
    topology_info: str,
    num_nodes: Optional[int] = None,
    num_edges: Optional[int] = None,
    output_file: str = 'nonces_by_threshold.png'
):
    """Plot cumulative nonces meeting each threshold, normalized per unit."""
    plt.figure(figsize=(12, 7))

    # Get device counts for normalization
    device_counts = get_device_counts(df)

    for miner_type in df['miner_type'].unique():
        display_type = normalize_miner_type(miner_type)
        miner_df = df[df['miner_type'] == miner_type]
        energies = miner_df['energy'].values

        # Calculate normalization factor
        # GPU: per GPU, CPU: per 32 cores, QPU: per QPU
        raw_count = device_counts.get(display_type, 1)
        if display_type == 'CPU':
            # Normalize to 32-core units (each worker is ~1 core)
            units = raw_count / 32
            unit_label = f"~{{:.0f}} nonces per CPU (32 cores)"
        elif display_type == 'GPU':
            units = raw_count
            unit_label = f"~{{:.0f}} nonces per GPU"
        else:  # QPU
            units = raw_count
            unit_label = f"{{:.0f}} nonces per QPU"

        nonces_at_threshold = []
        for threshold in thresholds:
            # Count all nonces with energy <= threshold
            num_nonces = np.sum(energies <= threshold)
            # Normalize to per-unit
            nonces_per_unit = num_nonces / units if units > 0 else 0
            nonces_at_threshold.append(nonces_per_unit)

        color = MINER_COLORS.get(display_type, '#95a5a6')
        # Get final nonce count for legend
        final_nonces = nonces_at_threshold[-1] if nonces_at_threshold else 0
        plt.plot(
            thresholds,
            nonces_at_threshold,
            marker='o',
            linewidth=2,
            markersize=6,
            label=f'{display_type} ({unit_label.format(final_nonces)})',
            color=color
        )

    plt.xlabel('Difficulty Threshold (Energy)', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Nonces Meeting Threshold Per Unit', fontsize=12, fontweight='bold')
    plt.title(f'Nonces Meeting Energy Difficulty Threshold\n{topology_info}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    _add_difficulty_annotations(plt.gca(), num_nodes, num_edges)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved nonces by threshold chart to {output_file}")


def plot_proportion_by_threshold(
    df: pd.DataFrame,
    thresholds: List[float],
    topology_info: str,
    num_nodes: Optional[int] = None,
    num_edges: Optional[int] = None,
    output_file: str = 'proportion_by_threshold.png'
):
    """Plot percentage of nonces that would mine at each threshold."""
    plt.figure(figsize=(12, 7))

    for miner_type in df['miner_type'].unique():
        display_type = normalize_miner_type(miner_type)
        miner_df = df[df['miner_type'] == miner_type]
        energies = miner_df['energy'].values
        total_attempts = len(energies)

        proportions = []
        for threshold in thresholds:
            num_blocks = np.sum(energies <= threshold)
            proportion = (num_blocks / total_attempts) * 100
            proportions.append(proportion)

        color = MINER_COLORS.get(display_type, '#95a5a6')
        plt.plot(
            thresholds,
            proportions,
            marker='o',
            linewidth=2,
            markersize=6,
            label=f'{display_type} (n={total_attempts:,})',
            color=color
        )

    plt.xlabel('Difficulty Threshold (Energy)', fontsize=12, fontweight='bold')
    plt.ylabel('% of Nonces Meeting Threshold', fontsize=12, fontweight='bold')
    plt.title(f'Proportion of Nonces Meeting Difficulty Threshold\n{topology_info}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    _add_difficulty_annotations(plt.gca(), num_nodes, num_edges)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved proportion by threshold chart to {output_file}")


def get_device_counts(df: pd.DataFrame, miner_groups: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, int]:
    """
    Count actual devices per miner type from the data.

    For GPU/CUDA: count unique (miner_machine, process) pairs = number of GPUs
    For CPU: count unique (miner_machine, process) pairs = number of CPU workers
    For QPU: count unique miner_machine = number of QPUs

    Args:
        df: DataFrame with mining data
        miner_groups: Optional pre-grouped dict from get_miner_groups() for efficiency
    """
    counts = {}

    if miner_groups is None:
        miner_groups = get_miner_groups(df)

    for miner_type, miner_df in miner_groups.items():
        display_type = normalize_miner_type(miner_type)

        if display_type == 'QPU':
            # QPU: count unique machines
            count = miner_df['miner_machine'].nunique()
        else:
            # CPU/GPU: count unique (machine, process) pairs
            count = miner_df.groupby(['miner_machine', 'process']).ngroups

        counts[display_type] = count

    return counts


def get_miner_tts_stats(df: pd.DataFrame, miner_groups: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Calculate TTS (time-to-solution) statistics for each miner type.

    Returns a dict keyed by display_type (CPU/GPU/QPU) with:
    - energies: all energy values (for probability calculations)
    - observed_tts: mean TTS at the observed threshold
    - observed_tts_std: std dev of TTS
    - tts_samples: raw TTS values for Monte Carlo sampling
    - p_actual: probability of success at actual threshold
    - device_count: number of devices
    - min_tts: minimum observed TTS (floor for current fleet)
    - spinup_time: time to produce first nonce (floor for projections)

    This is used by both win rate calculations and expected time charts.

    Args:
        df: DataFrame with mining data
        miner_groups: Optional pre-grouped dict from get_miner_groups() for efficiency
    """
    if miner_groups is None:
        miner_groups = get_miner_groups(df)

    device_counts = get_device_counts(df, miner_groups)
    stats = {}

    for miner_type, miner_df in miner_groups.items():
        display_type = normalize_miner_type(miner_type)
        energies = miner_df['energy'].values

        # Validate single threshold per miner type
        if 'threshold' in miner_df.columns and miner_df['threshold'].nunique() > 1:
            raise ValueError(
                f"Multiple thresholds found for miner type '{miner_type}'. "
                "This script requires single-threshold CSVs per miner type."
            )

        # Get observed TTS from successful block mining
        successful_df = miner_df[miner_df['valid'] > 0]
        tts_values = successful_df['time_to_solution'].values
        valid_tts = tts_values[tts_values > 0]

        if len(valid_tts) > 0:
            observed_tts = np.mean(valid_tts)
            observed_tts_std = np.std(valid_tts)
            tts_samples = valid_tts
            # Minimum TTS from actual data (floor for current fleet)
            min_tts = np.min(valid_tts)
        else:
            observed_tts = float('inf')
            observed_tts_std = 0
            tts_samples = np.array([])
            min_tts = float('inf')

        # Spinup time - time to produce first nonce on an attempt
        # For projections, this is the floor regardless of device count
        # Estimate: spinup is approximately min_tts since most of min_tts is initialization
        spinup_time = min_tts

        # Probability of success at the actual threshold
        actual_threshold = miner_df['threshold'].iloc[0] if ('threshold' in miner_df.columns and len(miner_df) > 0) else None
        if actual_threshold is not None:
            p_actual = np.sum(energies <= actual_threshold) / len(energies) if len(energies) > 0 else 0
        else:
            # Fallback: use success rate from data
            p_actual = len(successful_df) / len(miner_df) if len(miner_df) > 0 else 0

        stats[display_type] = {
            'energies': energies,
            'num_attempts': len(energies),
            'observed_tts': observed_tts,
            'observed_tts_std': observed_tts_std,
            'tts_samples': tts_samples,
            'p_actual': p_actual,
            'device_count': device_counts.get(display_type, 1),
            'min_tts': min_tts,
            'spinup_time': spinup_time
        }

    return stats


def calculate_expected_tts(miner_stats: Dict[str, Dict[str, Any]], threshold: float) -> Dict[str, float]:
    """
    Calculate expected TTS at a given threshold for all miner types.

    Uses the scaling formula: expected_tts = observed_tts * (p_actual / p_threshold)

    Returns dict of {display_type: expected_tts} or float('inf') if impossible.
    """
    expected = {}

    for display_type, stats in miner_stats.items():
        if stats['observed_tts'] == float('inf') or stats['p_actual'] == 0:
            expected[display_type] = float('inf')
            continue

        energies = stats['energies']
        if len(energies) == 0:
            expected[display_type] = float('inf')
            continue

        # Probability of meeting this threshold
        p_threshold = np.sum(energies <= threshold) / len(energies)

        if p_threshold > 0:
            # Scale TTS: easier threshold = faster, harder = slower
            scale = stats['p_actual'] / p_threshold
            scaled_tts = stats['observed_tts'] * scale
            # Apply floor: can't be faster than min_tts (minimum observed time to mine a block)
            expected[display_type] = max(stats['min_tts'], scaled_tts)
        else:
            expected[display_type] = float('inf')

    return expected


def _run_monte_carlo_race(
    scaled_params: Dict[str, Dict[str, Any]],
    n_simulations: int = 10000
) -> Dict[str, int]:
    """Run Monte Carlo simulation to determine race winners.

    Args:
        scaled_params: Dict mapping config names to TTS distribution parameters
            Each entry has: mean, std, floor, samples (optional), use_samples
        n_simulations: Number of simulations to run

    Returns:
        Dict mapping config names to win counts
    """
    wins = {name: 0 for name in scaled_params.keys()}

    for _ in range(n_simulations):
        # Sample TTS for each miner
        sampled_tts = {}
        for name, params in scaled_params.items():
            floor = params.get('floor', 0.001)
            if params.get('use_samples') and params.get('samples') is not None:
                # Sample from actual distribution, apply floor
                sampled_tts[name] = max(floor, np.random.choice(params['samples']))
            else:
                # Sample from normal distribution, apply floor
                sample = np.random.normal(params['mean'], params['std'])
                sampled_tts[name] = max(floor, sample)

        # Determine winner (lowest TTS)
        winner = min(sampled_tts, key=sampled_tts.get)
        wins[winner] += 1

    return wins


def _calculate_win_rates(
    df: pd.DataFrame,
    thresholds: List[float],
    configurations: List[Tuple[str, str, int]],
    debug: bool = False,
    n_simulations: int = 10000
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[float]]]:
    """
    Calculate win rates using Monte Carlo simulation with actual TTS distributions.

    For mining as a race: whoever finds a valid block first wins.

    Model: Sample from observed TTS distributions (incorporating variance),
    scale for different thresholds, and simulate races.

    For each miner type at threshold T:
    1. Get observed TTS distribution (mean and std) at actual test threshold
    2. Calculate p(T) = probability of meeting threshold T (from energy distribution)
    3. Scale TTS distribution: mean and std scale by p_actual / p(T)
    4. Simulate races by sampling from scaled distributions
    5. Count wins to get empirical win probability

    Returns:
        (miner_stats, win_rates) tuple
    """
    # Step 1: Get per-miner statistics using shared function
    miner_stats = get_miner_tts_stats(df)

    if debug:
        print("\n=== Win Rate Calculation Debug (Monte Carlo) ===")
        print("Miner statistics at observed threshold:")
        for name, stats in miner_stats.items():
            cv = stats['observed_tts_std'] / stats['observed_tts'] if stats['observed_tts'] > 0 else 0
            print(f"  {name}: TTS={stats['observed_tts']:.1f}s ± {stats['observed_tts_std']:.1f}s "
                  f"(CV={cv:.2f}), p_actual={stats['p_actual']:.4f}, "
                  f"devices={stats['device_count']}")

    # Step 2: Calculate win rates for each threshold via Monte Carlo
    win_rates = {config_name: [] for config_name, _, _ in configurations}

    for i, threshold in enumerate(thresholds):
        # Build scaled TTS parameters for each configuration
        scaled_params = {}

        for config_name, miner_type, count in configurations:
            if miner_type not in miner_stats:
                continue

            stats = miner_stats[miner_type]
            if stats['observed_tts'] == float('inf') or stats['p_actual'] == 0:
                continue
            if len(stats['energies']) == 0:
                continue

            # Probability at this threshold from energy distribution
            p_threshold = np.sum(stats['energies'] <= threshold) / len(stats['energies'])

            if p_threshold > 0:
                # Scale factor for TTS (both mean and std scale together)
                scale = stats['p_actual'] / p_threshold

                # Scale for device count (for projections)
                actual_count = stats['device_count']
                is_projection = (count != actual_count and count > 0)
                if is_projection:
                    scale *= actual_count / count

                # Choose floor based on whether this is a projection
                # - For actual fleet: use min_tts (observed minimum)
                # - For projections: use spinup_time (time to first nonce, doesn't scale with devices)
                if is_projection:
                    floor = stats['spinup_time']
                else:
                    floor = stats['min_tts']

                scaled_mean = max(floor, stats['observed_tts'] * scale)
                scaled_std = stats['observed_tts_std'] * scale

                scaled_params[config_name] = {
                    'mean': scaled_mean,
                    'std': scaled_std,
                    'floor': floor,  # Floor for sampling
                    'samples': stats['tts_samples'] * scale if len(stats['tts_samples']) > 0 else None,
                    'use_samples': len(stats['tts_samples']) >= 100  # Use samples if we have enough
                }

        if not scaled_params:
            for config_name, _, _ in configurations:
                win_rates[config_name].append(0.0)
            continue

        # Run Monte Carlo simulation
        wins = _run_monte_carlo_race(scaled_params, n_simulations)

        # Convert to win rates
        for config_name, _, _ in configurations:
            if config_name in wins:
                win_rates[config_name].append(wins[config_name] / n_simulations)
            else:
                win_rates[config_name].append(0.0)

        # Debug output for first, middle, and last threshold
        if debug and i in [0, len(thresholds) // 2, len(thresholds) - 1]:
            print(f"\nThreshold {threshold}:")
            for config_name in scaled_params.keys():
                params = scaled_params[config_name]
                print(f"  {config_name}: mean={params['mean']:.1f}s, std={params['std']:.1f}s, "
                      f"win={wins[config_name]/n_simulations*100:.1f}%")

    return miner_stats, win_rates


def plot_win_rate_by_threshold(
    df: pd.DataFrame,
    thresholds: List[float],
    topology_info: str,
    num_nodes: Optional[int] = None,
    num_edges: Optional[int] = None,
    output_file: str = 'win_rate_by_threshold.png',
    n_simulations: int = 10000,
):
    """
    Plot probability of winning the mining race at each threshold.

    Shows only actual device counts from the data (no projections).
    Labels show actual counts: "GPU (8 GPUs)", "CPU (32 CPUs)", "QPU (1)".
    """
    plt.figure(figsize=(12, 7))

    # Get actual device counts
    device_counts = get_device_counts(df)

    # Generate configurations with actual counts only
    configurations = []
    for display_type, count in device_counts.items():
        if display_type == 'QPU':
            label = f'QPU ({count})'
        elif display_type == 'GPU':
            label = f'GPU ({count} GPUs)'
        else:  # CPU
            label = f'CPU ({count} CPUs)'
        configurations.append((label, display_type, count))

    # Calculate win rates with Monte Carlo simulation (uses observed variance)
    _, win_rates = _calculate_win_rates(
        df, thresholds, configurations, debug=False, n_simulations=n_simulations
    )

    # Plot each configuration
    for config_name, miner_type, count in configurations:
        win_rate_data = win_rates[config_name]
        color = MINER_COLORS.get(miner_type, '#95a5a6')

        plt.plot(
            thresholds,
            [wr * 100 for wr in win_rate_data],
            marker='o',
            linewidth=2,
            markersize=6,
            label=config_name,
            color=color,
            linestyle='-',
            alpha=0.9
        )

    plt.gca().invert_xaxis()

    _add_difficulty_annotations(plt.gca(), num_nodes, num_edges)

    plt.xlabel('Easier                                                     Harder (More Negative Energy)', fontsize=12, fontweight='bold')
    plt.ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    plt.title(f'Probability of Winning Mining Race by Difficulty\n{topology_info}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved win rate by threshold chart to {output_file}")


def plot_win_rate_by_threshold_projection(
    df: pd.DataFrame,
    thresholds: List[float],
    topology_info: str,
    num_nodes: Optional[int] = None,
    num_edges: Optional[int] = None,
    output_file: str = 'win_rate_by_threshold_projection.png',
    projection_config: str = 'standard',
    n_simulations: int = 10000,
):
    """
    Plot probability of winning the mining race with projected device counts.

    Args:
        projection_config: 'standard' for CPU 2^20 + GPU 2^10,
                          'extreme' for CPU 2^40 + GPU 2^20
    """
    plt.figure(figsize=(12, 7))

    # Get actual device counts for base comparison
    device_counts = get_device_counts(df)

    # Generate configurations with projections
    configurations = []

    if projection_config == 'extreme':
        # Extreme projection: CPU 2^40 + GPU 2^20
        if 'QPU' in device_counts:
            configurations.append((f'QPU ({device_counts["QPU"]})', 'QPU', device_counts['QPU']))
        if 'CPU' in device_counts:
            configurations.extend([
                (f'CPU ({device_counts["CPU"]} CPUs)', 'CPU', device_counts['CPU']),
                ('CPU (2^40)', 'CPU', 1099511627776)
            ])
        if 'GPU' in device_counts:
            configurations.extend([
                (f'GPU ({device_counts["GPU"]} GPUs)', 'GPU', device_counts['GPU']),
                ('GPU (2^20)', 'GPU', 1048576)
            ])
    else:
        # Standard projection: CPU 2^20 + GPU 2^15
        if 'QPU' in device_counts:
            configurations.append((f'QPU ({device_counts["QPU"]})', 'QPU', device_counts['QPU']))
        if 'CPU' in device_counts:
            configurations.extend([
                (f'CPU ({device_counts["CPU"]} CPUs)', 'CPU', device_counts['CPU']),
                ('CPU (2^20)', 'CPU', 1048576)
            ])
        if 'GPU' in device_counts:
            configurations.extend([
                (f'GPU ({device_counts["GPU"]} GPUs)', 'GPU', device_counts['GPU']),
                ('GPU (2^10)', 'GPU', 1024)
            ])

    # Calculate win rates
    _, win_rates = _calculate_win_rates(df, thresholds, configurations, n_simulations=n_simulations)

    # Plot styles for projections
    style_map = {
        1024: {'linestyle': '--', 'marker': 's', 'linewidth': 2.5, 'alpha': 0.7},        # 2^10
        65536: {'linestyle': ':', 'marker': '^', 'linewidth': 2.5, 'alpha': 0.7},        # 2^16
        1048576: {'linestyle': '--', 'marker': 'D', 'linewidth': 2.5, 'alpha': 0.7},     # 2^20
        1099511627776: {'linestyle': ':', 'marker': 'X', 'linewidth': 2.5, 'alpha': 0.7} # 2^40
    }

    for config_name, miner_type, count in configurations:
        win_rate_data = win_rates[config_name]
        color = MINER_COLORS.get(miner_type, '#95a5a6')
        style = style_map.get(count, {'linestyle': '-', 'marker': 'o', 'linewidth': 2, 'alpha': 0.9})

        plt.plot(
            thresholds,
            [wr * 100 for wr in win_rate_data],
            marker=style['marker'],
            linewidth=style['linewidth'],
            markersize=6,
            label=config_name,
            color=color,
            linestyle=style['linestyle'],
            alpha=style['alpha']
        )

    plt.gca().invert_xaxis()

    _add_difficulty_annotations(plt.gca(), num_nodes, num_edges)

    plt.xlabel('Easier                                                     Harder (More Negative Energy)', fontsize=12, fontweight='bold')
    plt.ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    plt.title(f'Probability of Winning Mining Race (Projected)\n{topology_info}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved win rate projection chart to {output_file}")


def plot_expected_time_by_threshold(
    df: pd.DataFrame,
    thresholds: List[float],
    topology_info: str,
    num_nodes: Optional[int] = None,
    num_edges: Optional[int] = None,
    output_file: str = 'expected_time_by_threshold.png'
):
    """Plot expected time to mine a block at each threshold."""
    plt.figure(figsize=(12, 7))

    # Get TTS stats using shared function
    miner_stats = get_miner_tts_stats(df)

    for display_type, stats in miner_stats.items():
        if stats['observed_tts'] == float('inf') or stats['p_actual'] == 0:
            continue

        # Calculate expected TTS for each threshold
        expected_times = []
        for threshold in thresholds:
            expected = calculate_expected_tts({display_type: stats}, threshold)
            expected_times.append(expected.get(display_type, float('inf')))

        valid_data = [(t, et) for t, et in zip(thresholds, expected_times) if et != float('inf')]
        if not valid_data:
            continue

        valid_thresholds, valid_times = zip(*valid_data)

        color = MINER_COLORS.get(display_type, '#95a5a6')
        plt.plot(
            valid_thresholds,
            valid_times,
            marker='o',
            linewidth=2,
            markersize=6,
            label=f'{display_type} (mean: {stats["observed_tts"]:.1f}s, floor: {stats["min_tts"]:.1f}s)',
            color=color
        )

    plt.gca().invert_xaxis()

    _add_difficulty_annotations(plt.gca(), num_nodes, num_edges)

    plt.xlabel('Easier                                                     Harder (More Negative Energy)', fontsize=12, fontweight='bold')
    plt.ylabel('Expected Time to Mine Block (seconds)', fontsize=12, fontweight='bold')
    plt.title(f'Expected Mining Time by Difficulty\n{topology_info}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved expected time by threshold chart to {output_file}")


def plot_nonces_per_block(
    df: pd.DataFrame,
    topology_info: str,
    output_file: str = 'nonces_per_block.png'
):
    """Plot nonces per block with summary bar chart and distribution histograms."""
    # Get the threshold used for mining (from the data)
    threshold = df['threshold'].iloc[0] if ('threshold' in df.columns and len(df) > 0) else None

    # Create figure with 2 rows: bar chart on top, 3 histograms below
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.15, wspace=0.15)

    # Top row spans all 3 columns for the bar chart
    ax_bar = fig.add_subplot(gs[0, :])

    # Count attempts per (miner_machine, process, block_num) group
    # A "block" is mined when valid > 0
    nonces_data = {}
    block_counts = {}

    for miner_type in df['miner_type'].unique():
        display_type = normalize_miner_type(miner_type)
        miner_df = df[df['miner_type'] == miner_type]

        # Group by machine, process to count attempts between successful blocks
        nonces_list = []

        for (machine, process), group in miner_df.groupby(['miner_machine', 'process']):
            group = group.sort_values('end_time')
            attempt_count = 0

            for _, row in group.iterrows():
                attempt_count += 1
                if row['valid'] > 0:
                    nonces_list.append(attempt_count)
                    attempt_count = 0

        nonces_data[display_type] = nonces_list
        block_counts[display_type] = len(nonces_list)

    miner_types = sorted(nonces_data.keys())
    positions = np.arange(len(miner_types))

    # Calculate stats including skewness - use dict for safe access
    stats_by_type: Dict[str, Dict[str, float]] = {}
    for miner_type in miner_types:
        nonces = nonces_data[miner_type]
        if nonces:
            arr = np.array(nonces)
            skewness = scipy_stats.skew(arr)
            stats_by_type[miner_type] = {
                'mean': np.mean(arr),
                'median': np.median(arr),
                'min': np.min(arr),
                'max': np.max(arr),
                'std': np.std(arr),
                'skew': skewness,
                'p25': np.percentile(arr, 25),
                'p75': np.percentile(arr, 75),
                'p95': np.percentile(arr, 95)
            }
        else:
            stats_by_type[miner_type] = {
                'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0,
                'skew': 0, 'p25': 0, 'p75': 0, 'p95': 0
            }

    # Build stats_data list in same order as miner_types for bar chart
    stats_data = [stats_by_type[mt] for mt in miner_types]

    # ---- Top chart: Bar chart with summary stats ----
    means = [s['mean'] for s in stats_data]
    bars = ax_bar.bar(
        positions,
        means,
        color=[MINER_COLORS.get(mt, '#95a5a6') for mt in miner_types],
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5
    )

    # Use asymmetric error bars - can't go below 1 nonce (clamp lower error to non-negative)
    lower_errors = [min(s['std'], max(0.0, s['mean'] - 1.0)) for s in stats_data]
    upper_errors = [s['std'] for s in stats_data]
    ax_bar.errorbar(
        positions,
        means,
        yerr=[lower_errors, upper_errors],
        fmt='none',
        ecolor='black',
        capsize=5,
        capthick=2
    )

    for i, (bar, stat, mt) in enumerate(zip(bars, stats_data, miner_types)):
        height = bar.get_height()
        # Position text above the error bar
        text_y = height + stat['std'] + 0.5
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            f"μ={stat['mean']:.1f}, med={stat['median']:.0f}\nskew={stat['skew']:.2f}",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

    ax_bar.set_xlabel('Miner Type', fontsize=12, fontweight='bold')
    ax_bar.set_ylabel('Average Nonces per Block', fontsize=12, fontweight='bold')
    threshold_str = f' (threshold: {threshold:.0f})' if threshold else ''
    ax_bar.set_title(f'Nonces Required to Mine a Block{threshold_str}\n{topology_info}', fontsize=14, fontweight='bold')
    ax_bar.set_xticks(positions)
    labels_with_counts = [f'{mt}\n(n={block_counts.get(mt, 0):,} blocks)' for mt in miner_types]
    ax_bar.set_xticklabels(labels_with_counts, fontsize=11)
    ax_bar.grid(True, alpha=0.3, axis='y')

    # ---- Bottom row: Distribution histograms ----
    # Ensure consistent order: CPU, GPU, QPU
    ordered_types = ['CPU', 'GPU', 'QPU']
    for i, mt in enumerate(ordered_types):
        ax_hist = fig.add_subplot(gs[1, i])

        if mt in nonces_data and nonces_data[mt]:
            nonces = np.array(nonces_data[mt])
            stat = stats_by_type.get(mt)
            if stat is None:
                ax_hist.text(0.5, 0.5, f'No {mt} stats', ha='center', va='center', transform=ax_hist.transAxes)
                ax_hist.set_title(f'{mt} Distribution', fontsize=11, fontweight='bold')
                continue

            # Use integer bins, capped at 99th percentile for readability
            max_val = min(np.max(nonces), np.percentile(nonces, 99))
            bins = np.arange(1, max_val + 2)

            ax_hist.hist(
                nonces,
                bins=bins,
                color=MINER_COLORS.get(mt, '#95a5a6'),
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5
            )

            # Add vertical lines for mean and median
            ax_hist.axvline(stat['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stat["mean"]:.1f}')
            ax_hist.axvline(stat['median'], color='blue', linestyle='-', linewidth=2, label=f'Median: {stat["median"]:.0f}')

            # Add percentile info
            ax_hist.axvline(stat['p95'], color='orange', linestyle=':', linewidth=1.5, label=f'p95: {stat["p95"]:.0f}')

            ax_hist.set_xlabel('Nonces per Block', fontsize=10)
            ax_hist.set_ylabel('Frequency', fontsize=10)
            ax_hist.set_title(f'{mt} Distribution\n(skew={stat["skew"]:.2f})', fontsize=11, fontweight='bold')
            ax_hist.legend(fontsize=8, loc='upper right')
            ax_hist.grid(True, alpha=0.3, axis='y')

            # Limit x-axis to show most of the data clearly
            ax_hist.set_xlim(0, stat['p95'] * 1.5)
        else:
            ax_hist.text(0.5, 0.5, f'No {mt} data', ha='center', va='center', transform=ax_hist.transAxes)
            ax_hist.set_title(f'{mt} Distribution', fontsize=11, fontweight='bold')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved nonces per block chart to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize mining performance from CSV data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/visualize_mining_performance.py mining_data.csv
  python tools/visualize_mining_performance.py mining_data.csv --output-dir charts/
        """
    )
    parser.add_argument(
        'csv_file',
        type=Path,
        help='CSV file from process_mining_comparison.py'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Directory to save output charts (default: current directory)'
    )
    parser.add_argument(
        '--threshold-min',
        type=float,
        default=-15000,
        help='Minimum difficulty threshold (default: -15000)'
    )
    parser.add_argument(
        '--threshold-max',
        type=float,
        default=-14700,
        help='Maximum difficulty threshold (default: -14700)'
    )
    parser.add_argument(
        '--threshold-step',
        type=float,
        default=10,
        help='Step size for difficulty thresholds (default: 10)'
    )
    parser.add_argument(
        '--topology',
        type=str,
        default='Advantage2_system1.12',
        help='Topology name for chart titles (default: Advantage2_system1.12)'
    )
    parser.add_argument(
        '--mc-sims',
        type=int,
        default=10000,
        help='Number of Monte Carlo simulations for win rate calculation (default: 10000)'
    )
    parser.add_argument(
        '--mc-seed',
        type=int,
        default=42,
        help='Random seed for Monte Carlo reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Seed RNG for reproducibility
    np.random.seed(args.mc_seed)

    if not args.csv_file.exists():
        print(f"Error: CSV file '{args.csv_file}' not found")
        return 1

    # Load topology for node/edge counts
    num_nodes = None
    num_edges = None
    try:
        from dwave_topologies.topologies.json_loader import load_topology
        topology = load_topology(args.topology)
        num_nodes = len(topology.graph.nodes) if hasattr(topology.graph, 'nodes') else topology.num_nodes
        num_edges = len(topology.graph.edges) if hasattr(topology.graph, 'edges') else topology.num_edges
        topology_desc = f"{args.topology} - {num_nodes:,} nodes, {num_edges:,} edges"
        print(f"Loaded topology: {topology_desc}")
    except (ImportError, FileNotFoundError, KeyError, AttributeError) as e:
        print(f"Could not load topology '{args.topology}': {e}")
        topology_desc = args.topology

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate threshold range
    if args.threshold_min > args.threshold_max:
        step = -abs(args.threshold_step)
        thresholds = np.arange(args.threshold_min, args.threshold_max - abs(step), step)
    else:
        step = abs(args.threshold_step)
        thresholds = np.arange(args.threshold_min, args.threshold_max + step, step)
    thresholds = thresholds.tolist()

    if not thresholds:
        print(f"Error: No thresholds generated")
        return 1

    print(f"Loading {args.csv_file}...")
    df = load_mining_csv(str(args.csv_file))
    print(f"Loaded {len(df):,} records")

    print(f"\nThreshold range: {thresholds[0]} to {thresholds[-1]} ({len(thresholds)} points)")

    # Print summary
    print("\nData summary by miner type:")
    for miner_type in df['miner_type'].unique():
        miner_df = df[df['miner_type'] == miner_type]
        successful = miner_df[miner_df['valid'] > 0]
        print(f"  {normalize_miner_type(miner_type)}: {len(miner_df):,} attempts, "
              f"{len(successful):,} successful ({100*len(successful)/len(miner_df):.1f}%)")

    print("\nGenerating charts...")

    # Chart 1: Threshold probabilities
    plot_threshold_probabilities(
        df, thresholds, topology_desc, num_nodes, num_edges,
        str(args.output_dir / 'threshold_probabilities.png')
    )

    # Chart 2: Nonces by threshold
    plot_nonces_by_threshold(
        df, thresholds, topology_desc, num_nodes, num_edges,
        str(args.output_dir / 'nonces_by_threshold.png')
    )

    # Chart 3: Proportion by threshold
    plot_proportion_by_threshold(
        df, thresholds, topology_desc, num_nodes, num_edges,
        str(args.output_dir / 'proportion_by_threshold.png')
    )

    # Chart 4a: Win rate (actual device counts only)
    plot_win_rate_by_threshold(
        df, thresholds, topology_desc, num_nodes, num_edges,
        str(args.output_dir / 'win_rate_by_threshold.png'),
        n_simulations=args.mc_sims,
    )

    # Chart 4b: Win rate projection - standard (CPU 2^20, GPU 2^10)
    plot_win_rate_by_threshold_projection(
        df, thresholds, topology_desc, num_nodes, num_edges,
        str(args.output_dir / 'win_rate_by_threshold_projection.png'),
        projection_config='standard',
        n_simulations=args.mc_sims,
    )

    # Chart 4c: Win rate projection - extreme (CPU 2^40, GPU 2^20)
    plot_win_rate_by_threshold_projection(
        df, thresholds, topology_desc, num_nodes, num_edges,
        str(args.output_dir / 'win_rate_by_threshold_extreme_projection.png'),
        projection_config='extreme',
        n_simulations=args.mc_sims,
    )

    # Chart 5: Expected time
    plot_expected_time_by_threshold(
        df, thresholds, topology_desc, num_nodes, num_edges,
        str(args.output_dir / 'expected_time_by_threshold.png')
    )

    # Chart 6: Nonces per block
    plot_nonces_per_block(
        df, topology_desc,
        str(args.output_dir / 'nonces_per_block.png')
    )

    print(f"\nDone! Charts saved to {args.output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
