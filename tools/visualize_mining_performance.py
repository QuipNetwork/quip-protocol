#!/usr/bin/env python3
"""
Visualize mining performance across different miner types (CPU/GPU/QPU).

Reads structured CSV data from process_mining_comparison.py and generates
six analysis charts:

1. Probability of Nonce Meeting Threshold - Shows probability that a random nonce
   meets various difficulty thresholds for each miner type
2. Blocks by Threshold - Shows cumulative blocks mined at each threshold
3. Proportion by Threshold - Shows percentage of nonces that would mine at each
   threshold (normalized to proportion of blocks)
4. Win Rate by Threshold - Shows probability of winning the mining race at each
   difficulty level, accounting for different attempt rates
5. Expected Time by Threshold - Shows expected time to mine a block at each
   threshold, directly comparing mining performance between platforms
6. Nonces per Block - Histogram showing distribution of nonces required to mine
   a block for each miner type

Usage:
    python tools/visualize_mining_performance.py mining_data.csv --output-dir charts/
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_mining_csv(filepath: str) -> pd.DataFrame:
    """Load mining data CSV into DataFrame."""
    df = pd.read_csv(filepath, parse_dates=['start_time', 'end_time'])
    return df


def normalize_miner_type(miner_type: str) -> str:
    """Normalize miner type to CPU/GPU/QPU for display."""
    miner_type = miner_type.lower()
    if miner_type == 'cuda':
        return 'GPU'
    return miner_type.upper()


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

    colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71'
    }

    for miner_type in df['miner_type'].unique():
        display_type = normalize_miner_type(miner_type)
        miner_df = df[df['miner_type'] == miner_type]
        energies = miner_df['energy'].values

        probabilities = calculate_threshold_probabilities(energies, thresholds)

        color = colors.get(display_type, '#95a5a6')
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

    # Add difficulty annotations if possible
    if num_nodes and num_edges:
        try:
            from shared.energy_utils import energy_to_difficulty
            ax = plt.gca()
            xticks = ax.get_xticks()
            new_labels = []
            for energy in xticks:
                difficulty = energy_to_difficulty(energy, num_nodes, num_edges)
                new_labels.append(f'{energy:.0f}\n({difficulty:.2f})')
            ax.set_xticks(xticks)
            ax.set_xticklabels(new_labels)
        except ImportError:
            pass

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved threshold probability chart to {output_file}")


def plot_blocks_by_threshold(
    df: pd.DataFrame,
    thresholds: List[float],
    topology_info: str,
    num_nodes: Optional[int] = None,
    num_edges: Optional[int] = None,
    output_file: str = 'blocks_by_threshold.png'
):
    """Plot number of blocks that would be mined at each threshold."""
    plt.figure(figsize=(12, 7))

    colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71'
    }

    for miner_type in df['miner_type'].unique():
        display_type = normalize_miner_type(miner_type)
        miner_df = df[df['miner_type'] == miner_type]
        energies = miner_df['energy'].values

        blocks_at_threshold = []
        for threshold in thresholds:
            num_blocks = np.sum(energies <= threshold)
            blocks_at_threshold.append(num_blocks)

        color = colors.get(display_type, '#95a5a6')
        plt.plot(
            thresholds,
            blocks_at_threshold,
            marker='o',
            linewidth=2,
            markersize=6,
            label=f'{display_type} (total={len(energies):,})',
            color=color
        )

    plt.xlabel('Difficulty Threshold (Energy)', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Blocks Mined', fontsize=12, fontweight='bold')
    plt.title(f'Blocks Mined by Difficulty Threshold\n{topology_info}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if num_nodes and num_edges:
        try:
            from shared.energy_utils import energy_to_difficulty
            ax = plt.gca()
            xticks = ax.get_xticks()
            new_labels = []
            for energy in xticks:
                difficulty = energy_to_difficulty(energy, num_nodes, num_edges)
                new_labels.append(f'{energy:.0f}\n({difficulty:.2f})')
            ax.set_xticks(xticks)
            ax.set_xticklabels(new_labels)
        except ImportError:
            pass

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved blocks by threshold chart to {output_file}")


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

    colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71'
    }

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

        color = colors.get(display_type, '#95a5a6')
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

    if num_nodes and num_edges:
        try:
            from shared.energy_utils import energy_to_difficulty
            ax = plt.gca()
            xticks = ax.get_xticks()
            new_labels = []
            for energy in xticks:
                difficulty = energy_to_difficulty(energy, num_nodes, num_edges)
                new_labels.append(f'{energy:.0f}\n({difficulty:.2f})')
            ax.set_xticks(xticks)
            ax.set_xticklabels(new_labels)
        except ImportError:
            pass

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved proportion by threshold chart to {output_file}")


def get_device_counts(df: pd.DataFrame) -> dict:
    """
    Count actual devices per miner type from the data.

    For GPU/CUDA: count unique (miner_machine, process) pairs = number of GPUs
    For CPU: count unique (miner_machine, process) pairs = number of CPU workers
    For QPU: count unique miner_machine = number of QPUs
    """
    counts = {}

    for miner_type in df['miner_type'].unique():
        display_type = normalize_miner_type(miner_type)
        miner_df = df[df['miner_type'] == miner_type]

        if miner_type == 'qpu':
            # QPU: count unique machines
            count = miner_df['miner_machine'].nunique()
        else:
            # CPU/GPU: count unique (machine, process) pairs
            count = miner_df.groupby(['miner_machine', 'process']).ngroups

        counts[display_type] = count

    return counts


def _calculate_win_rates(
    df: pd.DataFrame,
    thresholds: List[float],
    configurations: List[tuple],
    debug: bool = False,
    n_simulations: int = 10000
) -> tuple:
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
    # Step 1: Calculate per-miner statistics at observed threshold
    miner_stats = {}
    device_counts = get_device_counts(df)

    for miner_type in df['miner_type'].unique():
        display_type = normalize_miner_type(miner_type)
        miner_df = df[df['miner_type'] == miner_type]
        energies = miner_df['energy'].values

        # Observed TTS distribution from successful mining
        successful_df = miner_df[miner_df['valid'] > 0]
        tts_values = successful_df['time_to_solution'].values
        valid_tts = tts_values[tts_values > 0]

        if len(valid_tts) > 0:
            observed_tts_mean = np.mean(valid_tts)
            observed_tts_std = np.std(valid_tts)
            # Store actual TTS samples for Monte Carlo
            tts_samples = valid_tts
        else:
            observed_tts_mean = float('inf')
            observed_tts_std = 0
            tts_samples = np.array([])

        # Success rate at actual threshold (proxy for p_actual)
        actual_probability = len(successful_df) / len(miner_df) if len(miner_df) > 0 else 0

        miner_stats[display_type] = {
            'energies': energies,
            'num_attempts': len(energies),
            'observed_tts_mean': observed_tts_mean,
            'observed_tts_std': observed_tts_std,
            'tts_samples': tts_samples,
            'actual_probability': actual_probability,
            'device_count': device_counts.get(display_type, 1)
        }

    if debug:
        print("\n=== Win Rate Calculation Debug (Monte Carlo) ===")
        print("Miner statistics at observed threshold:")
        for name, stats in miner_stats.items():
            cv = stats['observed_tts_std'] / stats['observed_tts_mean'] if stats['observed_tts_mean'] > 0 else 0
            print(f"  {name}: TTS={stats['observed_tts_mean']:.1f}s ± {stats['observed_tts_std']:.1f}s "
                  f"(CV={cv:.2f}), p_actual={stats['actual_probability']:.4f}, "
                  f"devices={stats['device_count']}")

    # Step 2: Calculate win rates for each threshold via Monte Carlo
    win_rates = {config_name: [] for config_name, _, _ in configurations}

    for i, threshold in enumerate(thresholds):
        # Build scaled TTS parameters for each configuration
        scaled_params = {}

        for config_name, miner_type, count in configurations:
            # Handle QPURT (hypothetical real-time QPU without queue latency)
            if miner_type == 'QPURT':
                if 'QPU' not in miner_stats:
                    continue
                stats = miner_stats['QPU']
                if len(stats['energies']) == 0:
                    continue

                p_threshold = np.sum(stats['energies'] <= threshold) / len(stats['energies'])
                if p_threshold > 0 and stats['actual_probability'] > 0:
                    # QPURT: ~45ms per nonce, scale by probability
                    scale = stats['actual_probability'] / p_threshold
                    # Very low variance for QPURT (network latency dominated)
                    scaled_params[config_name] = {
                        'mean': 0.045 * scale,
                        'std': 0.01 * scale,  # Low variance
                        'use_samples': False
                    }
                continue

            if miner_type not in miner_stats:
                continue

            stats = miner_stats[miner_type]
            if stats['observed_tts_mean'] == float('inf') or stats['actual_probability'] == 0:
                continue
            if len(stats['energies']) == 0:
                continue

            # Probability at this threshold from energy distribution
            p_threshold = np.sum(stats['energies'] <= threshold) / len(stats['energies'])

            if p_threshold > 0:
                # Scale factor for TTS (both mean and std scale together)
                scale = stats['actual_probability'] / p_threshold

                # Scale for device count (for projections)
                actual_count = stats['device_count']
                if count != actual_count and count > 0:
                    scale *= actual_count / count

                scaled_params[config_name] = {
                    'mean': stats['observed_tts_mean'] * scale,
                    'std': stats['observed_tts_std'] * scale,
                    'samples': stats['tts_samples'] * scale if len(stats['tts_samples']) > 0 else None,
                    'use_samples': len(stats['tts_samples']) >= 100  # Use samples if we have enough
                }

        if not scaled_params:
            for config_name, _, _ in configurations:
                win_rates[config_name].append(0.0)
            continue

        # Monte Carlo simulation
        wins = {name: 0 for name in scaled_params.keys()}

        for _ in range(n_simulations):
            # Sample TTS for each miner
            sampled_tts = {}
            for name, params in scaled_params.items():
                if params.get('use_samples') and params.get('samples') is not None:
                    # Sample from actual distribution
                    sampled_tts[name] = np.random.choice(params['samples'])
                else:
                    # Sample from normal distribution (truncated at 0)
                    sample = np.random.normal(params['mean'], params['std'])
                    sampled_tts[name] = max(0.001, sample)  # Ensure positive

            # Determine winner (lowest TTS)
            winner = min(sampled_tts, key=sampled_tts.get)
            wins[winner] += 1

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
    include_qpurt: bool = False
):
    """
    Plot probability of winning the mining race at each threshold.

    Shows only actual device counts from the data (no projections).
    Labels show actual counts: "GPU (8 GPUs)", "CPU (32 CPUs)", "QPU (1)".
    """
    plt.figure(figsize=(12, 7))

    base_colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71',
        'QPURT': '#27ae60'
    }

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

    if include_qpurt and 'QPU' in device_counts:
        configurations.append(('QPURT', 'QPURT', device_counts['QPU']))

    # Calculate win rates with Monte Carlo simulation (uses observed variance)
    _, win_rates = _calculate_win_rates(df, thresholds, configurations, debug=False)

    # Plot each configuration
    for config_name, miner_type, count in configurations:
        win_rate_data = win_rates[config_name]

        if miner_type == 'QPURT':
            color = base_colors.get('QPURT', '#27ae60')
            style = {'linestyle': '--', 'marker': '*', 'linewidth': 2.5, 'alpha': 0.85}
        else:
            color = base_colors.get(miner_type, '#95a5a6')
            style = {'linestyle': '-', 'marker': 'o', 'linewidth': 2, 'alpha': 0.9}

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

    if num_nodes and num_edges:
        try:
            from shared.energy_utils import energy_to_difficulty
            ax = plt.gca()
            xticks = ax.get_xticks()
            new_labels = []
            for energy in xticks:
                difficulty = energy_to_difficulty(energy, num_nodes, num_edges)
                new_labels.append(f'{energy:.0f}\n({difficulty:.2f})')
            ax.set_xticks(xticks)
            ax.set_xticklabels(new_labels)
        except ImportError:
            pass

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
    include_qpurt: bool = False
):
    """
    Plot probability of winning the mining race with projected device counts.

    Shows projections like GPU (2^8), GPU (2^16), CPU (2^20), CPU (2^30).
    """
    plt.figure(figsize=(12, 7))

    base_colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71',
        'QPURT': '#27ae60'
    }

    # Get actual device counts for base comparison
    device_counts = get_device_counts(df)

    # Generate configurations with projections
    configurations = []

    if include_qpurt:
        # QPURT projection version
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
        if 'QPU' in device_counts:
            configurations.append(('QPURT', 'QPURT', device_counts['QPU']))
    else:
        # Standard projection version
        if 'QPU' in device_counts:
            configurations.append((f'QPU ({device_counts["QPU"]})', 'QPU', device_counts['QPU']))
        if 'CPU' in device_counts:
            configurations.extend([
                (f'CPU ({device_counts["CPU"]} CPUs)', 'CPU', device_counts['CPU']),
                ('CPU (2^20)', 'CPU', 1048576),
                ('CPU (2^30)', 'CPU', 1073741824)
            ])
        if 'GPU' in device_counts:
            configurations.extend([
                (f'GPU ({device_counts["GPU"]} GPUs)', 'GPU', device_counts['GPU']),
                ('GPU (2^8)', 'GPU', 256),
                ('GPU (2^16)', 'GPU', 65536)
            ])

    # Calculate win rates
    _, win_rates = _calculate_win_rates(df, thresholds, configurations)

    # Plot styles for projections
    style_map = {
        256: {'linestyle': '--', 'marker': 's', 'linewidth': 2.5, 'alpha': 0.7},
        65536: {'linestyle': ':', 'marker': '^', 'linewidth': 2.5, 'alpha': 0.7},
        1048576: {'linestyle': '--', 'marker': 'D', 'linewidth': 2.5, 'alpha': 0.7},
        1073741824: {'linestyle': ':', 'marker': 'v', 'linewidth': 2.5, 'alpha': 0.7},
        1099511627776: {'linestyle': ':', 'marker': 'X', 'linewidth': 2.5, 'alpha': 0.7}
    }

    for config_name, miner_type, count in configurations:
        win_rate_data = win_rates[config_name]

        if miner_type == 'QPURT':
            color = base_colors.get('QPURT', '#27ae60')
            style = {'linestyle': '--', 'marker': '*', 'linewidth': 2.5, 'alpha': 0.85}
        else:
            color = base_colors.get(miner_type, '#95a5a6')
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

    if num_nodes and num_edges:
        try:
            from shared.energy_utils import energy_to_difficulty
            ax = plt.gca()
            xticks = ax.get_xticks()
            new_labels = []
            for energy in xticks:
                difficulty = energy_to_difficulty(energy, num_nodes, num_edges)
                new_labels.append(f'{energy:.0f}\n({difficulty:.2f})')
            ax.set_xticks(xticks)
            ax.set_xticklabels(new_labels)
        except ImportError:
            pass

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
    try:
        from shared.energy_utils import energy_to_difficulty
        has_difficulty_fn = True
    except ImportError:
        has_difficulty_fn = False

    plt.figure(figsize=(12, 7))

    colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71'
    }

    for miner_type in df['miner_type'].unique():
        display_type = normalize_miner_type(miner_type)
        miner_df = df[df['miner_type'] == miner_type]

        energies = miner_df['energy'].values
        tts_values = miner_df['time_to_solution'].values
        valid_tts = tts_values[tts_values > 0]

        if len(valid_tts) == 0:
            continue

        time_per_attempt = np.mean(valid_tts)

        expected_times = []
        for threshold in thresholds:
            meets_threshold = np.sum(energies <= threshold)
            probability = meets_threshold / len(energies)

            if probability > 0:
                expected_time = time_per_attempt / probability
                expected_times.append(expected_time)
            else:
                expected_times.append(float('inf'))

        valid_data = [(t, et) for t, et in zip(thresholds, expected_times) if et != float('inf')]
        if not valid_data:
            continue

        valid_thresholds, valid_times = zip(*valid_data)

        color = colors.get(display_type, '#95a5a6')
        plt.plot(
            valid_thresholds,
            valid_times,
            marker='o',
            linewidth=2,
            markersize=6,
            label=f'{display_type} ({time_per_attempt:.2f}s/attempt)',
            color=color
        )

    plt.gca().invert_xaxis()

    if has_difficulty_fn and num_nodes and num_edges:
        ax = plt.gca()
        xticks = ax.get_xticks()
        new_labels = []
        for energy in xticks:
            difficulty = energy_to_difficulty(energy, num_nodes, num_edges)
            new_labels.append(f'{energy:.0f}\n({difficulty:.2f})')
        ax.set_xticks(xticks)
        ax.set_xticklabels(new_labels)

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
    """Plot histogram of nonces per block for each miner type."""
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71'
    }

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

    stats_data = []
    for miner_type in miner_types:
        nonces = nonces_data[miner_type]
        if nonces:
            stats_data.append({
                'mean': np.mean(nonces),
                'median': np.median(nonces),
                'min': np.min(nonces),
                'max': np.max(nonces),
                'std': np.std(nonces)
            })
        else:
            stats_data.append({
                'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0
            })

    means = [s['mean'] for s in stats_data]
    bars = ax.bar(
        positions,
        means,
        color=[colors.get(mt, '#95a5a6') for mt in miner_types],
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5
    )

    stds = [s['std'] for s in stats_data]
    ax.errorbar(
        positions,
        means,
        yerr=stds,
        fmt='none',
        ecolor='black',
        capsize=5,
        capthick=2
    )

    for i, (bar, stat) in enumerate(zip(bars, stats_data)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{stat['mean']:.1f}\n({stat['std']:.1f})",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_xlabel('Miner Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Nonces per Block', fontsize=12, fontweight='bold')
    ax.set_title(f'Nonces Required to Mine a Block\n{topology_info}', fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
    labels_with_counts = [f'{mt}\n(n={block_counts.get(mt, 0):,} blocks)' for mt in miner_types]
    ax.set_xticklabels(labels_with_counts, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
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
        default='Advantage2_system1.8',
        help='Topology name for chart titles (default: Advantage2_system1.8)'
    )

    args = parser.parse_args()

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
    except Exception as e:
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

    # Chart 2: Blocks by threshold
    plot_blocks_by_threshold(
        df, thresholds, topology_desc, num_nodes, num_edges,
        str(args.output_dir / 'blocks_by_threshold.png')
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
        include_qpurt=False
    )

    # Chart 4b: Win rate with QPURT (actual device counts only)
    plot_win_rate_by_threshold(
        df, thresholds, topology_desc, num_nodes, num_edges,
        str(args.output_dir / 'win_rate_by_threshold_qpurt.png'),
        include_qpurt=True
    )

    # Chart 4c: Win rate projection (with 2^X device counts)
    plot_win_rate_by_threshold_projection(
        df, thresholds, topology_desc, num_nodes, num_edges,
        str(args.output_dir / 'win_rate_by_threshold_projection.png'),
        include_qpurt=False
    )

    # Chart 4d: Win rate projection with QPURT
    plot_win_rate_by_threshold_projection(
        df, thresholds, topology_desc, num_nodes, num_edges,
        str(args.output_dir / 'win_rate_by_threshold_projection_qpurt.png'),
        include_qpurt=True
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
