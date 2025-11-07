#!/usr/bin/env python3
"""
Visualize mining performance across different miner types (CPU/GPU/QPU).

Generates six charts:
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
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_miner_type(filename: str) -> str:
    """Extract miner type from filename using startswith."""
    name = Path(filename).stem.lower()

    # Check prefix of filename
    if name.startswith('metal'):
        return 'GPU'
    elif name.startswith('qpu'):
        return 'QPU'
    elif name.startswith('cpu'):
        return 'CPU'
    elif name.startswith('cuda'):
        return 'GPU'
    else:
        # Fallback: try first part before underscore
        parts = name.split('_')
        miner_type = parts[0].upper()
        return miner_type


def parse_log_file(filepath: str) -> Tuple[List[float], List[int], float]:
    """
    Parse log file to extract mining attempt energies, nonces per block, and timing.

    Returns:
        Tuple of (all_attempt_energies, nonces_per_block, total_time_seconds)
    """
    all_energies = []
    nonces_per_block = []
    current_block_attempts = 0

    # Try to extract total time from log
    total_time = 0.0
    first_timestamp = None
    last_timestamp = None

    with open(filepath, 'rb') as f:
        for line in f:
            try:
                line_str = line.decode('utf-8', errors='ignore')
            except:
                continue

            # Try to extract timestamps (format varies, look for common patterns)
            # Example: "2025-01-07 12:34:56" or "[12:34:56]" or "Elapsed: 123.45s"
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line_str)
            if timestamp_match and not first_timestamp:
                first_timestamp = timestamp_match.group(1)
            elif timestamp_match:
                last_timestamp = timestamp_match.group(1)

            # Look for explicit duration/elapsed time
            # Format 1: "Duration: 30.0 minutes"
            duration_min_match = re.search(r'Duration:\s*(\d+(?:\.\d+)?)\s*minutes?', line_str, re.IGNORECASE)
            if duration_min_match:
                total_time = float(duration_min_match.group(1)) * 60

            # Format 2: "Duration: 123.45s" or "Elapsed: 123s"
            duration_sec_match = re.search(r'(?:Duration|Elapsed|Total time):\s*(\d+(?:\.\d+)?)\s*s', line_str, re.IGNORECASE)
            if duration_sec_match:
                total_time = float(duration_sec_match.group(1))

            # Match mining attempt lines: "Mining attempt - Energy: -3504.00, Valid: ..."
            attempt_match = re.search(r'Mining attempt - Energy: (-?\d+(?:\.\d+)?)', line_str)
            if attempt_match:
                energy = float(attempt_match.group(1))
                all_energies.append(energy)
                current_block_attempts += 1

            # Match mined block lines: "[Block-X] Mined! Nonce: ..."
            block_match = re.search(r'\[Block-\d+\] Mined!', line_str)
            if block_match:
                if current_block_attempts > 0:
                    nonces_per_block.append(current_block_attempts)
                    current_block_attempts = 0

    return all_energies, nonces_per_block, total_time


def calculate_threshold_probabilities(
    energies: List[float],
    thresholds: List[float]
) -> List[float]:
    """
    Calculate probability that a random nonce meets each threshold.

    Args:
        energies: List of all mining attempt energies
        thresholds: List of difficulty thresholds to check

    Returns:
        List of probabilities (one per threshold)
    """
    if not energies:
        return [0.0] * len(thresholds)

    total_attempts = len(energies)
    probabilities = []

    for threshold in thresholds:
        # Count how many attempts meet or exceed threshold (more negative = better)
        meets_threshold = sum(1 for e in energies if e <= threshold)
        probability = meets_threshold / total_attempts
        probabilities.append(probability)

    return probabilities


def plot_threshold_probabilities(
    data: Dict[str, Tuple[List[float], List[float]]],
    topology_info: str,
    attempt_counts: Dict[str, int],
    num_nodes: int = None,
    num_edges: int = None,
    output_file: str = 'threshold_probabilities.png'
):
    """
    Plot probability of nonce meeting threshold for each miner type.

    Args:
        data: Dict mapping miner_type -> (thresholds, probabilities)
        topology_info: Topology information string (e.g., "Zephyr Z(9,2)")
        attempt_counts: Dict mapping miner_type -> total number of attempts
        num_nodes: Number of nodes in topology (for difficulty annotations)
        num_edges: Number of edges in topology (for difficulty annotations)
        output_file: Output filename for the plot
    """
    plt.figure(figsize=(12, 7))

    # Color palette
    colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71'
    }

    # Debug: check if we have any data
    if not data:
        print("⚠️  WARNING: No data to plot for threshold probabilities!")
        return

    for miner_type, (thresholds, probabilities) in sorted(data.items()):
        if not thresholds or not probabilities:
            print(f"⚠️  WARNING: {miner_type} has empty thresholds or probabilities")
            continue
        color = colors.get(miner_type, '#95a5a6')
        n_attempts = attempt_counts.get(miner_type, 0)

        plt.plot(
            thresholds,
            probabilities,
            marker='o',
            linewidth=2,
            markersize=8,
            label=f'{miner_type} (n={n_attempts})',
            color=color
        )

    plt.xlabel('Difficulty Threshold (Energy)', fontsize=12, fontweight='bold')
    plt.ylabel('Probability of Meeting Threshold', fontsize=12, fontweight='bold')
    plt.title(f'Probability of Nonce Meeting Difficulty Threshold\n{topology_info}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Format x-axis to show thresholds
    plt.gca().invert_xaxis()  # More negative = harder, so invert for intuitive reading

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
    print(f"✅ Saved threshold probability chart to {output_file}")


def plot_blocks_by_threshold(
    miner_energies: Dict[str, List[float]],
    thresholds: List[float],
    topology_info: str,
    num_nodes: int = None,
    num_edges: int = None,
    output_file: str = 'blocks_by_threshold.png'
):
    """
    Plot number of blocks that would be mined at each threshold.

    Args:
        miner_energies: Dict mapping miner_type -> list of all energy attempts
        thresholds: List of difficulty thresholds
        topology_info: Topology information string
        num_nodes: Number of nodes in topology (for difficulty annotations)
        num_edges: Number of edges in topology (for difficulty annotations)
        output_file: Output filename for the plot
    """
    plt.figure(figsize=(12, 7))

    # Color palette
    colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71'
    }

    for miner_type, energies in sorted(miner_energies.items()):
        if not energies:
            continue

        # Count blocks at each threshold
        blocks_at_threshold = []
        for threshold in thresholds:
            # Count attempts that meet threshold (energy <= threshold since more negative is better)
            num_blocks = sum(1 for e in energies if e <= threshold)
            blocks_at_threshold.append(num_blocks)

        color = colors.get(miner_type, '#95a5a6')
        plt.plot(
            thresholds,
            blocks_at_threshold,
            marker='o',
            linewidth=2,
            markersize=6,
            label=f'{miner_type} (total={len(energies)})',
            color=color
        )

    plt.xlabel('Difficulty Threshold (Energy) →', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Blocks Mined', fontsize=12, fontweight='bold')
    plt.title(f'Blocks Mined by Difficulty Threshold\n{topology_info}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Don't invert x-axis - let it go from more negative (left) to less negative (right)
    # This means moving right makes mining easier

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
    print(f"✅ Saved blocks by threshold chart to {output_file}")


def plot_proportion_by_threshold(
    miner_energies: Dict[str, List[float]],
    thresholds: List[float],
    topology_info: str,
    num_nodes: int = None,
    num_edges: int = None,
    output_file: str = 'proportion_by_threshold.png'
):
    """
    Plot percentage of nonces that would mine at each threshold (normalized to proportion).

    Args:
        miner_energies: Dict mapping miner_type -> list of all energy attempts
        thresholds: List of difficulty thresholds
        topology_info: Topology information string
        num_nodes: Number of nodes in topology (for difficulty annotations)
        num_edges: Number of edges in topology (for difficulty annotations)
        output_file: Output filename for the plot
    """
    plt.figure(figsize=(12, 7))

    # Color palette
    colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71'
    }

    for miner_type, energies in sorted(miner_energies.items()):
        if not energies:
            continue

        total_attempts = len(energies)
        # Calculate proportion at each threshold
        proportions = []
        for threshold in thresholds:
            # Count attempts that meet threshold (energy <= threshold since more negative is better)
            num_blocks = sum(1 for e in energies if e <= threshold)
            proportion = (num_blocks / total_attempts) * 100  # Convert to percentage
            proportions.append(proportion)

        color = colors.get(miner_type, '#95a5a6')
        plt.plot(
            thresholds,
            proportions,
            marker='o',
            linewidth=2,
            markersize=6,
            label=f'{miner_type} (n={total_attempts})',
            color=color
        )

    plt.xlabel('Difficulty Threshold (Energy) →', fontsize=12, fontweight='bold')
    plt.ylabel('% of Nonces Meeting Threshold', fontsize=12, fontweight='bold')
    plt.title(f'Proportion of Nonces Meeting Difficulty Threshold\n{topology_info}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

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
    print(f"✅ Saved proportion by threshold chart to {output_file}")


def plot_win_rate_by_threshold(
    miner_data: Dict[str, Tuple[List[float], float, int]],
    thresholds: List[float],
    topology_info: str,
    num_nodes: int = None,
    num_edges: int = None,
    output_file: str = 'win_rate_by_threshold.png',
    include_qpurt: bool = False
):
    """
    Plot probability of winning the mining race at each threshold for each miner type.

    Uses analytical calculation based on observed data:
    - For N parallel miners with probability p and time t per attempt:
      * Combined probability: p_combined = 1 - (1-p)^N
      * Expected time to success: E[T] = t / p_combined
    - Win probability modeled as exponential racing: P(A wins) = rate_A / sum(rates)

    Args:
        miner_data: Dict mapping miner_type -> (energies, total_time_seconds, num_attempts)
        thresholds: List of difficulty thresholds
        topology_info: Topology information string
        num_nodes: Number of nodes in topology (for difficulty annotations)
        num_edges: Number of edges in topology (for difficulty annotations)
        output_file: Output filename for the plot
        include_qpurt: If True, includes QPURT and uses extreme counts (GPU 2^20, CPU 2^40)
    """

    plt.figure(figsize=(12, 7))

    # Color palette
    base_colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71',
        'QPURT': '#27ae60'  # Darker green for QPU real-time (no network overhead)
    }

    # Generate configurations to test
    configurations = []

    if include_qpurt:
        # QPURT version: extreme parallelization vs QPU real-time
        # CPU: 1, 2^40 (1,099,511,627,776) - extreme parallelization
        # GPU: 1, 2^20 (1,048,576) - extreme parallelization
        # QPU: 1 - with network overhead
        # QPURT: 1 - QPU real-time (0.045s per nonce, no network overhead)
        for miner_type in miner_data.keys():
            if miner_type == 'QPU':
                counts_and_labels = [(1, 'QPU')]
            elif miner_type == 'CPU':
                counts_and_labels = [
                    (1, 'CPU'),
                    (1099511627776, 'CPU (2^40)')
                ]
            else:  # GPU
                counts_and_labels = [
                    (1, 'GPU'),
                    (1048576, 'GPU (2^20)')
                ]

            for count, label in counts_and_labels:
                configurations.append((label, miner_type, count))

        # Add QPURT (QPU Real-Time)
        if 'QPU' in miner_data:
            configurations.append(('QPURT', 'QPURT', 1))
    else:
        # Standard version: no QPURT
        # CPU: 1, 2^20 (1,048,576), 2^30 (1,073,741,824)
        # GPU: 1, 2^8 (256), 2^16 (65,536)
        # QPU: 1
        for miner_type in miner_data.keys():
            if miner_type == 'QPU':
                counts_and_labels = [(1, 'QPU')]
            elif miner_type == 'CPU':
                counts_and_labels = [
                    (1, 'CPU'),
                    (1048576, 'CPU (2^20)'),
                    (1073741824, 'CPU (2^30)')
                ]
            else:  # GPU
                counts_and_labels = [
                    (1, 'GPU'),
                    (256, 'GPU (2^8)'),
                    (65536, 'GPU (2^16)')
                ]

            for count, label in counts_and_labels:
                configurations.append((label, miner_type, count))

    # Calculate win rates for each threshold
    win_rates = {config_name: [] for config_name, _, _ in configurations}

    for threshold in thresholds:
        # Calculate success probability and time per attempt for each base miner type
        miner_params = {}
        for miner_type, (energies, total_time, num_attempts) in miner_data.items():
            if not energies or total_time <= 0 or num_attempts == 0:
                continue

            # Probability of success at this threshold
            meets_threshold = sum(1 for e in energies if e <= threshold)
            probability = meets_threshold / len(energies)

            # Time per attempt
            time_per_attempt = total_time / num_attempts

            if probability > 0:
                miner_params[miner_type] = {
                    'probability': probability,
                    'time_per_attempt': time_per_attempt
                }

        if len(miner_params) == 0:
            # No miners can mine at this threshold
            for config_name, _, _ in configurations:
                win_rates[config_name].append(0)
            continue

        # Calculate expected time to first success for each configuration
        # For N parallel miners with individual success probability p and time t per attempt:
        # Combined probability: p_combined = 1 - (1-p)^N
        # Expected time: E[T] = t / p_combined (expected attempts = 1/p_combined, time = attempts * t)
        expected_times = {}

        for config_name, miner_type, count in configurations:
            # Special handling for QPURT (synthetic QPU real-time)
            if miner_type == 'QPURT':
                # Use QPU's probability but with 0.045s per attempt (no network overhead)
                if 'QPU' not in miner_params:
                    continue
                params = miner_params['QPU']
                p = params['probability']
                t = 0.045  # QPU real-time: 0.045s per nonce
            elif miner_type not in miner_params:
                # This miner type can't mine at this threshold
                continue
            else:
                params = miner_params[miner_type]
                p = params['probability']
                t = params['time_per_attempt']

            # Combined probability for N parallel miners
            p_combined = 1 - (1 - p) ** count

            if p_combined > 0:
                # Expected time = time_per_attempt / combined_probability
                expected_time = t / p_combined
                expected_times[config_name] = expected_time

        if not expected_times:
            for config_name, _, _ in configurations:
                win_rates[config_name].append(0)
            continue

        # Calculate win rates based on expected times
        # The configuration with shortest expected time has highest win probability
        # Use exponential distribution for racing: P(A wins) = rate_A / sum(all_rates)
        # where rate = 1 / expected_time
        rates = {name: 1.0 / time for name, time in expected_times.items()}
        total_rate = sum(rates.values())

        for config_name, miner_type, count in configurations:
            if config_name in rates:
                win_rate = rates[config_name] / total_rate
                win_rates[config_name].append(win_rate)
            else:
                win_rates[config_name].append(0.0)

    # Plot win rates with different line styles and markers for different counts
    # Map counts to visual styles
    style_map = {
        1: {'linestyle': '-', 'marker': 'o', 'linewidth': 2, 'alpha': 0.9},
        256: {'linestyle': '--', 'marker': 's', 'linewidth': 2.5, 'alpha': 0.7},
        65536: {'linestyle': ':', 'marker': '^', 'linewidth': 2.5, 'alpha': 0.7},
        1048576: {'linestyle': '--', 'marker': 'D', 'linewidth': 2.5, 'alpha': 0.7},
        1073741824: {'linestyle': ':', 'marker': 'v', 'linewidth': 2.5, 'alpha': 0.7},
        1099511627776: {'linestyle': ':', 'marker': 'X', 'linewidth': 2.5, 'alpha': 0.7}  # 2^40
    }

    for config_name, miner_type, count in configurations:
        win_rate_data = win_rates[config_name]

        # Special styling for QPURT
        if miner_type == 'QPURT':
            color = base_colors.get('QPURT', '#27ae60')
            style = {'linestyle': '--', 'marker': '*', 'linewidth': 2.5, 'alpha': 0.85}
        else:
            color = base_colors.get(miner_type, '#95a5a6')
            style = style_map.get(count, style_map[1])

        plt.plot(
            thresholds,
            [wr * 100 for wr in win_rate_data],  # Convert to percentage
            marker=style['marker'],
            linewidth=style['linewidth'],
            markersize=6,
            label=config_name,
            color=color,
            linestyle=style['linestyle'],
            alpha=style['alpha']
        )

    # Reverse x-axis so difficulty increases left to right
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

    plt.xlabel('← Easier (Less Negative Energy)     Harder (More Negative Energy) →', fontsize=12, fontweight='bold')
    plt.ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    plt.title(f'Probability of Winning Mining Race by Difficulty\n{topology_info}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved win rate by threshold chart to {output_file}")


def plot_expected_time_by_threshold(
    miner_data: Dict[str, Tuple[List[float], float, int]],
    thresholds: List[float],
    topology_info: str,
    num_nodes: int = None,
    num_edges: int = None,
    output_file: str = 'expected_time_by_threshold.png'
):
    """
    Plot expected time to mine a block at each threshold for each miner type.

    Args:
        miner_data: Dict mapping miner_type -> (energies, total_time_seconds, num_attempts)
        thresholds: List of difficulty thresholds
        topology_info: Topology information string
        num_nodes: Number of nodes in topology (for difficulty calculation)
        num_edges: Number of edges in topology (for difficulty calculation)
        output_file: Output filename for the plot
    """
    # Import energy_to_difficulty for annotations
    try:
        from shared.energy_utils import energy_to_difficulty
        has_difficulty_fn = True
    except ImportError:
        has_difficulty_fn = False
        print("⚠️  Could not import energy_to_difficulty, skipping difficulty annotations")

    plt.figure(figsize=(12, 7))

    # Color palette
    colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71'
    }

    for miner_type, (energies, total_time, num_attempts) in sorted(miner_data.items()):
        if not energies or total_time <= 0 or num_attempts == 0:
            print(f"⚠️  Skipping {miner_type}: insufficient timing data (time={total_time}s, attempts={num_attempts})")
            continue

        # Calculate average time per attempt
        time_per_attempt = total_time / num_attempts

        # Calculate expected time to mine at each threshold
        expected_times = []
        for threshold in thresholds:
            # Probability of success at this threshold
            meets_threshold = sum(1 for e in energies if e <= threshold)
            probability = meets_threshold / len(energies)

            if probability > 0:
                # Expected time = time_per_attempt / probability
                expected_time = time_per_attempt / probability
                expected_times.append(expected_time)
            else:
                expected_times.append(float('inf'))  # No chance of success

        # Filter out infinite values for plotting
        valid_data = [(t, et) for t, et in zip(thresholds, expected_times) if et != float('inf')]
        if not valid_data:
            continue

        valid_thresholds, valid_times = zip(*valid_data)

        color = colors.get(miner_type, '#95a5a6')
        plt.plot(
            valid_thresholds,
            valid_times,
            marker='o',
            linewidth=2,
            markersize=6,
            label=f'{miner_type} ({time_per_attempt:.2f}s/attempt)',
            color=color
        )

    # Reverse x-axis so difficulty increases left to right (more negative = harder = left)
    plt.gca().invert_xaxis()

    # Add difficulty rating annotations on x-axis ticks
    if has_difficulty_fn and num_nodes and num_edges:
        from shared.energy_utils import energy_to_difficulty

        ax = plt.gca()

        # Get current x-tick locations
        xticks = ax.get_xticks()

        # Create new labels with difficulty ratings
        new_labels = []
        for energy in xticks:
            # Calculate difficulty for this energy
            difficulty = energy_to_difficulty(energy, num_nodes, num_edges)

            # Format: energy (difficulty as 0.0-1.0)
            new_labels.append(f'{energy:.0f}\n({difficulty:.2f})')

        # Set both ticks and labels to avoid warning
        ax.set_xticks(xticks)
        ax.set_xticklabels(new_labels)

    plt.xlabel('← Easier (Less Negative Energy)     Harder (More Negative Energy) →', fontsize=12, fontweight='bold')
    plt.ylabel('Expected Time to Mine Block (seconds)', fontsize=12, fontweight='bold')
    plt.title(f'Expected Mining Time by Difficulty\n{topology_info}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale since times can vary by orders of magnitude

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved expected time by threshold chart to {output_file}")


def plot_nonces_per_block(
    data: Dict[str, List[int]],
    topology_info: str,
    block_counts: Dict[str, int],
    output_file: str = 'nonces_per_block.png'
):
    """
    Plot histogram of nonces per block for each miner type.

    Args:
        data: Dict mapping miner_type -> list of nonces_per_block
        topology_info: Topology information string (e.g., "Zephyr Z(9,2)")
        block_counts: Dict mapping miner_type -> total number of blocks
        output_file: Output filename for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color palette
    colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71'
    }

    # Prepare data for grouped histogram
    miner_types = sorted(data.keys())
    positions = np.arange(len(miner_types))

    # Calculate statistics for each miner type
    stats_data = []
    for miner_type in miner_types:
        nonces = data[miner_type]
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
                'mean': 0,
                'median': 0,
                'min': 0,
                'max': 0,
                'std': 0
            })

    # Plot bars for mean values
    means = [s['mean'] for s in stats_data]
    bars = ax.bar(
        positions,
        means,
        color=[colors.get(mt, '#95a5a6') for mt in miner_types],
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5
    )

    # Add error bars for standard deviation
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

    # Add value labels on bars
    for i, (bar, stat) in enumerate(zip(bars, stats_data)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{stat['mean']:.1f}\n(±{stat['std']:.1f})",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_xlabel('Miner Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Nonces per Block', fontsize=12, fontweight='bold')
    ax.set_title(f'Nonces Required to Mine a Block\n{topology_info}', fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
    # Add block counts to x-axis labels
    labels_with_counts = [f'{mt}\n(n={block_counts.get(mt, 0)} blocks)' for mt in miner_types]
    ax.set_xticklabels(labels_with_counts, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved nonces per block chart to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize mining performance across CPU/GPU/QPU',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_mining_performance.py cpu_10min_at_cpu3min.log metal_10min_at_cpu3min.log qpu_*.log
  python visualize_mining_performance.py --output-dir charts/ *.log
        """
    )
    parser.add_argument(
        'logfiles',
        nargs='+',
        help='Log files to analyze (miner type extracted from filename)'
    )
    parser.add_argument(
        '--output-dir',
        default='.',
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
        default='Z(9,2)',
        help='Topology to load (e.g., "Z(9,2)", "Advantage2_system1.7", or path to .json/.json.gz file)'
    )

    args = parser.parse_args()

    # Load topology to get node/edge counts and description
    try:
        from dwave_topologies.topologies.json_loader import load_topology
        topology = load_topology(args.topology)

        # Extract topology information
        num_nodes = len(topology.graph.nodes) if hasattr(topology.graph, 'nodes') else topology.num_nodes
        num_edges = len(topology.graph.edges) if hasattr(topology.graph, 'edges') else topology.num_edges

        # Build topology description for chart title
        if hasattr(topology, 'topology_type'):
            # JSON topology with metadata
            topology_desc = f"{topology.solver_name} - {num_nodes:,} nodes, {num_edges:,} edges"
        elif hasattr(topology, 'm') and hasattr(topology, 't'):
            # Zephyr topology
            topology_desc = f"Zephyr Z({topology.m},{topology.t}) - {num_nodes:,} nodes, {num_edges:,} edges"
        else:
            # Generic
            topology_desc = f"{args.topology} - {num_nodes:,} nodes, {num_edges:,} edges"

        print(f"📍 Loaded topology: {topology_desc}")

    except Exception as e:
        print(f"⚠️  Could not load topology '{args.topology}': {e}")
        print(f"   Continuing without difficulty annotations")
        num_nodes = None
        num_edges = None
        topology_desc = args.topology

    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate threshold range
    # Auto-detect direction and adjust step if needed
    if args.threshold_min > args.threshold_max:
        # Going from less negative to more negative (e.g., -14700 to -15000)
        if args.threshold_step > 0:
            step = -abs(args.threshold_step)
        else:
            step = args.threshold_step
        thresholds = np.arange(args.threshold_min, args.threshold_max - abs(step), step)
    else:
        # Going from more negative to less negative (e.g., -15000 to -14700)
        if args.threshold_step < 0:
            step = abs(args.threshold_step)
        else:
            step = args.threshold_step
        thresholds = np.arange(args.threshold_min, args.threshold_max + step, step)

    thresholds = thresholds.tolist()

    if not thresholds:
        print(f"❌ Error: No thresholds generated with min={args.threshold_min}, max={args.threshold_max}, step={args.threshold_step}")
        print(f"   Hint: Make sure min < max when using positive step, or min > max when using negative step")
        return 1

    print(f"📊 Analyzing {len(args.logfiles)} log files...")
    print(f"   Threshold range: {thresholds[0]} to {thresholds[-1]} (step: {thresholds[1] - thresholds[0] if len(thresholds) > 1 else 0}, {len(thresholds)} points)")

    # Parse all log files and group by miner type
    miner_energies = defaultdict(list)  # miner_type -> list of all energies
    miner_nonces = defaultdict(list)    # miner_type -> list of nonces per block
    miner_times = {}                     # miner_type -> (total_time, num_attempts)

    for logfile in args.logfiles:
        if not Path(logfile).exists():
            print(f"⚠️  Warning: {logfile} not found, skipping")
            continue

        miner_type = parse_miner_type(logfile)
        print(f"   Parsing {logfile} -> {miner_type}")

        energies, nonces, total_time = parse_log_file(logfile)
        miner_energies[miner_type].extend(energies)
        miner_nonces[miner_type].extend(nonces)

        # Accumulate timing data
        if miner_type not in miner_times:
            miner_times[miner_type] = (0.0, 0)
        old_time, old_attempts = miner_times[miner_type]
        miner_times[miner_type] = (old_time + total_time, old_attempts + len(energies))

        print(f"      Found {len(energies)} mining attempts, {len(nonces)} blocks, {total_time:.1f}s")

    if not miner_energies:
        print("❌ No data found in log files")
        return 1

    print(f"\n📈 Generating charts...")

    # Chart 1: Threshold probabilities
    threshold_data = {}
    for miner_type, energies in miner_energies.items():
        probabilities = calculate_threshold_probabilities(energies, thresholds)
        threshold_data[miner_type] = (thresholds, probabilities)
        print(f"   {miner_type}: {len(energies)} attempts analyzed")
        if thresholds and probabilities:
            print(f"      Threshold range: {thresholds[0]:.0f} to {thresholds[-1]:.0f}")
            print(f"      Probability range: {min(probabilities):.3f} to {max(probabilities):.3f}")
        else:
            print(f"      ⚠️  No threshold data generated")

    # Collect attempt and block counts for legends
    attempt_counts = {miner_type: len(energies) for miner_type, energies in miner_energies.items()}
    block_counts = {miner_type: len(nonces) for miner_type, nonces in miner_nonces.items()}

    threshold_output = output_dir / 'threshold_probabilities.png'
    plot_threshold_probabilities(threshold_data, topology_desc, attempt_counts, num_nodes, num_edges, str(threshold_output))

    # Chart 2: Blocks by threshold
    blocks_threshold_output = output_dir / 'blocks_by_threshold.png'
    plot_blocks_by_threshold(dict(miner_energies), thresholds, topology_desc, num_nodes, num_edges, str(blocks_threshold_output))

    # Chart 3: Proportion by threshold
    proportion_output = output_dir / 'proportion_by_threshold.png'
    plot_proportion_by_threshold(dict(miner_energies), thresholds, topology_desc, num_nodes, num_edges, str(proportion_output))

    # Build miner_data dict: miner_type -> (energies, total_time, num_attempts)
    miner_timing_data = {}
    for miner_type in miner_energies.keys():
        total_time, num_attempts = miner_times.get(miner_type, (0.0, 0))
        miner_timing_data[miner_type] = (miner_energies[miner_type], total_time, num_attempts)

    # Chart 4a: Win rate by threshold (standard version, no QPURT)
    win_rate_output = output_dir / 'win_rate_by_threshold.png'
    plot_win_rate_by_threshold(
        miner_timing_data,
        thresholds,
        topology_desc,
        num_nodes=num_nodes,
        num_edges=num_edges,
        output_file=str(win_rate_output),
        include_qpurt=False
    )

    # Chart 4b: Win rate by threshold (QPURT version with extreme counts)
    win_rate_qpurt_output = output_dir / 'win_rate_by_threshold_qpurt.png'
    plot_win_rate_by_threshold(
        miner_timing_data,
        thresholds,
        topology_desc,
        num_nodes=num_nodes,
        num_edges=num_edges,
        output_file=str(win_rate_qpurt_output),
        include_qpurt=True
    )

    # Chart 5: Expected time by threshold
    expected_time_output = output_dir / 'expected_time_by_threshold.png'
    plot_expected_time_by_threshold(
        miner_timing_data,
        thresholds,
        topology_desc,
        num_nodes=num_nodes,
        num_edges=num_edges,
        output_file=str(expected_time_output)
    )

    # Chart 6: Nonces per block
    nonces_output = output_dir / 'nonces_per_block.png'
    plot_nonces_per_block(dict(miner_nonces), topology_desc, block_counts, str(nonces_output))

    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    for miner_type in sorted(miner_energies.keys()):
        energies = miner_energies[miner_type]
        nonces = miner_nonces[miner_type]
        print(f"\n{miner_type}:")
        print(f"  Total mining attempts: {len(energies)}")
        print(f"  Total blocks mined: {len(nonces)}")
        if energies:
            print(f"  Energy range: {min(energies):.0f} to {max(energies):.0f}")
            print(f"  Mean energy: {np.mean(energies):.1f}")
        if nonces:
            print(f"  Avg nonces per block: {np.mean(nonces):.1f} (±{np.std(nonces):.1f})")
            print(f"  Min/Max nonces: {min(nonces)} / {max(nonces)}")

    print(f"\n✅ Done! Charts saved to {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
