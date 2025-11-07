#!/usr/bin/env python3
"""Visualize QPU test results from test_qpu.py output.

Creates heatmap visualizations showing how num_reads and annealing_time
affect solution quality (minimum energy) across different intervals and seeds.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100


def load_results(filepath: str) -> Dict:
    """Load QPU test results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_heatmaps_per_seed(data: Dict, output_dir: str = '.'):
    """Create heatmap visualizations, one figure per seed.

    Each figure contains multiple subplots (one per interval) showing:
    - X-axis: annealing_time
    - Y-axis: num_reads
    - Color: energy_min

    Args:
        data: Loaded JSON data from test_qpu.py
        output_dir: Directory to save output figures
    """
    seeds = data['seeds']
    num_reads_list = data['num_reads_tested']
    annealing_time_list = data['annealing_time_tested']
    interval_list = data['interval_tested']
    cpu_baseline = data['cpu_baseline']['results']
    qpu_results = data['qpu_results']

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create one figure per seed
    for seed in seeds:
        n_intervals = len(interval_list)

        # Calculate grid layout (try to make roughly square)
        n_cols = min(3, n_intervals)
        n_rows = (n_intervals + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 5 * n_rows),
            squeeze=False
        )

        fig.suptitle(
            f'QPU Performance - Seed {seed} - Topology {data["topology"]}\n'
            f'CPU Baseline: {cpu_baseline[str(seed)]["energy_min"]:.1f}',
            fontsize=16,
            fontweight='bold'
        )

        for idx, interval in enumerate(interval_list):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Get results for this seed+interval
            key = f"seed_{seed}_interval_{interval}"
            if key not in qpu_results:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'Interval: {interval}s')
                continue

            results = qpu_results[key]['results']

            # Build energy matrix: rows=num_reads, cols=annealing_time
            energy_matrix = np.full((len(num_reads_list), len(annealing_time_list)), np.nan)

            for result in results:
                nr_idx = num_reads_list.index(result['num_reads'])
                at_idx = annealing_time_list.index(result['annealing_time'])
                energy_matrix[nr_idx, at_idx] = result['energy_min']

            # Create heatmap
            im = ax.imshow(
                energy_matrix,
                aspect='auto',
                cmap='RdYlGn_r',  # Red=high(bad), Green=low(good)
                interpolation='nearest'
            )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Energy (min)', rotation=270, labelpad=20)

            # Set ticks and labels
            ax.set_xticks(range(len(annealing_time_list)))
            ax.set_xticklabels([f'{int(at)}μs' for at in annealing_time_list], rotation=45, ha='right')
            ax.set_yticks(range(len(num_reads_list)))
            ax.set_yticklabels([str(nr) for nr in num_reads_list])

            ax.set_xlabel('Annealing Time')
            ax.set_ylabel('Num Reads')
            ax.set_title(f'Interval: {interval}s')

            # Add text annotations with energy values
            for i in range(len(num_reads_list)):
                for j in range(len(annealing_time_list)):
                    if not np.isnan(energy_matrix[i, j]):
                        text_color = 'white' if energy_matrix[i, j] < np.nanmean(energy_matrix) else 'black'
                        ax.text(j, i, f'{energy_matrix[i, j]:.0f}',
                               ha='center', va='center', color=text_color, fontsize=8)

        # Hide unused subplots
        for idx in range(n_intervals, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()

        # Save figure
        output_file = output_path / f'qpu_heatmap_seed_{seed}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved {output_file}")

        plt.close(fig)


def create_line_plots_per_seed(data: Dict, output_dir: str = '.'):
    """Create line plots showing energy vs num_reads, with lines per annealing_time.

    One figure per seed, with subplots for each interval.

    Args:
        data: Loaded JSON data from test_qpu.py
        output_dir: Directory to save output figures
    """
    seeds = data['seeds']
    num_reads_list = data['num_reads_tested']
    annealing_time_list = data['annealing_time_tested']
    interval_list = data['interval_tested']
    cpu_baseline = data['cpu_baseline']['results']
    qpu_results = data['qpu_results']

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create one figure per seed
    for seed in seeds:
        n_intervals = len(interval_list)

        # Calculate grid layout
        n_cols = min(3, n_intervals)
        n_rows = (n_intervals + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 5 * n_rows),
            squeeze=False
        )

        fig.suptitle(
            f'QPU Performance - Seed {seed} - Topology {data["topology"]}\n'
            f'CPU Baseline: {cpu_baseline[str(seed)]["energy_min"]:.1f}',
            fontsize=16,
            fontweight='bold'
        )

        for idx, interval in enumerate(interval_list):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Get results for this seed+interval
            key = f"seed_{seed}_interval_{interval}"
            if key not in qpu_results:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Interval: {interval}s')
                continue

            results = qpu_results[key]['results']

            # Plot lines for each annealing_time
            for at in annealing_time_list:
                energies = []
                for nr in num_reads_list:
                    # Find matching result
                    matching = [r for r in results
                               if r['num_reads'] == nr and r['annealing_time'] == at]
                    if matching:
                        energies.append(matching[0]['energy_min'])
                    else:
                        energies.append(np.nan)

                ax.plot(num_reads_list, energies, marker='o', label=f'{int(at)}μs', linewidth=2)

            # Add CPU baseline
            cpu_energy = cpu_baseline[str(seed)]["energy_min"]
            ax.axhline(cpu_energy, color='black', linestyle='--', linewidth=2,
                      label=f'CPU baseline ({cpu_energy:.0f})')

            ax.set_xlabel('Num Reads')
            ax.set_ylabel('Energy (min)')
            ax.set_title(f'Interval: {interval}s')
            ax.legend(title='Annealing Time', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_intervals, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()

        # Save figure
        output_file = output_path / f'qpu_lineplot_seed_{seed}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved {output_file}")

        plt.close(fig)


def create_summary_comparison(data: Dict, output_dir: str = '.'):
    """Create summary comparison showing best energy across all parameters.

    Shows how the best achievable energy varies across seeds and intervals.

    Args:
        data: Loaded JSON data from test_qpu.py
        output_dir: Directory to save output figures
    """
    seeds = data['seeds']
    interval_list = data['interval_tested']
    cpu_baseline = data['cpu_baseline']['results']
    qpu_results = data['qpu_results']

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Collect best energies
    best_energies = np.full((len(seeds), len(interval_list)), np.nan)

    for seed_idx, seed in enumerate(seeds):
        for interval_idx, interval in enumerate(interval_list):
            key = f"seed_{seed}_interval_{interval}"
            if key in qpu_results:
                results = qpu_results[key]['results']
                energies = [r['energy_min'] for r in results]
                best_energies[seed_idx, interval_idx] = min(energies)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        best_energies,
        aspect='auto',
        cmap='RdYlGn_r',
        interpolation='nearest'
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Best Energy Achieved', rotation=270, labelpad=20)

    # Set ticks and labels
    ax.set_xticks(range(len(interval_list)))
    ax.set_xticklabels([f'{interval}s' for interval in interval_list])
    ax.set_yticks(range(len(seeds)))
    ax.set_yticklabels([f'Seed {s}' for s in seeds])

    ax.set_xlabel('Interval Between QPU Queries')
    ax.set_ylabel('Problem Instance (Seed)')
    ax.set_title(f'Best QPU Energy vs CPU Baseline - Topology {data["topology"]}',
                fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(len(seeds)):
        for j in range(len(interval_list)):
            if not np.isnan(best_energies[i, j]):
                # Get CPU baseline for comparison
                cpu_energy = cpu_baseline[str(seeds[i])]["energy_min"]
                delta = best_energies[i, j] - cpu_energy

                text_color = 'white' if best_energies[i, j] < np.nanmean(best_energies) else 'black'
                ax.text(j, i, f'{best_energies[i, j]:.0f}\n({delta:+.0f})',
                       ha='center', va='center', color=text_color, fontsize=10)

    plt.tight_layout()

    # Save figure
    output_file = output_path / 'qpu_summary_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved {output_file}")

    plt.close(fig)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize QPU test results from test_qpu.py'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Input JSON file from test_qpu.py'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='.',
        help='Output directory for figures (default: current directory)'
    )
    parser.add_argument(
        '--plot-type',
        type=str,
        choices=['heatmap', 'lineplot', 'summary', 'all'],
        default='all',
        help='Type of plot to generate (default: all)'
    )

    args = parser.parse_args()

    # Load data
    print(f"📊 Loading results from {args.input_file}...")
    data = load_results(args.input_file)

    print(f"Found {len(data['seeds'])} seeds, {len(data['interval_tested'])} intervals")
    print(f"Parameters: {len(data['num_reads_tested'])} num_reads × {len(data['annealing_time_tested'])} annealing_time")
    print()

    # Generate plots
    if args.plot_type in ['heatmap', 'all']:
        print("Generating heatmaps...")
        create_heatmaps_per_seed(data, args.output_dir)
        print()

    if args.plot_type in ['lineplot', 'all']:
        print("Generating line plots...")
        create_line_plots_per_seed(data, args.output_dir)
        print()

    if args.plot_type in ['summary', 'all']:
        print("Generating summary comparison...")
        create_summary_comparison(data, args.output_dir)
        print()

    print("✨ Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
