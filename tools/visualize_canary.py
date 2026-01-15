#!/usr/bin/env python3
"""Visualize canary predictive power from test results.

This tool creates visualizations showing how canary energy predicts
mining success probability across different miner types (CPU/GPU/QPU).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

def load_canary_results(filepath: str) -> Dict:
    """Load canary test results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_success_probability(
    results_files: List[str],
    output_file: Optional[str] = None,
    title: str = "Canary Success Probability by Energy"
):
    """Plot success probability curves for multiple test results.

    Args:
        results_files: List of paths to canary test JSON files
        output_file: Optional path to save plot (if None, display interactively)
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, filepath in enumerate(results_files):
        data = load_canary_results(filepath)
        stats = data['statistics']

        # Extract metadata
        miner_type = data.get('miner_type', 'unknown').upper()
        topology = data.get('topology', 'unknown')
        canary_params = stats['canary']['params']
        label = f"{miner_type} (sweeps={canary_params['num_sweeps']}, reads={canary_params['num_reads']})"

        # Get nonce results
        nonce_results = stats['nonce_results']

        # Extract canary energies and success flags
        canary_energies = []
        success_flags = []

        for result in nonce_results:
            canary_e = result['canary'].get('energy')
            full_passed = result['full'].get('passed', False)

            if canary_e is not None:
                canary_energies.append(canary_e)
                success_flags.append(1 if full_passed else 0)

        if len(canary_energies) == 0:
            print(f"Warning: No data in {filepath}")
            continue

        # Sort by canary energy
        sorted_indices = np.argsort(canary_energies)
        sorted_energies = np.array(canary_energies)[sorted_indices]
        sorted_successes = np.array(success_flags)[sorted_indices]

        # === PLOT 1: Success Probability Curve (Rolling Average) ===
        # Use rolling window to estimate probability
        window_size = max(3, len(sorted_energies) // 10)

        smoothed_energies = []
        smoothed_probs = []

        for j in range(len(sorted_energies)):
            start = max(0, j - window_size // 2)
            end = min(len(sorted_energies), j + window_size // 2 + 1)

            window_successes = sorted_successes[start:end]
            window_energy = sorted_energies[start:end]

            prob = np.mean(window_successes)
            energy = np.mean(window_energy)

            smoothed_energies.append(energy)
            smoothed_probs.append(prob)

        ax1.plot(smoothed_energies, smoothed_probs,
                label=label, linewidth=2, color=colors[i % len(colors)], alpha=0.8)

        # Also plot raw data points
        ax1.scatter(sorted_energies, sorted_successes,
                   alpha=0.3, s=20, color=colors[i % len(colors)])

        # === PLOT 2: Cumulative Success Rate ===
        # If we accept all nonces with canary <= E, what's our success rate?
        cumulative_successes = np.cumsum(sorted_successes)
        cumulative_total = np.arange(1, len(sorted_successes) + 1)
        cumulative_rate = cumulative_successes / cumulative_total

        ax2.plot(sorted_energies, cumulative_rate * 100,
                label=label, linewidth=2, color=colors[i % len(colors)], alpha=0.8)

    # Configure Plot 1
    ax1.set_xlabel('Canary Energy', fontsize=12)
    ax1.set_ylabel('Success Probability', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    ax1.set_ylim([-0.05, 1.05])

    # Add reference line at 50%
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% probability')

    # Configure Plot 2
    ax2.set_xlabel('Canary Energy Threshold (accept if canary ≤ threshold)', fontsize=12)
    ax2.set_ylabel('Cumulative Success Rate (%)', fontsize=12)
    ax2.set_title('Success Rate vs Acceptance Threshold', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    ax2.set_ylim([0, 105])

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to {output_file}")
    else:
        plt.show()


def plot_energy_scatter(
    results_files: List[str],
    output_file: Optional[str] = None,
    title: str = "Canary vs Full Energy"
):
    """Create scatter plots of canary vs full energy.

    Args:
        results_files: List of paths to canary test JSON files
        output_file: Optional path to save plot
        title: Plot title
    """
    # Try to import energy_to_difficulty for annotations
    try:
        from shared.energy_utils import energy_to_difficulty
        has_difficulty_fn = True
    except ImportError:
        has_difficulty_fn = False

    # Try to load topology loader
    try:
        from dwave_topologies.topologies.json_loader import load_topology
        has_topology_loader = True
    except ImportError:
        has_topology_loader = False

    fig, axes = plt.subplots(1, len(results_files),
                            figsize=(6 * len(results_files), 5))

    if len(results_files) == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e']  # Blue for fail, orange for success

    for idx, filepath in enumerate(results_files):
        ax = axes[idx]
        data = load_canary_results(filepath)
        stats = data['statistics']

        # Extract metadata
        miner_type = data.get('miner_type', 'unknown').upper()
        topology_name = data.get('topology', 'unknown')
        difficulty = data['difficulty_energy']

        # Get correlation info
        corr = stats['analysis'].get('correlation', {})
        pearson_r = corr.get('pearson_r', 0)
        r_squared = corr['linear_regression']['r_squared'] if corr else 0
        equation = corr['linear_regression']['equation'] if corr else ''

        # Extract data
        canary_energies = []
        full_energies = []
        success_flags = []

        for result in stats['nonce_results']:
            canary_e = result['canary'].get('energy')
            full_e = result['full'].get('energy')
            passed = result['full'].get('passed', False)

            if canary_e is not None and full_e is not None:
                canary_energies.append(canary_e)
                full_energies.append(full_e)
                success_flags.append(passed)

        # Scatter plot colored by success
        for i, (ce, fe, success) in enumerate(zip(canary_energies, full_energies, success_flags)):
            color = colors[1] if success else colors[0]
            label = 'Passed' if success and i == 0 else ('Failed' if not success and i == 0 else None)
            ax.scatter(ce, fe, c=color, alpha=0.6, s=50, label=label)

        # Plot regression line if available
        if corr:
            x_range = np.array([min(canary_energies), max(canary_energies)])
            slope = corr['linear_regression']['slope']
            intercept = corr['linear_regression']['intercept']
            y_pred = slope * x_range + intercept
            ax.plot(x_range, y_pred, 'r--', linewidth=2, alpha=0.7, label='Linear fit')

        # Plot difficulty threshold
        ax.axhline(y=difficulty, color='green', linestyle='--',
                  linewidth=2, alpha=0.7, label=f'Difficulty: {difficulty:.0f}')

        # Set tighter axis limits based on data range with 5% padding
        canary_range = max(canary_energies) - min(canary_energies)
        full_range = max(full_energies) - min(full_energies)
        canary_padding = canary_range * 0.05
        full_padding = full_range * 0.05

        ax.set_xlim(min(canary_energies) - canary_padding, max(canary_energies) + canary_padding)
        ax.set_ylim(min(full_energies) - full_padding, max(full_energies) + full_padding)

        # Labels
        ax.set_xlabel('Canary Energy', fontsize=12)
        ax.set_ylabel('Full Energy', fontsize=12)
        ax.set_title(f'{miner_type}\nr={pearson_r:.3f}, R²={r_squared:.3f}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        # Add difficulty annotations if possible
        if has_difficulty_fn and has_topology_loader:
            try:
                # Load topology to get node/edge counts
                topology = load_topology(topology_name)
                num_nodes = len(topology.graph.nodes) if hasattr(topology.graph, 'nodes') else topology.num_nodes
                num_edges = len(topology.graph.edges) if hasattr(topology.graph, 'edges') else topology.num_edges

                # Get current y-tick locations (full energy axis)
                yticks = ax.get_yticks()

                # Create new labels with difficulty ratings
                new_labels = []
                for energy in yticks:
                    # Calculate difficulty for this energy
                    difficulty_val = energy_to_difficulty(energy, num_nodes, num_edges)
                    # Format: energy (difficulty as 0.0-1.0)
                    new_labels.append(f'{energy:.0f}\n({difficulty_val:.2f})')

                # Set both ticks and labels
                ax.set_yticks(yticks)
                ax.set_yticklabels(new_labels)
            except Exception as e:
                # If topology loading fails, continue without annotations
                pass

        # Add equation as text
        if equation:
            ax.text(0.05, 0.95, equation, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Scatter plot saved to {output_file}")
    else:
        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize canary predictive power from test results'
    )
    parser.add_argument(
        'results_files',
        nargs='+',
        help='Path(s) to canary test JSON result files'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path for plot (PNG, PDF, SVG supported)'
    )
    parser.add_argument(
        '--title',
        type=str,
        default='Canary Success Probability by Energy',
        help='Plot title'
    )
    parser.add_argument(
        '--type',
        type=str,
        choices=['probability', 'scatter', 'both'],
        default='both',
        help='Type of plot to generate (default: both)'
    )

    args = parser.parse_args()

    # Validate input files
    for filepath in args.results_files:
        if not Path(filepath).exists():
            print(f"❌ File not found: {filepath}")
            return 1

    print(f"📊 Visualizing {len(args.results_files)} canary test result(s)...")

    # Generate plots
    if args.type in ['probability', 'both']:
        output_file = args.output if args.output and args.type == 'probability' else None
        if args.type == 'both' and args.output:
            # Add suffix for probability plot
            base = Path(args.output)
            output_file = str(base.parent / f"{base.stem}_probability{base.suffix}")

        plot_success_probability(args.results_files, output_file, args.title)

    if args.type in ['scatter', 'both']:
        output_file = args.output if args.output and args.type == 'scatter' else None
        if args.type == 'both' and args.output:
            # Add suffix for scatter plot
            base = Path(args.output)
            output_file = str(base.parent / f"{base.stem}_scatter{base.suffix}")

        plot_energy_scatter(args.results_files, output_file, args.title)

    return 0


if __name__ == "__main__":
    sys.exit(main())
