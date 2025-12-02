#!/usr/bin/env python3
"""
Visualize comparative mining performance across CPU/GPU/QPU.

Reads structured CSV data from process_mining_comparison.py and generates
comparative analysis charts:

1. Blocks vs Time - Cumulative blocks mined over time for each miner type
2. Mining Efficiency - Blocks per minute by miner type
3. Energy Distribution - Histogram of mining attempt energies
4. Time to Solution - Distribution of mining times by miner type
5. Speedup vs CPU - Relative performance of GPU/QPU vs CPU baseline

Usage:
    python tools/visualize_comparative_performance.py mining_data.csv
    python tools/visualize_comparative_performance.py mining_data.csv --output charts/comparison.png
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

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


def plot_blocks_vs_time(ax, df: pd.DataFrame):
    """Plot cumulative blocks found vs time."""
    ax.set_title('Blocks Mined Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Cumulative Blocks Found', fontsize=12)
    ax.grid(True, alpha=0.3)

    colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71'
    }

    # Get successful blocks only
    successful_df = df[df['valid'] > 0].copy()

    for miner_type in successful_df['miner_type'].unique():
        display_type = normalize_miner_type(miner_type)
        miner_df = successful_df[successful_df['miner_type'] == miner_type].sort_values('end_time')

        if len(miner_df) == 0:
            continue

        # Calculate relative time from first block
        first_time = miner_df['end_time'].min()
        relative_times = (miner_df['end_time'] - first_time).dt.total_seconds() / 60.0  # minutes
        cumulative_blocks = np.arange(1, len(miner_df) + 1)

        ax.plot(
            relative_times,
            cumulative_blocks,
            marker='o',
            markersize=2,
            label=f"{display_type} ({len(miner_df):,} blocks)",
            color=colors.get(display_type, 'gray'),
            linewidth=2,
            alpha=0.8
        )

    ax.legend(fontsize=10)


def plot_mining_efficiency(ax, df: pd.DataFrame):
    """Plot mining efficiency (blocks per minute)."""
    ax.set_title('Mining Efficiency', fontsize=14, fontweight='bold')
    ax.set_ylabel('Blocks per Minute', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71'
    }

    successful_df = df[df['valid'] > 0]

    miner_names = []
    rates = []
    rate_colors = []

    for miner_type in sorted(df['miner_type'].unique()):
        display_type = normalize_miner_type(miner_type)
        miner_df = successful_df[successful_df['miner_type'] == miner_type]

        if len(miner_df) == 0:
            continue

        # Calculate total time span
        time_span_minutes = (miner_df['end_time'].max() - miner_df['start_time'].min()).total_seconds() / 60.0
        blocks_found = len(miner_df)

        if time_span_minutes > 0:
            blocks_per_min = blocks_found / time_span_minutes
        else:
            blocks_per_min = 0

        miner_names.append(display_type)
        rates.append(blocks_per_min)
        rate_colors.append(colors.get(display_type, 'gray'))

    if not rates:
        ax.text(0.5, 0.5, 'No successful blocks', ha='center', va='center', transform=ax.transAxes)
        return

    bars = ax.bar(miner_names, rates, color=rate_colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{rate:.2f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_xlabel('Miner Type', fontsize=12)


def plot_energy_distributions(ax, df: pd.DataFrame):
    """Plot energy distribution histograms."""
    ax.set_title('Energy Distribution (All Attempts)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Energy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71'
    }

    all_energies = df['energy'].values
    if len(all_energies) == 0:
        return

    # Determine bins from overall energy range
    bins = np.linspace(all_energies.min(), all_energies.max(), 50)

    for miner_type in df['miner_type'].unique():
        display_type = normalize_miner_type(miner_type)
        miner_df = df[df['miner_type'] == miner_type]
        energies = miner_df['energy'].values

        if len(energies) > 0:
            ax.hist(
                energies,
                bins=bins,
                alpha=0.5,
                label=f"{display_type} (n={len(energies):,}, mean={np.mean(energies):.0f})",
                color=colors.get(display_type, 'gray'),
                edgecolor='black',
                linewidth=0.5
            )

    ax.legend(fontsize=9)


def plot_time_to_solution(ax, df: pd.DataFrame):
    """Plot time to solution distribution with variance statistics."""
    ax.set_title('Time to Solution (Successful Blocks)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71'
    }

    # Only successful attempts
    successful_df = df[(df['valid'] > 0) & (df['time_to_solution'] > 0)]

    for miner_type in df['miner_type'].unique():
        display_type = normalize_miner_type(miner_type)
        miner_df = successful_df[successful_df['miner_type'] == miner_type]
        tts = miner_df['time_to_solution'].values

        if len(tts) > 0:
            mean_tts = np.mean(tts)
            std_tts = np.std(tts)
            cv = std_tts / mean_tts if mean_tts > 0 else 0
            ax.hist(
                tts,
                bins=30,
                alpha=0.5,
                label=f"{display_type} (μ={mean_tts:.1f}s, σ={std_tts:.1f}s, CV={cv:.2f})",
                color=colors.get(display_type, 'gray'),
                edgecolor='black',
                linewidth=0.5
            )

    ax.legend(fontsize=9)


def plot_cpu_model_breakdown(ax, df: pd.DataFrame):
    """Plot TTS breakdown by CPU model."""
    ax.set_title('CPU Performance by Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time to Solution (seconds)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    cpu_df = df[df['miner_type'] == 'cpu']
    successful_cpu = cpu_df[(cpu_df['valid'] > 0) & (cpu_df['time_to_solution'] > 0)]

    if len(successful_cpu) == 0:
        ax.text(0.5, 0.5, 'No CPU data', ha='center', va='center', transform=ax.transAxes)
        return

    # Get stats per model
    model_stats = []
    for model in successful_cpu['model'].unique():
        model_df = successful_cpu[successful_cpu['model'] == model]
        tts = model_df['time_to_solution'].values
        if len(tts) > 0:
            # Shorten model name for display
            short_name = model.replace('Processor', '').replace('Intel(R) ', '').replace('AMD ', '')
            short_name = short_name.replace('Xeon(R) CPU ', 'Xeon ').replace('-Core', 'c')
            short_name = short_name.strip()
            if len(short_name) > 25:
                short_name = short_name[:22] + '...'
            model_stats.append({
                'model': short_name,
                'mean': np.mean(tts),
                'std': np.std(tts),
                'n': len(tts)
            })

    # Sort by mean TTS
    model_stats.sort(key=lambda x: x['mean'])

    if not model_stats:
        ax.text(0.5, 0.5, 'No CPU model data', ha='center', va='center', transform=ax.transAxes)
        return

    models = [s['model'] for s in model_stats]
    means = [s['mean'] for s in model_stats]
    stds = [s['std'] for s in model_stats]
    positions = np.arange(len(models))

    # Color gradient from green (fast) to red (slow)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))

    bars = ax.bar(positions, means, color=colors, alpha=0.7, edgecolor='black')
    ax.errorbar(positions, means, yerr=stds, fmt='none', ecolor='black', capsize=3, capthick=1)

    # Add value labels
    for i, (bar, stat) in enumerate(zip(bars, model_stats)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + stat['std'] + 20,
                f"{stat['mean']:.0f}s\n(n={stat['n']})", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('CPU Model', fontsize=12)


def plot_speedup_vs_cpu(ax, df: pd.DataFrame):
    """Plot speedup relative to CPU baseline."""
    ax.set_title('Speedup vs CPU Baseline', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    colors = {
        'CPU': '#e74c3c',
        'GPU': '#3498db',
        'QPU': '#2ecc71'
    }

    successful_df = df[df['valid'] > 0]

    # Calculate CPU baseline rate
    cpu_df = successful_df[successful_df['miner_type'] == 'cpu']
    if len(cpu_df) == 0:
        ax.text(0.5, 0.5, 'No CPU baseline available', ha='center', va='center', transform=ax.transAxes)
        return

    cpu_time_span = (cpu_df['end_time'].max() - cpu_df['start_time'].min()).total_seconds() / 60.0
    cpu_rate = len(cpu_df) / cpu_time_span if cpu_time_span > 0 else 0

    if cpu_rate == 0:
        ax.text(0.5, 0.5, 'CPU rate is zero', ha='center', va='center', transform=ax.transAxes)
        return

    miner_names = []
    speedups = []
    speedup_colors = []

    for miner_type in sorted(df['miner_type'].unique()):
        if miner_type == 'cpu':
            continue

        display_type = normalize_miner_type(miner_type)
        miner_df = successful_df[successful_df['miner_type'] == miner_type]

        if len(miner_df) == 0:
            continue

        time_span = (miner_df['end_time'].max() - miner_df['start_time'].min()).total_seconds() / 60.0
        miner_rate = len(miner_df) / time_span if time_span > 0 else 0

        if miner_rate > 0:
            speedup = miner_rate / cpu_rate
            miner_names.append(display_type)
            speedups.append(speedup)
            speedup_colors.append(colors.get(display_type, 'gray'))

    if not speedups:
        ax.text(0.5, 0.5, 'No GPU/QPU results to compare', ha='center', va='center', transform=ax.transAxes)
        return

    bars = ax.bar(miner_names, speedups, color=speedup_colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{speedup:.1f}x',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    # Add reference line at 1x (CPU baseline)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='CPU baseline (1x)')
    ax.legend(fontsize=10)
    ax.set_xlabel('Miner Type', fontsize=12)


def generate_summary(df: pd.DataFrame) -> str:
    """Generate text summary of the data."""
    lines = []
    lines.append("COMPARATIVE MINING PERFORMANCE ANALYSIS")
    lines.append("=" * 60)
    lines.append("")

    successful_df = df[df['valid'] > 0]

    # Summary by miner type
    lines.append("Mining Performance by Type:")
    lines.append("-" * 40)

    cpu_rate = None

    for miner_type in sorted(df['miner_type'].unique()):
        display_type = normalize_miner_type(miner_type)
        miner_df = df[df['miner_type'] == miner_type]
        miner_successful = successful_df[successful_df['miner_type'] == miner_type]

        total_attempts = len(miner_df)
        successful_blocks = len(miner_successful)
        success_rate = successful_blocks / total_attempts if total_attempts > 0 else 0

        # Calculate rate
        if len(miner_successful) > 0:
            time_span = (miner_successful['end_time'].max() - miner_successful['start_time'].min()).total_seconds() / 60.0
            blocks_per_min = successful_blocks / time_span if time_span > 0 else 0

            if miner_type == 'cpu':
                cpu_rate = blocks_per_min
        else:
            blocks_per_min = 0

        # Energy stats
        energies = miner_df['energy'].values

        lines.append(f"\n{display_type}:")
        lines.append(f"  Total attempts: {total_attempts:,}")
        lines.append(f"  Successful blocks: {successful_blocks:,}")
        lines.append(f"  Success rate: {success_rate * 100:.1f}%")
        lines.append(f"  Mining rate: {blocks_per_min:.3f} blocks/min")
        lines.append(f"  Energy range: {np.min(energies):.0f} to {np.max(energies):.0f}")
        lines.append(f"  Energy mean: {np.mean(energies):.0f}")

        if len(miner_successful) > 0:
            tts_values = miner_successful['time_to_solution'].values
            valid_tts = tts_values[tts_values > 0]
            if len(valid_tts) > 0:
                mean_tts = np.mean(valid_tts)
                std_tts = np.std(valid_tts)
                cv = std_tts / mean_tts if mean_tts > 0 else 0
                lines.append(f"  Time to solution: {mean_tts:.1f}s ± {std_tts:.1f}s (CV={cv:.2f})")

    # Speedup analysis
    if cpu_rate and cpu_rate > 0:
        lines.append("\n" + "-" * 40)
        lines.append("Speedup vs CPU:")

        for miner_type in sorted(df['miner_type'].unique()):
            if miner_type == 'cpu':
                continue

            display_type = normalize_miner_type(miner_type)
            miner_successful = successful_df[successful_df['miner_type'] == miner_type]

            if len(miner_successful) > 0:
                time_span = (miner_successful['end_time'].max() - miner_successful['start_time'].min()).total_seconds() / 60.0
                miner_rate = len(miner_successful) / time_span if time_span > 0 else 0

                if miner_rate > 0:
                    speedup = miner_rate / cpu_rate
                    lines.append(f"  {display_type}: {speedup:.1f}x")

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize comparative mining performance from CSV data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/visualize_comparative_performance.py mining_data.csv
    python tools/visualize_comparative_performance.py mining_data.csv --output charts/comparison.png
        """
    )
    parser.add_argument(
        'csv_file',
        type=Path,
        help='CSV file from process_mining_comparison.py'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('comparative_analysis.png'),
        help='Output file for visualization (default: comparative_analysis.png)'
    )
    parser.add_argument(
        '--summary-file',
        type=Path,
        default=None,
        help='Optional text file for summary output'
    )

    args = parser.parse_args()

    if not args.csv_file.exists():
        print(f"Error: CSV file '{args.csv_file}' not found")
        return 1

    print(f"Loading {args.csv_file}...")
    df = load_mining_csv(str(args.csv_file))
    print(f"Loaded {len(df):,} records")

    # Print data summary
    print("\nData summary by miner type:")
    for miner_type in df['miner_type'].unique():
        miner_df = df[df['miner_type'] == miner_type]
        successful = miner_df[miner_df['valid'] > 0]
        print(f"  {normalize_miner_type(miner_type)}: {len(miner_df):,} attempts, "
              f"{len(successful):,} successful ({100*len(successful)/len(miner_df):.1f}%)")

    # Generate visualization
    print("\nGenerating charts...")

    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)

    # Plot 1: Blocks vs Time (full width)
    ax1 = fig.add_subplot(gs[0, :])
    plot_blocks_vs_time(ax1, df)

    # Plot 2: Mining Efficiency
    ax2 = fig.add_subplot(gs[1, 0])
    plot_mining_efficiency(ax2, df)

    # Plot 3: Energy Distributions
    ax3 = fig.add_subplot(gs[1, 1])
    plot_energy_distributions(ax3, df)

    # Plot 4: Time to Solution
    ax4 = fig.add_subplot(gs[2, 0])
    plot_time_to_solution(ax4, df)

    # Plot 5: Speedup vs CPU
    ax5 = fig.add_subplot(gs[2, 1])
    plot_speedup_vs_cpu(ax5, df)

    # Plot 6: CPU Model Breakdown (full width)
    ax6 = fig.add_subplot(gs[3, :])
    plot_cpu_model_breakdown(ax6, df)

    plt.suptitle(
        'Comparative Mining Performance Analysis',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {args.output}")

    # Generate text summary
    summary = generate_summary(df)

    if args.summary_file:
        with open(args.summary_file, 'w') as f:
            f.write(summary)
        print(f"Saved text summary to {args.summary_file}")

    # Print summary to console
    print("\n" + summary)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
