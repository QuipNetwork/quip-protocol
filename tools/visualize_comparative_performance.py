#!/usr/bin/env python3
"""Visualize comparative mining performance across CPU/GPU/QPU."""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def load_json(filepath: str) -> Optional[Dict]:
    """Load JSON file, return None on error."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading {filepath}: {e}")
        return None


def plot_blocks_vs_time(
    ax,
    mining_results: Dict[str, Dict],
    target_time: float
):
    """Plot cumulative blocks found vs time."""
    ax.set_title('Blocks Mined Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Cumulative Blocks Found', fontsize=12)
    ax.grid(True, alpha=0.3)

    colors = {
        'cpu': '#FF6B6B',
        'cuda': '#4ECDC4',
        'metal': '#95E1D3',
        'qpu': '#4285F4'
    }

    for miner_type, data in mining_results.items():
        stats = data.get('statistics', {})
        mining_times = stats.get('mining_time_stats', {}).get('all_times', [])

        if not mining_times:
            continue

        # Cumulative blocks over time
        cumulative_times = np.cumsum(mining_times) / 60.0  # Convert to minutes
        cumulative_blocks = np.arange(1, len(mining_times) + 1)

        ax.plot(
            cumulative_times,
            cumulative_blocks,
            marker='o',
            markersize=4,
            label=f"{miner_type.upper()} ({len(mining_times)} blocks)",
            color=colors.get(miner_type, 'gray'),
            linewidth=2
        )

    # Add target time reference line
    ax.axvline(x=target_time, color='red', linestyle='--', alpha=0.5, label=f'Target time ({target_time} min)')

    ax.legend(fontsize=10)


def plot_mining_efficiency(
    ax,
    mining_results: Dict[str, Dict]
):
    """Plot mining efficiency (blocks per minute)."""
    ax.set_title('Mining Efficiency', fontsize=14, fontweight='bold')
    ax.set_ylabel('Blocks per Minute', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    colors = {
        'cpu': '#FF6B6B',
        'cuda': '#4ECDC4',
        'metal': '#95E1D3',
        'qpu': '#4285F4'
    }

    miner_names = []
    rates = []
    rate_colors = []

    for miner_type, data in sorted(mining_results.items()):
        stats = data.get('statistics', {})
        blocks_per_min = stats.get('blocks_per_minute', 0)

        miner_names.append(miner_type.upper())
        rates.append(blocks_per_min)
        rate_colors.append(colors.get(miner_type, 'gray'))

    bars = ax.bar(miner_names, rates, color=rate_colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{rate:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_xlabel('Miner Type', fontsize=12)


def plot_energy_distributions(
    ax,
    mining_results: Dict[str, Dict]
):
    """Plot energy distribution histograms."""
    ax.set_title('Energy Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Energy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    colors = {
        'cpu': '#FF6B6B',
        'cuda': '#4ECDC4',
        'metal': '#95E1D3',
        'qpu': '#4285F4'
    }

    for miner_type, data in mining_results.items():
        stats = data.get('statistics', {})
        energies = stats.get('energy_stats', {}).get('all_energies', [])

        if energies:
            ax.hist(
                energies,
                bins=20,
                alpha=0.5,
                label=f"{miner_type.upper()} (μ={np.mean(energies):.1f})",
                color=colors.get(miner_type, 'gray'),
                edgecolor='black'
            )

    ax.legend(fontsize=10)


def plot_quantum_advantage(
    ax,
    mining_results: Dict[str, Dict],
    threshold_results: Dict[str, Dict]
):
    """Plot quantum advantage factors."""
    ax.set_title('Quantum Advantage Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup vs Classical', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Calculate speedup relative to CPU
    cpu_rate = None
    if 'cpu' in mining_results:
        cpu_rate = mining_results['cpu'].get('statistics', {}).get('blocks_per_minute', None)
    elif 'cpu' in threshold_results:
        # Estimate from threshold result (1 block / target_time)
        target_time = threshold_results['cpu'].get('target_time_minutes', 10.0)
        cpu_rate = 1.0 / target_time

    if cpu_rate is None or cpu_rate == 0:
        ax.text(
            0.5, 0.5,
            'No CPU baseline available',
            ha='center', va='center',
            transform=ax.transAxes,
            fontsize=12
        )
        return

    colors = {
        'cpu': '#FF6B6B',
        'cuda': '#4ECDC4',
        'metal': '#95E1D3',
        'qpu': '#4285F4'
    }

    miner_names = []
    speedups = []
    speedup_colors = []

    for miner_type, data in sorted(mining_results.items()):
        if miner_type == 'cpu':
            continue  # Skip CPU (baseline)

        stats = data.get('statistics', {})
        miner_rate = stats.get('blocks_per_minute', 0)

        if miner_rate > 0:
            speedup = miner_rate / cpu_rate
            miner_names.append(miner_type.upper())
            speedups.append(speedup)
            speedup_colors.append(colors.get(miner_type, 'gray'))

    if not speedups:
        ax.text(
            0.5, 0.5,
            'No quantum/GPU results to compare',
            ha='center', va='center',
            transform=ax.transAxes,
            fontsize=12
        )
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


def generate_report(
    threshold_results: Dict[str, Dict],
    mining_results: Dict[str, Dict],
    output_file: str
):
    """Generate comprehensive comparison report."""
    print("\n" + "=" * 60)
    print("📊 COMPARATIVE MINING PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Threshold results
    print("\n1️⃣  Block Time Thresholds:")
    print("-" * 40)
    for miner_type, data in sorted(threshold_results.items()):
        result = data.get('result', {})
        if result:
            print(f"{miner_type.upper()}:")
            print(f"  Difficulty energy: {result['difficulty_energy']:.1f}")
            print(f"  Average block time: {result['avg_time_minutes']:.1f} min")
            print(f"  Deviation: {result['deviation'] * 100:.1f}%")
            print(f"  Converged: {'Yes' if result.get('converged') else 'No'}")

    # Mining rates
    print("\n2️⃣  Mining Rates at Fixed Difficulty:")
    print("-" * 40)
    for miner_type, data in sorted(mining_results.items()):
        stats = data.get('statistics', {})
        print(f"{miner_type.upper()}:")
        print(f"  Duration: {data.get('duration_minutes', 0):.1f} min")
        print(f"  Difficulty: {data.get('difficulty_energy', 0):.1f}")
        print(f"  Blocks found: {stats.get('total_blocks_found', 0)}")
        print(f"  Success rate: {stats.get('success_rate', 0) * 100:.1f}%")
        print(f"  Mining rate: {stats.get('blocks_per_minute', 0):.3f} blocks/min")

        if stats.get('total_blocks_found', 0) > 0:
            energy_stats = stats.get('energy_stats', {})
            print(f"  Energy: min={energy_stats.get('min', 0):.1f}, "
                  f"max={energy_stats.get('max', 0):.1f}, "
                  f"mean={energy_stats.get('mean', 0):.1f}")

    # Quantum advantage
    print("\n3️⃣  Quantum Advantage Analysis:")
    print("-" * 40)

    cpu_rate = None
    if 'cpu' in mining_results:
        cpu_rate = mining_results['cpu'].get('statistics', {}).get('blocks_per_minute')
    elif 'cpu' in threshold_results:
        target_time = threshold_results['cpu'].get('target_time_minutes', 10.0)
        cpu_rate = 1.0 / target_time

    if cpu_rate and cpu_rate > 0:
        print(f"CPU baseline: {cpu_rate:.3f} blocks/min")

        for miner_type, data in sorted(mining_results.items()):
            if miner_type == 'cpu':
                continue

            stats = data.get('statistics', {})
            miner_rate = stats.get('blocks_per_minute', 0)

            if miner_rate > 0:
                speedup = miner_rate / cpu_rate
                print(f"{miner_type.upper()} speedup: {speedup:.1f}x")
    else:
        print("⚠️  No CPU baseline available for comparison")

    # Generate plots
    print("\n4️⃣  Generating visualizations...")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Blocks vs Time
    ax1 = fig.add_subplot(gs[0, :])
    target_time = 10.0  # Default
    if threshold_results:
        first_threshold = next(iter(threshold_results.values()))
        target_time = first_threshold.get('target_time_minutes', 10.0)
    plot_blocks_vs_time(ax1, mining_results, target_time)

    # Plot 2: Mining Efficiency
    ax2 = fig.add_subplot(gs[1, 0])
    plot_mining_efficiency(ax2, mining_results)

    # Plot 3: Energy Distributions
    ax3 = fig.add_subplot(gs[1, 1])
    plot_energy_distributions(ax3, mining_results)

    # Plot 4: Quantum Advantage
    ax4 = fig.add_subplot(gs[2, :])
    plot_quantum_advantage(ax4, mining_results, threshold_results)

    plt.suptitle(
        'Quantum Mining Performance Comparison',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to {output_file}")

    # Also save text summary
    summary_file = output_file.replace('.pdf', '_summary.txt').replace('.png', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("COMPARATIVE MINING PERFORMANCE ANALYSIS\n")
        f.write("=" * 60 + "\n\n")

        f.write("Block Time Thresholds:\n")
        f.write("-" * 40 + "\n")
        for miner_type, data in sorted(threshold_results.items()):
            result = data.get('result', {})
            if result:
                f.write(f"{miner_type.upper()}:\n")
                f.write(f"  Difficulty energy: {result['difficulty_energy']:.1f}\n")
                f.write(f"  Average block time: {result['avg_time_minutes']:.1f} min\n")
                f.write(f"  Deviation: {result['deviation'] * 100:.1f}%\n\n")

        f.write("\nMining Rates:\n")
        f.write("-" * 40 + "\n")
        for miner_type, data in sorted(mining_results.items()):
            stats = data.get('statistics', {})
            f.write(f"{miner_type.upper()}:\n")
            f.write(f"  Blocks per minute: {stats.get('blocks_per_minute', 0):.3f}\n")
            f.write(f"  Total blocks: {stats.get('total_blocks_found', 0)}\n\n")

        f.write("\nQuantum Advantage:\n")
        f.write("-" * 40 + "\n")
        if cpu_rate and cpu_rate > 0:
            for miner_type, data in sorted(mining_results.items()):
                if miner_type != 'cpu':
                    stats = data.get('statistics', {})
                    miner_rate = stats.get('blocks_per_minute', 0)
                    if miner_rate > 0:
                        speedup = miner_rate / cpu_rate
                        f.write(f"{miner_type.upper()}: {speedup:.1f}x speedup\n")

    print(f"✅ Saved text summary to {summary_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize comparative mining performance'
    )
    parser.add_argument(
        '--cpu-threshold',
        type=str,
        help='CPU threshold JSON file from find_block_time_threshold.py'
    )
    parser.add_argument(
        '--cuda-threshold',
        type=str,
        help='CUDA threshold JSON file from find_block_time_threshold.py'
    )
    parser.add_argument(
        '--metal-threshold',
        type=str,
        help='Metal threshold JSON file from find_block_time_threshold.py'
    )
    parser.add_argument(
        '--cpu-mining',
        type=str,
        help='CPU mining rate JSON file from compare_mining_rates.py'
    )
    parser.add_argument(
        '--cuda-mining',
        type=str,
        help='CUDA mining rate JSON file from compare_mining_rates.py'
    )
    parser.add_argument(
        '--metal-mining',
        type=str,
        help='Metal mining rate JSON file from compare_mining_rates.py'
    )
    parser.add_argument(
        '--qpu-mining',
        type=str,
        help='QPU mining rate JSON file from compare_mining_rates.py'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='comparative_analysis.pdf',
        help='Output file for visualization (default: comparative_analysis.pdf)'
    )

    args = parser.parse_args()

    print("🔬 Comparative Performance Visualization")
    print("=" * 50)

    # Load threshold results
    threshold_results = {}
    for miner_type in ['cpu', 'cuda', 'metal']:
        threshold_file = getattr(args, f'{miner_type}_threshold')
        if threshold_file:
            data = load_json(threshold_file)
            if data:
                threshold_results[miner_type] = data
                print(f"✅ Loaded {miner_type.upper()} threshold data")

    # Load mining rate results
    mining_results = {}
    for miner_type in ['cpu', 'cuda', 'metal', 'qpu']:
        mining_file = getattr(args, f'{miner_type}_mining')
        if mining_file:
            data = load_json(mining_file)
            if data:
                mining_results[miner_type] = data
                print(f"✅ Loaded {miner_type.upper()} mining rate data")

    # Check if we have any data
    if not threshold_results and not mining_results:
        print("❌ No input data provided. Use --help for usage information.")
        return 1

    # Generate report
    generate_report(threshold_results, mining_results, args.output)

    print("\n✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
