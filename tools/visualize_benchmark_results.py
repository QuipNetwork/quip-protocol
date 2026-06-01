#!/usr/bin/env python3
"""Post-hoc analysis of benchmark_prefilter_mining.py output.

Reads benchmark JSON, computes greedy energies for mined nonces and
random controls, then generates comparison plots and statistics.
No SA computation — purely analytical.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from dwave_topologies import DEFAULT_TOPOLOGY
from shared.energy_utils import energy_to_difficulty
from shared.nonce_prefilter import IsingTopologyCache


def load_benchmark(path: str) -> Dict[str, Any]:
    """Load and validate benchmark JSON."""
    with open(path) as f:
        data = json.load(f)

    required = ['difficulty_energy', 'per_type_stats']
    for key in required:
        assert key in data, f"Missing key: {key}"
    return data


def extract_block_details(
    data: Dict[str, Any],
) -> Dict[str, List[Dict]]:
    """Extract per-block details grouped by miner type.

    Returns:
        Dict mapping miner_type -> list of block detail dicts.
    """
    by_type: Dict[str, List[Dict]] = {}

    for mtype, type_data in data['per_type_stats'].items():
        blocks: List[Dict] = []
        for miner in type_data.get('per_miner_stats', []):
            for bd in miner.get('block_details', []):
                bd['miner_type'] = mtype
                bd['miner_id'] = miner.get('miner_id', 'unknown')
                blocks.append(bd)
        by_type[mtype] = blocks

    return by_type


def compute_greedy_energies(
    nonces: List[int],
    cache: IsingTopologyCache,
) -> List[float]:
    """Compute greedy descent energy for each nonce."""
    return [cache.greedy_descent_fast(n) for n in nonces]


def generate_random_nonces(count: int) -> List[int]:
    """Generate random nonces in the same 32-bit range."""
    rng = np.random.default_rng(42)
    return [int(x) for x in rng.integers(0, 2**32, size=count)]


def plot_performance_comparison(
    data: Dict[str, Any],
    by_type: Dict[str, List[Dict]],
    output_dir: Path,
):
    """Side-by-side bar chart of mining performance metrics."""
    types = sorted(by_type.keys())
    if len(types) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Blocks found
    blocks = [len(by_type[t]) for t in types]
    axes[0].bar(types, blocks, color=['#4C72B0', '#DD8452'])
    axes[0].set_title('Blocks Found')
    axes[0].set_ylabel('Count')

    # Blocks per minute
    duration_min = data.get('duration_minutes', 1)
    total_times = {}
    for t in types:
        ts = data['per_type_stats'][t]
        total_times[t] = ts.get('total_time_seconds', duration_min * 60)
    rates = [
        len(by_type[t]) / (total_times[t] / 60)
        if total_times[t] > 0 else 0
        for t in types
    ]
    axes[1].bar(types, rates, color=['#4C72B0', '#DD8452'])
    axes[1].set_title('Blocks / Minute')
    axes[1].set_ylabel('Rate')

    # Success rate
    success_rates = []
    for t in types:
        ts = data['per_type_stats'][t]
        sr = ts.get('success_rate', 0)
        success_rates.append(sr * 100)
    axes[2].bar(types, success_rates, color=['#4C72B0', '#DD8452'])
    axes[2].set_title('Success Rate')
    axes[2].set_ylabel('%')

    difficulty = data.get('difficulty_energy', 0)
    fig.suptitle(
        f'Performance Comparison  '
        f'(difficulty={difficulty:.0f})',
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(output_dir / 'performance_comparison.png', dpi=150)
    plt.close(fig)


def plot_greedy_vs_sa(
    by_type: Dict[str, List[Dict]],
    greedy_by_type: Dict[str, List[float]],
    output_dir: Path,
):
    """Scatter: greedy energy (x) vs SA energy (y) for mined blocks."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {'cpu': '#4C72B0', 'cpu-filtered': '#DD8452'}
    all_greedy: List[float] = []
    all_sa: List[float] = []

    for mtype in sorted(by_type.keys()):
        blocks = by_type[mtype]
        greedy = greedy_by_type[mtype]
        sa_energies = [b['energy'] for b in blocks]
        ax.scatter(
            greedy, sa_energies,
            c=colors.get(mtype, '#999'),
            label=mtype, alpha=0.6, s=30,
        )
        all_greedy.extend(greedy)
        all_sa.extend(sa_energies)

    if len(all_greedy) >= 3:
        r, p = scipy_stats.pearsonr(all_greedy, all_sa)
        # Regression line
        z = np.polyfit(all_greedy, all_sa, 1)
        x_line = np.linspace(min(all_greedy), max(all_greedy), 100)
        ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.5)
        ax.set_title(
            f'Greedy vs SA Energy (mined blocks)\n'
            f'Pearson r={r:.3f}, p={p:.2e}',
        )
    else:
        ax.set_title('Greedy vs SA Energy (mined blocks)')

    ax.set_xlabel('Greedy Energy')
    ax.set_ylabel('SA Energy')
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / 'greedy_vs_sa_mined.png', dpi=150)
    plt.close(fig)


def plot_greedy_mined_vs_random(
    all_mined_greedy: List[float],
    random_greedy: List[float],
    output_dir: Path,
):
    """Overlapping histograms: mined vs random greedy energies."""
    fig, ax = plt.subplots(figsize=(9, 6))

    bins = np.linspace(
        min(min(all_mined_greedy), min(random_greedy)),
        max(max(all_mined_greedy), max(random_greedy)),
        40,
    )
    ax.hist(
        random_greedy, bins=bins, alpha=0.5,
        label=f'Random nonces (n={len(random_greedy)})',
        color='#999', density=True,
    )
    ax.hist(
        all_mined_greedy, bins=bins, alpha=0.6,
        label=f'Mined nonces (n={len(all_mined_greedy)})',
        color='#DD8452', density=True,
    )

    ks_stat, ks_p = scipy_stats.ks_2samp(all_mined_greedy, random_greedy)
    ax.set_title(
        f'Greedy Energy: Mined vs Random Nonces\n'
        f'KS stat={ks_stat:.3f}, p={ks_p:.2e}',
    )
    ax.set_xlabel('Greedy Energy')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / 'greedy_mined_vs_random.png', dpi=150)
    plt.close(fig)

    return ks_stat, ks_p


def plot_energy_distributions(
    by_type: Dict[str, List[Dict]],
    greedy_by_type: Dict[str, List[float]],
    difficulty_energy: float,
    output_dir: Path,
):
    """Per-type histograms of SA and greedy energies."""
    types = sorted(by_type.keys())
    fig, axes = plt.subplots(1, len(types), figsize=(7 * len(types), 5))
    if len(types) == 1:
        axes = [axes]

    for ax, mtype in zip(axes, types):
        sa = [b['energy'] for b in by_type[mtype]]
        greedy = greedy_by_type[mtype]

        all_vals = sa + greedy
        lo = min(all_vals) - 200
        hi = max(all_vals) + 200
        bins = np.linspace(lo, hi, 30)

        ax.hist(
            sa, bins=bins, alpha=0.6,
            label='SA energy', color='#4C72B0',
        )
        ax.hist(
            greedy, bins=bins, alpha=0.5,
            label='Greedy energy', color='#DD8452',
        )
        ax.axvline(
            difficulty_energy, color='red', ls='--',
            label=f'Threshold ({difficulty_energy:.0f})',
        )
        ax.set_title(f'{mtype}')
        ax.set_xlabel('Energy')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    fig.suptitle('Energy Distributions by Miner Type', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / 'energy_distributions.png', dpi=150)
    plt.close(fig)


def plot_mining_timeline(
    by_type: Dict[str, List[Dict]],
    output_dir: Path,
):
    """Cumulative blocks over time per miner type."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {'cpu': '#4C72B0', 'cpu-filtered': '#DD8452'}

    for mtype in sorted(by_type.keys()):
        blocks = by_type[mtype]
        times = sorted(
            b['mining_time'] for b in blocks if b.get('mining_time')
        )
        if not times:
            continue
        # Cumulative: times are absolute timestamps; convert to minutes
        # from first block
        if times:
            cumulative = list(range(1, len(times) + 1))
            # Normalize to minutes from start
            t0 = times[0]
            minutes = [(t - t0) / 60 for t in times]
            ax.plot(
                minutes, cumulative,
                label=mtype, color=colors.get(mtype, '#999'),
                marker='.', markersize=4,
            )

    ax.set_xlabel('Time (minutes from first block)')
    ax.set_ylabel('Cumulative Blocks')
    ax.set_title('Mining Timeline')
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / 'mining_timeline.png', dpi=150)
    plt.close(fig)


def print_summary(
    data: Dict[str, Any],
    by_type: Dict[str, List[Dict]],
    greedy_by_type: Dict[str, List[float]],
    all_mined_greedy: List[float],
    random_greedy: List[float],
):
    """Print console summary with statistics."""
    difficulty = data.get('difficulty_energy', 0)
    diff_factor = energy_to_difficulty(difficulty)

    print("\n" + "=" * 60)
    print("VISUALIZATION SUMMARY")
    print("=" * 60)
    print(f"Difficulty energy: {difficulty:.1f}")
    print(f"Difficulty factor: {diff_factor:.3f}")
    print(f"Duration: {data.get('duration_spec', '?')}")

    for mtype in sorted(by_type.keys()):
        blocks = by_type[mtype]
        greedy = greedy_by_type[mtype]
        sa_energies = [b['energy'] for b in blocks]

        print(f"\n--- {mtype.upper()} ---")
        print(f"  Blocks found: {len(blocks)}")
        ts = data['per_type_stats'].get(mtype, {})
        total_sec = ts.get('total_time_seconds', 0)
        if total_sec > 0:
            rate = len(blocks) / (total_sec / 60)
            print(f"  Rate: {rate:.3f} blocks/min")
        if sa_energies:
            print(
                f"  SA energy:  mean={np.mean(sa_energies):.1f}, "
                f"std={np.std(sa_energies):.1f}",
            )
        if greedy:
            print(
                f"  Greedy:     mean={np.mean(greedy):.1f}, "
                f"std={np.std(greedy):.1f}",
            )

    # Correlation: greedy vs SA for mined blocks
    all_sa = []
    all_greedy_flat = []
    for mtype in by_type:
        for b, g in zip(by_type[mtype], greedy_by_type[mtype]):
            all_sa.append(b['energy'])
            all_greedy_flat.append(g)

    print("\n--- CORRELATION ---")
    if len(all_greedy_flat) >= 3:
        r, p = scipy_stats.pearsonr(all_greedy_flat, all_sa)
        print(f"  Pearson r (greedy vs SA, mined): {r:.3f} (p={p:.2e})")
    else:
        print("  Not enough blocks for correlation")

    # KS test: mined vs random greedy
    print("\n--- KS TEST: MINED vs RANDOM GREEDY ---")
    if all_mined_greedy and random_greedy:
        ks_stat, ks_p = scipy_stats.ks_2samp(
            all_mined_greedy, random_greedy,
        )
        print(f"  KS statistic: {ks_stat:.3f}")
        print(f"  p-value: {ks_p:.2e}")
        if ks_p < 0.05:
            print(
                "  Conclusion: Greedy distributions DIFFER "
                "(greedy has predictive value)",
            )
        else:
            print(
                "  Conclusion: No significant difference "
                "(greedy does NOT predict minable nonces)",
            )
    else:
        print("  Not enough data for KS test")

    print("=" * 60)


def run(args) -> int:
    """Main analysis pipeline."""
    data = load_benchmark(args.input)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_type = extract_block_details(data)
    total_blocks = sum(len(v) for v in by_type.values())
    if total_blocks == 0:
        print("No block_details found in benchmark JSON.")
        print("Re-run benchmark with updated code to include nonces.")
        return 1

    print(f"Loaded {total_blocks} blocks from {args.input}")

    # Build topology cache for greedy descent
    topo = DEFAULT_TOPOLOGY
    nodes = sorted(topo.graph.nodes)
    edges = list(topo.graph.edges)
    cache = IsingTopologyCache(nodes, edges)
    print("Topology cache built")

    # Compute greedy energies for mined nonces
    greedy_by_type: Dict[str, List[float]] = {}
    all_mined_greedy: List[float] = []
    for mtype, blocks in by_type.items():
        nonces = [b['nonce'] for b in blocks]
        greedy = compute_greedy_energies(nonces, cache)
        greedy_by_type[mtype] = greedy
        all_mined_greedy.extend(greedy)
        print(f"  {mtype}: {len(greedy)} greedy energies computed")

    # Random control group (same count as mined blocks)
    n_random = max(len(all_mined_greedy), 50)
    random_nonces = generate_random_nonces(n_random)
    random_greedy = compute_greedy_energies(random_nonces, cache)
    print(f"  random: {len(random_greedy)} control energies computed")

    # Generate plots
    print("\nGenerating plots...")
    plot_performance_comparison(data, by_type, output_dir)
    print("  performance_comparison.png")

    plot_greedy_vs_sa(by_type, greedy_by_type, output_dir)
    print("  greedy_vs_sa_mined.png")

    plot_greedy_mined_vs_random(
        all_mined_greedy, random_greedy, output_dir,
    )
    print("  greedy_mined_vs_random.png")

    difficulty = data.get('difficulty_energy', -14700)
    plot_energy_distributions(
        by_type, greedy_by_type, difficulty, output_dir,
    )
    print("  energy_distributions.png")

    plot_mining_timeline(by_type, output_dir)
    print("  mining_timeline.png")

    # Console summary
    print_summary(
        data, by_type, greedy_by_type,
        all_mined_greedy, random_greedy,
    )

    return 0


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze benchmark_prefilter_mining.py output',
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help='Benchmark JSON file to analyze',
    )
    parser.add_argument(
        '--output', '-o', default='.',
        help='Output directory for plots (default: current dir)',
    )
    args = parser.parse_args()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
