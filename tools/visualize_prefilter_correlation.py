#!/usr/bin/env python3
"""Visualize greedy descent vs SA energy correlation.

Generates a multi-panel figure with:
1. Scatter plot: greedy vs SA energy with regression line
2. Progressive correlation: Pearson/Spearman r vs greedy pass count
3. Filtering ROC: false negative rate vs rejection threshold
4. Energy distribution: overlapping histograms
5. 2D density heatmap: greedy vs SA with marginal histograms
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from dwave_topologies import DEFAULT_TOPOLOGY
from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
    ising_nonce_from_block,
)
from shared.nonce_prefilter import (
    IsingTopologyCache,
    greedy_descent_energy,
)
from CPU.sa_sampler import SimulatedAnnealingStructuredSampler


def collect_data(
    num_nonces: int,
    sa_sweeps: int,
    sa_reads: int,
    max_passes: int = 3,
):
    """Collect greedy and SA energies for correlation analysis.

    Args:
        num_nonces: Number of nonces to evaluate.
        sa_sweeps: SA num_sweeps parameter.
        sa_reads: SA num_reads parameter.
        max_passes: Maximum greedy passes to test (1..max_passes).

    Returns:
        Tuple of (greedy_by_pass, sa_energies) where
        greedy_by_pass is dict {pass_count: np.array}.
    """
    sampler = SimulatedAnnealingStructuredSampler()
    nodes = sampler.nodes
    edges = sampler.edges
    cache = IsingTopologyCache(nodes, edges)

    prev_hash = random.randbytes(32)
    miner_id = "viz-0"
    cur_index = 1

    greedy_by_pass = {p: [] for p in range(1, max_passes + 1)}
    sa_energies = []

    print(
        f"Collecting data: {num_nonces} nonces, "
        f"SA sweeps={sa_sweeps} reads={sa_reads}",
    )
    t0 = time.perf_counter()

    for i in range(num_nonces):
        salt = random.randbytes(32)
        nonce = ising_nonce_from_block(
            prev_hash, miner_id, cur_index, salt,
        )
        h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

        # Greedy at each pass count
        for p in range(1, max_passes + 1):
            ge = cache.greedy_descent(h, J, num_passes=p)
            greedy_by_pass[p].append(ge)

        # Full SA
        ss = sampler.sample_ising(
            h=h, J=J, num_reads=sa_reads, num_sweeps=sa_sweeps,
        )
        sa_energies.append(float(np.min(ss.record.energy)))

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (num_nonces - i - 1) / rate
            print(
                f"  [{i+1}/{num_nonces}] "
                f"{rate:.1f} nonces/s, ETA {eta:.0f}s",
            )

    elapsed = time.perf_counter() - t0
    print(f"Data collection complete in {elapsed:.1f}s")

    greedy_arrays = {
        p: np.array(v) for p, v in greedy_by_pass.items()
    }
    return greedy_arrays, np.array(sa_energies)


def plot_scatter(ax, greedy, sa, target_energy):
    """Plot 1: Scatter with regression line."""
    from scipy.stats import pearsonr

    ax.scatter(
        greedy, sa, alpha=0.4, s=12, c='steelblue', edgecolors='none',
    )

    # Regression line
    coeffs = np.polyfit(greedy, sa, 1)
    x_line = np.linspace(greedy.min(), greedy.max(), 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), 'r-', lw=1.5)

    r, p = pearsonr(greedy, sa)
    ax.set_xlabel('Greedy Energy (3 passes)')
    ax.set_ylabel('SA Best Energy')
    ax.set_title(f'Greedy vs SA Energy (r={r:.3f})')

    # Target energy line
    ax.axhline(
        y=target_energy, color='orange', linestyle='--',
        lw=1, alpha=0.7, label=f'Target {target_energy}',
    )
    ax.legend(fontsize=8)


def plot_progressive_correlation(
    ax, greedy_by_pass, sa,
):
    """Plot 2: Correlation vs greedy pass count."""
    from scipy.stats import pearsonr, spearmanr

    passes = sorted(greedy_by_pass.keys())
    pearson_vals = []
    spearman_vals = []

    for p in passes:
        r_p, _ = pearsonr(greedy_by_pass[p], sa)
        r_s, _ = spearmanr(greedy_by_pass[p], sa)
        pearson_vals.append(r_p)
        spearman_vals.append(r_s)

    x = np.arange(len(passes))
    width = 0.35
    ax.bar(
        x - width / 2, pearson_vals, width,
        label='Pearson r', color='steelblue',
    )
    ax.bar(
        x + width / 2, spearman_vals, width,
        label='Spearman rho', color='coral',
    )

    ax.set_xlabel('Greedy Passes')
    ax.set_ylabel('Correlation with SA')
    ax.set_title('Progressive Correlation Signal')
    ax.set_xticks(x)
    ax.set_xticklabels(passes)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)

    # Annotate values
    for i, (pv, sv) in enumerate(zip(pearson_vals, spearman_vals)):
        ax.text(
            i - width / 2, pv + 0.02, f'{pv:.2f}',
            ha='center', va='bottom', fontsize=7,
        )
        ax.text(
            i + width / 2, sv + 0.02, f'{sv:.2f}',
            ha='center', va='bottom', fontsize=7,
        )


def plot_roc(ax, greedy, sa, target_energy):
    """Plot 3: Filtering ROC curve."""
    sa_valid = sa < target_energy
    num_valid = np.sum(sa_valid)

    if num_valid == 0:
        ax.text(
            0.5, 0.5, f'No SA nonces below {target_energy}',
            transform=ax.transAxes, ha='center',
        )
        ax.set_title('Filtering ROC (no valid nonces)')
        return

    thresholds = np.linspace(0, 95, 50)
    false_neg_rates = []
    speedups = []
    greedy_sorted = np.sort(greedy)

    for reject_pct in thresholds:
        idx = int(len(greedy) * (1 - reject_pct / 100))
        idx = max(0, min(idx, len(greedy_sorted) - 1))
        threshold = greedy_sorted[idx]

        greedy_pass = greedy <= threshold
        false_neg = np.sum(sa_valid & ~greedy_pass)
        fn_rate = false_neg / num_valid
        false_neg_rates.append(fn_rate)

        kept = np.sum(greedy_pass)
        speedup = len(greedy) / kept if kept > 0 else len(greedy)
        speedups.append(speedup)

    color_fn = 'steelblue'
    color_sp = 'coral'

    ax.plot(
        thresholds, false_neg_rates,
        color=color_fn, lw=2, label='False negative rate',
    )
    ax.set_xlabel('Rejection %')
    ax.set_ylabel('False Negative Rate', color=color_fn)
    ax.tick_params(axis='y', labelcolor=color_fn)
    ax.set_title(f'Filter ROC (target={target_energy})')

    ax2 = ax.twinx()
    ax2.plot(
        thresholds, speedups,
        color=color_sp, lw=2, linestyle='--', label='Speedup',
    )
    ax2.set_ylabel('Speedup Factor', color=color_sp)
    ax2.tick_params(axis='y', labelcolor=color_sp)

    # Find sweet spot: maximum curvature in FN rate curve
    fn_arr = np.array(false_neg_rates)
    if len(fn_arr) >= 3:
        d2 = np.diff(fn_arr, n=2)
        knee_idx = np.argmax(d2) + 1  # offset for second derivative
        sweet_pct = thresholds[knee_idx]
        sweet_fn = fn_arr[knee_idx]
        sweet_sp = speedups[knee_idx]

        ax.axvline(
            x=sweet_pct, color='green', linestyle='-',
            lw=2, alpha=0.7,
        )
        ax.annotate(
            f'Sweet spot: {sweet_pct:.0f}%\n'
            f'FN={sweet_fn:.1%}, {sweet_sp:.1f}x',
            xy=(sweet_pct, sweet_fn),
            xytext=(sweet_pct + 8, sweet_fn + 0.15),
            fontsize=8, color='green', fontweight='bold',
            arrowprops=dict(
                arrowstyle='->', color='green', lw=1.5,
            ),
        )

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)


def plot_energy_distributions(ax, greedy, sa):
    """Plot 4: Overlapping energy histograms."""
    bins_greedy = np.linspace(greedy.min(), greedy.max(), 40)
    bins_sa = np.linspace(sa.min(), sa.max(), 40)

    ax.hist(
        greedy, bins=bins_greedy, alpha=0.5,
        label=f'Greedy (n={len(greedy)})', color='steelblue',
        density=True,
    )
    ax.hist(
        sa, bins=bins_sa, alpha=0.5,
        label=f'SA (n={len(sa)})', color='coral',
        density=True,
    )
    ax.set_xlabel('Energy')
    ax.set_ylabel('Density')
    ax.set_title('Energy Distributions')
    ax.legend(fontsize=8)


def plot_density_heatmap(
    fig, ax, greedy, sa, target_energy,
):
    """Plot 5: 2D density heatmap with marginal histograms."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", 0.6, pad=0.1, sharex=ax)
    ax_right = divider.append_axes("right", 0.6, pad=0.1, sharey=ax)

    # Hide tick labels on marginals
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)

    # Main hexbin
    hb = ax.hexbin(
        greedy, sa, gridsize=30, cmap='inferno',
        mincnt=1,
    )
    fig.colorbar(hb, ax=ax_right, pad=0.15, label='Count')

    ax.axhline(
        y=target_energy, color='cyan', linestyle='--',
        lw=1, alpha=0.8,
    )
    ax.set_xlabel('Greedy Energy (3 passes)')
    ax.set_ylabel('SA Best Energy')
    ax.set_title('2D Density: Greedy vs SA')

    # Marginal histograms
    ax_top.hist(greedy, bins=40, color='steelblue', alpha=0.7)
    ax_right.hist(
        sa, bins=40, orientation='horizontal',
        color='coral', alpha=0.7,
    )


def print_summary(greedy_by_pass, sa, target_energy):
    """Print correlation statistics to console."""
    from scipy.stats import pearsonr, spearmanr

    print("\n" + "=" * 50)
    print("CORRELATION SUMMARY")
    print("=" * 50)

    for p in sorted(greedy_by_pass.keys()):
        g = greedy_by_pass[p]
        r_p, _ = pearsonr(g, sa)
        r_s, _ = spearmanr(g, sa)
        print(
            f"  Pass {p}: Pearson r={r_p:.4f}, "
            f"Spearman rho={r_s:.4f}",
        )

    g3 = greedy_by_pass[max(greedy_by_pass.keys())]
    print(f"\n  Greedy range: [{g3.min():.0f}, {g3.max():.0f}]")
    print(f"  SA range:     [{sa.min():.0f}, {sa.max():.0f}]")

    sa_valid = np.sum(sa < target_energy)
    print(
        f"  SA valid at {target_energy}: "
        f"{sa_valid}/{len(sa)} ({sa_valid/len(sa)*100:.1f}%)",
    )


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize greedy descent vs SA correlation',
    )
    parser.add_argument(
        '--nonces', type=int, default=200,
        help='Number of nonces to evaluate (default: 200)',
    )
    parser.add_argument(
        '--sweeps', type=int, default=1024,
        help='SA num_sweeps (default: 1024)',
    )
    parser.add_argument(
        '--reads', type=int, default=20,
        help='SA num_reads (default: 20)',
    )
    parser.add_argument(
        '--target-energy', type=float, default=-14900.0,
        help='Target energy threshold (default: -14900)',
    )
    parser.add_argument(
        '--output', '-o', type=str,
        default='prefilter_correlation.png',
        help='Output PNG file (default: prefilter_correlation.png)',
    )
    args = parser.parse_args()

    # Collect data
    greedy_by_pass, sa = collect_data(
        num_nonces=args.nonces,
        sa_sweeps=args.sweeps,
        sa_reads=args.reads,
    )

    # Print stats
    print_summary(greedy_by_pass, sa, args.target_energy)

    # Import matplotlib (deferred to avoid import cost if data
    # collection fails)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 10))

    # Layout: 2 rows, 3 columns. Plot 5 (heatmap) is wider.
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)

    max_pass = max(greedy_by_pass.keys())
    g3 = greedy_by_pass[max_pass]

    plot_scatter(ax1, g3, sa, args.target_energy)
    plot_progressive_correlation(ax2, greedy_by_pass, sa)
    plot_roc(ax3, g3, sa, args.target_energy)
    plot_energy_distributions(ax4, g3, sa)
    plot_density_heatmap(fig, ax5, g3, sa, args.target_energy)

    fig.suptitle(
        f'Prefilter Correlation Analysis '
        f'(n={args.nonces}, SA sweeps={args.sweeps})',
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
