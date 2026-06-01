#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Sweep x Reads grid: streaming pipeline energy measurement.

Pre-generates N Ising models, then streams them through both
SA and Gibbs samplers at different (sweeps, reads) settings.
Collects per-model min energy and counts how many "blocks"
each config would have mined at every threshold.

Uses sample_ising_streaming() — the same code path as
production mine_block(). Results show the full success
curve from a single run per config.

Usage:
    # Quick grid (~5-10 min)
    python tools/sweep_reads_grid.py --quick

    # Full grid
    python tools/sweep_reads_grid.py --num-models 200

    # Custom sweep/read combos
    python tools/sweep_reads_grid.py \
        --sweeps 256,512,1024 --reads 64,128,256,512
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.ising_model import IsingModel
from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
    ising_nonce_from_block,
)
from shared.energy_utils import calc_energy_range
from tools.baseline_utils import load_baseline_topology

DEFAULT_SWEEPS = [256, 512, 768, 1024, 1536, 2048, 4096]
DEFAULT_READS = [64, 128, 256, 512, 1024, 2048]
DEFAULT_THRESHOLDS = list(range(-14500, -15125, -25))
DEFAULT_NUM_MODELS = 100

# SA kernel uses 256 threads per block (one thread per read)
SA_MAX_READS = 256

QUICK_SWEEPS = [256, 512, 1024]
QUICK_READS = [64, 128, 256]
QUICK_NUM_MODELS = 50


def generate_models(
    nodes: List[int],
    edges: list,
    num_models: int,
    h_values: List[float],
    seed: int = 42,
) -> List[IsingModel]:
    """Pre-generate deterministic IsingModels.

    Same models are reused across all configs so the only
    variable is the sampler/sweep/read setting.
    """
    rng = random.Random(seed)
    # Use a fake block context — we just need random
    # but reproducible Ising problems.
    prev_hash = rng.randbytes(32)
    miner_id = "grid-bench"
    cur_index = 1

    models = []
    for _ in range(num_models):
        salt = rng.randbytes(32)
        nonce = ising_nonce_from_block(
            prev_hash, miner_id, cur_index, salt,
        )
        h, J = generate_ising_model_from_nonce(
            nonce, nodes, edges, h_values=h_values,
        )
        models.append(IsingModel(
            h=h, J=J, nonce=nonce, salt=salt,
        ))
    return models


def _extract_model_result(ss) -> Dict:
    """Extract top-5 energies and diversity from a SampleSet."""
    from shared.quantum_proof_of_work import (
        calculate_diversity,
    )

    energies = np.array(ss.record.energy)
    top5_idx = np.argsort(energies)[:5]
    top5 = [float(energies[i]) for i in top5_idx]

    # Diversity among top-5 solutions
    samples = ss.record.sample
    top5_solutions = [list(samples[i]) for i in top5_idx]
    diversity = calculate_diversity(top5_solutions)

    return {
        'top5': top5,
        'diversity': round(diversity, 4),
    }


def run_streaming_config(
    sampler,
    models: List[IsingModel],
    sweeps: int,
    reads: int,
    num_kernels: Optional[int] = None,
) -> Dict:
    """Stream models through sampler, collect per-model results.

    Returns:
        Dict with per-model top-5 energies, diversity,
        runtime, and throughput.
    """
    start = time.time()
    model_results = []

    stream = sampler.sample_ising_streaming(
        iter(models),
        num_reads=reads,
        num_sweeps=sweeps,
        num_sweeps_per_beta=1,
        num_kernels=num_kernels,
        poll_timeout=300.0,
    )

    for model, ss in stream:
        model_results.append(_extract_model_result(ss))

    elapsed = time.time() - start

    return {
        'num_models': len(model_results),
        'runtime_s': round(elapsed, 2),
        'models_per_sec': round(
            len(model_results) / max(elapsed, 0.01), 2,
        ),
        'model_results': model_results,
    }


def count_blocks(
    model_results: List[Dict],
    thresholds: List[float],
) -> Dict[str, Dict]:
    """Count how many models would have mined a block.

    Uses the best (lowest) energy from each model's top5.
    """
    results = {}
    n = len(model_results)
    for t in thresholds:
        hits = sum(
            1 for m in model_results if m['top5'][0] <= t
        )
        results[str(t)] = {
            'blocks': hits,
            'rate': round(hits / n, 4) if n else 0,
        }
    return results


def run_grid(
    sweeps_range: List[int],
    reads_range: List[int],
    thresholds: List[float],
    num_models: int,
    h_values: List[float],
    topology: Optional[str],
) -> Dict:
    """Run the full streaming grid measurement."""
    from GPU.cuda_sa import CudaSASampler
    from GPU.cuda_gibbs_sa import CudaGibbsSampler

    nodes, edges, topo_desc = load_baseline_topology(
        topology_arg=topology,
    )

    sa_min, sa_knee, sa_max = calc_energy_range(
        num_nodes=len(nodes), num_edges=len(edges),
        h_values=tuple(h_values),
    )

    # Build config list: SA + Gibbs at each (sweeps, reads)
    # SA kernel is capped at 256 reads (one thread per read,
    # 256 threads per block).
    configs = []
    for sw in sweeps_range:
        for rd in reads_range:
            if rd <= SA_MAX_READS:
                configs.append({
                    'label': f'SA sw={sw} rd={rd}',
                    'sampler_type': 'sa',
                    'sweeps': sw,
                    'reads': rd,
                })
            configs.append({
                'label': f'Gibbs sw={sw} rd={rd}',
                'sampler_type': 'gibbs',
                'sweeps': sw,
                'reads': rd,
            })

    print("=" * 60)
    print("Sweep x Reads Streaming Grid")
    print("=" * 60)
    print(f"Topology: {topo_desc}")
    print(f"Models: {num_models} (identical across configs)")
    print(f"Sweeps: {sweeps_range}")
    print(f"Reads: {reads_range}")
    print(f"Thresholds: {thresholds}")
    sa_count = sum(
        1 for c in configs if c['sampler_type'] == 'sa'
    )
    gibbs_count = len(configs) - sa_count
    print(f"Configs: {len(configs)} "
          f"({sa_count} SA + {gibbs_count} Gibbs, "
          f"SA capped at {SA_MAX_READS} reads)")
    print()

    # Pre-generate models
    print("Generating models...", end=" ", flush=True)
    models = generate_models(
        nodes, edges, num_models, h_values,
    )
    print(f"{len(models)} models ready")
    print()

    # Create one sampler per type — reuse across configs.
    # max_sms defaults to device SM count, so streaming
    # saturates the GPU automatically.
    print("Initializing SA sampler...", end=" ", flush=True)
    sa_sampler = CudaSASampler()
    print(f"ready ({sa_sampler.max_sms} SMs)")
    print("Initializing Gibbs sampler...", end=" ", flush=True)
    gibbs_sampler = CudaGibbsSampler()
    print(f"ready ({gibbs_sampler.max_sms} SMs)")
    print()

    grid_data = []

    for i, cfg in enumerate(configs):
        sampler = (
            sa_sampler if cfg['sampler_type'] == 'sa'
            else gibbs_sampler
        )

        print(
            f"[{i + 1}/{len(configs)}] "
            f"{cfg['label']:30s} ... ",
            end="", flush=True,
        )

        result = run_streaming_config(
            sampler, models,
            cfg['sweeps'], cfg['reads'],
        )

        # Reset sampler state for next config
        sampler.close()

        mr = result['model_results']
        blocks = count_blocks(mr, thresholds)

        # Extract per-model best energies for summary stats
        best_per_model = [m['top5'][0] for m in mr]
        mean_div = float(np.mean(
            [m['diversity'] for m in mr],
        ))

        # Summary line
        t_mid = str(thresholds[len(thresholds) // 2])
        mid_blocks = blocks[t_mid]['blocks']
        best_e = min(best_per_model)
        print(
            f"{result['models_per_sec']:6.1f} m/s  "
            f"best={best_e:8.0f}  "
            f"div={mean_div:.3f}  "
            f"blocks@{t_mid}="
            f"{mid_blocks}/{result['num_models']}  "
            f"({result['runtime_s']:.1f}s)"
        )

        entry = {
            'label': cfg['label'],
            'sampler_type': cfg['sampler_type'],
            'sweeps': cfg['sweeps'],
            'reads': cfg['reads'],
            'runtime_s': result['runtime_s'],
            'models_per_sec': result['models_per_sec'],
            'best_energy': round(best_e, 1),
            'mean_min_energy': round(
                float(np.mean(best_per_model)), 1,
            ),
            'std_min_energy': round(
                float(np.std(best_per_model)), 1,
            ),
            'mean_diversity': round(mean_div, 4),
            'blocks_at_threshold': blocks,
            'models': [
                {
                    'top5': [round(e, 1) for e in m['top5']],
                    'diversity': m['diversity'],
                }
                for m in mr
            ],
        }
        grid_data.append(entry)

    return {
        'timestamp': int(time.time()),
        'topology': topo_desc,
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'h_values': h_values,
        'num_models': num_models,
        'sweeps_range': sweeps_range,
        'reads_range': reads_range,
        'thresholds': thresholds,
        'sa_reference': {
            'min_energy': round(sa_min, 1),
            'knee_energy': round(sa_knee, 1),
            'max_energy': round(sa_max, 1),
        },
        'grid_data': grid_data,
    }


def print_tables(results: Dict) -> None:
    """Print blocks-mined grids for SA and Gibbs."""
    grid = results['grid_data']
    thresholds = results['thresholds']
    sweeps_range = results['sweeps_range']
    reads_range = results['reads_range']
    num_models = results['num_models']

    for sampler_type in ('sa', 'gibbs'):
        name = 'SA' if sampler_type == 'sa' else 'Gibbs'
        print(f"\n{'=' * 60}")
        print(f"  {name} — blocks mined / {num_models} models")
        print(f"{'=' * 60}")

        for threshold in thresholds:
            t_str = str(threshold)
            print(f"\n  E <= {threshold:.0f}:")
            sw_rd = 'sw\\rd'
            header = f"  {sw_rd:>8}"
            for rd in reads_range:
                header += f" {rd:>6}"
            print(header)
            print("  " + "-" * (len(header) - 2))

            for sw in sweeps_range:
                row = f"  {sw:>8}"
                for rd in reads_range:
                    entry = next(
                        (e for e in grid
                         if e['sampler_type'] == sampler_type
                         and e['sweeps'] == sw
                         and e['reads'] == rd),
                        None,
                    )
                    if entry is None:
                        row += f" {'—':>6}"
                    else:
                        b = entry['blocks_at_threshold']
                        row += f" {b[t_str]['blocks']:>6}"
                row += f"  /{num_models}"
                print(row)

    # Throughput table
    print(f"\n{'=' * 60}")
    print("  Throughput (models/sec)")
    print(f"{'=' * 60}")
    for sampler_type in ('sa', 'gibbs'):
        name = 'SA' if sampler_type == 'sa' else 'Gibbs'
        print(f"\n  {name}:")
        sw_rd = 'sw\\rd'
        header = f"  {sw_rd:>8}"
        for rd in reads_range:
            header += f" {rd:>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for sw in sweeps_range:
            row = f"  {sw:>8}"
            for rd in reads_range:
                entry = next(
                    (e for e in grid
                     if e['sampler_type'] == sampler_type
                     and e['sweeps'] == sw
                     and e['reads'] == rd),
                    None,
                )
                if entry is None:
                    row += f" {'—':>6}"
                else:
                    row += f" {entry['models_per_sec']:>6.1f}"
            print(row)

    # Blocks-per-minute table (success rate × throughput × 60)
    print(f"\n{'=' * 60}")
    print("  Blocks per minute (rate x throughput x 60)")
    print(f"{'=' * 60}")
    t_mid = str(thresholds[len(thresholds) // 2])
    print(f"  At threshold E <= {t_mid}:")
    for sampler_type in ('sa', 'gibbs'):
        name = 'SA' if sampler_type == 'sa' else 'Gibbs'
        print(f"\n  {name}:")
        sw_rd = 'sw\\rd'
        header = f"  {sw_rd:>8}"
        for rd in reads_range:
            header += f" {rd:>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for sw in sweeps_range:
            row = f"  {sw:>8}"
            for rd in reads_range:
                entry = next(
                    (e for e in grid
                     if e['sampler_type'] == sampler_type
                     and e['sweeps'] == sw
                     and e['reads'] == rd),
                    None,
                )
                if entry is None:
                    row += f" {'—':>8}"
                else:
                    b = entry['blocks_at_threshold']
                    rate = b[t_mid]['rate']
                    bpm = rate * entry['models_per_sec'] * 60
                    row += f" {bpm:>8.2f}"
            print(row)


# --- SA vs Gibbs comparison: constants and helpers ---

_CMP_SA_COLOR = '#2196F3'
_CMP_GIBBS_COLORS = [
    '#FF7F0E', '#2CA02C', '#D62728',
    '#9467BD', '#8C564B', '#E377C2',
]
_CMP_GIBBS_MARKERS = ['s', '^', 'v', 'D', 'P', 'X']


def _entry_min_energies(entry):
    """Extract sorted min-energy list from a grid entry.

    Handles both formats: ``min_energies`` (flat list saved
    in JSON) and ``models`` (list of dicts with ``top5``
    produced by a live run).
    """
    if 'min_energies' in entry:
        return sorted(entry['min_energies'])
    return sorted(m['top5'][0] for m in entry['models'])


def _get_plt():
    """Import matplotlib with Agg backend, or None."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return None


def _best_gibbs_gap(sa_entry, gibbs_entries):
    """Mean absolute rate gap between SA and closest Gibbs.

    Lower = Gibbs is closer to matching SA performance.
    """
    thresholds = list(sa_entry['blocks_at_threshold'].keys())
    sa_rates = [
        sa_entry['blocks_at_threshold'][t]['rate']
        for t in thresholds
    ]

    best = float('inf')
    for g in gibbs_entries:
        g_rates = [
            g['blocks_at_threshold'][t]['rate']
            for t in thresholds
        ]
        gap = sum(
            abs(s - g) for s, g in zip(sa_rates, g_rates)
        ) / len(thresholds)
        best = min(best, gap)
    return best


def _select_comparison_pairs(
    grid_data, sweeps_range, reads_range,
):
    """Select up to 9 SA configs with Gibbs read-sweep pairs.

    Picks 3 sweep x 3 read levels from SA entries. For each,
    pairs with all Gibbs configs at the **same sweeps** across
    the full reads range. Sorted by best-Gibbs gap ascending
    so the closest matches appear first (top-left in the grid).

    Returns list of (sa_entry, [gibbs_entries]).
    """
    sa_by_key = {
        (e['sweeps'], e['reads']): e
        for e in grid_data if e['sampler_type'] == 'sa'
    }
    gibbs_by_key = {
        (e['sweeps'], e['reads']): e
        for e in grid_data if e['sampler_type'] == 'gibbs'
    }

    sa_sweeps = sorted(set(k[0] for k in sa_by_key))
    sa_reads = sorted(set(k[1] for k in sa_by_key))

    def _sample3(lst):
        if len(lst) <= 3:
            return lst
        return [lst[0], lst[len(lst) // 2], lst[-1]]

    sel_sw = _sample3(sa_sweeps)
    sel_rd = _sample3(sa_reads)

    pairs = []
    for sw in sel_sw:
        for rd in sel_rd:
            sa_entry = sa_by_key.get((sw, rd))
            if sa_entry is None:
                continue

            neighbors = [
                gibbs_by_key[(sw, r)]
                for r in reads_range
                if (sw, r) in gibbs_by_key
            ]

            pairs.append((sa_entry, neighbors))

    pairs.sort(key=lambda p: _best_gibbs_gap(p[0], p[1]))

    return pairs


def _draw_threshold_subplot(
    ax, sa_entry, gibbs_entries, thresholds, y_fn,
):
    """Draw one SA-vs-Gibbs subplot for threshold data.

    Args:
        y_fn: Callable(entry, threshold_str) -> float.
    """
    sa_ys = [y_fn(sa_entry, str(t)) for t in thresholds]
    lbl = (
        f"SA sw={sa_entry['sweeps']} "
        f"rd={sa_entry['reads']}"
    )
    ax.plot(
        thresholds, sa_ys,
        color=_CMP_SA_COLOR, linewidth=2.5,
        marker='o', markersize=5, label=lbl, zorder=3,
    )

    for gi, ge in enumerate(gibbs_entries):
        ys = [y_fn(ge, str(t)) for t in thresholds]
        ax.plot(
            thresholds, ys,
            color=_CMP_GIBBS_COLORS[gi % 6],
            marker=_CMP_GIBBS_MARKERS[gi % 6],
            markersize=3, linewidth=1.2,
            label=(
                f"G sw={ge['sweeps']} rd={ge['reads']}"
            ),
        )

    ax.set_title(lbl, fontsize=9)
    ax.legend(fontsize=5, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()


def plot_blocks_vs_threshold_cmp(results, output_path):
    """3x3: blocks mined vs threshold, SA + Gibbs neighbors."""
    plt = _get_plt()
    if plt is None:
        return

    pairs = _select_comparison_pairs(
        results['grid_data'],
        results['sweeps_range'],
        results['reads_range'],
    )
    if not pairs:
        return

    n = len(pairs)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    thresholds = results['thresholds']
    nm = results['num_models']

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 5 * nrows),
    )
    axes = np.atleast_1d(axes).flatten()

    def y_fn(entry, t_str):
        return entry['blocks_at_threshold'][t_str][
            'blocks'
        ]

    for idx, (sa, gibbs) in enumerate(pairs):
        _draw_threshold_subplot(
            axes[idx], sa, gibbs, thresholds, y_fn,
        )
        axes[idx].set_xlabel(
            'Energy Threshold', fontsize=8,
        )
        axes[idx].set_ylabel(
            f'Blocks / {nm}', fontsize=8,
        )

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        'Blocks Mined: SA Reference + '
        'Gibbs Neighbors',
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {output_path}")


def plot_success_probability_cmp(results, output_path):
    """3x3: P(success) vs threshold, SA + Gibbs neighbors."""
    plt = _get_plt()
    if plt is None:
        return

    pairs = _select_comparison_pairs(
        results['grid_data'],
        results['sweeps_range'],
        results['reads_range'],
    )
    if not pairs:
        return

    n = len(pairs)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    thresholds = results['thresholds']
    nm = max(results['num_models'], 1)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 5 * nrows),
    )
    axes = np.atleast_1d(axes).flatten()

    def y_fn(entry, t_str):
        rate = entry['blocks_at_threshold'][t_str][
            'rate'
        ]
        return rate if rate > 0 else float('nan')

    for idx, (sa, gibbs) in enumerate(pairs):
        _draw_threshold_subplot(
            axes[idx], sa, gibbs, thresholds, y_fn,
        )
        axes[idx].set_xlabel(
            'Energy Threshold', fontsize=8,
        )
        axes[idx].set_ylabel(
            'P(success)', fontsize=8,
        )
        axes[idx].set_yscale('log')
        axes[idx].set_ylim(bottom=0.5 / nm)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        'Success Probability: SA Reference + '
        'Gibbs Neighbors',
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {output_path}")


def plot_energy_cdf_cmp(results, output_path):
    """3x3: energy CDF, SA ref + Gibbs neighbors."""
    plt = _get_plt()
    if plt is None:
        return

    pairs = _select_comparison_pairs(
        results['grid_data'],
        results['sweeps_range'],
        results['reads_range'],
    )
    if not pairs:
        return

    n = len(pairs)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    thresholds = results['thresholds']

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 5 * nrows),
    )
    axes = np.atleast_1d(axes).flatten()

    for idx, (sa, gibbs_list) in enumerate(pairs):
        ax = axes[idx]

        sa_e = _entry_min_energies(sa)
        cdf = (
            np.arange(1, len(sa_e) + 1) / len(sa_e)
        )
        lbl = (
            f"SA sw={sa['sweeps']} "
            f"rd={sa['reads']}"
        )
        ax.plot(
            sa_e, cdf,
            color=_CMP_SA_COLOR, linewidth=2.5,
            label=lbl, zorder=3,
        )

        for gi, ge in enumerate(gibbs_list):
            ge_e = _entry_min_energies(ge)
            cdf_g = (
                np.arange(1, len(ge_e) + 1)
                / len(ge_e)
            )
            ax.plot(
                ge_e, cdf_g,
                color=_CMP_GIBBS_COLORS[gi % 6],
                linewidth=1.0,
                label=(
                    f"G sw={ge['sweeps']} "
                    f"rd={ge['reads']}"
                ),
            )

        for t in thresholds:
            ax.axvline(
                t, color='gray', alpha=0.2,
                linestyle=':',
            )

        ax.set_title(lbl, fontsize=9)
        ax.set_xlabel('Min Energy', fontsize=8)
        ax.set_ylabel('CDF', fontsize=8)
        ax.legend(fontsize=5, loc='upper left')
        ax.grid(True, alpha=0.3)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        'Energy CDF: SA Reference + '
        'Gibbs Neighbors',
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {output_path}")


def plot_sweep_multiplier(results, output_path):
    """Plot Gibbs sweep multiplier needed to match SA.

    Top: SA sweeps vs Gibbs sweeps needed (with 2x ref line).
    Bottom: residual gap at the best Gibbs match.
    """
    plt = _get_plt()
    if plt is None:
        return

    grid = results['grid_data']
    thresholds = results['thresholds']
    sweeps_range = results['sweeps_range']
    reads_range = results['reads_range']
    sa_max_reads = 256

    sa = {
        (e['sweeps'], e['reads']): e
        for e in grid if e['sampler_type'] == 'sa'
    }
    gb = {
        (e['sweeps'], e['reads']): e
        for e in grid if e['sampler_type'] == 'gibbs'
    }
    shared_reads = [r for r in reads_range if r <= sa_max_reads]

    def _rates(entry):
        return np.array([
            entry['blocks_at_threshold'][str(t)]['rate']
            for t in thresholds
        ])

    fig, (ax_sw, ax_gap) = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={'height_ratios': [2, 1]},
    )

    colors = ['#2196F3', '#FF9800', '#4CAF50']
    markers = ['o', 's', '^']

    for ri, rd in enumerate(shared_reads):
        sa_sws = []
        gb_sws = []
        gaps = []

        for sw_sa in sweeps_range:
            if (sw_sa, rd) not in sa:
                continue
            sa_r = _rates(sa[(sw_sa, rd)])

            best_gap = float('inf')
            best_sw = None
            for sw_g in sweeps_range:
                if (sw_g, rd) not in gb:
                    continue
                gap = np.mean(np.abs(
                    sa_r - _rates(gb[(sw_g, rd)]),
                ))
                if gap < best_gap:
                    best_gap = gap
                    best_sw = sw_g

            if best_sw is not None:
                sa_sws.append(sw_sa)
                gb_sws.append(best_sw)
                gaps.append(best_gap)

        ax_sw.plot(
            sa_sws, gb_sws,
            color=colors[ri % 3],
            marker=markers[ri % 3],
            markersize=6, linewidth=1.5,
            label=f'rd={rd}',
        )
        ax_gap.plot(
            sa_sws, gaps,
            color=colors[ri % 3],
            marker=markers[ri % 3],
            markersize=6, linewidth=1.5,
            label=f'rd={rd}',
        )

    # Reference lines
    sw_arr = np.array(sweeps_range)
    ax_sw.plot(
        sw_arr, sw_arr,
        'k--', alpha=0.3, label='1x (parity)',
    )
    ax_sw.plot(
        sw_arr, sw_arr * 2,
        'k:', alpha=0.3, label='2x',
    )

    ax_sw.set_ylabel('Gibbs sweeps to match SA')
    ax_sw.set_title(
        'Gibbs Sweep Multiplier to Match SA\n'
        '(same reads, best MAE match)',
    )
    ax_sw.legend(fontsize=8)
    ax_sw.grid(True, alpha=0.3)
    ax_sw.set_xscale('log', base=2)
    ax_sw.set_yscale('log', base=2)

    ax_gap.set_xlabel('SA sweeps')
    ax_gap.set_ylabel('Residual gap (MAE)')
    ax_gap.set_title('Residual rate gap at best match')
    ax_gap.legend(fontsize=8)
    ax_gap.grid(True, alpha=0.3)
    ax_gap.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {output_path}")


def plot_results(results: Dict, output_dir: str) -> None:
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    grid = results['grid_data']
    thresholds = results['thresholds']
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Blocks mined vs threshold ---
    # One line per config, x=threshold, y=blocks
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, sampler_type in enumerate(('sa', 'gibbs')):
        ax = axes[ax_idx]
        name = 'SA' if sampler_type == 'sa' else 'Gibbs'

        entries = [
            e for e in grid
            if e['sampler_type'] == sampler_type
        ]

        cmap = plt.cm.viridis
        n_entries = len(entries)

        for i, entry in enumerate(entries):
            blocks_list = []
            for t in thresholds:
                b = entry['blocks_at_threshold'][str(t)]
                blocks_list.append(b['blocks'])

            color = cmap(i / max(n_entries - 1, 1))
            label = (
                f"sw={entry['sweeps']} rd={entry['reads']}"
            )
            ax.plot(
                thresholds, blocks_list,
                marker='o', markersize=4,
                color=color, label=label,
                linewidth=1.5,
            )

        ax.set_xlabel('Energy Threshold')
        ax.set_ylabel(
            f'Blocks Mined / {results["num_models"]}',
        )
        ax.set_title(f'{name}: Blocks Mined vs Threshold')
        ax.legend(fontsize=7, ncol=2, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    plt.tight_layout()
    path = outdir / 'blocks_vs_threshold.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {path}")

    # --- Plot 2: Success probability vs threshold ---
    # Log-scale shows where each config transitions from
    # "usually succeeds" to "rarely succeeds". Guaranteed
    # monotonic because count_blocks uses <= threshold.
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    num_models = results['num_models']

    for ax_idx, sampler_type in enumerate(('sa', 'gibbs')):
        ax = axes[ax_idx]
        name = 'SA' if sampler_type == 'sa' else 'Gibbs'

        entries = [
            e for e in grid
            if e['sampler_type'] == sampler_type
        ]

        cmap = plt.cm.viridis
        n_entries = len(entries)

        for i, entry in enumerate(entries):
            probs = []
            for t in thresholds:
                rate = entry['blocks_at_threshold'][
                    str(t)
                ]['rate']
                probs.append(
                    rate if rate > 0 else float('nan'),
                )

            color = cmap(i / max(n_entries - 1, 1))
            label = (
                f"sw={entry['sweeps']} "
                f"rd={entry['reads']}"
            )
            ax.plot(
                thresholds, probs,
                marker='o', markersize=4,
                color=color, label=label,
                linewidth=1.5,
            )

        ax.set_xlabel('Energy Threshold')
        ax.set_ylabel('P(success)')
        ax.set_title(
            f'{name}: Success Probability vs Threshold',
        )
        ax.legend(fontsize=7, ncol=2, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.5 / max(num_models, 1))

    plt.tight_layout()
    path = outdir / 'success_probability.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {path}")

    # --- Plot 4: Energy CDF overlay ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, sampler_type in enumerate(('sa', 'gibbs')):
        ax = axes[ax_idx]
        name = 'SA' if sampler_type == 'sa' else 'Gibbs'

        entries = [
            e for e in grid
            if e['sampler_type'] == sampler_type
        ]

        cmap = plt.cm.viridis
        n_entries = len(entries)

        for i, entry in enumerate(entries):
            energies = _entry_min_energies(entry)
            cdf = np.arange(1, len(energies) + 1) / len(energies)
            color = cmap(i / max(n_entries - 1, 1))
            label = (
                f"sw={entry['sweeps']} rd={entry['reads']}"
            )
            ax.plot(
                energies, cdf,
                color=color, label=label,
                linewidth=1.2,
            )

        # Mark thresholds
        for t in thresholds:
            ax.axvline(
                t, color='gray', alpha=0.3, linestyle=':',
            )

        ax.set_xlabel('Min Energy per Model')
        ax.set_ylabel('CDF')
        ax.set_title(f'{name}: Energy CDF')
        ax.legend(fontsize=7, ncol=2, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = outdir / 'energy_cdf.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {path}")

    # --- Sweep multiplier analysis ---
    plot_sweep_multiplier(
        results,
        str(outdir / 'sweep_multiplier.png'),
    )

    # --- Comparison plots: SA ref + Gibbs neighbors ---
    plot_blocks_vs_threshold_cmp(
        results, str(outdir / 'blocks_vs_threshold_cmp.png'),
    )
    plot_success_probability_cmp(
        results,
        str(outdir / 'success_probability_cmp.png'),
    )
    plot_energy_cdf_cmp(
        results, str(outdir / 'energy_cdf_cmp.png'),
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sweep x Reads streaming grid",
    )
    parser.add_argument(
        '--sweeps', type=str,
        default=','.join(str(s) for s in DEFAULT_SWEEPS),
        help=f'Sweep counts (default: {DEFAULT_SWEEPS})',
    )
    parser.add_argument(
        '--reads', type=str,
        default=','.join(str(r) for r in DEFAULT_READS),
        help=f'Read counts (default: {DEFAULT_READS})',
    )
    parser.add_argument(
        '--thresholds', type=str,
        default=','.join(
            str(t) for t in DEFAULT_THRESHOLDS
        ),
        help=(
            'Energy thresholds '
            f'(default: {DEFAULT_THRESHOLDS[0]} to '
            f'{DEFAULT_THRESHOLDS[-1]} by 25)'
        ),
    )
    parser.add_argument(
        '--num-models', type=int,
        default=DEFAULT_NUM_MODELS,
        help=f'Models to stream (default: {DEFAULT_NUM_MODELS})',
    )
    parser.add_argument(
        '--output-dir', type=str,
        help='Output directory for JSON + plots',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode: smaller grid + fewer models',
    )
    parser.add_argument(
        '--h-values', type=str, default='-1,0,1',
        help='h values (default: -1,0,1)',
    )
    parser.add_argument(
        '--topology', type=str,
        help='Topology override',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='RNG seed for model generation (default: 42)',
    )
    return parser.parse_args()


def main():
    """Run the streaming grid measurement."""
    args = parse_args()

    h_values = [
        float(v.strip()) for v in args.h_values.split(',')
    ]

    if args.quick:
        sweeps_range = QUICK_SWEEPS
        reads_range = QUICK_READS
        num_models = QUICK_NUM_MODELS
    else:
        sweeps_range = [
            int(v.strip()) for v in args.sweeps.split(',')
        ]
        reads_range = [
            int(v.strip()) for v in args.reads.split(',')
        ]
        num_models = args.num_models

    thresholds = [
        float(v.strip())
        for v in args.thresholds.split(',')
    ]

    results = run_grid(
        sweeps_range=sweeps_range,
        reads_range=reads_range,
        thresholds=thresholds,
        num_models=num_models,
        h_values=h_values,
        topology=args.topology,
    )

    print_tables(results)

    # Save
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ts = int(time.time())
        output_dir = f"sweep_reads_grid_{ts}"

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    json_path = outdir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nOutput directory: {outdir}/")
    print(f"  {json_path}")

    try:
        plot_results(results, output_dir)
    except Exception as e:
        print(f"Plot generation failed: {e}")

    print(f"\nGrid measurement complete — {outdir}/")


if __name__ == '__main__':
    main()
