#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Sweep benchmark: CUDA SA vs Gibbs across num_sweeps.

Runs both kernels on the same set of Ising models at each
sweep count, measuring throughput (models/s) and solution
quality (average energy). Produces two comparison plots.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dwave_topologies import DEFAULT_TOPOLOGY
from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
)
from tools.baseline_utils import get_gpu_info


NUM_READS = 128
DEFAULT_SWEEP_VALUES = [128, 256, 512, 1024, 2048, 4096]


def generate_models(
    num_models: int,
) -> tuple:
    """Generate distinct Ising models from DEFAULT_TOPOLOGY.

    Args:
        num_models: Number of models to generate.

    Returns:
        (h_list, J_list) where each is a list of dicts.
    """
    topo = DEFAULT_TOPOLOGY
    nodes = list(topo.graph.nodes)
    edges = list(topo.graph.edges)
    h_list, J_list = [], []
    for seed in range(num_models):
        h, J = generate_ising_model_from_nonce(
            seed, nodes, edges,
            h_values=[-1.0, 0.0, 1.0],
        )
        h_list.append(h)
        J_list.append(J)
    return h_list, J_list


def collect_energies(results):
    """Extract per-model average energies from SampleSets.

    Args:
        results: List of dimod.SampleSet.

    Returns:
        List of per-model mean energies.
    """
    return [
        float(np.mean(ss.record.energy))
        for ss in results
    ]


def bench_sampler(sampler, h_list, J_list, sweeps):
    """Run sampler and return (wall_time, per_model_energies).

    Args:
        sampler: CudaSASampler or CudaGibbsSampler instance.
        h_list: List of h dicts.
        J_list: List of J dicts.
        sweeps: Number of sweeps.

    Returns:
        (elapsed_seconds, per_model_avg_energies).
    """
    start = time.perf_counter()
    results = sampler.sample_ising(
        h=h_list, J=J_list,
        num_reads=NUM_READS, num_sweeps=sweeps,
    )
    elapsed = time.perf_counter() - start
    energies = collect_energies(results)
    return elapsed, energies


def run_sweep(h_list, J_list, sweep_values):
    """Run SA then Gibbs at each sweep count.

    For each sweep value: create SA, run all models, close;
    then create Gibbs, run all models, close. Same models
    reused across all sweep values.

    Args:
        h_list: List of h dicts.
        J_list: List of J dicts.
        sweep_values: List of num_sweeps to test.

    Returns:
        Dict with 'sa' and 'gibbs' results.
    """
    from GPU.cuda_sa import CudaSASampler
    from GPU.cuda_gibbs_sa import CudaGibbsSampler

    n = len(h_list)
    results = {
        "sa": {
            "models_per_sec": [],
            "avg_energy": [],
            "per_model_energies": [],
        },
        "gibbs": {
            "models_per_sec": [],
            "avg_energy": [],
            "per_model_energies": [],
        },
    }

    for sweeps in sweep_values:
        print(f"--- num_sweeps={sweeps} ---")

        # SA
        sa = CudaSASampler()
        elapsed, energies = bench_sampler(
            sa, h_list, J_list, sweeps,
        )
        sa.close()
        mps = n / elapsed
        avg_e = float(np.mean(energies))
        results["sa"]["models_per_sec"].append(mps)
        results["sa"]["avg_energy"].append(avg_e)
        results["sa"]["per_model_energies"].append(energies)
        print(
            f"  SA:    {elapsed:6.3f}s  "
            f"{mps:7.2f} models/s  "
            f"avg_E={avg_e:.1f}"
        )

        # Gibbs
        gibbs = CudaGibbsSampler(
            update_mode="gibbs", parallel=True,
        )
        elapsed, energies = bench_sampler(
            gibbs, h_list, J_list, sweeps,
        )
        mps = n / elapsed
        avg_e = float(np.mean(energies))
        results["gibbs"]["models_per_sec"].append(mps)
        results["gibbs"]["avg_energy"].append(avg_e)
        results["gibbs"]["per_model_energies"].append(energies)
        print(
            f"  Gibbs: {elapsed:6.3f}s  "
            f"{mps:7.2f} models/s  "
            f"avg_E={avg_e:.1f}"
        )

    return results


def plot_throughput(
    sweep_values, sa_mps, gibbs_mps,
    gpu_name, output_dir,
):
    """Plot models/s vs num_sweeps for both kernels.

    Args:
        sweep_values: X-axis sweep counts.
        sa_mps: SA models/s values.
        gibbs_mps: Gibbs models/s values.
        gpu_name: GPU name for title.
        output_dir: Directory to save plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        sweep_values, sa_mps,
        'b-o', label='SA (1 SM/model)', linewidth=2,
    )
    ax.plot(
        sweep_values, gibbs_mps,
        'r-s', label='Gibbs (4 SMs/model)', linewidth=2,
    )
    ax.set_xscale('log', base=2)
    ax.set_xlabel('num_sweeps')
    ax.set_ylabel('Models/s')
    ax.set_title(
        f'Throughput: SA vs Gibbs\n'
        f'{gpu_name}, num_reads={NUM_READS}'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sweep_values)
    ax.set_xticklabels([str(s) for s in sweep_values])
    fig.tight_layout()
    path = output_dir / "models_per_sec_vs_sweeps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def _energy_stats(per_model_energies):
    """Compute mean, std, min, max per sweep point.

    Args:
        per_model_energies: List of lists, one per sweep
            value, each containing per-model mean energies.

    Returns:
        (means, stds, mins, maxs) as numpy arrays.
    """
    means, stds, mins, maxs = [], [], [], []
    for energies in per_model_energies:
        arr = np.array(energies)
        means.append(arr.mean())
        stds.append(arr.std())
        mins.append(arr.min())
        maxs.append(arr.max())
    return (
        np.array(means),
        np.array(stds),
        np.array(mins),
        np.array(maxs),
    )


def _plot_energy_series(ax, x, stats, color, marker, label):
    """Draw one series with shaded variance bands.

    Light band: min/max range across models.
    Dark band: +/- 1 std dev.
    Solid line: mean.

    Args:
        ax: Matplotlib axes.
        x: X-axis values.
        stats: (means, stds, mins, maxs) tuple.
        color: Line/marker color.
        marker: Marker style.
        label: Legend label.
    """
    means, stds, mins, maxs = stats
    ax.fill_between(
        x, mins, maxs,
        alpha=0.1, color=color,
    )
    ax.fill_between(
        x, means - stds, means + stds,
        alpha=0.25, color=color,
    )
    ax.plot(
        x, means,
        color=color, marker=marker,
        linewidth=2, label=label,
    )


def plot_energy(
    sweep_values, sa_data, gibbs_data,
    gpu_name, output_dir,
):
    """Plot average energy vs num_sweeps with shaded bands.

    Light band shows min/max across models.
    Dark band shows +/- 1 std dev across models.

    Args:
        sweep_values: X-axis sweep counts.
        sa_data: Dict with 'per_model_energies' key.
        gibbs_data: Dict with 'per_model_energies' key.
        gpu_name: GPU name for title.
        output_dir: Directory to save plot.
    """
    sa_stats = _energy_stats(sa_data["per_model_energies"])
    gibbs_stats = _energy_stats(
        gibbs_data["per_model_energies"],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_energy_series(
        ax, sweep_values, sa_stats,
        'b', 'o', 'SA (1 SM/model)',
    )
    _plot_energy_series(
        ax, sweep_values, gibbs_stats,
        'r', 's', 'Gibbs (4 SMs/model)',
    )
    ax.set_xscale('log', base=2)
    ax.set_xlabel('num_sweeps')
    ax.set_ylabel('Average Energy')
    ax.set_title(
        f'Solution Quality: SA vs Gibbs\n'
        f'{gpu_name}, num_reads={NUM_READS}'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sweep_values)
    ax.set_xticklabels([str(s) for s in sweep_values])
    ax.annotate(
        'lower is better',
        xy=(0.02, 0.02),
        xycoords='axes fraction',
        fontsize=10, fontstyle='italic',
        color='gray',
    )
    ax.annotate(
        'light band = min/max, dark band = \u00b11 std',
        xy=(0.98, 0.02),
        xycoords='axes fraction',
        fontsize=9, fontstyle='italic',
        color='gray', ha='right',
    )
    fig.tight_layout()
    path = output_dir / "avg_energy_vs_sweeps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Sweep benchmark: SA vs Gibbs across "
            "num_sweeps values"
        ),
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU device index (default: 0)',
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='benchmarks/sweep',
        help='Output directory (default: benchmarks/sweep)',
    )
    parser.add_argument(
        '--num-models', type=int, default=None,
        help='Number of Ising models (default: 4 * SM count)',
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Dump raw results to JSON',
    )
    return parser.parse_args()


def main():
    """Run sweep benchmark and generate plots."""
    args = parse_args()

    num_sms, gpu_name = get_gpu_info()
    print(f"GPU: {gpu_name} ({num_sms} SMs)")

    num_models = args.num_models if args.num_models else 4 * num_sms
    sweep_values = DEFAULT_SWEEP_VALUES
    print(
        f"Config: {num_models} models, "
        f"num_reads={NUM_READS}, "
        f"sweeps={sweep_values}"
    )

    h_list, J_list = generate_models(num_models)
    print(
        f"Generated {num_models} Ising models "
        f"({len(h_list[0])} vars, "
        f"{len(J_list[0])} couplings each)"
    )

    data = run_sweep(h_list, J_list, sweep_values)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_throughput(
        sweep_values,
        data["sa"]["models_per_sec"],
        data["gibbs"]["models_per_sec"],
        gpu_name, output_dir,
    )
    plot_energy(
        sweep_values,
        data["sa"], data["gibbs"],
        gpu_name, output_dir,
    )

    if args.json:
        json_data = {
            "gpu": gpu_name,
            "num_sms": num_sms,
            "num_reads": NUM_READS,
            "num_models": num_models,
            "sweep_values": sweep_values,
            "sa": data["sa"],
            "gibbs": data["gibbs"],
        }
        json_path = output_dir / "sweep_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
