#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Gibbs knee calibration tool.

Maps the CUDA Gibbs sampler's energy curve across sweep counts,
reads, and sweeps_per_beta values. Identifies the knee point
(diminishing returns) for each sweeps_per_beta setting and
recommends ADAPT_* parameters for CudaGibbsMiner.

Usage:
    # Quick validation run
    python tools/calibrate_gibbs_curve.py --quick

    # Full calibration
    python tools/calibrate_gibbs_curve.py \
        --output gibbs_knee_calibration.json \
        --plot gibbs_knee_calibration.png
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.quantum_proof_of_work import generate_ising_model_from_nonce
from shared.energy_utils import calc_energy_range
from tools.baseline_utils import load_baseline_topology
from GPU.cuda_gibbs_sa import CudaGibbsSampler

DEFAULT_SWEEPS = [
    64, 128, 256, 512, 768, 1024,
    1536, 2048, 3072, 4096, 6144, 8192,
]
DEFAULT_READS = [64, 128, 200]
DEFAULT_SPB = [1, 2, 4]
DEFAULT_NUM_NONCES = 5
DEFAULT_NUM_MODELS = 12

QUICK_SWEEPS = [256, 1024, 4096]
QUICK_READS = [64]
QUICK_SPB = [1]
QUICK_NUM_NONCES = 2

KNEE_THRESHOLD = 0.20  # 20% of initial improvement rate


def run_single_config(
    sampler: CudaGibbsSampler,
    nodes: List[int],
    edges: list,
    sweeps: int,
    reads: int,
    spb: int,
    num_nonces: int,
    num_models: int,
    h_values: List[float],
) -> Dict:
    """Run one (sweeps, reads, spb) configuration.

    Generates num_nonces random nonces. For each nonce,
    generates num_models Ising problems and samples them.
    Records per-model energies and cross-model min energy.

    Returns:
        Dict with sweep config and energy statistics.
    """
    nonce_min_energies = []
    nonce_avg_energies = []
    all_model_mins = []

    start_time = time.time()

    for _ in range(num_nonces):
        h_list = []
        J_list = []
        for _ in range(num_models):
            nonce = random.randint(0, 2**32 - 1)
            h_i, J_i = generate_ising_model_from_nonce(
                nonce, nodes, edges, h_values=h_values,
            )
            h_list.append(h_i)
            J_list.append(J_i)

        samplesets = sampler.sample_ising(
            h=h_list,
            J=J_list,
            num_reads=reads,
            num_sweeps=sweeps,
            num_sweeps_per_beta=spb,
        )

        model_mins = []
        model_avgs = []
        for ss in samplesets:
            energies = list(ss.record.energy)
            model_mins.append(float(min(energies)))
            model_avgs.append(float(np.mean(energies)))

        nonce_min = min(model_mins)
        nonce_avg = float(np.mean(model_avgs))
        nonce_min_energies.append(nonce_min)
        nonce_avg_energies.append(nonce_avg)
        all_model_mins.extend(model_mins)

    runtime = time.time() - start_time

    return {
        'sweeps': sweeps,
        'reads': reads,
        'sweeps_per_beta': spb,
        'num_nonces': num_nonces,
        'num_models': num_models,
        'runtime_seconds': round(runtime, 2),
        'min_energy_mean': round(
            float(np.mean(nonce_min_energies)), 1,
        ),
        'min_energy_std': round(
            float(np.std(nonce_min_energies)), 1,
        ),
        'min_energy_best': round(
            float(min(nonce_min_energies)), 1,
        ),
        'avg_energy_mean': round(
            float(np.mean(nonce_avg_energies)), 1,
        ),
        'avg_energy_std': round(
            float(np.std(nonce_avg_energies)), 1,
        ),
        'all_model_mins': [round(e, 1) for e in all_model_mins],
    }


def calibrate_gibbs_curve(
    sweeps_range: List[int],
    reads_range: List[int],
    spb_range: List[int],
    num_nonces: int,
    num_models: int,
    h_values: List[float],
    topology: Optional[str],
    timeout_minutes: float,
) -> Dict:
    """Run full Gibbs calibration experiment.

    Args:
        sweeps_range: Sweep counts to test.
        reads_range: Read counts to test.
        spb_range: sweeps_per_beta values to test.
        num_nonces: Nonces per configuration.
        num_models: Models per nonce (parallel).
        h_values: Ising h-field distribution.
        topology: Topology arg or None for default.
        timeout_minutes: Overall time limit.

    Returns:
        Results dict with curve_data, calibration, and
        sa_reference fields.
    """
    nodes, edges, topo_desc = load_baseline_topology(
        topology_arg=topology,
    )

    total_configs = (
        len(sweeps_range) * len(reads_range) * len(spb_range)
    )
    total_calls = total_configs * num_nonces

    print("=" * 60)
    print("Gibbs Knee Calibration")
    print("=" * 60)
    print(f"Topology: {topo_desc}")
    print(f"Sweeps: {sweeps_range}")
    print(f"Reads: {reads_range}")
    print(f"sweeps_per_beta: {spb_range}")
    print(f"Nonces per config: {num_nonces}")
    print(f"Models per nonce: {num_models}")
    print(
        f"Total: {total_configs} configs x "
        f"{num_nonces} nonces = {total_calls} sampler calls"
    )
    print(f"Timeout: {timeout_minutes} min")
    print()

    print("Initializing CUDA Gibbs sampler...")
    sampler = CudaGibbsSampler()
    print("Sampler ready")
    print()

    # SA reference from calc_energy_range
    sa_min, sa_knee, sa_max = calc_energy_range(
        num_nodes=len(nodes), num_edges=len(edges),
        h_values=tuple(h_values),
    )

    results = {
        'timestamp': int(time.time()),
        'topology': topo_desc,
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'h_values': h_values,
        'sweeps_range': sweeps_range,
        'reads_range': reads_range,
        'spb_range': spb_range,
        'num_nonces': num_nonces,
        'num_models': num_models,
        'sa_reference': {
            'min_energy': round(sa_min, 1),
            'knee_energy': round(sa_knee, 1),
            'max_energy': round(sa_max, 1),
        },
        'curve_data': [],
        'calibration': {},
    }

    timeout_seconds = timeout_minutes * 60
    experiment_start = time.time()
    completed = 0

    for spb in spb_range:
        for sweeps in sweeps_range:
            for reads in reads_range:
                elapsed = time.time() - experiment_start
                if elapsed > timeout_seconds:
                    print(
                        f"\nTimeout reached "
                        f"({timeout_minutes} min)"
                    )
                    break

                completed += 1
                print(
                    f"[{completed}/{total_configs}] "
                    f"sweeps={sweeps} reads={reads} "
                    f"spb={spb} ... ",
                    end="",
                    flush=True,
                )

                point = run_single_config(
                    sampler, nodes, edges,
                    sweeps, reads, spb,
                    num_nonces, num_models, h_values,
                )
                results['curve_data'].append(point)

                print(
                    f"min={point['min_energy_mean']:.0f} "
                    f"avg={point['avg_energy_mean']:.0f} "
                    f"({point['runtime_seconds']:.1f}s)"
                )
            else:
                continue
            break
        else:
            continue
        break

    # Analyze knee points per sweeps_per_beta
    results['calibration'] = find_all_knee_points(
        results['curve_data'], spb_range,
    )

    total_time = time.time() - experiment_start
    results['total_runtime_seconds'] = round(total_time, 1)

    print_calibration_summary(results)
    return results


def find_knee_point(
    sweep_energy_pairs: List[Tuple[int, float]],
) -> Optional[Dict]:
    """Find knee point in a sweep-vs-energy curve.

    Knee = where improvement rate drops to 20% of
    initial rate. Uses min_energy (not avg) since Gibbs
    avg_energy can regress at high sweeps.

    Args:
        sweep_energy_pairs: Sorted (sweeps, min_energy_mean)
            pairs.

    Returns:
        Dict with knee_sweeps, knee_energy, improvements,
        or None if insufficient data.
    """
    if len(sweep_energy_pairs) < 3:
        return None

    pairs = sorted(sweep_energy_pairs, key=lambda x: x[0])
    improvements = []

    for i in range(1, len(pairs)):
        s_prev, e_prev = pairs[i - 1]
        s_curr, e_curr = pairs[i]
        delta_sweeps = s_curr - s_prev
        assert delta_sweeps > 0, (
            f"Sweep values must be strictly increasing: "
            f"{s_prev} -> {s_curr}"
        )
        # Negative delta_energy = improvement (lower energy)
        rate = (e_curr - e_prev) / delta_sweeps
        improvements.append({
            'sweeps': s_curr,
            'from_sweeps': s_prev,
            'rate': round(rate, 6),
        })

    initial_rate = improvements[0]['rate']
    # initial_rate should be negative (energy improving)
    if initial_rate >= 0:
        return None

    threshold = initial_rate * KNEE_THRESHOLD
    knee_sweeps = None
    knee_energy = None

    for imp in improvements:
        # Rate becoming less negative = less improvement
        if imp['rate'] > threshold:
            knee_sweeps = imp['sweeps']
            knee_energy = dict(pairs)[knee_sweeps]
            break

    if knee_sweeps is None:
        # No knee found: use middle point
        mid = len(pairs) // 2
        knee_sweeps = pairs[mid][0]
        knee_energy = pairs[mid][1]

    return {
        'knee_sweeps': knee_sweeps,
        'knee_energy': round(knee_energy, 1),
        'threshold_rate': round(threshold, 6),
        'improvements': improvements,
    }


def find_all_knee_points(
    curve_data: List[Dict],
    spb_range: List[int],
) -> Dict:
    """Find knee points for each sweeps_per_beta value.

    Groups data by spb, picks best (reads, min_energy_mean)
    at each sweep level, then runs knee detection.

    Returns:
        Dict mapping spb values to knee results, plus
        'best_overall' recommendation.
    """
    calibration = {}

    best_spb = None
    best_knee_energy = float('inf')

    for spb in spb_range:
        spb_data = [
            p for p in curve_data
            if p['sweeps_per_beta'] == spb
        ]
        if not spb_data:
            continue

        # Best min_energy_mean at each sweep level
        sweeps_best = {}
        for p in spb_data:
            s = p['sweeps']
            e = p['min_energy_mean']
            if s not in sweeps_best or e < sweeps_best[s]:
                sweeps_best[s] = e

        pairs = sorted(sweeps_best.items())
        knee = find_knee_point(pairs)

        entry = {
            'sweep_energy_curve': [
                {'sweeps': s, 'min_energy': round(e, 1)}
                for s, e in pairs
            ],
        }
        if knee:
            entry.update(knee)
            if knee['knee_energy'] < best_knee_energy:
                best_knee_energy = knee['knee_energy']
                best_spb = spb

        calibration[f'spb_{spb}'] = entry

    if best_spb is not None:
        best_entry = calibration[f'spb_{best_spb}']
        calibration['best_overall'] = {
            'sweeps_per_beta': best_spb,
            'knee_sweeps': best_entry.get('knee_sweeps'),
            'knee_energy': best_entry.get('knee_energy'),
            'recommended_adapt_min': min(
                256, best_entry.get('knee_sweeps', 256),
            ),
            'recommended_adapt_max': best_entry.get(
                'knee_sweeps', 2048,
            ),
        }

    return calibration


def print_calibration_summary(results: Dict) -> None:
    """Print human-readable calibration summary."""
    cal = results.get('calibration', {})
    sa_ref = results.get('sa_reference', {})

    print()
    print("=" * 60)
    print("Calibration Results")
    print("=" * 60)

    # Per-spb knee points
    for key, entry in cal.items():
        if key.startswith('spb_'):
            spb_val = key.split('_')[1]
            knee_s = entry.get('knee_sweeps', '?')
            knee_e = entry.get('knee_energy', '?')
            print(
                f"  sweeps_per_beta={spb_val}: "
                f"knee at {knee_s} sweeps "
                f"(energy={knee_e})"
            )

    # Best overall
    best = cal.get('best_overall', {})
    if best:
        print()
        print(f"Best config:")
        print(
            f"  sweeps_per_beta = {best.get('sweeps_per_beta')}"
        )
        print(
            f"  Knee at {best.get('knee_sweeps')} sweeps "
            f"(energy={best.get('knee_energy')})"
        )
        print(
            f"  Recommended ADAPT_MIN_SWEEPS = "
            f"{best.get('recommended_adapt_min')}"
        )
        print(
            f"  Recommended ADAPT_MAX_SWEEPS = "
            f"{best.get('recommended_adapt_max')}"
        )

    # SA comparison
    if sa_ref:
        print()
        print("SA reference (calc_energy_range):")
        print(f"  min={sa_ref.get('min_energy')}")
        print(f"  knee={sa_ref.get('knee_energy')}")
        print(f"  max={sa_ref.get('max_energy')}")

    runtime = results.get('total_runtime_seconds', 0)
    print(f"\nTotal runtime: {runtime / 60:.1f} min")


def plot_gibbs_curve(
    results: Dict,
    output_file: str,
) -> None:
    """Generate 3-panel calibration plot.

    Panel 1: Energy vs Sweeps (lines per sweeps_per_beta)
    Panel 2: Improvement rate vs Sweeps with threshold
    Panel 3: Gibbs vs SA comparison
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cal = results.get('calibration', {})
    sa_ref = results.get('sa_reference', {})
    colors = sns.color_palette("tab10")

    # Panel 1: Energy vs Sweeps per spb
    ax1 = axes[0]
    color_idx = 0
    for key in sorted(cal.keys()):
        if not key.startswith('spb_'):
            continue
        spb_val = key.split('_')[1]
        entry = cal[key]
        curve = entry.get('sweep_energy_curve', [])
        if not curve:
            continue

        sweeps = [p['sweeps'] for p in curve]
        energies = [p['min_energy'] for p in curve]
        color = colors[color_idx % len(colors)]

        ax1.plot(
            sweeps, energies,
            marker='o', color=color,
            label=f'spb={spb_val}',
        )

        knee_s = entry.get('knee_sweeps')
        if knee_s:
            ax1.axvline(
                knee_s, color=color,
                linestyle=':', alpha=0.6,
            )
        color_idx += 1

    ax1.set_xlabel('Sweeps')
    ax1.set_ylabel('min_energy (mean across nonces)')
    ax1.set_title('Gibbs Energy vs Sweeps')
    ax1.set_xscale('log', base=2)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Improvement rate
    ax2 = axes[1]
    color_idx = 0
    for key in sorted(cal.keys()):
        if not key.startswith('spb_'):
            continue
        spb_val = key.split('_')[1]
        entry = cal[key]
        improvements = entry.get('improvements', [])
        if not improvements:
            continue

        imp_sweeps = [p['sweeps'] for p in improvements]
        imp_rates = [p['rate'] for p in improvements]
        color = colors[color_idx % len(colors)]

        ax2.plot(
            imp_sweeps, imp_rates,
            marker='s', color=color,
            label=f'spb={spb_val}',
        )

        threshold = entry.get('threshold_rate')
        if threshold is not None:
            ax2.axhline(
                threshold, color=color,
                linestyle='--', alpha=0.4,
            )
        color_idx += 1

    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Sweeps')
    ax2.set_ylabel('Improvement rate (dE/dSweeps)')
    ax2.set_title('Marginal Improvement Rate')
    ax2.set_xscale('log', base=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Gibbs vs SA comparison
    ax3 = axes[2]
    best_key = None
    best_info = cal.get('best_overall', {})
    if best_info:
        best_key = f"spb_{best_info.get('sweeps_per_beta')}"

    if best_key and best_key in cal:
        curve = cal[best_key].get('sweep_energy_curve', [])
        sweeps = [p['sweeps'] for p in curve]
        energies = [p['min_energy'] for p in curve]
        ax3.plot(
            sweeps, energies,
            marker='o', color=colors[0],
            label='Gibbs (best spb)',
            linewidth=2,
        )

    if sa_ref:
        sa_min = sa_ref.get('min_energy')
        sa_knee = sa_ref.get('knee_energy')
        sa_max = sa_ref.get('max_energy')
        if sa_min is not None:
            ax3.axhline(
                sa_min, color='red',
                linestyle='--', alpha=0.7,
                label=f'SA min ({sa_min:.0f})',
            )
        if sa_knee is not None:
            ax3.axhline(
                sa_knee, color='orange',
                linestyle='--', alpha=0.7,
                label=f'SA knee ({sa_knee:.0f})',
            )
        if sa_max is not None:
            ax3.axhline(
                sa_max, color='green',
                linestyle='--', alpha=0.7,
                label=f'SA max ({sa_max:.0f})',
            )

    ax3.set_xlabel('Sweeps')
    ax3.set_ylabel('min_energy')
    ax3.set_title('Gibbs vs SA Reference')
    ax3.set_xscale('log', base=2)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_file}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Gibbs knee calibration tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation
  python tools/calibrate_gibbs_curve.py --quick

  # Full calibration
  python tools/calibrate_gibbs_curve.py \\
      --output gibbs_knee_calibration.json \\
      --plot gibbs_knee_calibration.png

  # Custom sweep range
  python tools/calibrate_gibbs_curve.py \\
      --sweeps-range 256,512,1024,2048,4096 \\
      --sweeps-per-beta 1,2
        """,
    )

    parser.add_argument(
        '--sweeps-range', type=str,
        default=','.join(str(s) for s in DEFAULT_SWEEPS),
        help=(
            'Comma-separated sweep counts '
            f'(default: {DEFAULT_SWEEPS})'
        ),
    )
    parser.add_argument(
        '--reads-range', type=str,
        default=','.join(str(r) for r in DEFAULT_READS),
        help=(
            'Comma-separated read counts '
            f'(default: {DEFAULT_READS})'
        ),
    )
    parser.add_argument(
        '--sweeps-per-beta', type=str,
        default=','.join(str(s) for s in DEFAULT_SPB),
        help=(
            'Comma-separated sweeps_per_beta values '
            f'(default: {DEFAULT_SPB})'
        ),
    )
    parser.add_argument(
        '--nonces', type=int, default=DEFAULT_NUM_NONCES,
        help=(
            'Nonces per config '
            f'(default: {DEFAULT_NUM_NONCES})'
        ),
    )
    parser.add_argument(
        '--num-models', type=int,
        default=DEFAULT_NUM_MODELS,
        help=(
            'Models per nonce '
            f'(default: {DEFAULT_NUM_MODELS})'
        ),
    )
    parser.add_argument(
        '--timeout', '-t', type=float, default=120.0,
        help='Overall timeout in minutes (default: 120)',
    )
    parser.add_argument(
        '--output', '-o', type=str,
        help='Output JSON file (auto-generated if omitted)',
    )
    parser.add_argument(
        '--plot', type=str,
        help='Save plot to file (e.g., curve.png)',
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Disable plot generation',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode: fewer configs for validation',
    )
    parser.add_argument(
        '--h-values', type=str, default='-1,0,1',
        help='Comma-separated h values (default: -1,0,1)',
    )
    parser.add_argument(
        '--topology', type=str,
        help='Topology: file path, hardware name, or format',
    )

    args = parser.parse_args()

    h_values = [
        float(v.strip()) for v in args.h_values.split(',')
    ]

    if args.quick:
        sweeps_range = QUICK_SWEEPS
        reads_range = QUICK_READS
        spb_range = QUICK_SPB
        num_nonces = QUICK_NUM_NONCES
    else:
        sweeps_range = [
            int(v.strip())
            for v in args.sweeps_range.split(',')
        ]
        reads_range = [
            int(v.strip())
            for v in args.reads_range.split(',')
        ]
        spb_range = [
            int(v.strip())
            for v in args.sweeps_per_beta.split(',')
        ]
        num_nonces = args.nonces

    output_file = args.output
    if not output_file:
        timestamp = int(time.time())
        output_file = f"gibbs_knee_calibration_{timestamp}.json"

    results = calibrate_gibbs_curve(
        sweeps_range=sweeps_range,
        reads_range=reads_range,
        spb_range=spb_range,
        num_nonces=num_nonces,
        num_models=args.num_models,
        h_values=h_values,
        topology=args.topology,
        timeout_minutes=args.timeout,
    )

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    if not args.no_plot:
        plot_file = args.plot
        if not plot_file:
            plot_file = output_file.replace('.json', '.png')
        try:
            plot_gibbs_curve(results, plot_file)
        except Exception as e:
            print(f"Failed to generate plot: {e}")

    print("\nCalibration complete!")


if __name__ == "__main__":
    main()
