#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Benchmark: Gibbs adaptive curve vs SA baseline.

Runs full mine_block() with streaming GPU saturation for three
configurations at energy targets from -14600 to -15000.

1. SA baseline — CudaMiner(update_mode="sa")
2. Gibbs original — CudaMiner(update_mode="gibbs"), SA constants
3. Gibbs calibrated — CudaMiner(update_mode="gibbs"), Gibbs ADAPT_*

One long-lived process per GPU, each popping jobs from a shared
queue — mirrors production mining architecture.

Usage:
    python tools/benchmark_gibbs_curve.py --quick
    python tools/benchmark_gibbs_curve.py \
        --output gibbs_bench.json --plot gibbs_bench.png
"""

import argparse
import json
import multiprocessing as mp
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))


def _get_gpu_count():
    """Get GPU count without poisoning CUDA for child procs."""
    import subprocess
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '-L'], text=True,
        )
        return len([
            l for l in out.strip().splitlines() if l.strip()
        ])
    except Exception:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount()


# Config definitions: (label, update_mode, calibrated?)
CONFIG_DEFS = [
    ('SA baseline', 'sa', False),
    ('Gibbs original', 'gibbs', False),
    ('Gibbs calibrated', 'gibbs', True),
]

DEFAULT_ENERGY_START = -14600
DEFAULT_ENERGY_END = -15000
DEFAULT_ENERGY_STEP = 25
DEFAULT_TIMEOUT = 0
QUICK_TIMEOUT = 0


def build_energy_targets(start, end, step):
    """Generate energy targets from start to end (inclusive)."""
    targets = []
    e = start
    while e >= end:
        targets.append(e)
        e -= step
    if targets[-1] != end:
        targets.append(end)
    return targets


def gpu_worker(
    device,
    job_queue,
    result_queue,
    timeout_s,
):
    """Long-lived GPU worker process.

    Creates miners on demand, pops jobs from shared queue,
    runs mine_block, pushes results. One process per GPU,
    mirrors production architecture.
    """
    from GPU.cuda_miner import CudaMiner
    from shared.block import (
        BlockRequirements,
        MinerInfo,
        create_genesis_block,
    )

    class CudaMinerGibbsCalibrated(CudaMiner):
        """Gibbs with 2x sweep multiplier (inherited from CudaMiner).

        No ADAPT overrides needed — CudaMiner._adapt_mining_params
        applies GIBBS_SWEEP_MULTIPLIER=2 automatically.
        """

    dummy_key = b'\x00' * 32
    node_info = MinerInfo(
        miner_id='benchmark',
        miner_type='benchmark',
        reward_address=dummy_key,
        ecdsa_public_key=dummy_key,
        wots_public_key=dummy_key,
        next_wots_public_key=dummy_key,
    )

    while True:
        try:
            job = job_queue.get(timeout=1.0)
        except Exception:
            return

        if job is None:
            return  # Poison pill

        config_name = job['config']
        energy_target = job['energy_target']
        update_mode = job['update_mode']
        calibrated = job['calibrated']

        # Fresh miner per job — guarantees clean CUDA and
        # streaming state. No stale buffers, no leftover
        # kernel completions.
        if calibrated:
            miner = CudaMinerGibbsCalibrated(
                f'bench-{config_name[:8]}-d{device}',
                device=device,
                update_mode=update_mode,
            )
        else:
            miner = CudaMiner(
                f'bench-{config_name[:8]}-d{device}',
                device=device,
                update_mode=update_mode,
            )

        # Compute adaptive params for reporting
        params = miner.adapt_parameters(
            energy_target, 0.2, 5,
        )
        difficulty = miner.energy_to_difficulty(
            energy_target,
        )

        tag = (f"GPU{device} {config_name} "
               f"E={energy_target:.0f}")
        print(f"  [{tag}] d={difficulty:.3f} "
              f"sw={params['num_sweeps']} "
              f"rd={params['num_reads']} — mining...",
              flush=True)

        # Build synthetic block
        genesis = create_genesis_block()
        genesis.next_block_requirements = BlockRequirements(
            difficulty_energy=energy_target,
            min_diversity=0.2,
            min_solutions=5,
            timeout_to_difficulty_adjustment_decay=99999,
            h_values=[-1.0, 0.0, 1.0],
        )
        prev_block = genesis
        requirements = prev_block.next_block_requirements
        prev_timestamp = prev_block.header.timestamp

        stop_event = threading.Event()

        if timeout_s > 0:
            def timeout_handler():
                if not stop_event.wait(timeout=timeout_s):
                    stop_event.set()
            timer = threading.Thread(
                target=timeout_handler, daemon=True,
            )
            timer.start()

        start = time.time()

        result = miner.mine_block(
            prev_block, node_info, requirements,
            prev_timestamp, stop_event,
            feeder_seed=hash(
                (config_name, energy_target),
            ),
        )

        elapsed = time.time() - start
        stop_event.set()

        row = {
            'energy_target': energy_target,
            'config': config_name,
            'device': device,
            'difficulty': round(difficulty, 4),
            'num_sweeps': params['num_sweeps'],
            'num_reads': params['num_reads'],
            'elapsed_s': round(elapsed, 2),
            'mined': result is not None,
            'best_energy': None,
            'num_valid': 0,
            'diversity': 0.0,
        }

        if result is not None:
            row['best_energy'] = round(result.energy, 1)
            row['num_valid'] = result.num_valid
            row['diversity'] = round(result.diversity, 3)

        result_queue.put(row)

        status = (
            f"MINED ({row['best_energy']:.0f})"
            if row['mined']
            else "TIMEOUT"
        )
        print(
            f"GPU{device} "
            f"E={row['energy_target']:7.0f} "
            f"{row['config']:<18s} "
            f"d={row['difficulty']:.3f} "
            f"sw={row['num_sweeps']:5d} "
            f"rd={row['num_reads']:4d} "
            f"t={row['elapsed_s']:7.1f}s "
            f"{status}",
            flush=True,
        )

        del miner
        time.sleep(1.0)


def benchmark_all(
    energy_targets,
    timeout_s,
    devices,
):
    """Build job list, dispatch to GPU worker processes."""
    job_queue = mp.Queue()
    result_queue = mp.Queue()

    for energy_target in energy_targets:
        for config_name, update_mode, calibrated in CONFIG_DEFS:
            job_queue.put({
                'energy_target': energy_target,
                'config': config_name,
                'update_mode': update_mode,
                'calibrated': calibrated,
            })

    # Poison pills to signal workers to exit
    for _ in devices:
        job_queue.put(None)

    total_jobs = len(energy_targets) * len(CONFIG_DEFS)
    print(f"Dispatching {total_jobs} jobs across "
          f"GPUs {devices} (1 worker per GPU)...")
    print(flush=True)

    procs = []
    for dev_id in devices:
        p = mp.Process(
            target=gpu_worker,
            args=(
                str(dev_id), job_queue,
                result_queue, timeout_s,
            ),
        )
        p.start()
        procs.append(p)

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("\nInterrupted — killing workers...",
              flush=True)
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    config_order = {
        c[0]: i for i, c in enumerate(CONFIG_DEFS)
    }
    results.sort(
        key=lambda r: (
            r['energy_target'],
            config_order.get(r['config'], 99),
        ),
    )

    return results


def print_results_table(results):
    """Print formatted comparison table."""
    print()
    header = (
        f"{'Energy':>7} | {'Config':<18s} | "
        f"{'Diff':>5} | {'Sweep':>5} | {'Read':>4} | "
        f"{'Time':>7} | {'Result':<20s}"
    )
    print(header)
    print("-" * len(header))

    prev_energy = None
    for r in results:
        if (
            prev_energy is not None
            and r['energy_target'] != prev_energy
        ):
            print()
        prev_energy = r['energy_target']

        if r['mined']:
            status = (
                f"MINED ({r['best_energy']:.0f}, "
                f"n={r['num_valid']}, "
                f"d={r['diversity']:.2f})"
            )
        else:
            status = "TIMEOUT"

        print(
            f"{r['energy_target']:7.0f} | "
            f"{r['config']:<18s} | "
            f"{r['difficulty']:5.3f} | "
            f"{r['num_sweeps']:5d} | "
            f"{r['num_reads']:4d} | "
            f"{r['elapsed_s']:7.1f} | "
            f"{status:<20s}"
        )


def _get_plot_deps():
    """Import matplotlib or return None."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return None


_COLORS = ['#2196F3', '#FF9800', '#4CAF50']
_MARKERS = ['o', 's', '^']


def plot_mine_time(results, output_path):
    """Plot time-to-mine vs energy target."""
    plt = _get_plot_deps()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    config_names = [c[0] for c in CONFIG_DEFS]

    for i, name in enumerate(config_names):
        rows = [r for r in results if r['config'] == name]
        energies = [r['energy_target'] for r in rows]
        times = [r['elapsed_s'] for r in rows]
        mined = [r['mined'] for r in rows]

        mined_e = [e for e, m in zip(energies, mined) if m]
        mined_t = [t for t, m in zip(times, mined) if m]
        timeout_e = [
            e for e, m in zip(energies, mined) if not m
        ]
        timeout_t = [
            t for t, m in zip(times, mined) if not m
        ]

        if mined_t:
            ax.semilogy(
                mined_e, mined_t,
                color=_COLORS[i], marker=_MARKERS[i],
                label=name, linewidth=1.5, markersize=5,
            )
        if timeout_t:
            ax.semilogy(
                timeout_e, timeout_t,
                color=_COLORS[i], marker=_MARKERS[i],
                fillstyle='none', linewidth=0,
                markersize=8, markeredgewidth=2,
            )

    ax.set_xlabel('Energy Target')
    ax.set_ylabel('Time to Mine Block (s)')
    ax.set_title(
        'Mine Time vs Energy Target\n'
        '(open markers = timeout)'
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  {output_path}")


def plot_difficulty(results, output_path):
    """Plot difficulty mapping vs energy target."""
    plt = _get_plot_deps()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    config_names = [c[0] for c in CONFIG_DEFS]

    for i, name in enumerate(config_names):
        rows = [r for r in results if r['config'] == name]
        energies = [r['energy_target'] for r in rows]
        diffs = [r['difficulty'] for r in rows]
        ax.plot(
            energies, diffs,
            color=_COLORS[i], marker=_MARKERS[i],
            label=name, linewidth=1.5, markersize=5,
        )

    ax.set_xlabel('Energy Target')
    ax.set_ylabel('Difficulty Factor')
    ax.set_title('Difficulty Mapping vs Energy Target')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  {output_path}")


def plot_sa_vs_gibbs_overlay(results, output_path):
    """Plot SA as thick reference with Gibbs variants overlaid.

    Gap shading between SA and Gibbs calibrated shows the
    remaining performance delta. Crossover annotation marks
    where calibrated Gibbs beats SA (if it happens).
    """
    plt = _get_plot_deps()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    config_names = [c[0] for c in CONFIG_DEFS]

    line_styles = {
        'SA baseline': {
            'linewidth': 2.5, 'linestyle': '-', 'zorder': 3,
        },
        'Gibbs original': {
            'linewidth': 1.5, 'linestyle': '--', 'zorder': 2,
        },
        'Gibbs calibrated': {
            'linewidth': 1.5, 'linestyle': '-', 'zorder': 2,
        },
    }

    series = {}
    for i, name in enumerate(config_names):
        rows = [r for r in results if r['config'] == name]
        energies = [r['energy_target'] for r in rows]
        times = [r['elapsed_s'] for r in rows]
        mined = [r['mined'] for r in rows]

        mined_e = [e for e, m in zip(energies, mined) if m]
        mined_t = [t for t, m in zip(times, mined) if m]
        timeout_e = [
            e for e, m in zip(energies, mined) if not m
        ]
        timeout_t = [
            t for t, m in zip(times, mined) if not m
        ]

        style = line_styles.get(name, {})
        series[name] = {
            'energies': energies, 'times': times,
            'mined': mined,
        }

        if mined_t:
            ax.semilogy(
                mined_e, mined_t,
                color=_COLORS[i], marker=_MARKERS[i],
                label=name, markersize=5, **style,
            )
        if timeout_t:
            ax.semilogy(
                timeout_e, timeout_t,
                color=_COLORS[i], marker=_MARKERS[i],
                fillstyle='none', linewidth=0,
                markersize=8, markeredgewidth=2,
            )

    # Gap shading between SA and Gibbs calibrated
    sa = series.get('SA baseline')
    gcal = series.get('Gibbs calibrated')
    if sa and gcal:
        sa_lookup = {
            e: t for e, t, m in zip(
                sa['energies'], sa['times'], sa['mined'],
            ) if m
        }
        gcal_lookup = {
            e: t for e, t, m in zip(
                gcal['energies'], gcal['times'],
                gcal['mined'],
            ) if m
        }
        shared_e = sorted(
            set(sa_lookup) & set(gcal_lookup), reverse=True,
        )
        if len(shared_e) >= 2:
            sa_t = [sa_lookup[e] for e in shared_e]
            gcal_t = [gcal_lookup[e] for e in shared_e]
            ax.fill_between(
                shared_e, sa_t, gcal_t,
                alpha=0.12, color=_COLORS[2],
                label='SA\u2013Gibbs gap', zorder=1,
            )

        # Annotate crossover (Gibbs calibrated faster)
        for e in shared_e:
            if gcal_lookup[e] < sa_lookup[e]:
                cross_t = gcal_lookup[e]
                ax.annotate(
                    f'crossover\n({e:.0f})',
                    xy=(e, cross_t),
                    xytext=(e + 50, cross_t * 3),
                    arrowprops=dict(
                        arrowstyle='->', color='gray',
                        lw=1.2,
                    ),
                    fontsize=8, color='gray', ha='center',
                )
                break

    ax.set_xlabel('Energy Target')
    ax.set_ylabel('Time to Mine Block (s)')
    ax.set_title(
        'SA vs Gibbs Performance Overlay\n'
        '(open markers = timeout, '
        'shading = SA\u2013Gibbs gap)'
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  {output_path}")


def plot_runtime_settings_overlay(results, output_path):
    """Plot sweeps and reads vs energy target.

    Two stacked subplots reveal how each config allocates
    compute at each difficulty level.
    """
    plt = _get_plot_deps()
    if plt is None:
        return

    fig, (ax_sw, ax_rd) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
    )
    config_names = [c[0] for c in CONFIG_DEFS]

    for i, name in enumerate(config_names):
        rows = [r for r in results if r['config'] == name]
        energies = [r['energy_target'] for r in rows]
        sweeps = [r['num_sweeps'] for r in rows]
        reads = [r['num_reads'] for r in rows]

        ax_sw.plot(
            energies, sweeps,
            color=_COLORS[i], marker=_MARKERS[i],
            label=name, linewidth=1.5, markersize=5,
        )
        ax_rd.plot(
            energies, reads,
            color=_COLORS[i], marker=_MARKERS[i],
            label=name, linewidth=1.5, markersize=5,
        )

    # Reference lines at key thresholds
    ax_sw.axhline(
        256, color='gray', linestyle=':',
        linewidth=1, alpha=0.6,
        label='ADAPT_MIN_SWEEPS',
    )
    ax_sw.axhline(
        8192, color='gray', linestyle='--',
        linewidth=1, alpha=0.6,
        label='Gibbs MAX_SWEEPS',
    )

    ax_sw.set_ylabel('Num Sweeps')
    ax_sw.set_title(
        'Runtime Parameters vs Energy Target'
    )
    ax_sw.legend(fontsize=8, ncol=2)
    ax_sw.grid(True, alpha=0.3)

    ax_rd.set_xlabel('Energy Target')
    ax_rd.set_ylabel('Num Reads')
    ax_rd.legend(fontsize=8)
    ax_rd.grid(True, alpha=0.3)
    ax_rd.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  {output_path}")


def save_outputs(results, output_dir, meta):
    """Save results JSON and plots to output directory."""
    from pathlib import Path
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = outdir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump({**meta, 'results': results}, f, indent=2)

    print(f"\nOutputs saved to {outdir}/")
    print(f"  {json_path}")

    # Plots
    plot_mine_time(results, str(outdir / 'mine_time.png'))
    plot_difficulty(results, str(outdir / 'difficulty.png'))
    plot_sa_vs_gibbs_overlay(
        results, str(outdir / 'sa_vs_gibbs.png'),
    )
    plot_runtime_settings_overlay(
        results, str(outdir / 'runtime_settings.png'),
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Gibbs adaptive curve vs SA",
    )
    parser.add_argument(
        '--energy-start', type=float,
        default=DEFAULT_ENERGY_START,
        help='Starting energy target (default: %(default)s)',
    )
    parser.add_argument(
        '--energy-end', type=float,
        default=DEFAULT_ENERGY_END,
        help='Ending energy target (default: %(default)s)',
    )
    parser.add_argument(
        '--energy-step', type=float,
        default=DEFAULT_ENERGY_STEP,
        help='Energy step size (default: %(default)s)',
    )
    parser.add_argument(
        '--timeout', type=float, default=DEFAULT_TIMEOUT,
        help='Per-target timeout in seconds, 0=none '
             '(default: %(default)s)',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results.json and plots '
             '(default: gibbs_bench_<timestamp>)',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode: 5 targets',
    )
    parser.add_argument(
        '--gpus', type=int, default=None,
        help='Number of GPUs (default: auto-detect)',
    )
    parser.add_argument(
        '--devices', type=str, default=None,
        help='Comma-separated GPU device IDs '
             '(e.g., "0,1" or "1"). Overrides --gpus.',
    )
    return parser.parse_args()


def main():
    """Run the adaptive benchmark."""
    args = parse_args()

    if args.devices:
        devices = [int(d) for d in args.devices.split(',')]
    else:
        num_gpus = args.gpus or _get_gpu_count()
        devices = list(range(num_gpus))

    if args.quick:
        energy_targets = build_energy_targets(
            -14600, -15000, 100,
        )
        timeout = QUICK_TIMEOUT
    else:
        energy_targets = build_energy_targets(
            args.energy_start,
            args.energy_end,
            args.energy_step,
        )
        timeout = args.timeout

    total_jobs = len(energy_targets) * len(CONFIG_DEFS)

    print("Gibbs Adaptive Curve Benchmark (mine_block)")
    print("=" * 60)
    print(f"Energy targets: {energy_targets[0]} to "
          f"{energy_targets[-1]} "
          f"({len(energy_targets)} points)")
    print(f"Timeout per target: "
          f"{'none' if timeout == 0 else f'{timeout}s'}")
    print(f"GPUs: {devices}")
    print(f"Jobs: {total_jobs} "
          f"({len(CONFIG_DEFS)} configs x "
          f"{len(energy_targets)} targets)")
    print(f"Configs: "
          f"{', '.join(c[0] for c in CONFIG_DEFS)}")
    print()

    # Show adaptive params at endpoints
    from GPU.cuda_miner import CudaMiner as _CM

    print("Adaptive params at key energy targets:")
    for e in [energy_targets[0], energy_targets[-1]]:
        print(f"  E={e:.0f}:")
        for name, update_mode, _ in CONFIG_DEFS:
            p = _CM.adapt_parameters(e, 0.2, 5)
            d = _CM.energy_to_difficulty(e)
            if update_mode in ('gibbs', 'metropolis'):
                p['num_sweeps'] = min(
                    p['num_sweeps'] * _CM.GIBBS_SWEEP_MULTIPLIER,
                    _CM.ADAPT_MAX_SWEEPS
                    * _CM.GIBBS_SWEEP_MULTIPLIER,
                )
            print(f"    {name:<18s}: d={d:.3f} "
                  f"sw={p['num_sweeps']:5d} "
                  f"rd={p['num_reads']:4d}")
    print()

    # Run
    results = benchmark_all(
        energy_targets, timeout, devices,
    )

    # Print table
    print_results_table(results)

    # Save outputs
    if args.output_dir is None:
        ts = int(time.time())
        output_dir = f"gibbs_bench_{ts}"
    else:
        output_dir = args.output_dir

    meta = {
        'energy_targets': energy_targets,
        'timeout_s': timeout,
        'devices': devices,
        'configs': [c[0] for c in CONFIG_DEFS],
    }
    save_outputs(results, output_dir, meta)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
