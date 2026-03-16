#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""SA and Gibbs multi-nonce benchmark.

Calibrates kernels to target avg energy, then measures throughput
(models/s) over 20 batch launches. Both SA and Gibbs use
multi-nonce dispatch (one block per nonce).

Usage:
    # SA only
    python tools/benchmark_multi_nonce.py --solver sa

    # Gibbs only
    python tools/benchmark_multi_nonce.py --solver gibbs

    # Both (serial: SA first, then Gibbs)
    python tools/benchmark_multi_nonce.py
"""

import argparse
import time

import cupy as cp
import numpy as np

from dwave_topologies import DEFAULT_TOPOLOGY
from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
)


TARGET_ENERGY = -14900.0
NUM_BATCHES = 20


def get_topology_info():
    """Return nodes, edges from default topology."""
    topo = DEFAULT_TOPOLOGY
    nodes = list(topo.graph.nodes())
    edges = list(topo.graph.edges())
    return nodes, edges


def generate_nonce_batch(
    nodes, edges, num_nonces, base_nonce=1000,
):
    """Generate h/J for a batch of nonces."""
    h_list = []
    J_list = []
    for k in range(num_nonces):
        h, J = generate_ising_model_from_nonce(
            base_nonce + k, nodes, edges,
        )
        h_list.append(h)
        J_list.append(J)
    return h_list, J_list


def calibrate_sa(sampler, num_nonces, num_reads):
    """Find num_betas for SA to hit target avg energy."""
    nodes, edges = get_topology_info()
    cal_n = min(4, num_nonces)
    h_list, J_list = generate_nonce_batch(
        nodes, edges, cal_n,
    )

    best_betas = 50
    best_energy = 0.0

    for num_betas in [25, 50, 100, 200, 400]:
        results = sampler.sample_ising(
            h_list, J_list,
            num_reads=num_reads,
            num_betas=num_betas,
            num_sweeps_per_beta=1,
        )
        avg_e = np.mean([
            ss.record.energy.mean() for ss in results
        ])
        print(
            f"  SA calibrate: betas={num_betas:4d}, "
            f"avg_energy={avg_e:.1f}"
        )
        if avg_e <= TARGET_ENERGY:
            best_betas = num_betas
            best_energy = avg_e
            break
        best_betas = num_betas
        best_energy = avg_e

    return best_betas, best_energy


def calibrate_gibbs(
    sampler, num_nonces, num_reads, sms_per_nonce,
):
    """Find num_sweeps for Gibbs to hit target avg energy."""
    nodes, edges = get_topology_info()
    cal_n = min(4, num_nonces)
    h_list, J_list = generate_nonce_batch(
        nodes, edges, cal_n,
    )

    best_sweeps = 256
    best_energy = 0.0

    for num_sweeps in [128, 256, 512, 1024, 2048]:
        results = sampler.sample_multi_nonce(
            h_list, J_list,
            reads_per_nonce=num_reads,
            num_sweeps=num_sweeps,
            sms_per_nonce=sms_per_nonce,
        )
        avg_e = np.mean([
            ss.record.energy.mean() for ss in results
        ])
        print(
            f"  Gibbs calibrate: sweeps={num_sweeps:4d}, "
            f"avg_energy={avg_e:.1f}"
        )
        if avg_e <= TARGET_ENERGY:
            best_sweeps = num_sweeps
            best_energy = avg_e
            break
        best_sweeps = num_sweeps
        best_energy = avg_e

    return best_sweeps, best_energy


def benchmark_sa(
    sampler, sa_nonces, num_reads, num_betas,
):
    """Run SA multi-nonce benchmark and return metrics."""
    nodes, edges = get_topology_info()
    energies = []
    times = []

    for batch in range(NUM_BATCHES):
        h_b, J_b = generate_nonce_batch(
            nodes, edges, sa_nonces,
            base_nonce=batch * 100,
        )

        t0 = time.perf_counter()
        results = sampler.sample_ising(
            h_b, J_b,
            num_reads=num_reads,
            num_betas=num_betas,
            num_sweeps_per_beta=1,
        )
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()

        batch_e = np.mean([
            ss.record.energy.mean() for ss in results
        ])
        energies.append(batch_e)
        times.append(t1 - t0)

    avg_energy = np.mean(energies)
    avg_time = np.mean(times)
    models_per_sec = sa_nonces / avg_time

    return avg_energy, models_per_sec, avg_time


def benchmark_gibbs(
    sampler, num_nonces, num_reads,
    num_sweeps, sms_per_nonce,
):
    """Run Gibbs benchmark and return metrics."""
    nodes, edges = get_topology_info()
    energies = []
    times = []

    for batch in range(NUM_BATCHES):
        h_b, J_b = generate_nonce_batch(
            nodes, edges, num_nonces,
            base_nonce=batch * 100,
        )

        t0 = time.perf_counter()
        results = sampler.sample_multi_nonce(
            h_b, J_b,
            reads_per_nonce=num_reads,
            num_sweeps=num_sweeps,
            sms_per_nonce=sms_per_nonce,
        )
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()

        batch_e = np.mean([
            ss.record.energy.mean() for ss in results
        ])
        energies.append(batch_e)
        times.append(t1 - t0)

    avg_energy = np.mean(energies)
    avg_time = np.mean(times)
    models_per_sec = num_nonces / avg_time

    return avg_energy, models_per_sec, avg_time


def run_sa_benchmark(args, avail_sms):
    """Run SA benchmark and return metrics dict."""
    from GPU.cuda_kernel import CudaKernelRealSA
    from GPU.cuda_sa import CudaKernelAdapter, CudaSASamplerAsync

    # SA: 1 SM per nonce
    sa_nonces = args.sa_nonces or avail_sms

    print(
        f"=== SA Kernel (persistent, "
        f"{sa_nonces} nonces) ===",
    )
    sa_kernel = CudaKernelRealSA(max_N=5000, verbose=False)
    sa_sampler = CudaSASamplerAsync(
        CudaKernelAdapter(sa_kernel),
    )

    print("Calibrating SA...")
    sa_betas, sa_cal_e = calibrate_sa(
        sa_sampler, sa_nonces, args.sa_reads,
    )
    print(
        f"  -> betas={sa_betas}, "
        f"calibration energy={sa_cal_e:.1f}",
    )
    print()

    print("Benchmarking SA...")
    sa_avg_e, sa_mps, sa_time = benchmark_sa(
        sa_sampler, sa_nonces,
        args.sa_reads, sa_betas,
    )
    print(
        f"  SA: avg_energy={sa_avg_e:.1f}, "
        f"models/s={sa_mps:.1f}, "
        f"batch_time={sa_time*1000:.1f}ms",
    )
    print()

    # Tear down SA to free GPU memory
    sa_sampler.stop_immediate()
    del sa_kernel
    cp.get_default_memory_pool().free_all_blocks()

    return {
        'nonces': sa_nonces,
        'reads': args.sa_reads,
        'betas': sa_betas,
        'avg_energy': sa_avg_e,
        'models_per_sec': sa_mps,
        'batch_time': sa_time,
    }


def run_gibbs_benchmark(args, avail_sms):
    """Run Gibbs benchmark and return metrics dict."""
    from GPU.cuda_gibbs_sa import CudaGibbsSampler

    gibbs_nonces = avail_sms // args.sms_per_nonce
    if gibbs_nonces < 1:
        raise SystemExit(
            f"Not enough SMs: {avail_sms} available, "
            f"need {args.sms_per_nonce} per nonce"
        )

    print(
        f"=== Gibbs Kernel (multi-nonce, "
        f"{gibbs_nonces} nonces) ==="
    )
    gibbs_sampler = CudaGibbsSampler(parallel=True)
    gibbs_sampler.prepare(
        num_reads=args.gibbs_reads,
        num_sweeps=2048,
        num_sweeps_per_beta=1,
        max_nonces=gibbs_nonces,
    )

    print("Calibrating Gibbs...")
    gibbs_sweeps, gibbs_cal_e = calibrate_gibbs(
        gibbs_sampler, gibbs_nonces,
        args.gibbs_reads, args.sms_per_nonce,
    )
    print(
        f"  -> sweeps={gibbs_sweeps}, "
        f"calibration energy={gibbs_cal_e:.1f}",
    )
    print()

    print("Benchmarking Gibbs...")
    gibbs_avg_e, gibbs_mps, gibbs_time = benchmark_gibbs(
        gibbs_sampler, gibbs_nonces,
        args.gibbs_reads, gibbs_sweeps,
        args.sms_per_nonce,
    )
    print(
        f"  Gibbs: avg_energy={gibbs_avg_e:.1f}, "
        f"models/s={gibbs_mps:.1f}, "
        f"batch_time={gibbs_time*1000:.1f}ms",
    )
    print()

    # Tear down Gibbs to free GPU memory
    del gibbs_sampler
    cp.get_default_memory_pool().free_all_blocks()

    return {
        'nonces': gibbs_nonces,
        'reads': args.gibbs_reads,
        'sweeps': gibbs_sweeps,
        'sms_per_nonce': args.sms_per_nonce,
        'avg_energy': gibbs_avg_e,
        'models_per_sec': gibbs_mps,
        'batch_time': gibbs_time,
    }


def main():  # noqa: C901
    global TARGET_ENERGY  # noqa: PLW0604
    parser = argparse.ArgumentParser(
        description="SA and Gibbs multi-nonce benchmark",
    )
    parser.add_argument(
        "--solver", choices=["sa", "gibbs"],
        default=None,
        help="Run only SA or Gibbs (default: both serial)",
    )
    parser.add_argument(
        "--gpu-util", type=int, default=80,
        help="GPU utilization percent (default: 80)",
    )
    parser.add_argument(
        "--sms-per-nonce", type=int, default=4,
        help="SMs per Gibbs nonce (default: 4)",
    )
    parser.add_argument(
        "--sa-reads", type=int, default=32,
        help="Reads per nonce for SA (default: 32)",
    )
    parser.add_argument(
        "--gibbs-reads", type=int, default=32,
        help="Reads per nonce for Gibbs (default: 32)",
    )
    parser.add_argument(
        "--target-energy", type=float,
        default=TARGET_ENERGY,
        help=f"Target avg energy (default: {TARGET_ENERGY})",
    )
    parser.add_argument(
        "--sa-nonces", type=int, default=0,
        help="SA nonces per batch (0 = auto from SMs)",
    )
    args = parser.parse_args()

    TARGET_ENERGY = args.target_energy

    dev = cp.cuda.Device()
    device_sms = dev.attributes['MultiProcessorCount']
    max_sms = max(
        1, int(device_sms * args.gpu_util / 100),
    )
    avail_sms = max_sms

    print(f"Device: {dev.id}, SMs: {device_sms}")
    print(
        f"GPU util: {args.gpu_util}%, "
        f"available SMs: {avail_sms}",
    )
    print(f"Target avg energy: {TARGET_ENERGY}")
    print()

    sa_result = None
    gibbs_result = None

    if args.solver in (None, "sa"):
        sa_result = run_sa_benchmark(args, avail_sms)

    if args.solver in (None, "gibbs"):
        gibbs_result = run_gibbs_benchmark(args, avail_sms)

    # Comparison table (only if both ran)
    if sa_result and gibbs_result:
        print("=" * 60)
        print(f"{'Metric':<25} {'SA':>15} {'Gibbs':>15}")
        print("-" * 60)
        print(
            f"{'SMs/nonce':<25} {'1':>15} "
            f"{gibbs_result['sms_per_nonce']:>15}",
        )
        print(
            f"{'Nonces/batch':<25} "
            f"{sa_result['nonces']:>15} "
            f"{gibbs_result['nonces']:>15}",
        )
        print(
            f"{'Reads/nonce':<25} "
            f"{sa_result['reads']:>15} "
            f"{gibbs_result['reads']:>15}",
        )
        print(
            f"{'Sweeps/betas':<25} "
            f"{sa_result['betas']:>15} "
            f"{gibbs_result['sweeps']:>15}",
        )
        print(
            f"{'Avg energy':<25} "
            f"{sa_result['avg_energy']:>15.1f} "
            f"{gibbs_result['avg_energy']:>15.1f}",
        )
        print(
            f"{'Models/s':<25} "
            f"{sa_result['models_per_sec']:>15.1f} "
            f"{gibbs_result['models_per_sec']:>15.1f}",
        )
        print(
            f"{'Batch time (ms)':<25} "
            f"{sa_result['batch_time']*1000:>15.1f} "
            f"{gibbs_result['batch_time']*1000:>15.1f}",
        )
        print("=" * 60)


if __name__ == "__main__":
    main()
