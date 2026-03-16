#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Benchmark: CUDA Gibbs (persistent) vs CUDA SA vs CPU SA.

Runs all three samplers on the same Ising problem and prints
a comparison table of runtime, energy, and throughput.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
)
from dwave_topologies import DEFAULT_TOPOLOGY


def bench_cuda_gibbs(h, J, num_reads, num_sweeps, n_models):
    """Benchmark CUDA Gibbs sampler."""
    from GPU.cuda_gibbs_sa import CudaGibbsSampler

    sampler = CudaGibbsSampler(
        update_mode="gibbs", parallel=True,
    )
    # warmup
    sampler.sample_ising(
        h=[h], J=[J], num_reads=2, num_sweeps=100,
    )

    h_batch = [h] * n_models
    J_batch = [J] * n_models
    start = time.time()
    results = sampler.sample_ising(
        h=h_batch, J=J_batch,
        num_reads=num_reads, num_sweeps=num_sweeps,
    )
    elapsed = time.time() - start
    energies = []
    for ss in results:
        energies.extend(list(ss.record.energy))
    return elapsed, min(energies), sum(energies) / len(energies)


def bench_cuda_sa(h, J, num_reads, num_sweeps, n_models):
    """Benchmark CUDA SA sampler (self-feeding)."""
    from GPU.cuda_sa import CudaSASampler

    sampler = CudaSASampler()
    # warmup
    sampler.sample_ising(
        [h], [J], num_reads=2, num_sweeps=100,
    )

    h_batch = [h] * n_models
    J_batch = [J] * n_models
    start = time.time()
    results = sampler.sample_ising(
        h=h_batch, J=J_batch,
        num_reads=num_reads, num_sweeps=num_sweeps,
    )
    elapsed = time.time() - start
    energies = []
    for ss in results:
        energies.extend(list(ss.record.energy))
    sampler.close()
    return elapsed, min(energies), sum(energies) / len(energies)


def bench_cpu_sa(h, J, num_reads, num_sweeps, n_models):
    """Benchmark CPU SA sampler."""
    from CPU.sa_sampler import SimulatedAnnealingStructuredSampler

    sampler = SimulatedAnnealingStructuredSampler()
    start = time.time()
    all_e = []
    for _ in range(n_models):
        ss = sampler.sample_ising(
            h, J,
            num_reads=num_reads, num_sweeps=num_sweeps,
        )
        all_e.extend(list(ss.record.energy))
    elapsed = time.time() - start
    return elapsed, min(all_e), sum(all_e) / len(all_e)


def fmt_row(label, name, t, mine, avge, total_samples):
    """Format a single result row."""
    sps = total_samples / t if t > 0 else 0
    print(
        f"{label:<28} {name:<16} "
        f"{t:>7.2f}s {mine:>8.1f} {avge:>8.1f} "
        f"{sps:>10.1f}"
    )


def main():
    topo = DEFAULT_TOPOLOGY
    nodes = list(topo.graph.nodes)
    edges = list(topo.graph.edges)

    seed = 12345
    h, J = generate_ising_model_from_nonce(
        seed, nodes, edges, h_values=[-1.0, 0.0, 1.0],
    )
    print(f"Problem: {len(h)} variables, {len(J)} couplings")
    print()
    print(
        f"{'Config':<28} {'Sampler':<16} "
        f"{'Time':>7} {'Min E':>8} {'Avg E':>8} "
        f"{'Samples/s':>10}"
    )
    print("-" * 93)

    # --- Single model, 1024 sweeps ---
    label = "1 model, 1024sw, 100rd"
    sweeps, reads, nm = 1024, 100, 1
    ts = reads * nm

    t, mi, av = bench_cuda_gibbs(h, J, reads, sweeps, nm)
    fmt_row(label, "CUDA Gibbs", t, mi, av, ts)

    t, mi, av = bench_cuda_sa(h, J, reads, sweeps, nm)
    fmt_row(label, "CUDA SA", t, mi, av, ts)

    t, mi, av = bench_cpu_sa(h, J, reads, sweeps, nm)
    fmt_row(label, "CPU SA", t, mi, av, ts)
    print()

    # --- Single model, 2048 sweeps ---
    label = "1 model, 2048sw, 150rd"
    sweeps, reads, nm = 2048, 150, 1
    ts = reads * nm

    t, mi, av = bench_cuda_gibbs(h, J, reads, sweeps, nm)
    fmt_row(label, "CUDA Gibbs", t, mi, av, ts)

    t, mi, av = bench_cuda_sa(h, J, reads, sweeps, nm)
    fmt_row(label, "CUDA SA", t, mi, av, ts)

    t, mi, av = bench_cpu_sa(h, J, reads, sweeps, nm)
    fmt_row(label, "CPU SA", t, mi, av, ts)
    print()

    # --- 12 models, 1024 sweeps (skip CPU - too slow) ---
    label = "12 models, 1024sw, 100rd"
    sweeps, reads, nm = 1024, 100, 12
    ts = reads * nm

    t, mi, av = bench_cuda_gibbs(h, J, reads, sweeps, nm)
    fmt_row(label, "CUDA Gibbs", t, mi, av, ts)

    t, mi, av = bench_cuda_sa(h, J, reads, sweeps, nm)
    fmt_row(label, "CUDA SA", t, mi, av, ts)
    print()

    # --- 12 models, 2048 sweeps ---
    label = "12 models, 2048sw, 150rd"
    sweeps, reads, nm = 2048, 150, 12
    ts = reads * nm

    t, mi, av = bench_cuda_gibbs(h, J, reads, sweeps, nm)
    fmt_row(label, "CUDA Gibbs", t, mi, av, ts)

    t, mi, av = bench_cuda_sa(h, J, reads, sweeps, nm)
    fmt_row(label, "CUDA SA", t, mi, av, ts)
    print()


if __name__ == "__main__":
    main()
