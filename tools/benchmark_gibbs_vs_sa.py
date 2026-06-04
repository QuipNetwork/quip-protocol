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
from typing import Any


sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.quantum_proof_of_work import (
    generate_ising_model_from_nonce,
)
from dwave_topologies import DEFAULT_TOPOLOGY


def _summarize(
    elapsed: float, sample_sets: list[Any]
) -> tuple[float, float, float]:
    """Aggregate elapsed time and energies from a list of SampleSets."""
    energies: list[float] = []
    for ss in sample_sets:
        energies.extend(list(ss.record.energy))
    return elapsed, min(energies), sum(energies) / len(energies)


def _bench_batch(
    mode: str,
    h: dict,
    J: dict,
    num_reads: int,
    num_sweeps: int,
    n_models: int,
) -> tuple[float, float, float]:
    """Benchmark a CUDA batch sampler (mode='gibbs' or mode='sa').

    Returns (elapsed, min_energy, avg_energy).
    """
    if mode == "gibbs":
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        sampler = CudaGibbsSampler(update_mode="gibbs", parallel=True)
        sampler.sample_ising(h=[h], J=[J], num_reads=2, num_sweeps=100)
        h_batch = [h] * n_models
        J_batch = [J] * n_models
        start = time.time()
        results = sampler.sample_ising(
            h=h_batch, J=J_batch,
            num_reads=num_reads, num_sweeps=num_sweeps,
        )
        elapsed = time.time() - start
    else:
        from GPU.cuda_sa import CudaSASampler

        sampler = CudaSASampler()
        sampler.sample_ising([h], [J], num_reads=2, num_sweeps=100)
        h_batch = [h] * n_models
        J_batch = [J] * n_models
        start = time.time()
        results = sampler.sample_ising(
            h=h_batch, J=J_batch,
            num_reads=num_reads, num_sweeps=num_sweeps,
        )
        elapsed = time.time() - start
        sampler.close()

    return _summarize(elapsed, results)


def bench_cpu_sa(
    h: dict, J: dict, num_reads: int, num_sweeps: int, n_models: int
) -> tuple[float, float, float]:
    """Benchmark CPU SA sampler."""
    from CPU.sa_sampler import SimulatedAnnealingStructuredSampler

    sampler = SimulatedAnnealingStructuredSampler()
    start = time.time()
    all_ss = []
    for _ in range(n_models):
        ss = sampler.sample_ising(h, J, num_reads=num_reads, num_sweeps=num_sweeps)
        all_ss.append(ss)
    elapsed = time.time() - start
    return _summarize(elapsed, all_ss)


def fmt_row(
    label: str, name: str, t: float, mine: float, avge: float, total_samples: int
) -> None:
    """Format a single result row."""
    sps = total_samples / t if t > 0 else 0
    print(
        f"{label:<28} {name:<16} "
        f"{t:>7.2f}s {mine:>8.1f} {avge:>8.1f} "
        f"{sps:>10.1f}"
    )


def main() -> None:
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

    # (label, sweeps, reads, n_models, include_cpu)
    configs = [
        ("1 model, 1024sw, 100rd",   1024, 100,  1, True),
        ("1 model, 2048sw, 150rd",   2048, 150,  1, True),
        ("12 models, 1024sw, 100rd", 1024, 100, 12, False),
        ("12 models, 2048sw, 150rd", 2048, 150, 12, False),
    ]

    for label, sweeps, reads, nm, include_cpu in configs:
        ts = reads * nm
        t, mi, av = _bench_batch("gibbs", h, J, reads, sweeps, nm)
        fmt_row(label, "CUDA Gibbs", t, mi, av, ts)

        t, mi, av = _bench_batch("sa", h, J, reads, sweeps, nm)
        fmt_row(label, "CUDA SA", t, mi, av, ts)

        if include_cpu:
            t, mi, av = bench_cpu_sa(h, J, reads, sweeps, nm)
            fmt_row(label, "CPU SA", t, mi, av, ts)

        print()


if __name__ == "__main__":
    main()
