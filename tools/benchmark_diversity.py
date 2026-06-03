#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
"""Benchmark the diversity distance matrix: broadcast vs matmul.

Proves the matmul rewrite is (a) numerically identical to the pairwise
Hamming ground truth and (b) faster, at production shape (n=112 reads,
N=4578 nodes). Reports torch-CPU as an informational reference only — the
production function uses numpy (zero new dependency for the QPU image).

Run:  python tools/benchmark_diversity.py --n 112 --features 4578 --trials 5
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from shared.quantum_proof_of_work import (
    _compute_distance_matrix_vectorized,
    calculate_hamming_distance,
)


def _broadcast_matrix(solutions: list[list[int]]) -> np.ndarray:
    arr = np.array(solutions, dtype=np.int8)
    a1 = arr[:, np.newaxis, :]
    a2 = arr[np.newaxis, :, :]
    dist_normal = np.count_nonzero(a1 != a2, axis=2)
    dist_inverted = np.count_nonzero(a1 != -a2, axis=2)
    return np.minimum(dist_normal, dist_inverted).astype(np.float64)


def _ground_truth(solutions: list[list[int]]) -> np.ndarray:
    n = len(solutions)
    m = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            m[i, j] = calculate_hamming_distance(solutions[i], solutions[j])
    return m


def _time(fn, sols, trials: int) -> float:
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        fn(sols)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n", type=int, default=112)
    ap.add_argument("--features", type=int, default=4578)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sols = (
        rng.integers(0, 2, size=(args.n, args.features)) * 2 - 1
    ).tolist()

    matmul = _compute_distance_matrix_vectorized(sols)
    broadcast = _broadcast_matrix(sols)
    if not np.array_equal(
        np.rint(matmul).astype(np.int64), np.rint(broadcast).astype(np.int64)
    ):
        diff = matmul - broadcast
        raise AssertionError(
            f"matmul != broadcast: {int(np.count_nonzero(diff))} differing entries, "
            f"max|diff|={float(np.max(np.abs(diff)))}"
        )
    if args.n <= 40:
        truth = _ground_truth(sols)
        assert np.array_equal(
            np.rint(matmul).astype(np.int64), np.rint(truth).astype(np.int64)
        ), "matmul != truth"

    t_broadcast = _time(_broadcast_matrix, sols, args.trials)
    t_matmul = _time(_compute_distance_matrix_vectorized, sols, args.trials)
    print(f"shape: n={args.n} features={args.features} trials={args.trials}")
    print(f"broadcast (old): {t_broadcast * 1e3:8.2f} ms")
    print(f"matmul   (new):  {t_matmul * 1e3:8.2f} ms")
    print(f"speedup:         {t_broadcast / t_matmul:8.1f}x")

    try:
        import torch  # noqa: PLC0415 — optional reference only
        # arr built outside the timed loop, so torch excludes the list->tensor
        # conversion cost that numpy's np.asarray pays inside the function —
        # the torch number is an optimistic reference, not an apples-to-apples.
        arr = torch.tensor(sols, dtype=torch.float32)

        def _torch_cpu(_sols):
            g = arr @ arr.t()
            return ((args.features - g.abs()) / 2.0).numpy()

        t_torch = _time(_torch_cpu, sols, args.trials)
        print(f"matmul torch-CPU (ref): {t_torch * 1e3:8.2f} ms")
    except ImportError:
        print("matmul torch-CPU (ref): torch not installed — skipped")


if __name__ == "__main__":
    main()
