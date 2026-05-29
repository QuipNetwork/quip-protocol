# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np

from shared.quantum_proof_of_work import (
    _compute_distance_matrix_vectorized,
    calculate_hamming_distance,
)


def _ground_truth(solutions):
    n = len(solutions)
    m = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            m[i, j] = calculate_hamming_distance(solutions[i], solutions[j])
    return m


def test_distance_matrix_matches_pairwise_hamming():
    rng = np.random.default_rng(42)
    sols = (rng.integers(0, 2, size=(20, 64)) * 2 - 1).tolist()  # ±1
    got = _compute_distance_matrix_vectorized(sols)
    expected = _ground_truth(sols)
    assert np.array_equal(got, expected)


def test_distance_matrix_flip_symmetry():
    s = (np.random.default_rng(1).integers(0, 2, size=64) * 2 - 1)
    sols = [s.tolist(), (-s).tolist()]
    got = _compute_distance_matrix_vectorized(sols)
    assert got[0, 1] == 0.0
    assert got[0, 0] == 0.0


def test_distance_matrix_large_topology_exact():
    rng = np.random.default_rng(7)
    sols = (rng.integers(0, 2, size=(30, 4578)) * 2 - 1).tolist()
    got = _compute_distance_matrix_vectorized(sols)
    expected = _ground_truth(sols)
    assert np.array_equal(got, expected)
