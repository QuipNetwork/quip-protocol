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
    assert np.allclose(got, np.rint(got)), "distances must be integer-valued"
    assert np.array_equal(
        np.rint(got).astype(np.int64), np.rint(expected).astype(np.int64)
    )


def test_distance_matrix_flip_symmetry():
    s = (np.random.default_rng(1).integers(0, 2, size=64) * 2 - 1)
    sols = [s.tolist(), (-s).tolist()]
    got = _compute_distance_matrix_vectorized(sols)
    assert got[0, 1] == 0.0 and got[1, 0] == 0.0
    assert got[0, 0] == 0.0 and got[1, 1] == 0.0


def test_distance_matrix_singleton_and_identical():
    # n=1 → 1x1 zero matrix
    assert _compute_distance_matrix_vectorized([[1, -1, 1]]).tolist() == [[0.0]]
    # all-identical rows → all-zero matrix
    got = _compute_distance_matrix_vectorized([[1, -1, 1]] * 3)
    assert np.array_equal(got, np.zeros((3, 3)))


def test_distance_matrix_large_topology_exact():
    rng = np.random.default_rng(7)
    sols = (rng.integers(0, 2, size=(30, 4578)) * 2 - 1).tolist()
    got = _compute_distance_matrix_vectorized(sols)
    expected = _ground_truth(sols)
    assert np.allclose(got, np.rint(got)), "distances must be integer-valued"
    assert np.array_equal(
        np.rint(got).astype(np.int64), np.rint(expected).astype(np.int64)
    )


# ---------------------------------------------------------------------------
# _ising_from_requirements: complete-topology contract (fail loud)
# ---------------------------------------------------------------------------

import pytest  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from shared.allowed_value_spec import AllowedValueSet  # noqa: E402
from shared.miner_types import BlockRequirements  # noqa: E402
from shared.quantum_proof_of_work import _ising_from_requirements  # noqa: E402

_ZERO_H = AllowedValueSet((0,))
_BIN_J = AllowedValueSet((-1000, 1000))
_NODES = [0, 1, 2]
_EDGES = [(0, 1), (1, 2)]


def _reqs(allowed_h, allowed_j):
    return SimpleNamespace(allowed_h_values=allowed_h, allowed_j_values=allowed_j)


def test_ising_from_requirements_crashes_on_missing_allowed_h():
    """No silent ternary fallback: a missing h spec must crash, not default."""
    with pytest.raises(ValueError, match="incomplete topology reference"):
        _ising_from_requirements(_reqs(None, _BIN_J), 0, _NODES, _EDGES)


def test_ising_from_requirements_crashes_on_missing_allowed_j():
    with pytest.raises(ValueError, match="incomplete topology reference"):
        _ising_from_requirements(_reqs(_ZERO_H, None), 0, _NODES, _EDGES)


def test_ising_from_requirements_crashes_on_missing_attributes():
    """A requirements object lacking the attributes entirely also crashes."""
    with pytest.raises(ValueError, match="incomplete topology reference"):
        _ising_from_requirements(SimpleNamespace(), 0, _NODES, _EDGES)


def test_ising_from_requirements_crashes_on_empty_nodes_or_edges():
    with pytest.raises(ValueError, match="nodes=0"):
        _ising_from_requirements(_reqs(_ZERO_H, _BIN_J), 0, [], _EDGES)
    with pytest.raises(ValueError, match="edges=0"):
        _ising_from_requirements(_reqs(_ZERO_H, _BIN_J), 0, _NODES, [])


def test_ising_from_requirements_succeeds_with_complete_zero_field_spec():
    """A complete h=0 reference generates an all-zero field, non-empty J."""
    h, J = _ising_from_requirements(_reqs(_ZERO_H, _BIN_J), 0, _NODES, _EDGES)
    assert set(h.values()) == {0.0}
    assert len(J) == len(_EDGES)


def test_block_requirements_does_not_silently_default_allowed_values():
    """The masking ternary default is gone: unset allowed_* stays None so the
    incomplete-topology guard can fire instead of scoring a ternary problem."""
    reqs = BlockRequirements(
        difficulty_energy=-14000.0,
        min_diversity=0.0,
        min_solutions=1,
        timeout_to_difficulty_adjustment_decay=2**31 - 1,
    )
    assert reqs.allowed_h_values is None
    assert reqs.allowed_j_values is None
