"""Consensus-energy tests for shared.problem_prep array reduction.

The array transport (feeder → ProblemView → submitter) must not change a single
energy: validators recompute energy on the full topology, so a reduced sample's
energy + the clamp offset MUST equal the full-topology energy with the clamped
spins inserted. These tests prove that round-trip without the D-Wave SDK.
"""

from __future__ import annotations

import itertools

import numpy as np

from shared.problem_prep import (
    clamp_fixed_variables,
    live_topology,
    rebuild_ising,
    reduce_to_arrays,
)

_NODES = [0, 1, 2, 3]
_EDGES = [(0, 1), (1, 2), (2, 3), (0, 3)]
_H = {0: 0.5, 1: -1.0, 2: 0.3, 3: 0.8}
_J = {(0, 1): 1.0, (1, 2): -0.5, (2, 3): 0.7, (0, 3): -0.2}


def _ising_energy(h, J, spins):
    """Full Ising energy of a {node: ±1} assignment."""
    e = sum(h.get(n, 0.0) * spins[n] for n in spins)
    e += sum(val * spins[u] * spins[v] for (u, v), val in J.items())
    return e


def _milli(e):
    """Energy at chain-consensus precision (validators compare integer milli).

    Direct guards against float summation-order ULP noise that is irrelevant to
    consensus — the chain ranks on ``best_energy_milli``, an int.
    """
    return round(e * 1000)


class TestArrayRoundTrip:
    def test_rebuild_equals_clamp_output_with_defects(self):
        dq, de = [1], set()
        h_red, J_red, fixed, offset, removed = clamp_fixed_variables(
            _H, _J, 42, dq, de
        )
        live_nodes, live_edges = live_topology(_NODES, _EDGES, dq, de)
        h_vec, j_vec, di = reduce_to_arrays(_H, _J, 42, dq, de, live_nodes, live_edges)
        h_rb, J_rb = rebuild_ising(h_vec, j_vec, live_nodes, live_edges)

        # Arrays rebuild to exactly the clamp's reduced problem.
        assert h_rb == h_red
        assert J_rb == J_red
        # defect_info matches the clamp metadata used by reconstruction.
        assert di is not None
        assert di.fixed_spins == fixed
        assert di.energy_offset == offset
        assert di.removed_edges == removed

    def test_no_defects_round_trips_full_problem(self):
        live_nodes, live_edges = live_topology(_NODES, _EDGES, [], set())
        h_vec, j_vec, di = reduce_to_arrays(_H, _J, 7, [], set(), live_nodes, live_edges)
        h_rb, J_rb = rebuild_ising(h_vec, j_vec, live_nodes, live_edges)
        assert di is None
        assert h_rb == _H
        assert J_rb == _J
        assert h_vec.dtype == np.float64 and j_vec.dtype == np.float64
        assert h_vec.shape == (len(_NODES),) and j_vec.shape == (len(_EDGES),)

    def test_array_shapes_match_problemview_contract(self):
        dq, de = [1], set()
        live_nodes, live_edges = live_topology(_NODES, _EDGES, dq, de)
        h_vec, j_vec, _ = reduce_to_arrays(_H, _J, 1, dq, de, live_nodes, live_edges)
        # ProblemView is sized (len(live_nodes), len(live_edges)).
        assert h_vec.shape == (len(live_nodes),)
        assert j_vec.shape == (len(live_edges),)


class TestEnergyConsensus:
    def test_reduced_plus_offset_equals_full_energy_with_clamp(self):
        """For every live assignment, reduced energy + offset == full energy."""
        dq, de = [1], set()
        live_nodes, live_edges = live_topology(_NODES, _EDGES, dq, de)
        h_vec, j_vec, di = reduce_to_arrays(_H, _J, 99, dq, de, live_nodes, live_edges)
        h_rb, J_rb = rebuild_ising(h_vec, j_vec, live_nodes, live_edges)
        s_clamped = di.fixed_spins  # {1: ±1}, deterministic from nonce_seed=99

        for combo in itertools.product((-1, 1), repeat=len(live_nodes)):
            live_spins = dict(zip(live_nodes, combo))
            reduced_e = _ising_energy(h_rb, J_rb, live_spins)
            full_spins = {**live_spins, **s_clamped}
            full_e = _ising_energy(_H, _J, full_spins)
            assert _milli(reduced_e + di.energy_offset) == _milli(full_e), (
                f"consensus break at {full_spins}: "
                f"{reduced_e} + {di.energy_offset} != {full_e}"
            )

    def test_defective_coupler_contribution_tracked_in_removed_edges(self):
        """A defective coupler (both endpoints live) leaves the reduced J but its
        value is preserved in removed_edges for per-sample reconstruction."""
        dq, de = [], {(0, 3)}
        live_nodes, live_edges = live_topology(_NODES, _EDGES, dq, de)
        h_vec, j_vec, di = reduce_to_arrays(_H, _J, 5, dq, de, live_nodes, live_edges)
        h_rb, J_rb = rebuild_ising(h_vec, j_vec, live_nodes, live_edges)

        assert (0, 3) not in J_rb              # excluded from the submitted problem
        assert di.removed_edges == {(0, 3): _J[(0, 3)]}  # but preserved for energy

        # Full energy == reduced energy + per-sample defective-coupler term.
        for combo in itertools.product((-1, 1), repeat=len(live_nodes)):
            spins = dict(zip(live_nodes, combo))
            reduced_e = _ising_energy(h_rb, J_rb, spins)
            removed_term = _J[(0, 3)] * spins[0] * spins[3]
            full_e = _ising_energy(_H, _J, spins)
            assert _milli(reduced_e + di.energy_offset + removed_term) == _milli(full_e)
