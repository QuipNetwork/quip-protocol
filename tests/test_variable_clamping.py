"""Tests for QPU variable clamping (defect-tolerant mining).

When a QPU qubit goes offline, the DWaveSamplerWrapper clamps it to a
deterministic spin and absorbs its coupling energy into neighbors' h-fields.
This file tests that the clamping math is correct and consensus-safe.
"""

import numpy as np
import pytest
import dimod

from shared.quantum_proof_of_work import generate_ising_model_from_nonce


# ---------------------------------------------------------------------------
# Helpers — lightweight stand-ins so we don't need a live D-Wave connection
# ---------------------------------------------------------------------------

def _make_small_topology():
    """Create a small 6-node graph for testing.

    Graph:
        0 -- 1 -- 2
        |    |    |
        3 -- 4 -- 5
    """
    nodes = [0, 1, 2, 3, 4, 5]
    edges = [
        (0, 1), (1, 2),
        (0, 3), (1, 4), (2, 5),
        (3, 4), (4, 5),
    ]
    return nodes, edges


def _ising_energy(sample, h, J):
    """Compute Ising energy: E = sum(h_i * s_i) + sum(J_ij * s_i * s_j)."""
    energy = sum(h.get(i, 0.0) * sample[i] for i in sample)
    for (u, v), val in J.items():
        if u in sample and v in sample:
            energy += val * sample[u] * sample[v]
    return energy


# ---------------------------------------------------------------------------
# Import the clamping logic — we instantiate the class methods directly
# by calling them as plain functions with a mock self.
# ---------------------------------------------------------------------------

class MockSamplerWrapper:
    """Minimal mock that has the clamping methods from DWaveSamplerWrapper."""

    def __init__(self, defective_qubits, defective_edges=None):
        self._defective_qubits = defective_qubits
        self._defective_edges = defective_edges or set()

    def _clamp_defective_qubits(self, h, J, nonce_seed):
        """Copied interface — import the real implementation."""
        from QPU.dwave_sampler import DWaveSamplerWrapper
        # Call the unbound method with self substituted
        return DWaveSamplerWrapper._clamp_defective_qubits(self, h, J, nonce_seed)

    def _reconstruct_full_sampleset(self, reduced_ss, fixed_spins, full_h, full_J):
        from QPU.dwave_sampler import DWaveSamplerWrapper
        return DWaveSamplerWrapper._reconstruct_full_sampleset(
            self, reduced_ss, fixed_spins, full_h, full_J
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClampDefectiveQubits:
    """Tests for _clamp_defective_qubits()."""

    def test_no_defects_returns_unchanged(self):
        """With no defective qubits, h and J pass through unchanged."""
        nodes, edges = _make_small_topology()
        h = {i: 0.5 for i in nodes}
        J = {e: -1.0 for e in edges}

        wrapper = MockSamplerWrapper(defective_qubits=[])
        h_r, J_r, fixed = wrapper._clamp_defective_qubits(h, J, nonce_seed=42)

        assert h_r == h
        assert J_r == J
        assert fixed == {}

    def test_single_defect_removes_qubit(self):
        """A single defective qubit is removed from h and J."""
        nodes, edges = _make_small_topology()
        h = {i: 0.0 for i in nodes}
        J = {e: -1.0 for e in edges}

        wrapper = MockSamplerWrapper(defective_qubits=[4])
        h_r, J_r, fixed = wrapper._clamp_defective_qubits(h, J, nonce_seed=42)

        # Qubit 4 should not be in reduced h
        assert 4 not in h_r
        # Qubit 4's neighbors: 1, 3, 5 (from edges (1,4), (3,4), (4,5))
        # Their h should be adjusted
        assert len(h_r) == 5  # 6 - 1

        # No edge should reference qubit 4
        for u, v in J_r:
            assert u != 4 and v != 4

        # fixed_spins should have qubit 4
        assert 4 in fixed
        assert fixed[4] in (-1, 1)

    def test_neighbor_h_adjustment(self):
        """Clamped qubit's coupling is correctly absorbed into neighbors."""
        # Simple: 3 nodes in a line: 0 -- 1 -- 2
        h = {0: 0.0, 1: 0.0, 2: 0.0}
        J = {(0, 1): -1.0, (1, 2): 1.0}

        # Clamp node 1 (the middle one)
        wrapper = MockSamplerWrapper(defective_qubits=[1])
        h_r, J_r, fixed = wrapper._clamp_defective_qubits(h, J, nonce_seed=123)

        s1 = fixed[1]  # deterministic spin for node 1

        # Node 0's h should be adjusted by J[0,1] * s1 = -1.0 * s1
        assert h_r[0] == pytest.approx(-1.0 * s1)
        # Node 2's h should be adjusted by J[1,2] * s1 = 1.0 * s1
        assert h_r[2] == pytest.approx(1.0 * s1)

        # No edges should remain (both edges touch node 1)
        assert len(J_r) == 0

    def test_deterministic_clamping(self):
        """Same nonce_seed always produces same fixed spins."""
        nodes, edges = _make_small_topology()
        h = {i: 0.0 for i in nodes}
        J = {e: -1.0 for e in edges}

        wrapper = MockSamplerWrapper(defective_qubits=[2, 4])

        _, _, fixed1 = wrapper._clamp_defective_qubits(h, J, nonce_seed=999)
        _, _, fixed2 = wrapper._clamp_defective_qubits(h, J, nonce_seed=999)

        assert fixed1 == fixed2

    def test_different_nonce_different_spins(self):
        """Different nonce_seeds produce different fixed spins (statistical)."""
        nodes, edges = _make_small_topology()
        h = {i: 0.0 for i in nodes}
        J = {e: -1.0 for e in edges}

        wrapper = MockSamplerWrapper(defective_qubits=[0, 1, 2, 3, 4, 5])

        results = set()
        for seed in range(100):
            _, _, fixed = wrapper._clamp_defective_qubits(h, J, seed)
            results.add(tuple(fixed[i] for i in sorted(fixed)))

        # With 6 qubits, 2^6 = 64 possible spin configs. 100 seeds should
        # produce more than 1 unique config.
        assert len(results) > 1

    def test_multiple_defects(self):
        """Multiple defective qubits are all removed and neighbors adjusted."""
        nodes, edges = _make_small_topology()
        h = {i: 1.0 for i in nodes}
        J = {e: -1.0 for e in edges}

        # Remove nodes 0 and 5 (corners)
        wrapper = MockSamplerWrapper(defective_qubits=[0, 5])
        h_r, J_r, fixed = wrapper._clamp_defective_qubits(h, J, nonce_seed=7)

        assert 0 not in h_r
        assert 5 not in h_r
        assert len(fixed) == 2
        assert set(fixed.keys()) == {0, 5}

        # Remaining nodes: 1, 2, 3, 4
        assert set(h_r.keys()) == {1, 2, 3, 4}

        # Edges touching 0 or 5 should be gone
        for u, v in J_r:
            assert u not in (0, 5) and v not in (0, 5)


class TestReconstructFullSampleset:
    """Tests for _reconstruct_full_sampleset()."""

    def test_reconstruction_inserts_fixed_spins(self):
        """Fixed spins appear in reconstructed samples."""
        h = {0: 0.0, 1: 0.5, 2: -0.5}
        J = {(0, 1): -1.0, (1, 2): 1.0, (0, 2): -1.0}

        # Simulate QPU returned samples for nodes 0, 2 (node 1 was clamped)
        reduced_samples = [{0: 1, 2: -1}, {0: -1, 2: 1}]
        reduced_ss = dimod.SampleSet.from_samples(
            reduced_samples, vartype=dimod.SPIN, energy=[0, 0]
        )

        fixed_spins = {1: 1}

        wrapper = MockSamplerWrapper(defective_qubits=[1])
        full_ss = wrapper._reconstruct_full_sampleset(
            reduced_ss, fixed_spins, h, J
        )

        # All samples should have all 3 variables
        for sample in full_ss.samples():
            assert set(sample.keys()) == {0, 1, 2}
            assert sample[1] == 1  # Fixed spin

    def test_energy_recomputed_correctly(self):
        """Reconstructed energies match manual Ising energy calculation."""
        h = {0: 1.0, 1: -1.0, 2: 0.5}
        J = {(0, 1): -1.0, (1, 2): 1.0}

        # Node 1 clamped to +1
        fixed_spins = {1: 1}

        # QPU solved for nodes 0, 2
        reduced_samples = [{0: 1, 2: -1}]
        reduced_ss = dimod.SampleSet.from_samples(
            reduced_samples, vartype=dimod.SPIN, energy=[0]
        )

        wrapper = MockSamplerWrapper(defective_qubits=[1])
        full_ss = wrapper._reconstruct_full_sampleset(
            reduced_ss, fixed_spins, h, J
        )

        # Manual energy: h[0]*1 + h[1]*1 + h[2]*(-1) + J[0,1]*1*1 + J[1,2]*1*(-1)
        # = 1.0 + (-1.0) + (-0.5) + (-1.0) + (-1.0) = -2.5
        expected_energy = _ising_energy({0: 1, 1: 1, 2: -1}, h, J)
        assert full_ss.first.energy == pytest.approx(expected_energy)

    def test_timing_info_preserved(self):
        """QPU timing info survives reconstruction."""
        h = {0: 0.0, 1: 0.0}
        J = {(0, 1): -1.0}
        fixed_spins = {1: 1}

        reduced_ss = dimod.SampleSet.from_samples(
            [{0: 1}], vartype=dimod.SPIN, energy=[0],
            info={'timing': {'qpu_access_time': 12345}},
        )

        wrapper = MockSamplerWrapper(defective_qubits=[1])
        full_ss = wrapper._reconstruct_full_sampleset(
            reduced_ss, fixed_spins, h, J
        )

        assert 'timing' in full_ss.info
        assert full_ss.info['timing']['qpu_access_time'] == 12345


class TestClampingEnergyConsistency:
    """End-to-end tests verifying energy consistency with clamping."""

    def test_clamped_energy_matches_full_topology(self):
        """Energy of clamped solution equals energy on full Ising model.

        This is the critical consensus test: validators compute energy on
        the full topology, so the reconstructed sampleset must match.
        """
        nodes, edges = _make_small_topology()
        nonce = 42
        h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

        # Clamp node 4
        wrapper = MockSamplerWrapper(defective_qubits=[4])
        h_r, J_r, fixed = wrapper._clamp_defective_qubits(
            dict(h), dict(J), nonce_seed=nonce
        )

        # Simulate QPU returning a solution for the reduced problem
        reduced_sample = {i: 1 if i % 2 == 0 else -1 for i in h_r}
        reduced_ss = dimod.SampleSet.from_samples(
            [reduced_sample], vartype=dimod.SPIN, energy=[0]
        )

        full_ss = wrapper._reconstruct_full_sampleset(
            reduced_ss, fixed, dict(h), dict(J)
        )

        # Verify energy matches manual calculation
        full_sample = dict(full_ss.first.sample)
        expected = _ising_energy(full_sample, h, J)
        assert full_ss.first.energy == pytest.approx(expected)

    def test_clamped_energy_vs_direct_energy(self):
        """Clamped+reconstructed energy equals direct evaluation.

        Use the same spin configuration for both direct and clamped paths
        to verify mathematical equivalence.
        """
        nodes, edges = _make_small_topology()
        nonce = 77
        h, J = generate_ising_model_from_nonce(nonce, nodes, edges)

        defective = [2]
        wrapper = MockSamplerWrapper(defective_qubits=defective)
        h_r, J_r, fixed = wrapper._clamp_defective_qubits(
            dict(h), dict(J), nonce_seed=nonce
        )

        # Choose a spin config for the reduced problem
        reduced_sample = {i: 1 for i in h_r}

        # Build full sample by inserting fixed spins
        full_sample = dict(reduced_sample)
        full_sample.update(fixed)

        # Energy computed directly on full model
        direct_energy = _ising_energy(full_sample, h, J)

        # Energy via reconstruction
        reduced_ss = dimod.SampleSet.from_samples(
            [reduced_sample], vartype=dimod.SPIN, energy=[0]
        )
        full_ss = wrapper._reconstruct_full_sampleset(
            reduced_ss, fixed, dict(h), dict(J)
        )

        assert full_ss.first.energy == pytest.approx(direct_energy)
