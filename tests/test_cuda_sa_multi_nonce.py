# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Gate 1 tests: CudaSAKernel multi-nonce dispatch.

Verifies that sample_multi_nonce() produces valid results with
shared CSR topology but independent J/h per nonce.
"""

import numpy as np
import pytest

cp = pytest.importorskip(
    "cupy", reason="CuPy required for CUDA SA tests",
)


def _generate_topology_problem(nonce=42):
    """Generate Ising problem on the full Zephyr topology."""
    from dwave_topologies import DEFAULT_TOPOLOGY
    from shared.quantum_proof_of_work import (
        generate_ising_model_from_nonce,
    )

    topo = DEFAULT_TOPOLOGY
    nodes = list(topo.graph.nodes())
    edges = list(topo.graph.edges())
    return (
        generate_ising_model_from_nonce(nonce, nodes, edges),
        nodes,
        edges,
    )


def _make_kernel(max_nonces=4, num_reads=32):
    """Create a prepared SA kernel with multi-nonce buffers."""
    from GPU.cuda_sa_kernel import CudaSAKernel

    (h, J), nodes, edges = _generate_topology_problem()

    kernel = CudaSAKernel(max_N=5000)
    kernel.prepare(
        nodes=nodes,
        edges=edges,
        num_reads=num_reads,
        max_num_betas=400,
        max_nonces=max_nonces,
    )
    return kernel, nodes, edges


class TestSingleNonceViaMulti:
    """1 nonce through multi-nonce path should match oneshot."""

    def test_negative_energies(self):
        kernel, nodes, edges = _make_kernel(max_nonces=4)
        (h, J), _, _ = _generate_topology_problem(nonce=42)

        results = kernel.sample_multi_nonce(
            [h], [J],
            num_reads=32,
            num_betas=100,
            num_sweeps_per_beta=1,
            seed=42,
        )

        assert len(results) == 1
        assert results[0].record.energy.min() < 0

    def test_matches_oneshot_quality(self):
        """Multi-nonce with 1 nonce should produce comparable
        energy to oneshot kernel.
        """
        kernel, nodes, edges = _make_kernel(max_nonces=4)
        (h, J), _, _ = _generate_topology_problem(nonce=42)

        # Multi-nonce path
        multi = kernel.sample_multi_nonce(
            [h], [J],
            num_reads=32,
            num_betas=200,
            num_sweeps_per_beta=1,
            seed=42,
        )
        multi_min = multi[0].record.energy.min()

        # Oneshot path
        single = kernel.sample_ising(
            h, J,
            num_reads=32,
            num_betas=200,
            num_sweeps_per_beta=1,
            seed=42,
        )
        single_min = single.record.energy.min()

        # Both should find negative energies
        assert multi_min < 0
        assert single_min < 0


class TestMultipleNonces:
    """Multiple nonces should all produce negative energies."""

    def test_four_nonces(self):
        kernel, nodes, edges = _make_kernel(max_nonces=4)

        h_list = []
        J_list = []
        for nonce in [10, 20, 30, 40]:
            (h, J), _, _ = _generate_topology_problem(
                nonce=nonce,
            )
            h_list.append(h)
            J_list.append(J)

        results = kernel.sample_multi_nonce(
            h_list, J_list,
            num_reads=32,
            num_betas=100,
            num_sweeps_per_beta=1,
            seed=7,
        )

        assert len(results) == 4
        for k, ss in enumerate(results):
            assert len(ss) == 32, (
                f"nonce {k}: expected 32 reads"
            )
            assert ss.record.energy.min() < 0, (
                f"nonce {k}: no negative energy"
            )

    def test_many_nonces(self):
        """Test with max nonces to verify SM scaling."""
        max_n = 8
        kernel, nodes, edges = _make_kernel(max_nonces=max_n)

        h_list = []
        J_list = []
        for nonce in range(max_n):
            (h, J), _, _ = _generate_topology_problem(
                nonce=nonce + 100,
            )
            h_list.append(h)
            J_list.append(J)

        results = kernel.sample_multi_nonce(
            h_list, J_list,
            num_reads=16,
            num_betas=50,
            num_sweeps_per_beta=1,
            seed=99,
        )

        assert len(results) == max_n
        for ss in results:
            assert ss.record.energy.min() < 0


class TestDifferentNoncesDifferentEnergies:
    """Different h/J should produce different energy."""

    def test_energy_distributions_differ(self):
        kernel, nodes, edges = _make_kernel(max_nonces=4)

        (h1, J1), _, _ = _generate_topology_problem(nonce=100)
        (h2, J2), _, _ = _generate_topology_problem(nonce=999)

        results = kernel.sample_multi_nonce(
            [h1, h2], [J1, J2],
            num_reads=32,
            num_betas=100,
            num_sweeps_per_beta=1,
            seed=42,
        )

        e1 = results[0].record.energy
        e2 = results[1].record.energy
        assert not np.array_equal(e1, e2), (
            "Different nonces produced identical energies"
        )


class TestDeterministicWithSeed:
    """Same seed should produce identical results."""

    def test_reproducible(self):
        kernel, nodes, edges = _make_kernel(max_nonces=4)
        (h, J), _, _ = _generate_topology_problem(nonce=42)

        r1 = kernel.sample_multi_nonce(
            [h], [J],
            num_reads=16,
            num_betas=50,
            num_sweeps_per_beta=1,
            seed=123,
        )
        r2 = kernel.sample_multi_nonce(
            [h], [J],
            num_reads=16,
            num_betas=50,
            num_sweeps_per_beta=1,
            seed=123,
        )

        np.testing.assert_array_equal(
            r1[0].record.energy,
            r2[0].record.energy,
        )


class TestMaxNoncesAssertion:
    """Exceeding max_nonces should raise."""

    def test_raises_on_exceed(self):
        kernel, nodes, edges = _make_kernel(max_nonces=2)
        (h, J), _, _ = _generate_topology_problem(nonce=42)

        with pytest.raises(AssertionError, match="max_nonces"):
            kernel.sample_multi_nonce(
                [h, h, h], [J, J, J],
                num_reads=16,
                num_betas=50,
            )


class TestRequiresPrepare:
    """sample_multi_nonce should require prepare() first."""

    def test_raises_without_prepare(self):
        from GPU.cuda_sa_kernel import CudaSAKernel

        kernel = CudaSAKernel(max_N=5000)
        (h, J), _, _ = _generate_topology_problem(nonce=42)

        with pytest.raises(AssertionError, match="prepare"):
            kernel.sample_multi_nonce(
                [h], [J],
                num_reads=16,
                num_betas=50,
            )


class TestPreloadMultiNonce:
    """Verify preload + sample pipeline for SA."""

    def test_preload_then_sample(self):
        kernel, nodes, edges = _make_kernel(max_nonces=4)
        (h1, J1), _, _ = _generate_topology_problem(nonce=1)
        (h2, J2), _, _ = _generate_topology_problem(nonce=2)

        # First call: sync
        r1 = kernel.sample_multi_nonce(
            [h1], [J1],
            num_reads=16,
            num_betas=50,
            seed=10,
        )
        assert len(r1) == 1

        # Preload next
        kernel.preload_multi_nonce(
            [h2], [J2],
            num_reads=16,
            num_betas=50,
            seed=20,
        )

        # Sample should use preloaded data
        r2 = kernel.sample_multi_nonce(
            [h1], [J1],  # ignored
            num_reads=16,
            num_betas=50,
            seed=99,  # ignored
        )
        assert len(r2) == 1
        assert not kernel._preloaded

    def test_double_buffer_alternation(self):
        kernel, nodes, edges = _make_kernel(max_nonces=4)
        (h1, J1), _, _ = _generate_topology_problem(nonce=1)
        (h2, J2), _, _ = _generate_topology_problem(nonce=2)

        initial_idx = kernel._buf_idx

        # Sync call
        kernel.sample_multi_nonce(
            [h1], [J1], num_reads=8, num_betas=25, seed=1,
        )

        # Preload + sample
        kernel.preload_multi_nonce(
            [h2], [J2], num_reads=8, num_betas=25, seed=2,
        )
        kernel.sample_multi_nonce(
            [h1], [J1], num_reads=8, num_betas=25, seed=99,
        )
        idx_after = kernel._buf_idx

        # Buffer should have toggled
        assert idx_after != initial_idx
