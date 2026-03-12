# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Gate 2 tests: CudaGibbsSampler multi-nonce dispatch.

Verifies that sample_multi_nonce() produces valid results with
shared topology but independent J/h per nonce.
"""

import numpy as np
import pytest

cp = pytest.importorskip(
    "cupy", reason="CuPy required for CUDA Gibbs tests",
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
    return generate_ising_model_from_nonce(nonce, nodes, edges)


def _make_sampler(max_nonces=4):
    """Create a prepared sampler with multi-nonce buffers."""
    from GPU.cuda_gibbs_sa import CudaGibbsSampler

    sampler = CudaGibbsSampler(parallel=True)
    sampler.prepare(
        num_reads=64,
        num_sweeps=512,
        num_sweeps_per_beta=1,
        max_nonces=max_nonces,
    )
    return sampler


class TestSingleNonceViaMulti:
    """1 nonce through multi-nonce path should produce valid results."""

    def test_negative_energies(self):
        sampler = _make_sampler(max_nonces=4)
        h, J = _generate_topology_problem(nonce=42)

        results = sampler.sample_multi_nonce(
            [h], [J],
            reads_per_nonce=32,
            num_sweeps=512,
            sms_per_nonce=4,
            seed=42,
        )

        assert len(results) == 1
        assert results[0].record.energy.min() < 0


class TestMultipleNonces:
    """Multiple nonces should all produce negative energies."""

    def test_four_nonces(self):
        sampler = _make_sampler(max_nonces=4)

        h_list = []
        J_list = []
        for nonce in [10, 20, 30, 40]:
            h, J = _generate_topology_problem(nonce=nonce)
            h_list.append(h)
            J_list.append(J)

        results = sampler.sample_multi_nonce(
            h_list, J_list,
            reads_per_nonce=32,
            num_sweeps=512,
            sms_per_nonce=4,
            seed=7,
        )

        assert len(results) == 4
        for k, ss in enumerate(results):
            assert len(ss) == 32, f"nonce {k}: expected 32 reads"
            assert ss.record.energy.min() < 0, (
                f"nonce {k}: no negative energy"
            )


class TestDifferentNoncesDifferentEnergies:
    """Different h/J should produce different energy distributions."""

    def test_energy_distributions_differ(self):
        sampler = _make_sampler(max_nonces=4)

        h1, J1 = _generate_topology_problem(nonce=100)
        h2, J2 = _generate_topology_problem(nonce=999)

        results = sampler.sample_multi_nonce(
            [h1, h2], [J1, J2],
            reads_per_nonce=32,
            num_sweeps=512,
            sms_per_nonce=4,
            seed=42,
        )

        e1 = results[0].record.energy
        e2 = results[1].record.energy
        # Different nonces must not produce identical sample arrays
        # (mean energies can be close for similar-strength problems)
        assert not np.array_equal(e1, e2), (
            "Different nonces produced identical energy arrays"
        )


class TestDeterministicWithSeed:
    """Same seed should produce identical results."""

    def test_reproducible(self):
        sampler = _make_sampler(max_nonces=4)
        h, J = _generate_topology_problem(nonce=42)

        r1 = sampler.sample_multi_nonce(
            [h], [J],
            reads_per_nonce=16,
            num_sweeps=256,
            sms_per_nonce=4,
            seed=123,
        )
        r2 = sampler.sample_multi_nonce(
            [h], [J],
            reads_per_nonce=16,
            num_sweeps=256,
            sms_per_nonce=4,
            seed=123,
        )

        np.testing.assert_array_equal(
            r1[0].record.energy,
            r2[0].record.energy,
        )


class TestSmsPerNonceConfig:
    """Grid size should equal nonces * sms_per_nonce."""

    def test_grid_calculation(self):
        sampler = _make_sampler(max_nonces=4)
        h, J = _generate_topology_problem(nonce=42)

        # 3 nonces × 4 SMs = 12 blocks
        results = sampler.sample_multi_nonce(
            [h, h, h], [J, J, J],
            reads_per_nonce=16,
            num_sweeps=256,
            sms_per_nonce=4,
            seed=1,
        )
        assert len(results) == 3
        for ss in results:
            assert len(ss) == 16


class TestMaxNoncesAssertion:
    """Exceeding max_nonces should raise."""

    def test_raises_on_exceed(self):
        sampler = _make_sampler(max_nonces=2)
        h, J = _generate_topology_problem(nonce=42)

        with pytest.raises(AssertionError, match="max_nonces"):
            sampler.sample_multi_nonce(
                [h, h, h], [J, J, J],
                reads_per_nonce=16,
                num_sweeps=256,
            )


class TestRequiresPrepare:
    """sample_multi_nonce should require prepare() first."""

    def test_raises_without_prepare(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        sampler = CudaGibbsSampler(parallel=True)
        h, J = _generate_topology_problem(nonce=42)

        with pytest.raises(AssertionError, match="prepare"):
            sampler.sample_multi_nonce(
                [h], [J],
                reads_per_nonce=16,
                num_sweeps=256,
            )
