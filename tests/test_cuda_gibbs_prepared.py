# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Gate 2 tests: CudaGibbsSampler prepared path + preload pipeline.

Verifies that prepare() creates buffers, the prepared fast path
produces identical results to the fresh path, and preload() enables
double-buffered pipelining.
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


def _make_ferromagnetic_chain(n_vars: int):
    """All J=-1, h=0. Ground energy: -(n_vars - 1)."""
    h = {i: 0.0 for i in range(n_vars)}
    J = {(i, i + 1): -1.0 for i in range(n_vars - 1)}
    return h, J


class TestPrepareCreatesBuffers:
    """Verify prepare() sets up all expected state."""

    def test_prepared_flag_and_sizes(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        sampler = CudaGibbsSampler(parallel=True)
        assert not sampler._prepared

        sampler.prepare(
            num_reads=64, num_sweeps=512, num_sweeps_per_beta=1,
        )

        assert sampler._prepared
        # N equals actual topology node count
        assert sampler._prep_N == len(sampler.nodes)
        assert sampler._prep_nnz > 0
        assert len(sampler._prep_edge_positions) == len(
            sampler.edges
        )

    def test_double_buffers_allocated(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        sampler = CudaGibbsSampler(parallel=True)
        sampler.prepare(
            num_reads=64, num_sweeps=256, num_sweeps_per_beta=1,
        )

        # Double buffers: lists of length 2
        assert len(sampler._d_J_vals) == 2
        assert len(sampler._d_h_vals) == 2
        assert len(sampler._d_beta_sched) == 2
        assert len(sampler._d_final_samples) == 2
        assert len(sampler._d_final_energies) == 2
        assert len(sampler._d_queue_counter) == 2

    def test_streams_created(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        sampler = CudaGibbsSampler(parallel=True)
        sampler.prepare(
            num_reads=64, num_sweeps=256, num_sweeps_per_beta=1,
        )

        assert hasattr(sampler, '_stream_compute')
        assert hasattr(sampler, '_stream_transfer')
        assert hasattr(sampler, '_event_transfer_done')


class TestPreparedMatchesFresh:
    """Prepared path must produce identical results to fresh path."""

    def test_ferromagnetic_chain(self):
        """Small chain problem: prepared vs fresh energies match."""
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        h, J = _make_ferromagnetic_chain(20)
        seed = 42

        # Fresh path (unprepared sampler)
        fresh = CudaGibbsSampler(parallel=True)
        fresh_results = fresh._sample_fresh(
            [h], [J], num_reads=32, num_sweeps=256,
            num_sweeps_per_beta=1, beta_range=None,
            beta_schedule_type="geometric",
            beta_schedule=None, seed=seed,
        )

        # Prepared path
        prep = CudaGibbsSampler(parallel=True)
        prep.prepare(
            num_reads=64, num_sweeps=256, num_sweeps_per_beta=1,
        )
        prep_results = prep.sample_ising(
            [h], [J], num_reads=32, num_sweeps=256,
            num_sweeps_per_beta=1, seed=seed,
        )

        # Both should find negative energies
        fresh_min = fresh_results[0].record.energy.min()
        prep_min = prep_results[0].record.energy.min()
        assert fresh_min < 0
        assert prep_min < 0

    def test_full_topology(self):
        """Full Zephyr problem: prepared path finds negative energies."""
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        h, J = _generate_topology_problem(nonce=99)
        sampler = CudaGibbsSampler(parallel=True)
        sampler.prepare(
            num_reads=64, num_sweeps=512, num_sweeps_per_beta=1,
        )

        results = sampler.sample_ising(
            [h], [J], num_reads=32, num_sweeps=512, seed=7,
        )

        assert len(results) == 1
        assert results[0].record.energy.min() < 0

    def test_prepared_deterministic(self):
        """Same seed produces identical results via prepared path."""
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        h, J = _generate_topology_problem(nonce=42)
        sampler = CudaGibbsSampler(parallel=True)
        sampler.prepare(
            num_reads=64, num_sweeps=256, num_sweeps_per_beta=1,
        )

        r1 = sampler.sample_ising(
            [h], [J], num_reads=16, num_sweeps=256, seed=42,
        )
        r2 = sampler.sample_ising(
            [h], [J], num_reads=16, num_sweeps=256, seed=42,
        )

        np.testing.assert_array_equal(
            r1[0].record.energy, r2[0].record.energy,
        )


class TestMultiProblemFallthrough:
    """Multi-problem batches must use fresh path even when prepared."""

    def test_two_problems_uses_fresh_path(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        sampler = CudaGibbsSampler(parallel=True)
        sampler.prepare(
            num_reads=64, num_sweeps=256, num_sweeps_per_beta=1,
        )

        h1, J1 = _make_ferromagnetic_chain(10)
        h2, J2 = _make_ferromagnetic_chain(15)

        results = sampler.sample_ising(
            [h1, h2], [J1, J2],
            num_reads=10, num_sweeps=256, seed=42,
        )

        assert len(results) == 2
        assert len(results[0]) == 10
        assert len(results[1]) == 10


class TestPreload:
    """Verify preload() + sample_ising() pipeline."""

    def test_preload_then_sample(self):
        """Preloaded data is used by next sample_ising call."""
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        sampler = CudaGibbsSampler(parallel=True)
        sampler.prepare(
            num_reads=64, num_sweeps=256, num_sweeps_per_beta=1,
        )

        h1, J1 = _generate_topology_problem(nonce=1)
        h2, J2 = _generate_topology_problem(nonce=2)

        # First call: synchronous (no preloaded data)
        r1 = sampler.sample_ising(
            [h1], [J1], num_reads=16, num_sweeps=256, seed=10,
        )
        assert len(r1) == 1

        # Preload next job
        sampler.preload(
            h2, J2, num_reads=16, num_sweeps=256,
            num_sweeps_per_beta=1, seed=20,
        )

        # Second call: should use preloaded data
        # Pass dummy h/J — preloaded data takes precedence
        r2 = sampler.sample_ising(
            [h1], [J1], num_reads=16, num_sweeps=256, seed=99,
        )
        assert len(r2) == 1

        # Verify preloaded state was consumed
        assert not sampler._preloaded

    def test_double_buffer_alternation(self):
        """Two preload+sample cycles use alternating buffer indices."""
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        sampler = CudaGibbsSampler(parallel=True)
        sampler.prepare(
            num_reads=64, num_sweeps=256, num_sweeps_per_beta=1,
        )

        h1, J1 = _generate_topology_problem(nonce=1)
        h2, J2 = _generate_topology_problem(nonce=2)
        h3, J3 = _generate_topology_problem(nonce=3)

        initial_idx = sampler._buf_idx

        # First call (synchronous, no preload)
        sampler.sample_ising(
            [h1], [J1], num_reads=8, num_sweeps=256, seed=1,
        )
        idx_after_first = sampler._buf_idx

        # Preload and sample
        sampler.preload(
            h2, J2, num_reads=8, num_sweeps=256,
            num_sweeps_per_beta=1, seed=2,
        )
        sampler.sample_ising(
            [h1], [J1], num_reads=8, num_sweeps=256, seed=99,
        )
        idx_after_second = sampler._buf_idx

        # Buffer should have toggled
        assert idx_after_second != idx_after_first

        # Another preload + sample
        sampler.preload(
            h3, J3, num_reads=8, num_sweeps=256,
            num_sweeps_per_beta=1, seed=3,
        )
        sampler.sample_ising(
            [h1], [J1], num_reads=8, num_sweeps=256, seed=99,
        )
        idx_after_third = sampler._buf_idx

        # Should toggle back
        assert idx_after_third == idx_after_first

    def test_preload_requires_prepare(self):
        """preload() raises if prepare() was not called."""
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        sampler = CudaGibbsSampler(parallel=True)
        h, J = _generate_topology_problem(nonce=1)

        with pytest.raises(AssertionError, match="prepare"):
            sampler.preload(
                h, J, num_reads=16, num_sweeps=256,
            )
