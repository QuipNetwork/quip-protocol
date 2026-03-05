# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for CUDA block Gibbs sampler."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy", reason="CuPy required for CUDA Gibbs tests")


def make_ferromagnetic_chain(n_vars: int):
    """Create a ferromagnetic chain: all J=-1, h=0.

    Ground state: all spins aligned (+1 or -1).
    Ground energy: -(n_vars - 1).
    """
    h = {i: 0.0 for i in range(n_vars)}
    J = {(i, i + 1): -1.0 for i in range(n_vars - 1)}
    return h, J


def make_biased_ferromagnet(n_vars: int):
    """Ferromagnetic chain with positive h bias.

    Ground state: all spins +1.
    Ground energy: -(n_vars - 1) - n_vars = -(2*n_vars - 1).
    """
    h = {i: -1.0 for i in range(n_vars)}
    J = {(i, i + 1): -1.0 for i in range(n_vars - 1)}
    return h, J


class TestKernelCompilation:
    """Smoke tests for kernel compilation."""

    def test_sampler_creates_successfully(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler
        sampler = CudaGibbsSampler(update_mode="gibbs", parallel=True)
        assert sampler is not None
        assert sampler.update_mode == 0
        assert sampler.parallel is True

    def test_sampler_metropolis_mode(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler
        sampler = CudaGibbsSampler(
            update_mode="metropolis", parallel=False
        )
        assert sampler.update_mode == 1
        assert sampler.parallel is False

    def test_invalid_update_mode(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler
        with pytest.raises(ValueError, match="update_mode"):
            CudaGibbsSampler(update_mode="invalid")


class TestSequentialKernel:
    """Tests for the sequential (Gauss-Seidel) kernel."""

    def test_small_problem_runs(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler
        sampler = CudaGibbsSampler(parallel=False)

        h, J = make_ferromagnetic_chain(10)
        results = sampler.sample_ising(
            [h], [J], num_reads=5, num_sweeps=100, seed=42
        )

        assert len(results) == 1
        ss = results[0]
        assert len(ss) == 5
        # Energies should be negative (ferromagnetic coupling)
        assert all(e <= 0 for e in ss.record.energy)

    def test_ferromagnetic_finds_ground_state(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler
        sampler = CudaGibbsSampler(parallel=False)

        n = 20
        h, J = make_ferromagnetic_chain(n)
        ground_energy = -(n - 1)

        results = sampler.sample_ising(
            [h], [J], num_reads=50, num_sweeps=500, seed=123
        )

        min_energy = results[0].record.energy.min()
        # Should find ground state or very close
        assert min_energy <= ground_energy + 2, (
            f"Expected energy <= {ground_energy + 2}, got {min_energy}"
        )

    def test_metropolis_mode(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler
        sampler = CudaGibbsSampler(
            update_mode="metropolis", parallel=False
        )

        h, J = make_ferromagnetic_chain(10)
        results = sampler.sample_ising(
            [h], [J], num_reads=10, num_sweeps=200, seed=42
        )

        assert len(results) == 1
        assert all(e <= 0 for e in results[0].record.energy)

    def test_biased_problem(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler
        sampler = CudaGibbsSampler(parallel=False)

        n = 15
        h, J = make_biased_ferromagnet(n)
        ground_energy = -(2 * n - 1)

        results = sampler.sample_ising(
            [h], [J], num_reads=30, num_sweeps=500, seed=99
        )

        min_energy = results[0].record.energy.min()
        assert min_energy <= ground_energy + 4


class TestParallelKernel:
    """Tests for the Jacobi-style parallel kernel."""

    def test_small_problem_runs(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler
        sampler = CudaGibbsSampler(parallel=True)

        h, J = make_ferromagnetic_chain(10)
        results = sampler.sample_ising(
            [h], [J], num_reads=10, num_sweeps=200, seed=42
        )

        assert len(results) == 1
        assert len(results[0]) == 10
        # At least some samples should find low energy
        assert results[0].record.energy.min() < 0

    def test_ferromagnetic_finds_ground_state(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler
        sampler = CudaGibbsSampler(parallel=True)

        n = 20
        h, J = make_ferromagnetic_chain(n)
        ground_energy = -(n - 1)

        results = sampler.sample_ising(
            [h], [J], num_reads=50, num_sweeps=500, seed=123
        )

        min_energy = results[0].record.energy.min()
        assert min_energy <= ground_energy + 2

    def test_metropolis_mode(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler
        sampler = CudaGibbsSampler(
            update_mode="metropolis", parallel=True
        )

        h, J = make_ferromagnetic_chain(10)
        results = sampler.sample_ising(
            [h], [J], num_reads=10, num_sweeps=200, seed=42
        )

        assert len(results) == 1


class TestSeedReproducibility:
    """Verify that same seed produces same results."""

    def test_sequential_deterministic(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        h, J = make_ferromagnetic_chain(15)

        sampler = CudaGibbsSampler(parallel=False)
        r1 = sampler.sample_ising(
            [h], [J], num_reads=10, num_sweeps=100, seed=42
        )
        r2 = sampler.sample_ising(
            [h], [J], num_reads=10, num_sweeps=100, seed=42
        )

        np.testing.assert_array_equal(
            r1[0].record.energy, r2[0].record.energy
        )

    def test_parallel_deterministic(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        h, J = make_ferromagnetic_chain(15)

        sampler = CudaGibbsSampler(parallel=True)
        r1 = sampler.sample_ising(
            [h], [J], num_reads=10, num_sweeps=100, seed=42
        )
        r2 = sampler.sample_ising(
            [h], [J], num_reads=10, num_sweeps=100, seed=42
        )

        np.testing.assert_array_equal(
            r1[0].record.energy, r2[0].record.energy
        )


class TestMultiProblem:
    """Test batch processing of multiple problems."""

    def test_two_problems(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler
        sampler = CudaGibbsSampler(parallel=False)

        h1, J1 = make_ferromagnetic_chain(10)
        h2, J2 = make_biased_ferromagnet(10)

        results = sampler.sample_ising(
            [h1, h2], [J1, J2],
            num_reads=10, num_sweeps=200, seed=42,
        )

        assert len(results) == 2
        assert len(results[0]) == 10
        assert len(results[1]) == 10


class TestBetaSchedule:
    """Test different beta schedule configurations."""

    def test_linear_schedule(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler
        sampler = CudaGibbsSampler(parallel=False)

        h, J = make_ferromagnetic_chain(10)
        results = sampler.sample_ising(
            [h], [J], num_reads=5, num_sweeps=100,
            beta_schedule_type="linear", seed=42,
        )
        assert len(results[0]) == 5

    def test_custom_schedule(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler
        sampler = CudaGibbsSampler(parallel=False)

        h, J = make_ferromagnetic_chain(10)
        custom_betas = np.linspace(0.1, 5.0, 50, dtype=np.float32)
        results = sampler.sample_ising(
            [h], [J], num_reads=5, num_sweeps=50,
            num_sweeps_per_beta=1,
            beta_schedule_type="custom",
            beta_schedule=custom_betas, seed=42,
        )
        assert len(results[0]) == 5

    def test_explicit_beta_range(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler
        sampler = CudaGibbsSampler(parallel=False)

        h, J = make_ferromagnetic_chain(10)
        results = sampler.sample_ising(
            [h], [J], num_reads=5, num_sweeps=100,
            beta_range=(0.1, 10.0), seed=42,
        )
        assert len(results[0]) == 5


def _generate_topology_problem(nonce=42):
    """Generate Ising problem on the full Zephyr topology."""
    from dwave_topologies import DEFAULT_TOPOLOGY
    from shared.quantum_proof_of_work import generate_ising_model_from_nonce

    topo = DEFAULT_TOPOLOGY
    nodes = list(topo.graph.nodes())
    edges = list(topo.graph.edges())
    return generate_ising_model_from_nonce(nonce, nodes, edges)


class TestFullTopology:
    """Test with full Zephyr Z(9,2) topology."""

    def test_zephyr_topology_parallel(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        sampler = CudaGibbsSampler(parallel=True)
        h, J = _generate_topology_problem(nonce=42)

        results = sampler.sample_ising(
            [h], [J], num_reads=10, num_sweeps=200, seed=42
        )

        assert len(results) == 1
        assert len(results[0]) == 10
        assert results[0].record.energy.min() < 0

    def test_zephyr_topology_sequential(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        sampler = CudaGibbsSampler(parallel=False)
        h, J = _generate_topology_problem(nonce=42)

        results = sampler.sample_ising(
            [h], [J], num_reads=5, num_sweeps=200, seed=42
        )

        assert len(results) == 1
        assert len(results[0]) == 5
        assert results[0].record.energy.min() < 0


class TestJacobiVsSequentialQuality:
    """Verify both kernel variants produce comparable quality."""

    def test_similar_energy_distributions(self):
        from GPU.cuda_gibbs_sa import CudaGibbsSampler

        h, J = _generate_topology_problem(nonce=7)

        seq_sampler = CudaGibbsSampler(parallel=False)
        par_sampler = CudaGibbsSampler(parallel=True)

        seq_results = seq_sampler.sample_ising(
            [h], [J], num_reads=20, num_sweeps=500, seed=42
        )
        par_results = par_sampler.sample_ising(
            [h], [J], num_reads=20, num_sweeps=500, seed=42
        )

        seq_min = seq_results[0].record.energy.min()
        par_min = par_results[0].record.energy.min()

        # Both should find reasonably low energies
        # Allow tolerance since Jacobi and Gauss-Seidel converge
        # differently
        assert par_min < seq_min * 0.5 or par_min < -100, (
            f"Parallel min={par_min} much worse than "
            f"sequential min={seq_min}"
        )
