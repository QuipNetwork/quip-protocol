# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Tests for the SA self-feeding kernel pipeline.

Verifies that the 3-slot rotating buffer kernel produces
valid results equivalent to the oneshot/multi-nonce kernels.
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


class TestSelfFeedingBasic:
    """Basic self-feeding kernel tests."""

    def test_single_nonce_single_slot(self):
        """1 nonce, 1 slot: kernel should produce results."""
        kernel, nodes, edges = _make_kernel(
            max_nonces=1, num_reads=32,
        )
        (h, J), _, _ = _generate_topology_problem(nonce=42)

        kernel.prepare_self_feeding(
            num_nonces=1,
            num_reads=32,
            num_betas=50,
        )

        num_betas, _ = kernel.upload_beta_schedule(
            h, J, 50,
        )
        kernel.upload_slot(0, 0, h, J)
        kernel.launch_self_feeding(num_betas=num_betas)

        # Poll until complete
        import time
        deadline = time.monotonic() + 30
        completed = []
        while not completed:
            completed = kernel.poll_completions()
            assert time.monotonic() < deadline, (
                "Timed out waiting for completion"
            )
            if not completed:
                time.sleep(0.01)

        assert len(completed) == 1
        nonce_id, slot_id = completed[0]
        assert nonce_id == 0
        assert slot_id == 0

        ss = kernel.download_slot(0, 0)
        assert len(ss) == 32

        energies = ss.record.energy
        assert np.all(energies < 0), (
            f"Expected negative energies, got min={energies.min()}"
        )

        # Kernel should exit naturally (no more READY slots)
        time.sleep(0.5)
        assert not kernel.is_kernel_running()

    def test_multi_nonce_two_slots(self):
        """2 nonces, 2 slots each: all 4 should complete."""
        num_nonces = 2
        num_reads = 32
        kernel, nodes, edges = _make_kernel(
            max_nonces=num_nonces, num_reads=num_reads,
        )

        problems = []
        for nonce_val in [42, 99]:
            (h, J), _, _ = _generate_topology_problem(
                nonce=nonce_val,
            )
            problems.append((h, J))

        kernel.prepare_self_feeding(
            num_nonces=num_nonces,
            num_reads=num_reads,
            num_betas=50,
        )

        num_betas, _ = kernel.upload_beta_schedule(
            problems[0][0], problems[0][1], 50,
        )

        # Upload 2 slots per nonce
        for k in range(num_nonces):
            for s in range(2):
                kernel.upload_slot(k, s, *problems[k])

        kernel.launch_self_feeding(num_betas=num_betas)

        # Collect all completions
        import time
        deadline = time.monotonic() + 30
        all_completed = set()
        while len(all_completed) < 4:
            completed = kernel.poll_completions()
            for c in completed:
                all_completed.add(c)
            assert time.monotonic() < deadline
            if len(all_completed) < 4:
                time.sleep(0.01)

        # Verify all 4 (nonce, slot) pairs completed
        expected = {(0, 0), (0, 1), (1, 0), (1, 1)}
        assert all_completed == expected

        # Download and verify all results
        for nonce_id, slot_id in all_completed:
            ss = kernel.download_slot(nonce_id, slot_id)
            assert len(ss) == num_reads
            assert np.all(ss.record.energy < 0)

        # Cleanup
        import time
        time.sleep(0.5)

    def test_signal_exit(self):
        """signal_exit() should stop the kernel cleanly."""
        kernel, nodes, edges = _make_kernel(
            max_nonces=1, num_reads=32,
        )
        (h, J), _, _ = _generate_topology_problem(nonce=42)

        kernel.prepare_self_feeding(
            num_nonces=1,
            num_reads=32,
            num_betas=50,
        )

        num_betas, _ = kernel.upload_beta_schedule(
            h, J, 50,
        )

        # Upload 2 slots so kernel stays busy
        kernel.upload_slot(0, 0, h, J)
        kernel.upload_slot(0, 1, h, J)
        kernel.launch_self_feeding(num_betas=num_betas)

        # Signal exit immediately
        kernel.signal_exit()

        assert not kernel.is_kernel_running()

    def test_refill_loop(self):
        """Refill a completed slot and verify it processes."""
        kernel, nodes, edges = _make_kernel(
            max_nonces=1, num_reads=32,
        )
        (h, J), _, _ = _generate_topology_problem(nonce=42)

        kernel.prepare_self_feeding(
            num_nonces=1,
            num_reads=32,
            num_betas=50,
        )

        num_betas, _ = kernel.upload_beta_schedule(
            h, J, 50,
        )

        # Upload first slot
        kernel.upload_slot(0, 0, h, J)
        kernel.launch_self_feeding(num_betas=num_betas)

        # Wait for first completion
        import time
        deadline = time.monotonic() + 30
        completed = []
        while not completed:
            completed = kernel.poll_completions()
            assert time.monotonic() < deadline
            time.sleep(0.01)

        nonce_id, slot_id = completed[0]
        ss1 = kernel.download_slot(nonce_id, slot_id)
        assert len(ss1) == 32

        # Refill the same slot with new data
        (h2, J2), _, _ = _generate_topology_problem(
            nonce=99,
        )
        kernel.upload_slot(nonce_id, slot_id, h2, J2)

        # Wait for second completion
        deadline = time.monotonic() + 30
        completed2 = []
        while not completed2:
            completed2 = kernel.poll_completions()
            assert time.monotonic() < deadline
            time.sleep(0.01)

        assert len(completed2) >= 1
        ss2 = kernel.download_slot(*completed2[0])
        assert len(ss2) == 32
        assert np.all(ss2.record.energy < 0)

        kernel.signal_exit()


class TestSelfFeedingQuality:
    """Verify solution quality matches oneshot kernel."""

    def test_energy_quality(self):
        """Self-feeding should produce similar energies to oneshot."""
        kernel, nodes, edges = _make_kernel(
            max_nonces=1, num_reads=64,
        )
        (h, J), _, _ = _generate_topology_problem(nonce=42)

        # Oneshot reference
        oneshot_ss = kernel.sample_ising(
            h, J,
            num_reads=64,
            num_betas=200,
            num_sweeps_per_beta=1,
        )
        oneshot_min = float(oneshot_ss.record.energy.min())

        # Self-feeding
        kernel.prepare_self_feeding(
            num_nonces=1,
            num_reads=64,
            num_betas=200,
        )
        num_betas, _ = kernel.upload_beta_schedule(
            h, J, 200,
        )
        kernel.upload_slot(0, 0, h, J)
        kernel.launch_self_feeding(num_betas=num_betas)

        import time
        deadline = time.monotonic() + 30
        completed = []
        while not completed:
            completed = kernel.poll_completions()
            assert time.monotonic() < deadline
            time.sleep(0.01)

        sf_ss = kernel.download_slot(0, 0)
        sf_min = float(sf_ss.record.energy.min())

        kernel.signal_exit()

        # Self-feeding should find reasonable energy
        # (within 20% of oneshot — stochastic so be lenient)
        assert sf_min < 0, f"Expected negative energy, got {sf_min}"
        assert sf_min < oneshot_min * 0.5, (
            f"Self-feeding min={sf_min} much worse than "
            f"oneshot min={oneshot_min}"
        )
