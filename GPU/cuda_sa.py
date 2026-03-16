# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""CUDA Simulated Annealing Sampler - self-feeding persistent kernel.

3-slot rotating buffer architecture: the kernel autonomously grabs
READY slots via atomicCAS, processes SA sweeps with thread-local
unpacked state, marks COMPLETE, and grabs the next slot. No host
signaling needed.

1 block per nonce, 1 SM per block. 48 SMs → 48 concurrent nonces.
"""
import time
from typing import Dict, List, Optional, Tuple

import cupy as cp
import dimod
import numpy as np

from GPU.base_cuda_sampler import BaseCudaSampler


SA_NUM_REGIONS = 10


class CudaSASampler(BaseCudaSampler):
    """Self-feeding SA sampler using CUDA GPU.

    Each nonce gets 1 block (1 SM) with 3 rotating slots.
    Threads within the block process reads independently
    using thread-local state + delta_energy workspace.
    """

    def __init__(
        self,
        topology=None,
        max_sms: int = 0,
        profile: bool = False,
    ):
        super().__init__(
            topology=topology,
            max_sms=max_sms,
            profile=profile,
            sampler_type="cuda-sa",
        )

    # -- BaseCudaSampler hooks --

    def _kernel_filename(self) -> str:
        return 'cuda_sa.cu'

    def _kernel_function_name(self) -> str:
        return 'cuda_sa_self_feeding'

    def _num_profile_regions(self) -> int:
        return SA_NUM_REGIONS

    @property
    def _sms_per_nonce(self) -> int:
        return 1

    def _allocate_kernel_buffers(
        self,
        num_nonces: int,
        reads_per_nonce: int,
        num_sweeps: int,
        num_sweeps_per_beta: int,
        **kwargs,
    ) -> None:
        """Allocate SA-specific delta energy workspace."""
        N = self._prep_N
        total_threads = num_nonces * 256
        self._d_sf_delta_energy = cp.zeros(
            total_threads * N, dtype=cp.int8,
        )

    def _kernel_launch_args(
        self,
        active: int,
        num_betas: int,
        seed: int,
    ) -> tuple:
        N = self._prep_N
        nnz = self._prep_nnz
        max_packed_size = self._prep_max_packed_size
        num_nonces = self._sf_num_nonces

        return (
            self._d_row_ptr,
            self._d_col_ind,
            self._d_sf_J,
            self._d_sf_h,
            self._d_sf_samples,
            self._d_sf_energies,
            self._d_sf_beta,
            np.int32(num_betas),
            np.int32(self._sf_num_sweeps_per_beta),
            self._d_sf_ctrl,
            np.int32(num_nonces),
            np.int32(self._sf_reads_per_nonce),
            np.int32(N),
            np.int32(nnz),
            np.int32(max_packed_size),
            np.uint32(seed),
            self._d_sf_delta_energy,
            np.int32(N),
        )

    # -- SA-specific sample_ising --

    def sample_ising(
        self,
        h: List[Dict[int, float]],
        J: List[Dict[Tuple[int, int], float]],
        num_reads: int = 200,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[dimod.SampleSet]:
        """Sample from Ising model using self-feeding kernel.

        Args:
            h: List of linear biases per problem.
            J: List of quadratic biases per problem.
            num_reads: Number of independent samples per
                problem.
            num_sweeps: Total number of sweeps.
            num_sweeps_per_beta: Sweeps per beta value.
            beta_range: (hot_beta, cold_beta) or None for
                auto.
            beta_schedule_type: Schedule type.
            seed: RNG seed.

        Returns:
            List of dimod.SampleSet, one per problem.
        """
        num_problems = len(h)
        assert len(J) == num_problems, (
            f"h and J must have same length: "
            f"{num_problems} vs {len(J)}"
        )

        if not self._prepared:
            self.prepare(
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                num_sweeps_per_beta=num_sweeps_per_beta,
            )

        if not self._sf_prepared:
            self.prepare_self_feeding(
                num_nonces=num_problems,
                reads_per_nonce=num_reads,
                num_sweeps=num_sweeps,
                num_sweeps_per_beta=num_sweeps_per_beta,
            )

        # Reset ctrl array (clears stale EXIT_NOW from
        # previous sample_ising call)
        self._d_sf_ctrl[:] = 0

        # Upload beta schedule
        num_betas, beta_range = self.upload_beta_schedule(
            h[0], J[0], num_sweeps,
            num_sweeps_per_beta, beta_range,
            beta_schedule_type,
        )

        # Upload one model per nonce to slot 0
        for i in range(num_problems):
            self.upload_slot(i, 0, h[i], J[i])

        # Launch kernel
        self._sf_kernel_running = False  # allow re-launch
        self.launch_self_feeding(
            num_betas=num_betas,
            seed=seed,
            active_nonce_count=num_problems,
        )

        # Poll until all nonces complete
        completed = set()
        deadline = time.time() + 300.0
        while len(completed) < num_problems:
            assert time.time() < deadline, (
                "sample_ising timed out waiting for kernel"
            )
            for nonce_id, slot_id in self.poll_completions():
                if nonce_id not in completed:
                    completed.add(nonce_id)
            if len(completed) < num_problems:
                time.sleep(0.001)

        # Download results
        results = []
        for i in range(num_problems):
            ss = self.download_slot(i, 0)
            results.append(ss)

        # Signal exit and wait
        self.signal_exit()

        return results
