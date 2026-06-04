# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""CUDA Simulated Annealing Sampler - self-feeding persistent kernel.

3-slot rotating buffer architecture: the kernel autonomously grabs
READY slots via atomicCAS, processes SA sweeps with thread-local
unpacked state, marks COMPLETE, and grabs the next slot. No host
signaling needed.

1 block per nonce, 1 SM per block. 48 SMs → 48 concurrent nonces.
"""
import cupy as cp
import numpy as np

from GPU.base_cuda_sampler import BaseCudaSampler


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

    def _profiling_mode(self) -> str:
        return "per_thread"

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
