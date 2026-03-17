# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Per-job CUDA SA kernel launcher.

Simple synchronous interface: build CSR, launch kernel, return results.
No ring buffers, no polling, no persistent kernel state.
"""

import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import cupy as cp
import dimod
import numpy as np

from shared.beta_schedule import _default_ising_beta_range


class CudaSAKernel:
    """Per-job CUDA simulated annealing kernel.

    Each call to sample_ising() launches a fresh kernel, waits for
    completion, and returns results. Uses 1 block with up to 256
    threads (one thread per SA read).

    Args:
        max_N: Maximum problem size (nodes). Default 5000.
    """

    def __init__(self, max_N: int = 5000):
        self.logger = logging.getLogger(__name__)
        self.max_N = max_N

        # Compile kernel
        kernel_path = os.path.join(
            os.path.dirname(__file__), 'cuda_sa.cu'
        )
        with open(kernel_path, 'r') as f:
            kernel_code = f.read()

        self._module = cp.RawModule(
            code=kernel_code,
            options=('--use_fast_math',),
        )
        self._kernel = self._module.get_function('cuda_sa_oneshot')

        # Query device
        dev = cp.cuda.Device()
        self.device_sms = dev.attributes['MultiProcessorCount']
        self.logger.debug(
            f"CudaSAKernel compiled, device has {self.device_sms} SMs"
        )

    def sample_ising(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        num_reads: int = 100,
        num_betas: int = 50,
        num_sweeps_per_beta: int = 1,
        seed: Optional[int] = None,
        beta_range: Optional[Tuple[float, float]] = None,
    ) -> dimod.SampleSet:
        """Run SA on a single Ising problem.

        Args:
            h: Linear biases {node: bias}.
            J: Quadratic couplings {(i, j): coupling}.
            num_reads: Number of independent SA runs.
            num_betas: Number of temperature steps.
            num_sweeps_per_beta: Sweeps per temperature step.
            seed: RNG seed.
            beta_range: (hot_beta, cold_beta) or None for auto.

        Returns:
            dimod.SampleSet with num_reads samples.
        """
        num_reads = min(num_reads, 256)

        # Determine N from h and J
        max_node = 0
        if h:
            max_node = max(max_node, max(h.keys()))
        if J:
            max_node = max(
                max_node,
                max(max(i, j) for i, j in J.keys()),
            )
        N = max_node + 1

        # Build symmetric CSR
        csr_row_ptr, csr_col_ind, csr_J_vals = self._build_csr(
            J, N
        )

        # Build h array
        h_arr = np.zeros(N, dtype=np.float32)
        for node, val in h.items():
            if node < N:
                h_arr[node] = float(val)

        # Beta schedule
        if beta_range is None:
            beta_range = _default_ising_beta_range(h, J)
        hot_beta, cold_beta = beta_range
        if num_betas == 1:
            beta_sched = np.array([cold_beta], dtype=np.float32)
        else:
            beta_sched = np.geomspace(
                hot_beta, cold_beta, num=num_betas,
                dtype=np.float32,
            )

        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        # Transfer to GPU
        d_row_ptr = cp.asarray(csr_row_ptr, dtype=cp.int32)
        d_col_ind = cp.asarray(csr_col_ind, dtype=cp.int32)
        d_J_vals = cp.asarray(csr_J_vals, dtype=cp.int8)
        d_h = cp.asarray(h_arr, dtype=cp.float32)
        d_beta = cp.asarray(beta_sched, dtype=cp.float32)

        # Workspace and output buffers
        d_delta_ws = cp.zeros(
            num_reads * N, dtype=cp.int8
        )
        d_samples = cp.zeros(
            num_reads * N, dtype=cp.float32
        )
        d_energies = cp.zeros(num_reads, dtype=cp.float32)

        # Launch: 1 block, num_reads threads
        grid = (1,)
        block = (num_reads,)
        self._kernel(
            grid, block, (
                d_row_ptr, d_col_ind, d_J_vals, d_h,
                d_beta,
                np.int32(N),
                np.int32(num_reads),
                np.int32(num_betas),
                np.int32(num_sweeps_per_beta),
                np.uint32(seed),
                d_delta_ws,
                d_samples,
                d_energies,
            ),
        )
        cp.cuda.Stream.null.synchronize()

        # Read results
        samples = cp.asnumpy(d_samples).reshape(num_reads, N)
        energies = cp.asnumpy(d_energies)

        return dimod.SampleSet.from_samples(
            samples.astype(np.int8),
            vartype='SPIN',
            energy=energies,
            info={
                'beta_range': beta_range,
                'num_betas': num_betas,
                'min_energy': float(energies.min()),
            },
        )

    def _build_csr(
        self,
        J: Dict[Tuple[int, int], float],
        N: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build symmetric CSR from J couplings."""
        rows = [[] for _ in range(N)]
        vals = [[] for _ in range(N)]
        for (i, j), Jij in J.items():
            if i >= N or j >= N:
                continue
            rows[i].append(j)
            vals[i].append(int(Jij))
            rows[j].append(i)
            vals[j].append(int(Jij))

        csr_row_ptr = np.zeros(N + 1, dtype=np.int32)
        nnz = 0
        for i in range(N):
            csr_row_ptr[i] = nnz
            if rows[i]:
                order = np.argsort(rows[i])
                rows[i] = [rows[i][k] for k in order]
                vals[i] = [vals[i][k] for k in order]
            nnz += len(rows[i])
        csr_row_ptr[N] = nnz

        csr_col_ind = np.array(
            [c for row in rows for c in row], dtype=np.int32
        )
        csr_J_vals = np.array(
            [v for row in vals for v in row], dtype=np.int8
        )
        return csr_row_ptr, csr_col_ind, csr_J_vals
