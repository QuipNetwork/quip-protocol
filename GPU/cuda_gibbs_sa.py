# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""CUDA Block Gibbs Sampler.

Chromatic parallel block Gibbs sampling on GPU via CuPy.
Colors processed sequentially (Gauss-Seidel), nodes within each
color updated in parallel (independent set). One CUDA block per
sample with 256 threads.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import cupy as cp
import dimod
import numpy as np

from GPU.sampler_utils import (
    build_csr_from_ising,
    compute_beta_schedule,
    compute_color_blocks,
    unpack_packed_results,
)


class CudaGibbsSampler:
    """Block Gibbs sampler using CUDA GPU.

    Chromatic parallel: colors processed sequentially
    (Gauss-Seidel), nodes within each color updated in
    parallel by 256 threads. Also supports a fully sequential
    mode for validation.
    """

    def __init__(
        self,
        topology=None,
        update_mode: str = "gibbs",
        parallel: bool = True,
    ):
        """Initialize CUDA Gibbs sampler.

        Args:
            topology: Topology object (default: DEFAULT_TOPOLOGY).
            update_mode: "gibbs" or "metropolis".
            parallel: Use chromatic parallel kernel (True) or
                fully sequential kernel (False).
        """
        self.logger = logging.getLogger(__name__)

        if update_mode.lower() not in ("gibbs", "metropolis"):
            raise ValueError(
                f"update_mode must be 'gibbs' or 'metropolis', "
                f"got {update_mode}"
            )
        self.update_mode = 0 if update_mode.lower() == "gibbs" else 1
        self.update_mode_name = update_mode.lower()
        self.parallel = parallel

        # Set up topology
        from dwave_topologies import DEFAULT_TOPOLOGY
        topology_obj = (
            topology if topology is not None else DEFAULT_TOPOLOGY
        )
        topology_graph = topology_obj.graph
        self.nodes = list(topology_graph.nodes())
        self.edges = list(topology_graph.edges())
        self.nodelist = self.nodes
        self.edgelist = self.edges
        self.properties = topology_obj.properties

        # Extract Zephyr parameters
        topo_shape = self.properties.get(
            'topology', {}
        ).get('shape', [9, 2])
        self.m = topo_shape[0]
        self.t = topo_shape[1]

        # Precompute color blocks
        self.block_starts, self.block_counts, self.color_node_indices = \
            compute_color_blocks(self.nodes, self.m, self.t)
        self.num_colors = 4

        self.logger.debug(
            f"Color block sizes: {self.block_counts}"
        )

        # Compile CUDA kernels
        kernel_path = os.path.join(
            os.path.dirname(__file__), 'cuda_gibbs.cu'
        )
        with open(kernel_path, 'r') as f:
            kernel_code = f.read()

        self._module = cp.RawModule(
            code=kernel_code,
            options=('--use_fast_math',),
        )
        self._parallel_kernel = self._module.get_function(
            'cuda_block_gibbs_parallel'
        )
        self._sequential_kernel = self._module.get_function(
            'cuda_block_gibbs_sequential'
        )

    def sample_ising(
        self,
        h: List[Dict[int, float]],
        J: List[Dict[Tuple[int, int], float]],
        num_reads: int = 200,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
        beta_schedule: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[dimod.SampleSet]:
        """Sample from Ising model using block Gibbs sampling.

        Args:
            h: List of linear biases per problem.
            J: List of quadratic biases per problem.
            num_reads: Number of independent samples per problem.
            num_sweeps: Total number of sweeps.
            num_sweeps_per_beta: Sweeps per beta value.
            beta_range: (hot_beta, cold_beta) or None for auto.
            beta_schedule_type: "linear", "geometric", or "custom".
            beta_schedule: Custom schedule (requires type="custom").
            seed: RNG seed.

        Returns:
            List of dimod.SampleSet, one per problem.
        """
        num_problems = len(h)
        assert len(J) == num_problems, (
            f"h and J must have same length: "
            f"{num_problems} vs {len(J)}"
        )

        # Build CSR for all problems
        (all_row_ptr, all_col_ind, all_J_vals, all_h_vals,
         row_ptr_offsets, col_ind_offsets,
         node_to_idx_list, N_list) = build_csr_from_ising(h, J)

        # Compute beta schedule (using first problem for auto range)
        sched, beta_range = compute_beta_schedule(
            h[0], J[0], num_sweeps, num_sweeps_per_beta,
            beta_range, beta_schedule_type, beta_schedule,
        )

        if seed is None:
            seed = np.random.randint(0, 2**31)

        self.logger.debug(
            f"[CudaGibbs] {num_problems} problems, "
            f"{num_reads} reads, {num_sweeps} sweeps, "
            f"mode={self.update_mode_name}, "
            f"parallel={self.parallel}"
        )

        # Process each problem separately (kernel is single-problem)
        all_samplesets = []
        for prob_idx in range(num_problems):
            sampleset = self._sample_single_problem(
                prob_idx, all_row_ptr, all_col_ind, all_J_vals,
                all_h_vals, row_ptr_offsets, col_ind_offsets,
                node_to_idx_list, N_list,
                num_reads, sched, num_sweeps_per_beta,
                seed + prob_idx, beta_range, beta_schedule_type,
            )
            all_samplesets.append(sampleset)

        return all_samplesets

    def _sample_single_problem(
        self,
        prob_idx: int,
        all_row_ptr: np.ndarray,
        all_col_ind: np.ndarray,
        all_J_vals: np.ndarray,
        all_h_vals: np.ndarray,
        row_ptr_offsets: np.ndarray,
        col_ind_offsets: np.ndarray,
        node_to_idx_list: list,
        N_list: list,
        num_reads: int,
        beta_sched: np.ndarray,
        sweeps_per_beta: int,
        seed: int,
        beta_range: Optional[Tuple[float, float]],
        beta_schedule_type: str,
    ) -> dimod.SampleSet:
        """Run kernel for a single Ising problem."""
        N = N_list[prob_idx]
        num_betas = len(beta_sched)
        packed_size = (N + 7) // 8

        # Extract this problem's CSR slice
        rp_start = int(row_ptr_offsets[prob_idx])
        rp_end = int(row_ptr_offsets[prob_idx + 1])
        ci_start = int(col_ind_offsets[prob_idx])
        ci_end = int(col_ind_offsets[prob_idx + 1])

        prob_row_ptr = all_row_ptr[rp_start:rp_end]
        prob_col_ind = all_col_ind[ci_start:ci_end]
        prob_J_vals = all_J_vals[ci_start:ci_end]
        prob_h_vals = all_h_vals[prob_idx * N:(prob_idx + 1) * N]

        # Compute color blocks for this problem's nodes and remap
        # to dense CSR indices
        node_to_idx = node_to_idx_list[prob_idx]
        prob_nodes = sorted(node_to_idx.keys())
        block_starts, block_counts, color_indices = \
            compute_color_blocks(prob_nodes, self.m, self.t)

        # Remap original node IDs to dense CSR indices
        remapped_colors = np.array(
            [node_to_idx[n] for n in color_indices],
            dtype=np.int32,
        )

        # Transfer to GPU
        d_row_ptr = cp.asarray(prob_row_ptr)
        d_col_ind = cp.asarray(prob_col_ind)
        d_J_vals = cp.asarray(prob_J_vals)
        d_h_vals = cp.asarray(prob_h_vals)

        d_block_starts = cp.asarray(block_starts)
        d_block_counts = cp.asarray(block_counts)
        d_color_indices = cp.asarray(remapped_colors)

        d_beta_sched = cp.asarray(beta_sched)

        # Output buffers
        d_final_samples = cp.zeros(
            num_reads * packed_size, dtype=cp.int8
        )
        d_final_energies = cp.zeros(num_reads, dtype=cp.int32)

        if self.parallel:
            self._launch_parallel(
                d_row_ptr, d_col_ind, d_J_vals, d_h_vals, N,
                d_block_starts, d_block_counts, d_color_indices,
                d_beta_sched, num_betas, sweeps_per_beta,
                d_final_samples, d_final_energies,
                num_reads, seed,
            )
        else:
            self._launch_sequential(
                d_row_ptr, d_col_ind, d_J_vals, d_h_vals, N,
                d_block_starts, d_block_counts, d_color_indices,
                d_beta_sched, num_betas, sweeps_per_beta,
                d_final_samples, d_final_energies,
                num_reads, seed,
            )

        cp.cuda.Stream.null.synchronize()

        # Read results
        packed_data = cp.asnumpy(d_final_samples).reshape(
            num_reads, packed_size
        )
        energies = cp.asnumpy(d_final_energies)

        self.logger.debug(
            f"[CudaGibbs] Problem {prob_idx}: "
            f"energy range [{energies.min()}, {energies.max()}]"
        )

        # Unpack single problem
        samplesets = unpack_packed_results(
            packed_data, energies, 1, num_reads, N,
            [node_to_idx_list[prob_idx]],
            info={
                "beta_range": beta_range,
                "beta_schedule_type": beta_schedule_type,
                "update_mode": self.update_mode_name,
            },
        )
        return samplesets[0]

    def _launch_parallel(
        self,
        d_row_ptr, d_col_ind, d_J_vals, d_h_vals, N,
        d_block_starts, d_block_counts, d_color_indices,
        d_beta_sched, num_betas, sweeps_per_beta,
        d_final_samples, d_final_energies,
        num_reads, seed,
    ):
        """Launch chromatic parallel kernel."""
        d_state = cp.zeros(num_reads * N, dtype=cp.int8)

        grid = (num_reads,)
        block = (256,)

        self._parallel_kernel(
            grid, block,
            (
                d_row_ptr, d_col_ind, d_J_vals, d_h_vals,
                np.int32(N),
                d_block_starts, d_block_counts, d_color_indices,
                np.int32(self.num_colors),
                d_beta_sched,
                np.int32(num_betas),
                np.int32(sweeps_per_beta),
                d_state,
                d_final_samples, d_final_energies,
                np.int32(num_reads),
                np.uint32(seed),
                np.int32(self.update_mode),
            ),
        )

    def _launch_sequential(
        self,
        d_row_ptr, d_col_ind, d_J_vals, d_h_vals, N,
        d_block_starts, d_block_counts, d_color_indices,
        d_beta_sched, num_betas, sweeps_per_beta,
        d_final_samples, d_final_energies,
        num_reads, seed,
    ):
        """Launch the sequential (Gauss-Seidel) kernel."""
        d_state = cp.zeros(num_reads * N, dtype=cp.int8)

        grid = (num_reads,)
        block = (1,)

        self._sequential_kernel(
            grid, block,
            (
                d_row_ptr, d_col_ind, d_J_vals, d_h_vals,
                np.int32(N),
                d_block_starts, d_block_counts, d_color_indices,
                np.int32(self.num_colors),
                d_beta_sched,
                np.int32(num_betas),
                np.int32(sweeps_per_beta),
                d_state,
                d_final_samples, d_final_energies,
                np.int32(num_reads),
                np.uint32(seed),
                np.int32(self.update_mode),
            ),
        )
