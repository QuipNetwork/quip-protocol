# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""CUDA Block Gibbs Sampler - persistent kernel with work queue.

Chromatic parallel block Gibbs sampling on GPU via CuPy.
Colors processed sequentially (Gauss-Seidel), nodes within each
color updated in parallel (independent set).

Single persistent kernel: blocks grab work units (model + read
chunk) from atomic queue, process all sweeps/colors using shared
memory, then grab the next unit. Work-stealing balances load
across models.
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

    Persistent kernel with work queue: blocks grab work units
    from an atomic queue, process all sweeps/colors using
    shared memory state, then grab the next unit. 256 threads
    per block parallelize nodes within each color.

    Also supports a fully sequential mode for validation.
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
        self.num_colors = 4

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
        self._persistent_kernel = self._module.get_function(
            'cuda_gibbs_persistent'
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

        All problems are dispatched in a single kernel launch.

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

        num_betas = len(sched)
        max_N = max(N_list)
        max_packed_size = (max_N + 7) // 8

        # Build per-problem color blocks
        color_data = self._build_batched_color_blocks(
            node_to_idx_list, N_list, num_problems
        )

        self.logger.debug(
            f"[CudaGibbs] {num_problems} problems batched, "
            f"{num_reads} reads, {num_sweeps} sweeps, "
            f"mode={self.update_mode_name}, "
            f"parallel={self.parallel}"
        )

        # Build per-problem offset arrays
        problem_rp_offsets = np.array(
            [int(row_ptr_offsets[i]) for i in range(num_problems)],
            dtype=np.int32,
        )
        problem_ci_offsets = np.array(
            [int(col_ind_offsets[i]) for i in range(num_problems)],
            dtype=np.int32,
        )
        # h_vals offset: sum of N values for problems before this one
        h_offsets = np.zeros(num_problems, dtype=np.int32)
        running = 0
        for i in range(num_problems):
            h_offsets[i] = running
            running += N_list[i]
        problem_N = np.array(N_list, dtype=np.int32)

        # Transfer to GPU
        d_row_ptr = cp.asarray(all_row_ptr)
        d_col_ind = cp.asarray(all_col_ind)
        d_J_vals = cp.asarray(all_J_vals)
        d_h_vals = cp.asarray(all_h_vals)

        d_problem_N = cp.asarray(problem_N)
        d_problem_rp = cp.asarray(problem_rp_offsets)
        d_problem_ci = cp.asarray(problem_ci_offsets)
        d_problem_h = cp.asarray(h_offsets)

        d_block_starts = cp.asarray(color_data[0])
        d_block_counts = cp.asarray(color_data[1])
        d_color_nodes = cp.asarray(color_data[2])

        d_beta_sched = cp.asarray(sched)

        # Output buffers
        total_samples = num_problems * num_reads
        d_final_samples = cp.zeros(
            total_samples * max_packed_size, dtype=cp.int8
        )
        d_final_energies = cp.zeros(
            total_samples, dtype=cp.int32
        )

        if self.parallel:
            # Persistent kernel with work queue.
            # Blocks grab (model, read_chunk) from atomic queue.
            dev = cp.cuda.Device()
            num_sms = dev.attributes['MultiProcessorCount']
            num_blocks = num_sms  # 1 persistent block per SM

            chunks_per_model = max(
                1, num_blocks // num_problems
            )
            reads_per_chunk = (
                (num_reads + chunks_per_model - 1)
                // chunks_per_model
            )
            total_work_units = (
                num_problems * chunks_per_model
            )

            # Atomic work queue counter
            d_queue_counter = cp.zeros(1, dtype=cp.int32)

            self.logger.debug(
                f"[CudaGibbs] persistent: "
                f"{num_blocks} blocks, "
                f"{chunks_per_model} chunks/model, "
                f"{reads_per_chunk} reads/chunk, "
                f"{total_work_units} total units"
            )

            grid = (num_blocks,)
            block = (256,)
            self._persistent_kernel(
                grid, block, (
                    d_row_ptr, d_col_ind,
                    d_J_vals, d_h_vals,
                    d_problem_N, d_problem_rp,
                    d_problem_ci, d_problem_h,
                    d_block_starts, d_block_counts,
                    d_color_nodes,
                    np.int32(self.num_colors),
                    d_beta_sched,
                    np.int32(num_betas),
                    np.int32(num_sweeps // num_betas),
                    d_final_samples, d_final_energies,
                    np.int32(num_reads),
                    np.int32(max_N),
                    np.int32(max_packed_size),
                    np.int32(num_problems),
                    d_queue_counter,
                    np.int32(chunks_per_model),
                    np.int32(reads_per_chunk),
                    np.int32(total_work_units),
                    np.uint32(seed),
                    np.int32(self.update_mode),
                ),
            )
        else:
            # Sequential: 1 thread/block, reads_per_block=1
            d_state = cp.zeros(
                total_samples * max_N, dtype=cp.int8
            )
            seq_args = (
                d_row_ptr, d_col_ind, d_J_vals, d_h_vals,
                d_problem_N, d_problem_rp,
                d_problem_ci, d_problem_h,
                d_block_starts, d_block_counts,
                d_color_nodes,
                np.int32(self.num_colors),
                d_beta_sched,
                np.int32(num_betas),
                np.int32(num_sweeps // num_betas),
                d_state,
                d_final_samples, d_final_energies,
                np.int32(num_reads),
                np.int32(max_N),
                np.int32(max_packed_size),
                np.int32(num_problems),
                np.int32(1),  # reads_per_block=1
                np.uint32(seed),
                np.int32(self.update_mode),
            )
            grid = (num_reads, num_problems)
            block = (1,)
            self._sequential_kernel(grid, block, seq_args)

        cp.cuda.Stream.null.synchronize()

        # Read results
        packed_raw = cp.asnumpy(d_final_samples)
        energies_raw = cp.asnumpy(d_final_energies)

        packed_data = packed_raw.reshape(
            total_samples, max_packed_size
        )
        energies_data = energies_raw

        for prob_idx in range(num_problems):
            start = prob_idx * num_reads
            end = (prob_idx + 1) * num_reads
            e = energies_data[start:end]
            self.logger.debug(
                f"[CudaGibbs] Problem {prob_idx}: "
                f"energy range [{e.min()}, {e.max()}]"
            )

        return unpack_packed_results(
            packed_data, energies_data,
            num_problems, num_reads, max_N,
            node_to_idx_list,
            info={
                "beta_range": beta_range,
                "beta_schedule_type": beta_schedule_type,
                "update_mode": self.update_mode_name,
            },
        )

    def _build_batched_color_blocks(
        self,
        node_to_idx_list: list,
        N_list: list,
        num_problems: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build flattened color block arrays for all problems.

        Returns:
            Tuple of:
            - all_block_starts: [num_problems * 4] global starts
            - all_block_counts: [num_problems * 4] counts
            - all_color_nodes: concatenated remapped node indices
        """
        all_block_starts = np.zeros(
            num_problems * self.num_colors, dtype=np.int32
        )
        all_block_counts = np.zeros(
            num_problems * self.num_colors, dtype=np.int32
        )
        all_color_nodes = []
        global_offset = 0

        for prob_idx in range(num_problems):
            node_to_idx = node_to_idx_list[prob_idx]
            prob_nodes = sorted(node_to_idx.keys())

            starts, counts, color_indices = compute_color_blocks(
                prob_nodes, self.m, self.t
            )

            # Remap to dense CSR indices
            remapped = np.array(
                [node_to_idx[n] for n in color_indices],
                dtype=np.int32,
            )

            base = prob_idx * self.num_colors
            for c in range(self.num_colors):
                all_block_starts[base + c] = (
                    global_offset + int(starts[c])
                )
                all_block_counts[base + c] = int(counts[c])

            all_color_nodes.append(remapped)
            global_offset += len(remapped)

        all_color_nodes = np.concatenate(all_color_nodes)
        return all_block_starts, all_block_counts, all_color_nodes
