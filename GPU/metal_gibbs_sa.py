# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""
Metal Block Gibbs Sampler - Based on dwave-pytorch-plugin BlockSampler

Implements block-parallel Gibbs sampling where nodes are partitioned by
graph coloring. All nodes in a color block can be updated simultaneously
because no two adjacent nodes share the same color.

Key algorithm (from BlockSampler):
1. Graph coloring partitions nodes into independent blocks (4 colors for Zephyr)
2. For each temperature step, for each color block:
   - Compute effective field for all nodes in block
   - Sample new spins (Gibbs) or accept/reject flips (Metropolis)
3. Effective field: h_eff = h_i + sum_j(J_ij * x_j)
4. Gibbs update: P(spin=+1) = 1 / (1 + exp(2 * beta * h_eff))
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import dimod
import Metal
import numpy as np

from GPU.sampler_utils import (
    build_csr_from_ising,
    compute_beta_schedule,
    compute_color_blocks,
    unpack_packed_results,
)


class MetalGibbsSampler:
    """
    Block Gibbs sampler using Metal GPU.

    Based on dwave-pytorch-plugin BlockSampler algorithm:
    1. Partition nodes by graph coloring (4 colors for Zephyr)
    2. For each temperature step:
       - For each color block:
         - Compute effective field for all nodes in block
         - Sample new spins (Gibbs) or accept/reject flips (Metropolis)
    """

    def __init__(self, topology=None, update_mode: str = "gibbs"):
        """Initialize Metal Gibbs sampler.

        Args:
            topology: Topology object (default: DEFAULT_TOPOLOGY)
            update_mode: "gibbs" or "metropolis" (default: "gibbs")
        """
        self.logger = logging.getLogger(__name__)
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError("Metal is not supported on this device")

        if update_mode.lower() not in ("gibbs", "metropolis"):
            raise ValueError(f"update_mode must be 'gibbs' or 'metropolis', got {update_mode}")
        self.update_mode = 0 if update_mode.lower() == "gibbs" else 1
        self.update_mode_name = update_mode.lower()

        # Set up topology
        from dwave_topologies import DEFAULT_TOPOLOGY
        topology_obj = topology if topology is not None else DEFAULT_TOPOLOGY
        topology_graph = topology_obj.graph
        self.nodes = list(topology_graph.nodes())
        self.edges = list(topology_graph.edges())
        self.nodelist = self.nodes
        self.edgelist = self.edges
        self.properties = topology_obj.properties

        # Extract Zephyr parameters from topology
        topo_shape = self.properties.get('topology', {}).get('shape', [9, 2])
        self.m = topo_shape[0]
        self.t = topo_shape[1]

        # Precompute color blocks
        self.block_starts, self.block_counts, self.color_node_indices = \
            compute_color_blocks(self.nodes, self.m, self.t)
        self.num_colors = 4

        self.logger.debug(f"Color block sizes: {self.block_counts}")

        # Load and compile Metal kernel
        kernel_path = os.path.join(os.path.dirname(__file__), "metal_gibbs.metal")
        with open(kernel_path, 'r') as f:
            kernel_source = f.read()

        lib, err = self.device.newLibraryWithSource_options_error_(kernel_source, None, None)
        if err:
            raise RuntimeError(f"Failed to compile Metal kernels: {err}")
        if not lib:
            raise RuntimeError("Failed to create Metal library (no error reported)")

        self._kernel = lib.newFunctionWithName_("block_gibbs_sampler")
        if not self._kernel:
            function_names = [lib.functionNames()[i] for i in range(len(lib.functionNames()))]
            raise RuntimeError(f"Failed to find block_gibbs_sampler kernel. Available: {function_names}")

        self._pipeline, err = self.device.newComputePipelineStateWithFunction_error_(self._kernel, None)
        if err or not self._pipeline:
            raise RuntimeError(f"Failed to create pipeline: {err}")

        self._command_queue = self.device.newCommandQueue()

    def _create_buffer(self, data: np.ndarray, label: str = ""):
        """Create a Metal buffer from numpy array."""
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        byte_data = data.tobytes()
        byte_length = len(byte_data)
        buf = self.device.newBufferWithBytes_length_options_(
            byte_data, byte_length, Metal.MTLResourceStorageModeShared
        )
        if not buf:
            raise RuntimeError(f"Failed to create buffer: {label}")
        return buf

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
        **kwargs
    ) -> List[dimod.SampleSet]:
        """
        Sample from Ising model using block Gibbs sampling.

        Args:
            h: List of linear biases [{node: bias}, ...] for each problem
            J: List of quadratic biases [{(node1, node2): coupling}, ...] for each problem
            num_reads: Number of independent sampling runs per problem
            num_sweeps: Total number of sweeps (default 1000)
            num_sweeps_per_beta: Sweeps per beta value (default 1)
            beta_range: (hot_beta, cold_beta) or None for auto
            beta_schedule_type: "linear", "geometric", or "custom"
            beta_schedule: Custom beta schedule (requires beta_schedule_type="custom")
            seed: RNG seed

        Returns:
            List of dimod.SampleSet with samples and energies for each problem
        """
        num_problems = len(h)
        if len(J) != num_problems:
            raise ValueError(f"h and J must have same length: {num_problems} vs {len(J)}")

        self.logger.debug(f"[MetalGibbs] Processing {num_problems} problems, {num_reads} reads each, {num_sweeps} sweeps")

        # Build concatenated CSR arrays for all problems
        (all_csr_row_ptr, all_csr_col_ind, all_csr_J_vals, all_h_vals,
         row_ptr_offsets, col_ind_offsets, node_to_idx_list, N_list) = \
            build_csr_from_ising(h, J)

        N = N_list[0]
        if not all(n == N for n in N_list):
            raise ValueError(f"All problems must have same N: {N_list}")

        # Compute beta schedule
        beta_schedule, beta_range = compute_beta_schedule(
            h[0], J[0], num_sweeps, num_sweeps_per_beta,
            beta_range, beta_schedule_type, beta_schedule,
        )

        self.logger.debug(f"[MetalGibbs] Beta schedule: {len(beta_schedule)} betas from {beta_schedule[0]:.4f} to {beta_schedule[-1]:.4f}")

        # RNG seed
        if seed is None:
            seed = np.random.randint(0, 2**31)

        # Create Metal buffers
        csr_row_ptr_buf = self._create_buffer(all_csr_row_ptr, "csr_row_ptr")
        csr_col_ind_buf = self._create_buffer(all_csr_col_ind, "csr_col_ind")
        csr_J_vals_buf = self._create_buffer(all_csr_J_vals, "csr_J_vals")
        row_ptr_offsets_buf = self._create_buffer(row_ptr_offsets, "row_ptr_offsets")
        col_ind_offsets_buf = self._create_buffer(col_ind_offsets, "col_ind_offsets")
        csr_h_vals_buf = self._create_buffer(all_h_vals, "csr_h_vals")
        beta_schedule_buf = self._create_buffer(beta_schedule, "beta_schedule")

        # Color block buffers
        color_block_starts_buf = self._create_buffer(self.block_starts, "color_block_starts")
        color_block_counts_buf = self._create_buffer(self.block_counts, "color_block_counts")
        color_node_indices_buf = self._create_buffer(self.color_node_indices, "color_node_indices")

        # Scalar parameters
        N_bytes = np.int32(N).tobytes()
        num_betas_bytes = np.int32(len(beta_schedule)).tobytes()
        sweeps_per_beta_bytes = np.int32(num_sweeps_per_beta).tobytes()
        base_seed_bytes = np.uint32(seed).tobytes()
        update_mode_bytes = np.int32(self.update_mode).tobytes()
        num_colors_bytes = np.int32(self.num_colors).tobytes()

        # Batch parameters
        num_threads = num_problems * num_reads
        num_threads_bytes = np.int32(num_threads).tobytes()
        num_problems_bytes = np.int32(num_problems).tobytes()
        num_reads_bytes = np.int32(num_reads).tobytes()

        # Output buffers
        packed_size = (N + 7) // 8

        final_samples_buf = self.device.newBufferWithLength_options_(
            num_threads * packed_size, Metal.MTLResourceStorageModeShared
        )
        final_energies_buf = self.device.newBufferWithLength_options_(
            num_threads * 4, Metal.MTLResourceStorageModeShared
        )

        # Execute kernel
        cmd_buf = self._command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()

        encoder.setComputePipelineState_(self._pipeline)

        # Buffer bindings (must match kernel parameter order)
        encoder.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(row_ptr_offsets_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(col_ind_offsets_buf, 0, 4)

        encoder.setBytes_length_atIndex_(N_bytes, len(N_bytes), 5)
        encoder.setBytes_length_atIndex_(num_betas_bytes, len(num_betas_bytes), 6)
        encoder.setBytes_length_atIndex_(sweeps_per_beta_bytes, len(sweeps_per_beta_bytes), 7)
        encoder.setBytes_length_atIndex_(base_seed_bytes, len(base_seed_bytes), 8)

        encoder.setBuffer_offset_atIndex_(beta_schedule_buf, 0, 9)
        encoder.setBuffer_offset_atIndex_(final_samples_buf, 0, 10)
        encoder.setBuffer_offset_atIndex_(final_energies_buf, 0, 11)

        encoder.setBytes_length_atIndex_(num_threads_bytes, len(num_threads_bytes), 12)
        encoder.setBytes_length_atIndex_(num_problems_bytes, len(num_problems_bytes), 13)
        encoder.setBytes_length_atIndex_(num_reads_bytes, len(num_reads_bytes), 14)

        encoder.setBuffer_offset_atIndex_(csr_h_vals_buf, 0, 15)

        # Color block buffers
        encoder.setBuffer_offset_atIndex_(color_block_starts_buf, 0, 16)
        encoder.setBuffer_offset_atIndex_(color_block_counts_buf, 0, 17)
        encoder.setBuffer_offset_atIndex_(color_node_indices_buf, 0, 18)

        encoder.setBytes_length_atIndex_(update_mode_bytes, len(update_mode_bytes), 19)
        encoder.setBytes_length_atIndex_(num_colors_bytes, len(num_colors_bytes), 20)

        # Dispatch configuration
        max_threadgroups = self._pipeline.maxTotalThreadsPerThreadgroup()

        if num_problems > max_threadgroups:
            raise ValueError(f"Too many problems ({num_problems}) for device capacity ({max_threadgroups})")

        num_threadgroups_width = num_problems
        threads_per_threadgroup_width = num_reads

        threads_per_threadgroup = Metal.MTLSize(width=threads_per_threadgroup_width, height=1, depth=1)
        num_threadgroups = Metal.MTLSize(width=num_threadgroups_width, height=1, depth=1)

        self.logger.debug(f"[MetalGibbs] Dispatch: {num_threadgroups.width} threadgroups x {threads_per_threadgroup.width} threads")

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups, threads_per_threadgroup)

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        # Check for errors
        if cmd_buf.status() != Metal.MTLCommandBufferStatusCompleted:
            error = cmd_buf.error()
            raise RuntimeError(f"Metal command buffer failed: {error}")

        # Read results
        packed_data = np.frombuffer(
            final_samples_buf.contents().as_buffer(num_threads * packed_size),
            dtype=np.int8
        ).reshape(num_threads, packed_size)

        energies_data = np.frombuffer(
            final_energies_buf.contents().as_buffer(num_threads * 4),
            dtype=np.int32
        )

        self.logger.debug(f"[MetalGibbs] Energy range: [{energies_data.min()}, {energies_data.max()}]")

        return unpack_packed_results(
            packed_data, energies_data, num_problems, num_reads, N,
            node_to_idx_list,
            info={
                "beta_range": beta_range,
                "beta_schedule_type": beta_schedule_type,
                "update_mode": self.update_mode_name,
            },
        )
