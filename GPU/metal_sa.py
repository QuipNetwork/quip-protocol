"""
Simulated Annealing Metal Sampler - Exact D-Wave Implementation

This module provides a Metal GPU implementation that exactly mimics D-Wave's
SimulatedAnnealingSampler from cpu_sa.cpp, including:

1. Delta energy array optimization (pre-compute, update incrementally)
2. xorshift32 RNG
3. Sequential variable ordering (spins 0..N-1)
4. Metropolis criterion with threshold optimization (skip if delta_E > 22.18/beta)
5. Beta schedule computation matching _default_ising_beta_range
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import dimod
import Metal
import numpy as np

from GPU.sampler_utils import (
    _default_ising_beta_range,
    build_csr_from_ising,
    compute_beta_schedule,
    unpack_packed_results,
)


class MetalSASampler:
    """
    Simulated Annealing sampler using Metal GPU.

    Exactly mimics D-Wave's SimulatedAnnealingSampler implementation.
    """

    def __init__(self, topology=None):
        self.logger = logging.getLogger(__name__)
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError("Metal is not supported on this device")

        # Set up topology for mining compatibility
        from dwave_topologies import DEFAULT_TOPOLOGY
        topology_obj = topology if topology is not None else DEFAULT_TOPOLOGY
        topology_graph = topology_obj.graph
        self.nodes = list(topology_graph.nodes())
        self.edges = list(topology_graph.edges())
        self.nodelist = self.nodes
        self.edgelist = self.edges
        self.properties = topology_obj.properties

        # Load Metal library
        kernel_path = os.path.join(os.path.dirname(__file__), "metal_kernels.metal")
        with open(kernel_path, 'r') as f:
            kernel_source = f.read()

        lib, err = self.device.newLibraryWithSource_options_error_(kernel_source, None, None)
        if err:
            raise RuntimeError(f"Failed to compile Metal kernels: {err}")
        if not lib:
            raise RuntimeError("Failed to create Metal library (no error reported)")

        # List all functions in library for debugging
        function_names = [lib.functionNames()[i] for i in range(len(lib.functionNames()))]
        self.logger.debug(f"Available Metal functions: {function_names}")

        # Get SA kernel
        self._kernel = lib.newFunctionWithName_("pure_simulated_annealing")
        if not self._kernel:
            raise RuntimeError(f"Failed to find pure_simulated_annealing kernel. Available: {function_names}")

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
        Sample from Ising model using pure simulated annealing.

        Args:
            h: List of linear biases [{node: bias}, ...] for each problem
            J: List of quadratic biases [{(node1, node2): coupling}, ...] for each problem
            num_reads: Number of independent SA runs per problem
            num_sweeps: Total number of sweeps (default 1000)
            num_sweeps_per_beta: Sweeps per beta value (default 1)
            beta_range: (hot_beta, cold_beta) or None for auto (uses first problem for auto)
            beta_schedule_type: "linear", "geometric", or "custom"
            beta_schedule: Custom beta schedule (requires beta_schedule_type="custom")
            seed: RNG seed

        Returns:
            List of dimod.SampleSet with samples and energies for each problem
        """
        num_problems = len(h)
        if len(J) != num_problems:
            raise ValueError(f"h and J must have same length: {num_problems} vs {len(J)}")

        self.logger.debug(f"[MetalSA] Processing {num_problems} problems, {num_reads} reads each, {num_sweeps} sweeps")

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

        self.logger.debug(f"[MetalSA] Beta schedule: {len(beta_schedule)} betas from {beta_schedule[0]:.4f} to {beta_schedule[-1]:.4f}")

        # RNG seed
        if seed is None:
            seed = np.random.randint(0, 2**31)

        # Create Metal buffers for concatenated CSR arrays
        csr_row_ptr_buf = self._create_buffer(all_csr_row_ptr, "csr_row_ptr")
        csr_col_ind_buf = self._create_buffer(all_csr_col_ind, "csr_col_ind")
        csr_J_vals_buf = self._create_buffer(all_csr_J_vals, "csr_J_vals")
        csr_h_vals_buf = self._create_buffer(all_h_vals, "csr_h_vals")
        row_ptr_offsets_buf = self._create_buffer(row_ptr_offsets, "row_ptr_offsets")
        col_ind_offsets_buf = self._create_buffer(col_ind_offsets, "col_ind_offsets")

        beta_schedule_buf = self._create_buffer(beta_schedule, "beta_schedule")

        # Scalar parameters for batched problems
        N_bytes = np.int32(N).tobytes()
        num_betas_bytes = np.int32(len(beta_schedule)).tobytes()
        sweeps_per_beta_bytes = np.int32(num_sweeps_per_beta).tobytes()
        base_seed_bytes = np.uint32(seed).tobytes()

        # Batched parameters
        num_threads = num_problems * num_reads
        num_threads_bytes = np.int32(num_threads).tobytes()
        num_problems_bytes = np.int32(num_problems).tobytes()
        num_reads_bytes = np.int32(num_reads).tobytes()

        self.logger.debug(f"[MetalSA] Batch config: {num_problems} problems × {num_reads} reads = {num_threads} total reads")

        # Output buffers for all problems
        packed_size = (N + 7) // 8  # Bit-packed state size

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
        # Batched CSR buffers with separate offsets
        encoder.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(row_ptr_offsets_buf, 0, 3)  # Offsets into csr_row_ptr
        encoder.setBuffer_offset_atIndex_(col_ind_offsets_buf, 0, 4)  # Offsets into csr_col_ind

        # Scalar parameters (passed as bytes)
        encoder.setBytes_length_atIndex_(N_bytes, len(N_bytes), 5)
        encoder.setBytes_length_atIndex_(num_betas_bytes, len(num_betas_bytes), 6)
        encoder.setBytes_length_atIndex_(sweeps_per_beta_bytes, len(sweeps_per_beta_bytes), 7)
        encoder.setBytes_length_atIndex_(base_seed_bytes, len(base_seed_bytes), 8)

        # Beta schedule array
        encoder.setBuffer_offset_atIndex_(beta_schedule_buf, 0, 9)

        # Output buffers
        encoder.setBuffer_offset_atIndex_(final_samples_buf, 0, 10)
        encoder.setBuffer_offset_atIndex_(final_energies_buf, 0, 11)

        # Batch parameters
        encoder.setBytes_length_atIndex_(num_threads_bytes, len(num_threads_bytes), 12)
        encoder.setBytes_length_atIndex_(num_problems_bytes, len(num_problems_bytes), 13)
        encoder.setBytes_length_atIndex_(num_reads_bytes, len(num_reads_bytes), 14)

        # h field values (buffer 15)
        encoder.setBuffer_offset_atIndex_(csr_h_vals_buf, 0, 15)

        # Dispatch configuration for batched problems
        # One threadgroup per problem - optimal for cache locality
        max_threadgroups = self._pipeline.maxTotalThreadsPerThreadgroup()

        if num_problems > max_threadgroups:
            raise ValueError(f"Too many problems ({num_problems}) for device capacity ({max_threadgroups} threadgroups). Use batches of <= {max_threadgroups} problems.")

        num_threadgroups_width = num_problems
        threads_per_threadgroup_width = num_reads

        threads_per_threadgroup = Metal.MTLSize(width=threads_per_threadgroup_width, height=1, depth=1)
        num_threadgroups = Metal.MTLSize(width=num_threadgroups_width, height=1, depth=1)

        self.logger.debug(f"[MetalSA] Dispatch: {num_threadgroups.width} threadgroups × {threads_per_threadgroup.width} threads = {num_threadgroups.width * threads_per_threadgroup.width} total threads for {num_threads} total reads ({num_problems} problems × {num_reads} reads)")

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups, threads_per_threadgroup)

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        # Check for errors
        if cmd_buf.status() != Metal.MTLCommandBufferStatusCompleted:
            error = cmd_buf.error()
            raise RuntimeError(f"Metal command buffer failed: {error}")

        # Read batched results and parse into separate SampleSets
        # Read all results for all problems
        packed_data = np.frombuffer(
            final_samples_buf.contents().as_buffer(num_threads * packed_size),
            dtype=np.int8
        ).reshape(num_threads, packed_size)

        energies_data = np.frombuffer(
            final_energies_buf.contents().as_buffer(num_threads * 4),
            dtype=np.int32
        )

        self.logger.debug(f"[MetalSA] Energy range: [{energies_data.min()}, {energies_data.max()}]")

        return unpack_packed_results(
            packed_data, energies_data, num_problems, num_reads, N,
            node_to_idx_list,
            info={"beta_range": beta_range, "beta_schedule_type": beta_schedule_type},
        )

