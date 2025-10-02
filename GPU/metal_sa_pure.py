"""
Pure Simulated Annealing Metal Sampler - Exact D-Wave Implementation

This module provides a Metal GPU implementation that exactly mimics D-Wave's
SimulatedAnnealingSampler from cpu_sa.cpp, including:

1. Delta energy array optimization (pre-compute, update incrementally)
2. xorshift128+ RNG matching D-Wave
3. Sequential variable ordering (spins 0..N-1)
4. Metropolis criterion with threshold optimization (skip if delta_E > 44.36/beta)
5. Beta schedule computation matching _default_ising_beta_range
"""

import logging
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple
import warnings

import dimod
import Metal
import numpy as np


def _default_ising_beta_range(
    h: Dict[int, float],
    J: Dict[tuple, float],
    max_single_qubit_excitation_rate: float = 0.01,
    scale_T_with_N: bool = True
) -> Tuple[float, float]:
    """
    Exact replica of D-Wave's _default_ising_beta_range function.

    Determine the starting and ending beta from h, J.

    Args:
        h: External field of Ising model (linear bias)
        J: Couplings of Ising model (quadratic biases)
        max_single_qubit_excitation_rate: Targeted single qubit excitation rate at final temperature
        scale_T_with_N: Whether to scale temperature with system size

    Returns:
        [hot_beta, cold_beta] - tuple of starting and ending inverse temperatures
    """
    if not 0 < max_single_qubit_excitation_rate < 1:
        raise ValueError('Targeted single qubit excitations rates must be in range (0,1)')

    # Approximate worst and best cases of the [non-zero] energy signal
    sum_abs_bias_dict = defaultdict(int, {k: abs(v) for k, v in h.items()})
    if sum_abs_bias_dict:
        min_abs_bias_dict = {k: v for k, v in sum_abs_bias_dict.items() if v != 0}
    else:
        min_abs_bias_dict = {}

    # Build bias dictionaries from J
    for (k1, k2), v in J.items():
        for k in [k1, k2]:
            sum_abs_bias_dict[k] += abs(v)
            if v != 0:
                if k in min_abs_bias_dict:
                    min_abs_bias_dict[k] = min(abs(v), min_abs_bias_dict[k])
                else:
                    min_abs_bias_dict[k] = abs(v)

    if not min_abs_bias_dict:
        # Null problem - all biases are zero
        warn_msg = ('All bqm biases are zero (all energies are zero), this is '
                   'likely a value error. Temperature range is set arbitrarily '
                   'to [0.1,1]. Metropolis-Hastings update is non-ergodic.')
        warnings.warn(warn_msg)
        return (0.1, 1.0)

    # Hot temp: 50% flip probability for worst case
    max_effective_field = max(sum_abs_bias_dict.values(), default=0)

    if max_effective_field == 0:
        hot_beta = 1.0
    else:
        hot_beta = np.log(2) / (2 * max_effective_field)

    # Cold temp: Low excitation probability at end
    if len(min_abs_bias_dict) == 0:
        cold_beta = hot_beta
    else:
        values_array = np.array(list(min_abs_bias_dict.values()), dtype=float)
        min_effective_field = np.min(values_array)
        if scale_T_with_N:
            number_min_gaps = np.sum(min_effective_field == values_array)
        else:
            number_min_gaps = 1
        cold_beta = np.log(number_min_gaps / max_single_qubit_excitation_rate) / (2 * min_effective_field)

    return (hot_beta, cold_beta)


class PureMetalSASampler:
    """
    Pure Simulated Annealing sampler using Metal GPU.

    Exactly mimics D-Wave's SimulatedAnnealingSampler implementation.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError("Metal is not supported on this device")

        # Load Metal library (Pure SA implementation)
        kernel_path = os.path.join(os.path.dirname(__file__), "metal_kernels_pure.metal")
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

        # Get pure SA kernel
        self._kernel = lib.newFunctionWithName_("pure_simulated_annealing")
        if not self._kernel:
            raise RuntimeError(f"Failed to find pure_simulated_annealing kernel. Available: {function_names}")

        self._pipeline, err = self.device.newComputePipelineStateWithFunction_error_(self._kernel, None)
        if err or not self._pipeline:
            raise RuntimeError(f"Failed to create pipeline: {err}")

        # Get pure SA with coloring kernel (Phase 1)
        self._kernel_coloring = lib.newFunctionWithName_("pure_sa_with_coloring")
        if self._kernel_coloring:
            self._pipeline_coloring, err = self.device.newComputePipelineStateWithFunction_error_(self._kernel_coloring, None)
            if err or not self._pipeline_coloring:
                self.logger.warning(f"Failed to create coloring pipeline: {err}")
                self._pipeline_coloring = None
        else:
            self.logger.warning("pure_sa_with_coloring kernel not found")
            self._pipeline_coloring = None

        # Get pure SA with double buffering kernel (Phase 2)
        self._kernel_double_buffering = lib.newFunctionWithName_("pure_sa_with_double_buffering")
        if self._kernel_double_buffering:
            self._pipeline_double_buffering, err = self.device.newComputePipelineStateWithFunction_error_(self._kernel_double_buffering, None)
            if err or not self._pipeline_double_buffering:
                self.logger.warning(f"Failed to create double buffering pipeline: {err}")
                self._pipeline_double_buffering = None
        else:
            self.logger.warning("pure_sa_with_double_buffering kernel not found")
            self._pipeline_double_buffering = None

        # Get pure SA with per-color precomputation kernel (Phase 3)
        self._kernel_per_color_precomp = lib.newFunctionWithName_("pure_sa_with_per_color_precomputation")
        if self._kernel_per_color_precomp:
            self._pipeline_per_color_precomp, err = self.device.newComputePipelineStateWithFunction_error_(self._kernel_per_color_precomp, None)
            if err or not self._pipeline_per_color_precomp:
                self.logger.warning(f"Failed to create per-color precomputation pipeline: {err}")
                self._pipeline_per_color_precomp = None
        else:
            self.logger.warning("pure_sa_with_per_color_precomputation kernel not found")
            self._pipeline_per_color_precomp = None

        # Get pure SA with color shuffling kernel (Phase 4)
        self._kernel_color_shuffling = lib.newFunctionWithName_("pure_sa_with_color_shuffling")
        if self._kernel_color_shuffling:
            self._pipeline_color_shuffling, err = self.device.newComputePipelineStateWithFunction_error_(self._kernel_color_shuffling, None)
            if err or not self._pipeline_color_shuffling:
                self.logger.warning(f"Failed to create color shuffling pipeline: {err}")
                self._pipeline_color_shuffling = None
        else:
            self.logger.warning("pure_sa_with_color_shuffling kernel not found")
            self._pipeline_color_shuffling = None

        # Get pure SA with multiple replicas kernel (Phase 5)
        self._kernel_multiple_replicas = lib.newFunctionWithName_("pure_sa_with_multiple_replicas")
        if self._kernel_multiple_replicas:
            self._pipeline_multiple_replicas, err = self.device.newComputePipelineStateWithFunction_error_(self._kernel_multiple_replicas, None)
            if err or not self._pipeline_multiple_replicas:
                self.logger.warning(f"Failed to create multiple replicas pipeline: {err}")
                self._pipeline_multiple_replicas = None
        else:
            self.logger.warning("pure_sa_with_multiple_replicas kernel not found")
            self._pipeline_multiple_replicas = None

        # Get pure SA with replica exchange kernel (Phase 6)
        self._kernel_replica_exchange = lib.newFunctionWithName_("pure_sa_with_replica_exchange")
        if self._kernel_replica_exchange:
            self._pipeline_replica_exchange, err = self.device.newComputePipelineStateWithFunction_error_(self._kernel_replica_exchange, None)
            if err or not self._pipeline_replica_exchange:
                self.logger.warning(f"Failed to create replica exchange pipeline: {err}")
                self._pipeline_replica_exchange = None
        else:
            self.logger.warning("pure_sa_with_replica_exchange kernel not found")
            self._pipeline_replica_exchange = None

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
        h: Dict[int, float],
        J: Dict[tuple, float],
        num_reads: int = 200,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
        beta_schedule: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> dimod.SampleSet:
        """
        Sample from Ising model using pure simulated annealing.

        Args:
            h: Linear biases {node: bias}
            J: Quadratic biases {(node1, node2): coupling}
            num_reads: Number of independent SA runs
            num_sweeps: Total number of sweeps (default 1000)
            num_sweeps_per_beta: Sweeps per beta value (default 1)
            beta_range: (hot_beta, cold_beta) or None for auto
            beta_schedule_type: "linear", "geometric", or "custom"
            beta_schedule: Custom beta schedule (requires beta_schedule_type="custom")
            seed: RNG seed

        Returns:
            dimod.SampleSet with samples and energies
        """
        # Get all nodes
        all_nodes = set(h.keys()) | set(n for edge in J.keys() for n in edge)
        N = len(all_nodes)
        node_list = sorted(all_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        self.logger.debug(f"[PureMetalSA] N={N}, num_reads={num_reads}, num_sweeps={num_sweeps}")

        # Build CSR representation
        csr_row_ptr = np.zeros(N + 1, dtype=np.int32)
        csr_col_ind = []
        csr_J_vals = []

        # Count degrees
        degree = np.zeros(N, dtype=np.int32)
        for (i, j) in J.keys():
            if i in node_to_idx and j in node_to_idx:
                degree[node_to_idx[i]] += 1
                degree[node_to_idx[j]] += 1

        # Build CSR
        csr_row_ptr[1:] = np.cumsum(degree)
        current_pos = np.zeros(N, dtype=np.int32)

        adjacency = [[] for _ in range(N)]
        for (i, j), Jij in J.items():
            if i in node_to_idx and j in node_to_idx:
                idx_i = node_to_idx[i]
                idx_j = node_to_idx[j]
                adjacency[idx_i].append((idx_j, Jij))
                adjacency[idx_j].append((idx_i, Jij))

        for i in range(N):
            adjacency[i].sort()  # Ensure deterministic ordering
            for j, Jij in adjacency[i]:
                csr_col_ind.append(j)
                csr_J_vals.append(int(Jij))  # Convert to int8

        csr_col_ind = np.array(csr_col_ind, dtype=np.int32)
        csr_J_vals = np.array(csr_J_vals, dtype=np.int8)

        # Compute beta schedule (matching D-Wave exactly)
        if beta_schedule_type == "custom":
            if beta_schedule is None:
                raise ValueError("'beta_schedule' must be provided for beta_schedule_type = 'custom'")
            beta_schedule = np.array(beta_schedule, dtype=np.float32)
            num_betas = len(beta_schedule)
            if num_sweeps != num_betas * num_sweeps_per_beta:
                raise ValueError(f"num_sweeps ({num_sweeps}) must equal len(beta_schedule) * num_sweeps_per_beta")
        else:
            num_betas, rem = divmod(num_sweeps, num_sweeps_per_beta)
            if rem > 0 or num_betas < 0:
                raise ValueError("'num_sweeps' must be divisible by 'num_sweeps_per_beta'")

            if beta_range is None:
                beta_range = _default_ising_beta_range(h, J)
            elif len(beta_range) != 2 or min(beta_range) < 0:
                raise ValueError("'beta_range' should be a 2-tuple of positive numbers")

            if num_betas == 1:
                beta_schedule = np.array([beta_range[-1]], dtype=np.float32)
            else:
                if beta_schedule_type == "linear":
                    beta_schedule = np.linspace(beta_range[0], beta_range[1], num=num_betas, dtype=np.float32)
                elif beta_schedule_type == "geometric":
                    if min(beta_range) <= 0:
                        raise ValueError("'beta_range' must contain non-zero values for geometric schedule")
                    beta_schedule = np.geomspace(beta_range[0], beta_range[1], num=num_betas, dtype=np.float32)
                else:
                    raise ValueError(f"Beta schedule type {beta_schedule_type} not implemented")

        self.logger.debug(f"[PureMetalSA] Beta schedule: {len(beta_schedule)} betas from {beta_schedule[0]:.4f} to {beta_schedule[-1]:.4f}")

        # RNG seed (initial states will be generated in kernel)
        if seed is None:
            seed = np.random.randint(0, 2**31)

        # Create Metal buffers for arrays
        csr_row_ptr_buf = self._create_buffer(csr_row_ptr, "csr_row_ptr")
        csr_col_ind_buf = self._create_buffer(csr_col_ind, "csr_col_ind")
        csr_J_vals_buf = self._create_buffer(csr_J_vals, "csr_J_vals")

        beta_schedule_buf = self._create_buffer(beta_schedule, "beta_schedule")

        # Scalar parameters (passed as bytes, not buffers)
        N_bytes = np.int32(N).tobytes()
        num_betas_bytes = np.int32(len(beta_schedule)).tobytes()
        sweeps_per_beta_bytes = np.int32(num_sweeps_per_beta).tobytes()
        base_seed_bytes = np.uint32(seed).tobytes()

        # Working states buffer (initial state generated in kernel, not transferred from CPU)
        working_states_buf = self.device.newBufferWithLength_options_(
            num_reads * N, Metal.MTLResourceStorageModeShared
        )

        final_samples_buf = self.device.newBufferWithLength_options_(
            num_reads * N, Metal.MTLResourceStorageModeShared
        )
        final_energies_buf = self.device.newBufferWithLength_options_(
            num_reads * 4, Metal.MTLResourceStorageModeShared
        )

        # Delta energy buffer (int32 per spin per read)
        delta_energies_buf = self.device.newBufferWithLength_options_(
            num_reads * N * 4, Metal.MTLResourceStorageModeShared
        )

        # Execute kernel
        cmd_buf = self._command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()

        encoder.setComputePipelineState_(self._pipeline)
        # Buffers for arrays
        encoder.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 2)

        # Scalar parameters (passed as bytes)
        encoder.setBytes_length_atIndex_(N_bytes, len(N_bytes), 3)
        encoder.setBytes_length_atIndex_(num_betas_bytes, len(num_betas_bytes), 4)
        encoder.setBytes_length_atIndex_(sweeps_per_beta_bytes, len(sweeps_per_beta_bytes), 5)
        encoder.setBytes_length_atIndex_(base_seed_bytes, len(base_seed_bytes), 6)

        # Beta schedule array
        encoder.setBuffer_offset_atIndex_(beta_schedule_buf, 0, 7)

        # Working and output buffers
        encoder.setBuffer_offset_atIndex_(working_states_buf, 0, 8)
        encoder.setBuffer_offset_atIndex_(final_samples_buf, 0, 9)
        encoder.setBuffer_offset_atIndex_(final_energies_buf, 0, 10)
        encoder.setBuffer_offset_atIndex_(delta_energies_buf, 0, 11)

        # Dispatch one thread per read
        # Use larger threadgroup size for better GPU utilization
        # Apple GPUs work best with 256-1024 threads per threadgroup
        max_threads_per_threadgroup = 1024  # Maximum for Apple GPUs
        threads_per_threadgroup = Metal.MTLSize(width=min(num_reads, max_threads_per_threadgroup), height=1, depth=1)
        num_threadgroups = Metal.MTLSize(
            width=(num_reads + threads_per_threadgroup.width - 1) // threads_per_threadgroup.width,
            height=1,
            depth=1
        )
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups, threads_per_threadgroup)

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        # Check for errors
        if cmd_buf.status() != Metal.MTLCommandBufferStatusCompleted:
            error = cmd_buf.error()
            raise RuntimeError(f"Metal command buffer failed: {error}")

        # Read results
        samples_data = np.frombuffer(
            final_samples_buf.contents().as_buffer(num_reads * N),
            dtype=np.int8
        ).reshape(num_reads, N)

        energies_data = np.frombuffer(
            final_energies_buf.contents().as_buffer(num_reads * 4),
            dtype=np.int32
        )

        self.logger.debug(f"[PureMetalSA] Energy range: [{energies_data.min()}, {energies_data.max()}]")

        # Build SampleSet (continued below)
        samples_dict = []
        for sample in samples_data:
            samples_dict.append({node: int(sample[idx]) for node, idx in node_to_idx.items()})

        return dimod.SampleSet.from_samples(
            samples_dict,
            energy=energies_data.astype(float),
            vartype=dimod.SPIN,
            info={"beta_range": beta_range, "beta_schedule_type": beta_schedule_type}
        )

    def _compute_graph_coloring(self, csr_row_ptr: np.ndarray, csr_col_ind: np.ndarray, N: int):
        """
        Compute graph coloring using greedy algorithm.
        Returns (node_colors, num_colors) or None if coloring not available.
        """
        # Simple Python greedy coloring (could use Metal kernel later)
        node_colors = np.full(N, -1, dtype=np.int32)
        degrees = csr_row_ptr[1:] - csr_row_ptr[:-1]

        # Sort nodes by degree (descending)
        nodes_by_degree = sorted(range(N), key=lambda i: degrees[i], reverse=True)

        num_colors = 0
        for node in nodes_by_degree:
            # Find colors used by neighbors
            neighbor_colors = set()
            start = csr_row_ptr[node]
            end = csr_row_ptr[node + 1]
            for p in range(start, end):
                neighbor = csr_col_ind[p]
                if node_colors[neighbor] >= 0:
                    neighbor_colors.add(node_colors[neighbor])

            # Find first available color
            color = 0
            while color in neighbor_colors:
                color += 1

            node_colors[node] = color
            num_colors = max(num_colors, color + 1)

        self.logger.debug(f"[PureMetalSA] Graph coloring: {num_colors} colors for {N} nodes")
        return node_colors, num_colors

    def sample_ising_with_coloring(
        self,
        h: Dict[int, float],
        J: Dict[tuple, float],
        num_reads: int = 200,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
        beta_schedule: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> dimod.SampleSet:
        """
        PHASE 1: Sample using Pure SA with graph coloring.

        This adds parallel color updates while keeping everything else the same:
        - Sequential color order (no shuffling)
        - Delta energy array (incremental updates, not per-color precomputation)
        - Threshold skipping
        - Same beta schedule
        """
        if not self._pipeline_coloring:
            raise RuntimeError("Coloring kernel not available. Use sample_ising() instead.")

        # Get all nodes
        all_nodes = set(h.keys()) | set(n for edge in J.keys() for n in edge)
        N = len(all_nodes)
        node_list = sorted(all_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        self.logger.debug(f"[PureMetalSA-Coloring] N={N}, num_reads={num_reads}, num_sweeps={num_sweeps}")

        # Build CSR representation (same as original)
        csr_row_ptr = np.zeros(N + 1, dtype=np.int32)
        degree = np.zeros(N, dtype=np.int32)
        for (i, j) in J.keys():
            if i in node_to_idx and j in node_to_idx:
                degree[node_to_idx[i]] += 1
                degree[node_to_idx[j]] += 1

        csr_row_ptr[1:] = np.cumsum(degree)
        adjacency = [[] for _ in range(N)]
        for (i, j), Jij in J.items():
            if i in node_to_idx and j in node_to_idx:
                idx_i = node_to_idx[i]
                idx_j = node_to_idx[j]
                adjacency[idx_i].append((idx_j, Jij))
                adjacency[idx_j].append((idx_i, Jij))

        csr_col_ind = []
        csr_J_vals = []
        for i in range(N):
            adjacency[i].sort()
            for j, Jij in adjacency[i]:
                csr_col_ind.append(j)
                csr_J_vals.append(int(Jij))

        csr_col_ind = np.array(csr_col_ind, dtype=np.int32)
        csr_J_vals = np.array(csr_J_vals, dtype=np.int8)

        # Compute graph coloring
        node_colors, num_colors = self._compute_graph_coloring(csr_row_ptr, csr_col_ind, N)

        # Compute beta schedule (same as original)
        if beta_schedule_type == "custom":
            if beta_schedule is None:
                raise ValueError("'beta_schedule' must be provided for beta_schedule_type = 'custom'")
            beta_schedule = np.array(beta_schedule, dtype=np.float32)
            num_betas = len(beta_schedule)
        else:
            num_betas, rem = divmod(num_sweeps, num_sweeps_per_beta)
            if rem > 0 or num_betas < 0:
                raise ValueError("'num_sweeps' must be divisible by 'num_sweeps_per_beta'")

            if beta_range is None:
                beta_range = _default_ising_beta_range(h, J)

            if num_betas == 1:
                beta_schedule = np.array([beta_range[-1]], dtype=np.float32)
            else:
                if beta_schedule_type == "linear":
                    beta_schedule = np.linspace(beta_range[0], beta_range[1], num=num_betas, dtype=np.float32)
                elif beta_schedule_type == "geometric":
                    beta_schedule = np.geomspace(beta_range[0], beta_range[1], num=num_betas, dtype=np.float32)

        # Create Metal buffers
        csr_row_ptr_buf = self._create_buffer(csr_row_ptr, "csr_row_ptr")
        csr_col_ind_buf = self._create_buffer(csr_col_ind, "csr_col_ind")
        csr_J_vals_buf = self._create_buffer(csr_J_vals, "csr_J_vals")
        beta_schedule_buf = self._create_buffer(beta_schedule, "beta_schedule")
        node_colors_buf = self._create_buffer(node_colors, "node_colors")

        # Allocate output buffers
        working_states_buf = self._create_buffer(np.zeros(num_reads * N, dtype=np.int8), "working_states")
        final_samples_buf = self._create_buffer(np.zeros(num_reads * N, dtype=np.int8), "final_samples")
        final_energies_buf = self._create_buffer(np.zeros(num_reads, dtype=np.int32), "final_energies")
        delta_energies_buf = self._create_buffer(np.zeros(num_reads * N, dtype=np.int32), "delta_energies")

        # Dispatch kernel
        cmd_buf = self._command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipeline_coloring)

        # Set buffers (matching kernel signature)
        encoder.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 2)
        encoder.setBytes_length_atIndex_(np.int32(N).tobytes(), 4, 3)
        encoder.setBytes_length_atIndex_(np.int32(num_betas).tobytes(), 4, 4)
        encoder.setBytes_length_atIndex_(np.int32(num_sweeps_per_beta).tobytes(), 4, 5)
        encoder.setBytes_length_atIndex_(np.uint32(seed if seed else 1).tobytes(), 4, 6)
        encoder.setBuffer_offset_atIndex_(beta_schedule_buf, 0, 7)
        encoder.setBuffer_offset_atIndex_(working_states_buf, 0, 8)
        encoder.setBuffer_offset_atIndex_(final_samples_buf, 0, 9)
        encoder.setBuffer_offset_atIndex_(final_energies_buf, 0, 10)
        encoder.setBuffer_offset_atIndex_(delta_energies_buf, 0, 11)
        encoder.setBuffer_offset_atIndex_(node_colors_buf, 0, 12)
        encoder.setBytes_length_atIndex_(np.int32(num_colors).tobytes(), 4, 13)

        # Dispatch
        max_threads_per_threadgroup = 1024
        threads_per_threadgroup = Metal.MTLSize(width=min(num_reads, max_threads_per_threadgroup), height=1, depth=1)
        num_threadgroups = Metal.MTLSize(
            width=(num_reads + threads_per_threadgroup.width - 1) // threads_per_threadgroup.width,
            height=1,
            depth=1
        )
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups, threads_per_threadgroup)

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        if cmd_buf.status() != Metal.MTLCommandBufferStatusCompleted:
            error = cmd_buf.error()
            raise RuntimeError(f"Metal command buffer failed: {error}")

        # Read results
        samples_data = np.frombuffer(
            final_samples_buf.contents().as_buffer(num_reads * N),
            dtype=np.int8
        ).reshape(num_reads, N)

        energies_data = np.frombuffer(
            final_energies_buf.contents().as_buffer(num_reads * 4),
            dtype=np.int32
        )

        self.logger.debug(f"[PureMetalSA-Coloring] Energy range: [{energies_data.min()}, {energies_data.max()}]")

        # Build SampleSet
        samples_dict = []
        for sample in samples_data:
            samples_dict.append({node: int(sample[idx]) for node, idx in node_to_idx.items()})

        return dimod.SampleSet.from_samples(
            samples_dict,
            energy=energies_data.astype(float),
            vartype=dimod.SPIN,
            info={"beta_range": beta_range, "beta_schedule_type": beta_schedule_type, "num_colors": num_colors}
        )

    def sample_ising_with_double_buffering(
        self,
        h: Dict[int, float],
        J: Dict[tuple, float],
        num_reads: int = 200,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
        beta_schedule: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> dimod.SampleSet:
        """
        PHASE 2: Sample using Pure SA with graph coloring + double buffering.

        This adds src/dst buffer swaps after each color while keeping:
        - Graph coloring with sequential color order
        - Delta energy array with incremental updates
        - Threshold skipping
        - Same beta schedule
        """
        if not self._pipeline_double_buffering:
            raise RuntimeError("Double buffering kernel not available. Use sample_ising() instead.")

        # Get all nodes (same as Phase 1)
        all_nodes = set(h.keys()) | set(n for edge in J.keys() for n in edge)
        N = len(all_nodes)
        node_list = sorted(all_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        self.logger.debug(f"[PureMetalSA-DoubleBuffer] N={N}, num_reads={num_reads}, num_sweeps={num_sweeps}")

        # Build CSR (same as Phase 1)
        csr_row_ptr = np.zeros(N + 1, dtype=np.int32)
        degree = np.zeros(N, dtype=np.int32)
        for (i, j) in J.keys():
            if i in node_to_idx and j in node_to_idx:
                degree[node_to_idx[i]] += 1
                degree[node_to_idx[j]] += 1

        csr_row_ptr[1:] = np.cumsum(degree)
        adjacency = [[] for _ in range(N)]
        for (i, j), Jij in J.items():
            if i in node_to_idx and j in node_to_idx:
                idx_i = node_to_idx[i]
                idx_j = node_to_idx[j]
                adjacency[idx_i].append((idx_j, Jij))
                adjacency[idx_j].append((idx_i, Jij))

        csr_col_ind = []
        csr_J_vals = []
        for i in range(N):
            adjacency[i].sort()
            for j, Jij in adjacency[i]:
                csr_col_ind.append(j)
                csr_J_vals.append(int(Jij))

        csr_col_ind = np.array(csr_col_ind, dtype=np.int32)
        csr_J_vals = np.array(csr_J_vals, dtype=np.int8)

        # Compute graph coloring
        node_colors, num_colors = self._compute_graph_coloring(csr_row_ptr, csr_col_ind, N)

        # Compute beta schedule (same as Phase 1)
        if beta_schedule_type == "custom":
            if beta_schedule is None:
                raise ValueError("'beta_schedule' must be provided for beta_schedule_type = 'custom'")
            beta_schedule = np.array(beta_schedule, dtype=np.float32)
            num_betas = len(beta_schedule)
        else:
            num_betas, rem = divmod(num_sweeps, num_sweeps_per_beta)
            if rem > 0 or num_betas < 0:
                raise ValueError("'num_sweeps' must be divisible by 'num_sweeps_per_beta'")

            if beta_range is None:
                beta_range = _default_ising_beta_range(h, J)

            if num_betas == 1:
                beta_schedule = np.array([beta_range[-1]], dtype=np.float32)
            else:
                if beta_schedule_type == "linear":
                    beta_schedule = np.linspace(beta_range[0], beta_range[1], num=num_betas, dtype=np.float32)
                elif beta_schedule_type == "geometric":
                    beta_schedule = np.geomspace(beta_range[0], beta_range[1], num=num_betas, dtype=np.float32)

        # Create Metal buffers
        csr_row_ptr_buf = self._create_buffer(csr_row_ptr, "csr_row_ptr")
        csr_col_ind_buf = self._create_buffer(csr_col_ind, "csr_col_ind")
        csr_J_vals_buf = self._create_buffer(csr_J_vals, "csr_J_vals")
        beta_schedule_buf = self._create_buffer(beta_schedule, "beta_schedule")
        node_colors_buf = self._create_buffer(node_colors, "node_colors")

        # PHASE 2: Allocate TWO working state buffers for double buffering
        working_states_src_buf = self._create_buffer(np.zeros(num_reads * N, dtype=np.int8), "working_states_src")
        working_states_dst_buf = self._create_buffer(np.zeros(num_reads * N, dtype=np.int8), "working_states_dst")
        final_samples_buf = self._create_buffer(np.zeros(num_reads * N, dtype=np.int8), "final_samples")
        final_energies_buf = self._create_buffer(np.zeros(num_reads, dtype=np.int32), "final_energies")
        delta_energies_buf = self._create_buffer(np.zeros(num_reads * N, dtype=np.int32), "delta_energies")

        # Dispatch kernel
        cmd_buf = self._command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipeline_double_buffering)

        # Set buffers (matching Phase 2 kernel signature)
        encoder.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 2)
        encoder.setBytes_length_atIndex_(np.int32(N).tobytes(), 4, 3)
        encoder.setBytes_length_atIndex_(np.int32(num_betas).tobytes(), 4, 4)
        encoder.setBytes_length_atIndex_(np.int32(num_sweeps_per_beta).tobytes(), 4, 5)
        encoder.setBytes_length_atIndex_(np.uint32(seed if seed else 1).tobytes(), 4, 6)
        encoder.setBuffer_offset_atIndex_(beta_schedule_buf, 0, 7)
        encoder.setBuffer_offset_atIndex_(working_states_src_buf, 0, 8)  # src buffer
        encoder.setBuffer_offset_atIndex_(working_states_dst_buf, 0, 9)  # dst buffer
        encoder.setBuffer_offset_atIndex_(final_samples_buf, 0, 10)
        encoder.setBuffer_offset_atIndex_(final_energies_buf, 0, 11)
        encoder.setBuffer_offset_atIndex_(delta_energies_buf, 0, 12)
        encoder.setBuffer_offset_atIndex_(node_colors_buf, 0, 13)
        encoder.setBytes_length_atIndex_(np.int32(num_colors).tobytes(), 4, 14)

        # Dispatch
        max_threads_per_threadgroup = 1024
        threads_per_threadgroup = Metal.MTLSize(width=min(num_reads, max_threads_per_threadgroup), height=1, depth=1)
        num_threadgroups = Metal.MTLSize(
            width=(num_reads + threads_per_threadgroup.width - 1) // threads_per_threadgroup.width,
            height=1,
            depth=1
        )
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups, threads_per_threadgroup)

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        if cmd_buf.status() != Metal.MTLCommandBufferStatusCompleted:
            error = cmd_buf.error()
            raise RuntimeError(f"Metal command buffer failed: {error}")

        # Read results
        samples_data = np.frombuffer(
            final_samples_buf.contents().as_buffer(num_reads * N),
            dtype=np.int8
        ).reshape(num_reads, N)

        energies_data = np.frombuffer(
            final_energies_buf.contents().as_buffer(num_reads * 4),
            dtype=np.int32
        )

        self.logger.debug(f"[PureMetalSA-DoubleBuffer] Energy range: [{energies_data.min()}, {energies_data.max()}]")

        # Build SampleSet
        samples_dict = []
        for sample in samples_data:
            samples_dict.append({node: int(sample[idx]) for node, idx in node_to_idx.items()})

        return dimod.SampleSet.from_samples(
            samples_dict,
            energy=energies_data.astype(float),
            vartype=dimod.SPIN,
            info={"beta_range": beta_range, "beta_schedule_type": beta_schedule_type, "num_colors": num_colors, "phase": "double_buffering"}
        )

    def sample_ising_with_per_color_precomputation(
        self,
        h: Dict[int, float],
        J: Dict[tuple, float],
        num_reads: int = 200,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
        beta_schedule: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> dimod.SampleSet:
        """
        PHASE 3: Sample using Pure SA with per-color delta energy precomputation.

        This adds per-color delta energy recomputation from src buffer while keeping:
        - Graph coloring with sequential color order
        - Double buffering (src/dst swap after each color)
        - Threshold skipping
        - Same beta schedule

        Key change: Instead of incrementally updating delta energies, we recompute them
        from the src buffer for each color (matching PT's approach).
        """
        if not self._pipeline_per_color_precomp:
            raise RuntimeError("Per-color precomputation kernel not available. Use sample_ising() instead.")

        # Get all nodes (same as Phase 2)
        all_nodes = set(h.keys()) | set(n for edge in J.keys() for n in edge)
        N = len(all_nodes)
        node_list = sorted(all_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        self.logger.debug(f"[PureMetalSA-PerColorPrecomp] N={N}, num_reads={num_reads}, num_sweeps={num_sweeps}")

        # Build CSR (same as Phase 2)
        csr_row_ptr = np.zeros(N + 1, dtype=np.int32)
        degree = np.zeros(N, dtype=np.int32)
        for (i, j) in J.keys():
            if i in node_to_idx and j in node_to_idx:
                degree[node_to_idx[i]] += 1
                degree[node_to_idx[j]] += 1

        csr_row_ptr[1:] = np.cumsum(degree)
        adjacency = [[] for _ in range(N)]
        for (i, j), Jij in J.items():
            if i in node_to_idx and j in node_to_idx:
                idx_i = node_to_idx[i]
                idx_j = node_to_idx[j]
                adjacency[idx_i].append((idx_j, Jij))
                adjacency[idx_j].append((idx_i, Jij))

        csr_col_ind = []
        csr_J_vals = []
        for i in range(N):
            adjacency[i].sort()
            for j, Jij in adjacency[i]:
                csr_col_ind.append(j)
                csr_J_vals.append(int(Jij))

        csr_col_ind = np.array(csr_col_ind, dtype=np.int32)
        csr_J_vals = np.array(csr_J_vals, dtype=np.int8)

        # Compute graph coloring
        node_colors, num_colors = self._compute_graph_coloring(csr_row_ptr, csr_col_ind, N)

        # Compute beta schedule (same as Phase 2)
        if beta_schedule_type == "custom":
            if beta_schedule is None:
                raise ValueError("'beta_schedule' must be provided for beta_schedule_type = 'custom'")
            beta_schedule = np.array(beta_schedule, dtype=np.float32)
            num_betas = len(beta_schedule)
        else:
            num_betas, rem = divmod(num_sweeps, num_sweeps_per_beta)
            if rem > 0 or num_betas < 0:
                raise ValueError("'num_sweeps' must be divisible by 'num_sweeps_per_beta'")

            if beta_range is None:
                beta_range = _default_ising_beta_range(h, J)

            if num_betas == 1:
                beta_schedule = np.array([beta_range[-1]], dtype=np.float32)
            else:
                if beta_schedule_type == "linear":
                    beta_schedule = np.linspace(beta_range[0], beta_range[1], num=num_betas, dtype=np.float32)
                elif beta_schedule_type == "geometric":
                    beta_schedule = np.geomspace(beta_range[0], beta_range[1], num=num_betas, dtype=np.float32)

        # Create Metal buffers
        csr_row_ptr_buf = self._create_buffer(csr_row_ptr, "csr_row_ptr")
        csr_col_ind_buf = self._create_buffer(csr_col_ind, "csr_col_ind")
        csr_J_vals_buf = self._create_buffer(csr_J_vals, "csr_J_vals")
        beta_schedule_buf = self._create_buffer(beta_schedule, "beta_schedule")
        node_colors_buf = self._create_buffer(node_colors, "node_colors")

        # Allocate TWO working state buffers for double buffering
        working_states_src_buf = self._create_buffer(np.zeros(num_reads * N, dtype=np.int8), "working_states_src")
        working_states_dst_buf = self._create_buffer(np.zeros(num_reads * N, dtype=np.int8), "working_states_dst")
        final_samples_buf = self._create_buffer(np.zeros(num_reads * N, dtype=np.int8), "final_samples")
        final_energies_buf = self._create_buffer(np.zeros(num_reads, dtype=np.int32), "final_energies")
        delta_energies_buf = self._create_buffer(np.zeros(num_reads * N, dtype=np.int32), "delta_energies")

        # Dispatch kernel
        cmd_buf = self._command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipeline_per_color_precomp)

        # Set buffers (matching Phase 3 kernel signature)
        encoder.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 2)
        encoder.setBytes_length_atIndex_(np.int32(N).tobytes(), 4, 3)
        encoder.setBytes_length_atIndex_(np.int32(num_betas).tobytes(), 4, 4)
        encoder.setBytes_length_atIndex_(np.int32(num_sweeps_per_beta).tobytes(), 4, 5)
        encoder.setBytes_length_atIndex_(np.uint32(seed if seed else 1).tobytes(), 4, 6)
        encoder.setBuffer_offset_atIndex_(beta_schedule_buf, 0, 7)
        encoder.setBuffer_offset_atIndex_(working_states_src_buf, 0, 8)  # src buffer
        encoder.setBuffer_offset_atIndex_(working_states_dst_buf, 0, 9)  # dst buffer
        encoder.setBuffer_offset_atIndex_(final_samples_buf, 0, 10)
        encoder.setBuffer_offset_atIndex_(final_energies_buf, 0, 11)
        encoder.setBuffer_offset_atIndex_(delta_energies_buf, 0, 12)
        encoder.setBuffer_offset_atIndex_(node_colors_buf, 0, 13)
        encoder.setBytes_length_atIndex_(np.int32(num_colors).tobytes(), 4, 14)

        # Dispatch
        max_threads_per_threadgroup = 1024
        threads_per_threadgroup = Metal.MTLSize(width=min(num_reads, max_threads_per_threadgroup), height=1, depth=1)
        num_threadgroups = Metal.MTLSize(
            width=(num_reads + threads_per_threadgroup.width - 1) // threads_per_threadgroup.width,
            height=1,
            depth=1
        )
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups, threads_per_threadgroup)

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        if cmd_buf.status() != Metal.MTLCommandBufferStatusCompleted:
            error = cmd_buf.error()
            raise RuntimeError(f"Metal command buffer failed: {error}")

        # Read results
        samples_data = np.frombuffer(
            final_samples_buf.contents().as_buffer(num_reads * N),
            dtype=np.int8
        ).reshape(num_reads, N)

        energies_data = np.frombuffer(
            final_energies_buf.contents().as_buffer(num_reads * 4),
            dtype=np.int32
        )

        self.logger.debug(f"[PureMetalSA-PerColorPrecomp] Energy range: [{energies_data.min()}, {energies_data.max()}]")

        # Build SampleSet
        samples_dict = []
        for sample in samples_data:
            samples_dict.append({node: int(sample[idx]) for node, idx in node_to_idx.items()})

        return dimod.SampleSet.from_samples(
            samples_dict,
            energy=energies_data.astype(float),
            vartype=dimod.SPIN,
            info={"beta_range": beta_range, "beta_schedule_type": beta_schedule_type, "num_colors": num_colors, "phase": "per_color_precomputation"}
        )

    def sample_ising_with_color_shuffling(
        self,
        h: Dict[int, float],
        J: Dict[tuple, float],
        num_reads: int = 200,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
        beta_schedule: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> dimod.SampleSet:
        """
        PHASE 4: Sample using Pure SA with color shuffling.

        This adds Fisher-Yates color order shuffling each sweep while keeping:
        - Graph coloring
        - Double buffering (src/dst swap after each color)
        - Per-color delta energy precomputation
        - Threshold skipping
        - Same beta schedule

        Key change: Randomize the color order each sweep using Fisher-Yates shuffle.
        """
        if not self._pipeline_color_shuffling:
            raise RuntimeError("Color shuffling kernel not available. Use sample_ising() instead.")

        # Get all nodes (same as Phase 3)
        all_nodes = set(h.keys()) | set(n for edge in J.keys() for n in edge)
        N = len(all_nodes)
        node_list = sorted(all_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        self.logger.debug(f"[PureMetalSA-ColorShuffling] N={N}, num_reads={num_reads}, num_sweeps={num_sweeps}")

        # Build CSR (same as Phase 3)
        csr_row_ptr = np.zeros(N + 1, dtype=np.int32)
        degree = np.zeros(N, dtype=np.int32)
        for (i, j) in J.keys():
            if i in node_to_idx and j in node_to_idx:
                degree[node_to_idx[i]] += 1
                degree[node_to_idx[j]] += 1

        csr_row_ptr[1:] = np.cumsum(degree)
        adjacency = [[] for _ in range(N)]
        for (i, j), Jij in J.items():
            if i in node_to_idx and j in node_to_idx:
                idx_i = node_to_idx[i]
                idx_j = node_to_idx[j]
                adjacency[idx_i].append((idx_j, Jij))
                adjacency[idx_j].append((idx_i, Jij))

        csr_col_ind = []
        csr_J_vals = []
        for i in range(N):
            adjacency[i].sort()
            for j, Jij in adjacency[i]:
                csr_col_ind.append(j)
                csr_J_vals.append(int(Jij))

        csr_col_ind = np.array(csr_col_ind, dtype=np.int32)
        csr_J_vals = np.array(csr_J_vals, dtype=np.int8)

        # Compute graph coloring
        node_colors, num_colors = self._compute_graph_coloring(csr_row_ptr, csr_col_ind, N)

        # Compute beta schedule (same as Phase 3)
        if beta_schedule_type == "custom":
            if beta_schedule is None:
                raise ValueError("'beta_schedule' must be provided for beta_schedule_type = 'custom'")
            beta_schedule = np.array(beta_schedule, dtype=np.float32)
            num_betas = len(beta_schedule)
        else:
            num_betas, rem = divmod(num_sweeps, num_sweeps_per_beta)
            if rem > 0 or num_betas < 0:
                raise ValueError("'num_sweeps' must be divisible by 'num_sweeps_per_beta'")

            if beta_range is None:
                beta_range = _default_ising_beta_range(h, J)

            if num_betas == 1:
                beta_schedule = np.array([beta_range[-1]], dtype=np.float32)
            else:
                if beta_schedule_type == "linear":
                    beta_schedule = np.linspace(beta_range[0], beta_range[1], num=num_betas, dtype=np.float32)
                elif beta_schedule_type == "geometric":
                    beta_schedule = np.geomspace(beta_range[0], beta_range[1], num=num_betas, dtype=np.float32)

        # Create Metal buffers
        csr_row_ptr_buf = self._create_buffer(csr_row_ptr, "csr_row_ptr")
        csr_col_ind_buf = self._create_buffer(csr_col_ind, "csr_col_ind")
        csr_J_vals_buf = self._create_buffer(csr_J_vals, "csr_J_vals")
        beta_schedule_buf = self._create_buffer(beta_schedule, "beta_schedule")
        node_colors_buf = self._create_buffer(node_colors, "node_colors")

        # Allocate TWO working state buffers for double buffering
        working_states_src_buf = self._create_buffer(np.zeros(num_reads * N, dtype=np.int8), "working_states_src")
        working_states_dst_buf = self._create_buffer(np.zeros(num_reads * N, dtype=np.int8), "working_states_dst")
        final_samples_buf = self._create_buffer(np.zeros(num_reads * N, dtype=np.int8), "final_samples")
        final_energies_buf = self._create_buffer(np.zeros(num_reads, dtype=np.int32), "final_energies")
        delta_energies_buf = self._create_buffer(np.zeros(num_reads * N, dtype=np.int32), "delta_energies")

        # Dispatch kernel
        cmd_buf = self._command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipeline_color_shuffling)

        # Set buffers (matching Phase 4 kernel signature)
        encoder.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 2)
        encoder.setBytes_length_atIndex_(np.int32(N).tobytes(), 4, 3)
        encoder.setBytes_length_atIndex_(np.int32(num_betas).tobytes(), 4, 4)
        encoder.setBytes_length_atIndex_(np.int32(num_sweeps_per_beta).tobytes(), 4, 5)
        encoder.setBytes_length_atIndex_(np.uint32(seed if seed else 1).tobytes(), 4, 6)
        encoder.setBuffer_offset_atIndex_(beta_schedule_buf, 0, 7)
        encoder.setBuffer_offset_atIndex_(working_states_src_buf, 0, 8)  # src buffer
        encoder.setBuffer_offset_atIndex_(working_states_dst_buf, 0, 9)  # dst buffer
        encoder.setBuffer_offset_atIndex_(final_samples_buf, 0, 10)
        encoder.setBuffer_offset_atIndex_(final_energies_buf, 0, 11)
        encoder.setBuffer_offset_atIndex_(delta_energies_buf, 0, 12)
        encoder.setBuffer_offset_atIndex_(node_colors_buf, 0, 13)
        encoder.setBytes_length_atIndex_(np.int32(num_colors).tobytes(), 4, 14)

        # Dispatch
        max_threads_per_threadgroup = 1024
        threads_per_threadgroup = Metal.MTLSize(width=min(num_reads, max_threads_per_threadgroup), height=1, depth=1)
        num_threadgroups = Metal.MTLSize(
            width=(num_reads + threads_per_threadgroup.width - 1) // threads_per_threadgroup.width,
            height=1,
            depth=1
        )
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups, threads_per_threadgroup)

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        if cmd_buf.status() != Metal.MTLCommandBufferStatusCompleted:
            error = cmd_buf.error()
            raise RuntimeError(f"Metal command buffer failed: {error}")

        # Read results
        samples_data = np.frombuffer(
            final_samples_buf.contents().as_buffer(num_reads * N),
            dtype=np.int8
        ).reshape(num_reads, N)

        energies_data = np.frombuffer(
            final_energies_buf.contents().as_buffer(num_reads * 4),
            dtype=np.int32
        )

        self.logger.debug(f"[PureMetalSA-ColorShuffling] Energy range: [{energies_data.min()}, {energies_data.max()}]")

        # Build SampleSet
        samples_dict = []
        for sample in samples_data:
            samples_dict.append({node: int(sample[idx]) for node, idx in node_to_idx.items()})

        return dimod.SampleSet.from_samples(
            samples_dict,
            energy=energies_data.astype(float),
            vartype=dimod.SPIN,
            info={"beta_range": beta_range, "beta_schedule_type": beta_schedule_type, "num_colors": num_colors, "phase": "color_shuffling"}
        )

    def sample_ising_with_multiple_replicas(
        self,
        h: Dict[int, float],
        J: Dict[tuple, float],
        num_replicas: int = 16,
        num_sweeps: int = 1000,
        beta_range: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> dimod.SampleSet:
        """
        PHASE 5: Sample using Pure SA with multiple temperature replicas (NO replica exchange).

        This runs multiple independent chains at different temperatures while keeping:
        - Graph coloring
        - Double buffering
        - Per-color delta energy precomputation
        - Color shuffling

        Key change: Each thread runs ONE replica at ONE fixed temperature.
        NO replica exchange - just independent parallel chains.
        """
        if not self._pipeline_multiple_replicas:
            raise RuntimeError("Multiple replicas kernel not available. Use sample_ising() instead.")

        # Get all nodes
        all_nodes = set(h.keys()) | set(n for edge in J.keys() for n in edge)
        N = len(all_nodes)
        node_list = sorted(all_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        self.logger.debug(f"[PureMetalSA-MultiReplica] N={N}, num_replicas={num_replicas}, num_sweeps={num_sweeps}")

        # Build CSR
        csr_row_ptr = np.zeros(N + 1, dtype=np.int32)
        degree = np.zeros(N, dtype=np.int32)
        for (i, j) in J.keys():
            if i in node_to_idx and j in node_to_idx:
                degree[node_to_idx[i]] += 1
                degree[node_to_idx[j]] += 1

        csr_row_ptr[1:] = np.cumsum(degree)
        adjacency = [[] for _ in range(N)]
        for (i, j), Jij in J.items():
            if i in node_to_idx and j in node_to_idx:
                idx_i = node_to_idx[i]
                idx_j = node_to_idx[j]
                adjacency[idx_i].append((idx_j, Jij))
                adjacency[idx_j].append((idx_i, Jij))

        csr_col_ind = []
        csr_J_vals = []
        for i in range(N):
            adjacency[i].sort()
            for j, Jij in adjacency[i]:
                csr_col_ind.append(j)
                csr_J_vals.append(int(Jij))

        csr_col_ind = np.array(csr_col_ind, dtype=np.int32)
        csr_J_vals = np.array(csr_J_vals, dtype=np.int8)

        # Compute graph coloring
        node_colors, num_colors = self._compute_graph_coloring(csr_row_ptr, csr_col_ind, N)

        # Create temperature ladder (one beta per replica)
        if beta_range is None:
            beta_range = _default_ising_beta_range(h, J)

        # Geometric schedule from hot to cold
        beta_schedule = np.geomspace(beta_range[0], beta_range[1], num=num_replicas, dtype=np.float32)

        # Create Metal buffers
        csr_row_ptr_buf = self._create_buffer(csr_row_ptr, "csr_row_ptr")
        csr_col_ind_buf = self._create_buffer(csr_col_ind, "csr_col_ind")
        csr_J_vals_buf = self._create_buffer(csr_J_vals, "csr_J_vals")
        beta_schedule_buf = self._create_buffer(beta_schedule, "beta_schedule")
        node_colors_buf = self._create_buffer(node_colors, "node_colors")

        # Allocate buffers for all replicas
        working_states_src_buf = self._create_buffer(np.zeros(num_replicas * N, dtype=np.int8), "working_states_src")
        working_states_dst_buf = self._create_buffer(np.zeros(num_replicas * N, dtype=np.int8), "working_states_dst")
        final_samples_buf = self._create_buffer(np.zeros(num_replicas * N, dtype=np.int8), "final_samples")
        final_energies_buf = self._create_buffer(np.zeros(num_replicas, dtype=np.int32), "final_energies")
        delta_energies_buf = self._create_buffer(np.zeros(num_replicas * N, dtype=np.int32), "delta_energies")

        # Dispatch kernel
        cmd_buf = self._command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipeline_multiple_replicas)

        # Set buffers (matching Phase 5 kernel signature)
        encoder.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 2)
        encoder.setBytes_length_atIndex_(np.int32(N).tobytes(), 4, 3)
        encoder.setBytes_length_atIndex_(np.int32(num_sweeps).tobytes(), 4, 4)
        # Skip buffer 5 (was num_sweeps_per_beta)
        encoder.setBytes_length_atIndex_(np.uint32(seed if seed else 1).tobytes(), 4, 6)
        encoder.setBuffer_offset_atIndex_(beta_schedule_buf, 0, 7)
        encoder.setBytes_length_atIndex_(np.int32(num_replicas).tobytes(), 4, 8)
        encoder.setBuffer_offset_atIndex_(working_states_src_buf, 0, 9)
        encoder.setBuffer_offset_atIndex_(working_states_dst_buf, 0, 10)
        encoder.setBuffer_offset_atIndex_(final_samples_buf, 0, 11)
        encoder.setBuffer_offset_atIndex_(final_energies_buf, 0, 12)
        encoder.setBuffer_offset_atIndex_(delta_energies_buf, 0, 13)
        encoder.setBuffer_offset_atIndex_(node_colors_buf, 0, 14)
        encoder.setBytes_length_atIndex_(np.int32(num_colors).tobytes(), 4, 15)

        # Dispatch - one thread per replica
        max_threads_per_threadgroup = 1024
        threads_per_threadgroup = Metal.MTLSize(width=min(num_replicas, max_threads_per_threadgroup), height=1, depth=1)
        num_threadgroups = Metal.MTLSize(
            width=(num_replicas + threads_per_threadgroup.width - 1) // threads_per_threadgroup.width,
            height=1,
            depth=1
        )
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups, threads_per_threadgroup)

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        if cmd_buf.status() != Metal.MTLCommandBufferStatusCompleted:
            error = cmd_buf.error()
            raise RuntimeError(f"Metal command buffer failed: {error}")

        # Read results
        samples_data = np.frombuffer(
            final_samples_buf.contents().as_buffer(num_replicas * N),
            dtype=np.int8
        ).reshape(num_replicas, N)

        energies_data = np.frombuffer(
            final_energies_buf.contents().as_buffer(num_replicas * 4),
            dtype=np.int32
        )

        self.logger.debug(f"[PureMetalSA-MultiReplica] Energy range: [{energies_data.min()}, {energies_data.max()}]")

        # Build SampleSet
        samples_dict = []
        for sample in samples_data:
            samples_dict.append({node: int(sample[idx]) for node, idx in node_to_idx.items()})

        return dimod.SampleSet.from_samples(
            samples_dict,
            energy=energies_data.astype(float),
            vartype=dimod.SPIN,
            info={"beta_range": beta_range, "num_replicas": num_replicas, "num_colors": num_colors, "phase": "multiple_replicas"}
        )

    def sample_ising_with_replica_exchange(
        self,
        h: Dict[int, float],
        J: Dict[tuple, float],
        num_replicas: int = 16,
        num_sweeps: int = 16,
        num_exchanges: int = 16,
        beta_range: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> dimod.SampleSet:
        """
        PHASE 6: Sample using Pure SA with replica exchange.

        This is the complete parallel tempering algorithm:
        - Multiple temperature replicas
        - Replica exchange (swapping states between adjacent temperatures)
        - Graph coloring
        - Double buffering
        - Per-color delta energy precomputation
        - Color shuffling

        Key change: Adds replica exchange to allow information flow between temperatures.

        Args:
            h: Linear biases
            J: Quadratic biases
            num_replicas: Number of temperature replicas
            num_sweeps: Number of sweeps per exchange attempt
            num_exchanges: Number of exchange attempts
            beta_range: (beta_min, beta_max) temperature range
            seed: Random seed

        Returns:
            SampleSet with num_replicas samples
        """
        if not self._pipeline_replica_exchange:
            raise RuntimeError("Replica exchange pipeline not available")

        # Get all nodes
        all_nodes = set(h.keys()) | set(n for edge in J.keys() for n in edge)
        N = len(all_nodes)
        node_list = sorted(all_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        self.logger.debug(f"[PureMetalSA-ReplicaExchange] N={N}, num_replicas={num_replicas}, num_sweeps={num_sweeps}, num_exchanges={num_exchanges}")

        # Build CSR
        csr_row_ptr = np.zeros(N + 1, dtype=np.int32)
        degree = np.zeros(N, dtype=np.int32)
        for (i, j) in J.keys():
            if i in node_to_idx and j in node_to_idx:
                degree[node_to_idx[i]] += 1
                degree[node_to_idx[j]] += 1

        csr_row_ptr[1:] = np.cumsum(degree)
        adjacency = [[] for _ in range(N)]
        for (i, j), Jij in J.items():
            if i in node_to_idx and j in node_to_idx:
                idx_i = node_to_idx[i]
                idx_j = node_to_idx[j]
                adjacency[idx_i].append((idx_j, Jij))
                adjacency[idx_j].append((idx_i, Jij))

        csr_col_ind = []
        csr_J_vals = []
        for i in range(N):
            adjacency[i].sort()
            for j, Jij in adjacency[i]:
                csr_col_ind.append(j)
                csr_J_vals.append(int(Jij))

        csr_col_ind = np.array(csr_col_ind, dtype=np.int32)
        csr_J_vals = np.array(csr_J_vals, dtype=np.int8)

        # Compute graph coloring
        node_colors, num_colors = self._compute_graph_coloring(csr_row_ptr, csr_col_ind, N)
        self.logger.debug(f"[PureMetalSA-ReplicaExchange] Graph coloring: {num_colors} colors")

        # Create temperature ladder (one beta per replica)
        if beta_range is None:
            beta_range = _default_ising_beta_range(h, J)

        # Geometric schedule from hot to cold
        beta_schedule = np.geomspace(beta_range[0], beta_range[1], num=num_replicas, dtype=np.float32)

        # Create buffers
        csr_row_ptr_buf = self._create_buffer(csr_row_ptr, "csr_row_ptr")
        csr_col_ind_buf = self._create_buffer(csr_col_ind, "csr_col_ind")
        csr_J_vals_buf = self._create_buffer(csr_J_vals, "csr_J_vals")
        beta_schedule_buf = self._create_buffer(beta_schedule, "beta_schedule")
        node_colors_buf = self._create_buffer(node_colors, "node_colors")

        # Working memory
        working_states_buf = self.device.newBufferWithLength_options_(
            num_replicas * N, Metal.MTLResourceStorageModeShared
        )

        # Output buffers
        final_samples_buf = self.device.newBufferWithLength_options_(
            num_replicas * N, Metal.MTLResourceStorageModeShared
        )
        replica_energies_buf = self.device.newBufferWithLength_options_(
            num_replicas * 4, Metal.MTLResourceStorageModeShared
        )

        # Scalar buffers
        N_buf = self._create_buffer(np.array([N], dtype=np.int32), "N")
        num_sweeps_buf = self._create_buffer(np.array([num_sweeps], dtype=np.int32), "num_sweeps")
        num_exchanges_buf = self._create_buffer(np.array([num_exchanges], dtype=np.int32), "num_exchanges")
        base_seed_buf = self._create_buffer(np.array([seed if seed is not None else np.random.randint(1, 2**31)], dtype=np.uint32), "base_seed")
        num_replicas_buf = self._create_buffer(np.array([num_replicas], dtype=np.int32), "num_replicas")
        num_colors_buf = self._create_buffer(np.array([num_colors], dtype=np.int32), "num_colors")

        # Dispatch kernel
        cmd_buf = self._command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipeline_replica_exchange)

        # Set buffers
        encoder.setBuffer_offset_atIndex_(csr_row_ptr_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(csr_col_ind_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(csr_J_vals_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(N_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(num_sweeps_buf, 0, 4)
        encoder.setBuffer_offset_atIndex_(num_exchanges_buf, 0, 5)
        encoder.setBuffer_offset_atIndex_(base_seed_buf, 0, 6)
        encoder.setBuffer_offset_atIndex_(beta_schedule_buf, 0, 7)
        encoder.setBuffer_offset_atIndex_(num_replicas_buf, 0, 8)
        encoder.setBuffer_offset_atIndex_(working_states_buf, 0, 9)
        encoder.setBuffer_offset_atIndex_(final_samples_buf, 0, 10)
        encoder.setBuffer_offset_atIndex_(replica_energies_buf, 0, 11)
        encoder.setBuffer_offset_atIndex_(node_colors_buf, 0, 12)
        encoder.setBuffer_offset_atIndex_(num_colors_buf, 0, 13)

        # Dispatch: one thread per replica
        threads_per_group = Metal.MTLSizeMake(min(num_replicas, 256), 1, 1)
        num_groups = (num_replicas + 255) // 256
        threadgroups = Metal.MTLSizeMake(num_groups, 1, 1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadgroups, threads_per_group)

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        # Read results
        samples_data = np.frombuffer(
            final_samples_buf.contents().as_buffer(num_replicas * N),
            dtype=np.int8
        ).reshape(num_replicas, N)

        energies_data = np.frombuffer(
            replica_energies_buf.contents().as_buffer(num_replicas * 4),
            dtype=np.int32
        )

        self.logger.debug(f"[PureMetalSA-ReplicaExchange] Energy range: [{energies_data.min()}, {energies_data.max()}]")

        # Build SampleSet
        samples_dict = []
        for sample in samples_data:
            samples_dict.append({node: int(sample[idx]) for node, idx in node_to_idx.items()})

        return dimod.SampleSet.from_samples(
            samples_dict,
            energy=energies_data.astype(float),
            vartype=dimod.SPIN,
            info={"beta_range": beta_range, "num_replicas": num_replicas, "num_colors": num_colors, "num_sweeps": num_sweeps, "num_exchanges": num_exchanges, "phase": "replica_exchange"}
        )
