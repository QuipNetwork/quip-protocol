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
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
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
        all_csr_row_ptr = []
        all_csr_col_ind = []
        all_csr_J_vals = []
        all_h_vals = []  # Concatenated h values for all problems
        row_ptr_offsets = [0]  # Offsets into csr_row_ptr array
        col_ind_offsets = [0]  # Offsets into csr_col_ind array
        node_to_idx_list = []
        N_list = []

        for prob_idx, (h_prob, J_prob) in enumerate(zip(h, J)):
            # Get all nodes for this problem
            all_nodes = set(h_prob.keys()) | set(n for edge in J_prob.keys() for n in edge)
            N = len(all_nodes)
            N_list.append(N)
            node_list = sorted(all_nodes)
            node_to_idx = {node: idx for idx, node in enumerate(node_list)}
            node_to_idx_list.append(node_to_idx)

            # Build CSR representation for this problem
            csr_row_ptr = np.zeros(N + 1, dtype=np.int32)
            csr_col_ind = []
            csr_J_vals = []

            # Extract h values in node order
            h_vals_array = np.zeros(N, dtype=np.int8)
            for node, h_val in h_prob.items():
                if node in node_to_idx:
                    h_vals_array[node_to_idx[node]] = int(h_val)

            # Count degrees
            degree = np.zeros(N, dtype=np.int32)
            for (i, j) in J_prob.keys():
                if i in node_to_idx and j in node_to_idx:
                    degree[node_to_idx[i]] += 1
                    degree[node_to_idx[j]] += 1

            # Build CSR
            csr_row_ptr[1:] = np.cumsum(degree)

            adjacency = [[] for _ in range(N)]
            for (i, j), Jij in J_prob.items():
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

            # Append to concatenated arrays
            all_csr_row_ptr.extend(csr_row_ptr)
            all_csr_col_ind.extend(csr_col_ind)
            all_csr_J_vals.extend(csr_J_vals)
            all_h_vals.extend(h_vals_array)

            # Track offsets for next problem
            row_ptr_offsets.append(len(all_csr_row_ptr))
            col_ind_offsets.append(len(all_csr_col_ind))

            self.logger.debug(f"[MetalSA] Problem {prob_idx}: N={N}, edges={len(csr_col_ind)}, row_ptr_offset={row_ptr_offsets[-2]}, col_ind_offset={col_ind_offsets[-2]}")

        # Convert to numpy arrays
        all_csr_row_ptr = np.array(all_csr_row_ptr, dtype=np.int32)
        all_csr_col_ind = np.array(all_csr_col_ind, dtype=np.int32)
        all_csr_J_vals = np.array(all_csr_J_vals, dtype=np.int8)
        all_h_vals = np.array(all_h_vals, dtype=np.int8)
        row_ptr_offsets = np.array(row_ptr_offsets, dtype=np.int32)
        col_ind_offsets = np.array(col_ind_offsets, dtype=np.int32)

        # Use first problem's N for uniform sizing (all problems should have same N)
        N = N_list[0]
        if not all(n == N for n in N_list):
            raise ValueError(f"All problems must have same N: {N_list}")

        # Compute beta schedule (matching D-Wave exactly) - use first problem for auto range
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
                # Use first problem to determine beta range
                beta_range = _default_ising_beta_range(h[0], J[0])
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

        # Parse into separate SampleSets
        samplesets = []
        for prob_idx in range(num_problems):
            start_idx = prob_idx * num_reads
            end_idx = (prob_idx + 1) * num_reads

            # Extract this problem's results
            prob_packed = packed_data[start_idx:end_idx]
            prob_energies = energies_data[start_idx:end_idx]

            # Unpack bit-packed samples using this problem's node mapping
            samples_data = np.zeros((num_reads, N), dtype=np.int8)
            for read_idx in range(num_reads):
                for var in range(N):
                    byte_idx = var >> 3  # var / 8
                    bit_idx = var & 7    # var % 8
                    bit = (prob_packed[read_idx, byte_idx] >> bit_idx) & 1
                    samples_data[read_idx, var] = -1 if bit else 1

            # Build SampleSet using this problem's node_to_idx mapping
            samples_dict = []
            for sample in samples_data:
                samples_dict.append({node: int(sample[idx]) for node, idx in node_to_idx_list[prob_idx].items()})

            sampleset = dimod.SampleSet.from_samples(
                samples_dict,
                energy=prob_energies.astype(float),
                vartype=dimod.SPIN,
                info={"beta_range": beta_range, "beta_schedule_type": beta_schedule_type}
            )
            samplesets.append(sampleset)

            self.logger.debug(f"[MetalSA] Problem {prob_idx}: energy range [{prob_energies.min()}, {prob_energies.max()}]")

        return samplesets

