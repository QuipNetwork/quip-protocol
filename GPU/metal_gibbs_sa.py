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
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
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
    Determine the starting and ending beta from h, J.

    Exact replica of D-Wave's _default_ising_beta_range function.
    """
    if not 0 < max_single_qubit_excitation_rate < 1:
        raise ValueError('Targeted single qubit excitations rates must be in range (0,1)')

    sum_abs_bias_dict = defaultdict(int, {k: abs(v) for k, v in h.items()})
    if sum_abs_bias_dict:
        min_abs_bias_dict = {k: v for k, v in sum_abs_bias_dict.items() if v != 0}
    else:
        min_abs_bias_dict = {}

    for (k1, k2), v in J.items():
        for k in [k1, k2]:
            sum_abs_bias_dict[k] += abs(v)
            if v != 0:
                if k in min_abs_bias_dict:
                    min_abs_bias_dict[k] = min(abs(v), min_abs_bias_dict[k])
                else:
                    min_abs_bias_dict[k] = abs(v)

    if not min_abs_bias_dict:
        warn_msg = ('All bqm biases are zero (all energies are zero), this is '
                   'likely a value error. Temperature range is set arbitrarily '
                   'to [0.1,1]. Metropolis-Hastings update is non-ergodic.')
        warnings.warn(warn_msg)
        return (0.1, 1.0)

    max_effective_field = max(sum_abs_bias_dict.values(), default=0)

    if max_effective_field == 0:
        hot_beta = 1.0
    else:
        hot_beta = np.log(2) / (2 * max_effective_field)

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


def zephyr_four_color_linear(linear_idx: int, m: int = 9, t: int = 2) -> int:
    """Compute 4-color for Zephyr node given linear index.

    Converts linear index to Zephyr coordinates, then applies coloring.
    Based on dwave_networkx.zephyr_four_color scheme 0.

    The Zephyr linear index encoding is:
        r = u * M * t * 2 * m + w * t * 2 * m + k * 2 * m + j * m + z
    where M = 2*m + 1

    We reverse this to get (u, w, k, j, z), then apply:
        color = j + ((w + 2*(z+u) + j) & 2)

    Args:
        linear_idx: Linear node index
        m: Zephyr m parameter (default 9 for Z(9,2))
        t: Zephyr t parameter (default 2)

    Returns:
        Color index (0-3)
    """
    M = 2 * m + 1  # = 19 for m=9

    # Decode linear index to Zephyr coordinates
    r = linear_idx
    r, z = divmod(r, m)
    r, j = divmod(r, 2)
    r, k = divmod(r, t)
    u, w = divmod(r, M)

    # Apply zephyr_four_color scheme 0
    return j + ((w + 2 * (z + u) + j) & 2)


def compute_color_blocks(nodes: List[int], m: int = 9, t: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute color block partitions for Zephyr topology.

    Partitions nodes by their graph coloring. For Zephyr topologies,
    this produces 4 independent sets where no two adjacent nodes
    share the same color.

    Args:
        nodes: List of node indices
        m: Zephyr m parameter
        t: Zephyr t parameter

    Returns:
        Tuple of (block_starts, block_counts, color_node_indices)
        - block_starts: [4] start indices into color_node_indices
        - block_counts: [4] number of nodes per color
        - color_node_indices: [N] nodes sorted by color
    """
    # Compute color for each node
    node_colors = {node: zephyr_four_color_linear(node, m, t) for node in nodes}

    # Group nodes by color
    color_groups = defaultdict(list)
    for node in nodes:
        color_groups[node_colors[node]].append(node)

    # Sort each color group for determinism
    for color in color_groups:
        color_groups[color].sort()

    # Build output arrays
    num_colors = 4
    block_starts = np.zeros(num_colors, dtype=np.int32)
    block_counts = np.zeros(num_colors, dtype=np.int32)

    # Concatenate all color groups
    color_node_indices = []
    current_start = 0
    for color in range(num_colors):
        nodes_in_color = color_groups.get(color, [])
        block_starts[color] = current_start
        block_counts[color] = len(nodes_in_color)
        color_node_indices.extend(nodes_in_color)
        current_start += len(nodes_in_color)

    color_node_indices = np.array(color_node_indices, dtype=np.int32)

    return block_starts, block_counts, color_node_indices


class MetalGibbsSampler:
    """
    Block Gibbs sampler using Metal GPU.

    Based on dwave-pytorch-plugin BlockSampler algorithm:
    1. Partition nodes by graph coloring (4 colors for Zephyr)
    2. For each temperature step:
       - For each color block:
         - Compute effective field for all nodes in block
         - Sample new spins (Gibbs) or accept/reject flips (Metropolis)

    Two execution modes:
    - Sequential (parallel=False): One thread per sample, nodes updated sequentially within colors
    - Parallel (parallel=True): One threadgroup per sample, threads divide nodes within colors,
      with barriers between colors for true parallel Gibbs updates
    """

    def __init__(self, topology=None, update_mode: str = "gibbs", parallel: bool = False):
        """Initialize Metal Gibbs sampler.

        Args:
            topology: Topology object (default: DEFAULT_TOPOLOGY)
            update_mode: "gibbs" or "metropolis" (default: "gibbs")
            parallel: Use parallel kernel with threadgroup barriers (default: False)
        """
        self.logger = logging.getLogger(__name__)
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError("Metal is not supported on this device")

        if update_mode.lower() not in ("gibbs", "metropolis"):
            raise ValueError(f"update_mode must be 'gibbs' or 'metropolis', got {update_mode}")
        self.update_mode = 0 if update_mode.lower() == "gibbs" else 1
        self.update_mode_name = update_mode.lower()
        self.parallel = parallel

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
        self.logger.debug(f"Parallel mode: {self.parallel}")

        # Load and compile Metal kernels
        kernel_path = os.path.join(os.path.dirname(__file__), "metal_gibbs.metal")
        with open(kernel_path, 'r') as f:
            kernel_source = f.read()

        lib, err = self.device.newLibraryWithSource_options_error_(kernel_source, None, None)
        if err:
            raise RuntimeError(f"Failed to compile Metal kernels: {err}")
        if not lib:
            raise RuntimeError("Failed to create Metal library (no error reported)")

        # Load sequential kernel
        self._kernel = lib.newFunctionWithName_("block_gibbs_sampler")
        if not self._kernel:
            function_names = [lib.functionNames()[i] for i in range(len(lib.functionNames()))]
            raise RuntimeError(f"Failed to find block_gibbs_sampler kernel. Available: {function_names}")

        self._pipeline, err = self.device.newComputePipelineStateWithFunction_error_(self._kernel, None)
        if err or not self._pipeline:
            raise RuntimeError(f"Failed to create sequential pipeline: {err}")

        # Load parallel kernel
        self._kernel_parallel = lib.newFunctionWithName_("block_gibbs_parallel")
        if not self._kernel_parallel:
            function_names = [lib.functionNames()[i] for i in range(len(lib.functionNames()))]
            raise RuntimeError(f"Failed to find block_gibbs_parallel kernel. Available: {function_names}")

        self._pipeline_parallel, err = self.device.newComputePipelineStateWithFunction_error_(self._kernel_parallel, None)
        if err or not self._pipeline_parallel:
            raise RuntimeError(f"Failed to create parallel pipeline: {err}")

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
        all_csr_row_ptr = []
        all_csr_col_ind = []
        all_csr_J_vals = []
        all_h_vals = []
        row_ptr_offsets = [0]
        col_ind_offsets = [0]
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
                adjacency[i].sort()
                for j, Jij in adjacency[i]:
                    csr_col_ind.append(j)
                    csr_J_vals.append(int(Jij))

            # Append to concatenated arrays
            all_csr_row_ptr.extend(csr_row_ptr)
            all_csr_col_ind.extend(csr_col_ind)
            all_csr_J_vals.extend(csr_J_vals)
            all_h_vals.extend(h_vals_array)

            row_ptr_offsets.append(len(all_csr_row_ptr))
            col_ind_offsets.append(len(all_csr_col_ind))

        # Convert to numpy arrays
        all_csr_row_ptr = np.array(all_csr_row_ptr, dtype=np.int32)
        all_csr_col_ind = np.array(all_csr_col_ind, dtype=np.int32)
        all_csr_J_vals = np.array(all_csr_J_vals, dtype=np.int8)
        all_h_vals = np.array(all_h_vals, dtype=np.int8)
        row_ptr_offsets = np.array(row_ptr_offsets, dtype=np.int32)
        col_ind_offsets = np.array(col_ind_offsets, dtype=np.int32)

        N = N_list[0]
        if not all(n == N for n in N_list):
            raise ValueError(f"All problems must have same N: {N_list}")

        # Compute beta schedule
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
        # Remap color_node_indices from topology node IDs to dense CSR indices.
        # Topology node IDs can be non-contiguous (e.g., Advantage2 has 4582 nodes
        # with IDs spanning 0-4799 due to dead qubits). The CSR structure uses dense
        # indices 0..N-1, so we must translate.
        node_to_idx = node_to_idx_list[0]  # All problems share same topology
        csr_color_node_indices = np.array(
            [node_to_idx[n] for n in self.color_node_indices],
            dtype=np.int32
        )
        color_block_starts_buf = self._create_buffer(self.block_starts, "color_block_starts")
        color_block_counts_buf = self._create_buffer(self.block_counts, "color_block_counts")
        color_node_indices_buf = self._create_buffer(csr_color_node_indices, "color_node_indices")

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

        # Select kernel based on parallel mode
        if self.parallel:
            pipeline = self._pipeline_parallel
            kernel_name = "block_gibbs_parallel"
        else:
            pipeline = self._pipeline
            kernel_name = "block_gibbs_sampler"

        encoder.setComputePipelineState_(pipeline)

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

        # Dispatch configuration depends on kernel mode
        if self.parallel:
            # Parallel kernel: one threadgroup per sample, multiple threads per group
            # Each threadgroup collaboratively updates one sample
            num_samples = num_problems * num_reads

            # Use 256 threads per threadgroup (good balance for color block parallelism)
            # For ~342 nodes/color, 256 threads means each thread handles ~1-2 nodes
            threads_per_group = min(256, pipeline.maxTotalThreadsPerThreadgroup())

            threads_per_threadgroup = Metal.MTLSize(width=threads_per_group, height=1, depth=1)
            num_threadgroups_size = Metal.MTLSize(width=num_samples, height=1, depth=1)

            self.logger.debug(f"[MetalGibbs] Parallel dispatch ({kernel_name}): {num_samples} threadgroups x {threads_per_group} threads")
        else:
            # Sequential kernel: one thread per sample
            max_threads_per_group = pipeline.maxTotalThreadsPerThreadgroup()

            # Group samples by problem for locality
            num_threadgroups_width = num_problems
            threads_per_threadgroup_width = min(num_reads, max_threads_per_group)

            threads_per_threadgroup = Metal.MTLSize(width=threads_per_threadgroup_width, height=1, depth=1)
            num_threadgroups_size = Metal.MTLSize(width=num_threadgroups_width, height=1, depth=1)

            self.logger.debug(f"[MetalGibbs] Sequential dispatch ({kernel_name}): {num_threadgroups_width} threadgroups x {threads_per_threadgroup_width} threads")

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups_size, threads_per_threadgroup)

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

        # Parse into separate SampleSets
        samplesets = []
        for prob_idx in range(num_problems):
            start_idx = prob_idx * num_reads
            end_idx = (prob_idx + 1) * num_reads

            prob_packed = packed_data[start_idx:end_idx]
            prob_energies = energies_data[start_idx:end_idx]

            # Unpack bit-packed samples
            samples_data = np.zeros((num_reads, N), dtype=np.int8)
            for read_idx in range(num_reads):
                for var in range(N):
                    byte_idx = var >> 3
                    bit_idx = var & 7
                    bit = (prob_packed[read_idx, byte_idx] >> bit_idx) & 1
                    samples_data[read_idx, var] = -1 if bit else 1

            # Build SampleSet
            samples_dict = []
            for sample in samples_data:
                samples_dict.append({node: int(sample[idx]) for node, idx in node_to_idx_list[prob_idx].items()})

            sampleset = dimod.SampleSet.from_samples(
                samples_dict,
                energy=prob_energies.astype(float),
                vartype=dimod.SPIN,
                info={
                    "beta_range": beta_range,
                    "beta_schedule_type": beta_schedule_type,
                    "update_mode": self.update_mode_name
                }
            )
            samplesets.append(sampleset)

        return samplesets
