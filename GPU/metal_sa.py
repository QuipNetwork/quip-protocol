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
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import dimod
import Metal
import numpy as np

from shared.ising_model import IsingModel
from GPU.metal_utils import _create_buffer, build_csr_from_ising, compute_beta_schedule, unpack_metal_results


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

        # Cached topology CSR structure (set by prepare_topology)
        self._topo_prepared = False
        self._topo_N = 0
        self._topo_node_to_idx: Dict[int, int] = {}
        self._topo_row_ptr: Optional[np.ndarray] = None
        self._topo_col_ind: Optional[np.ndarray] = None
        # edge_positions[i][j] = position in CSR col_ind for
        # node i's neighbor j. Used for fast J value filling.
        self._topo_edge_pos: Optional[Dict[int, Dict[int, int]]] = None

    def prepare_topology(self) -> None:
        """Precompute topology CSR structure for streaming.

        Builds the graph structure (row_ptr, col_ind, node_to_idx)
        once from the sampler's topology. Only the J and h values
        change per nonce — the structure is invariant.
        """
        if self._topo_prepared:
            return

        all_nodes = set(self.nodes)
        for u, v in self.edges:
            all_nodes.add(u)
            all_nodes.add(v)

        N = len(all_nodes)
        node_list = sorted(all_nodes)
        node_to_idx = {n: i for i, n in enumerate(node_list)}

        # Build adjacency lists (sorted, like build_csr_from_ising)
        adjacency: List[List[int]] = [[] for _ in range(N)]
        for u, v in self.edges:
            idx_u = node_to_idx[u]
            idx_v = node_to_idx[v]
            adjacency[idx_u].append(idx_v)
            adjacency[idx_v].append(idx_u)

        for i in range(N):
            adjacency[i].sort()

        # Build CSR row_ptr and col_ind
        degree = np.array(
            [len(adjacency[i]) for i in range(N)],
            dtype=np.int32,
        )
        csr_row_ptr = np.zeros(N + 1, dtype=np.int32)
        csr_row_ptr[1:] = np.cumsum(degree)

        nnz = int(csr_row_ptr[N])
        csr_col_ind = np.zeros(nnz, dtype=np.int32)

        # Build edge position index for fast J filling
        edge_pos: Dict[int, Dict[int, int]] = {}
        pos = 0
        for i in range(N):
            edge_pos[i] = {}
            for j in adjacency[i]:
                csr_col_ind[pos] = j
                edge_pos[i][j] = pos
                pos += 1

        self._topo_N = N
        self._topo_node_to_idx = node_to_idx
        self._topo_row_ptr = csr_row_ptr
        self._topo_col_ind = csr_col_ind
        self._topo_edge_pos = edge_pos
        self._topo_prepared = True

        self.logger.debug(
            "[MetalSA] Topology prepared: N=%d, nnz=%d",
            N, nnz,
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
            # Build CSR representation for this problem
            csr_row_ptr, csr_col_ind, csr_J_vals, h_vals_array, node_to_idx, N = build_csr_from_ising(
                h_prob, J_prob, use_float=False
            )
            N_list.append(N)
            node_to_idx_list.append(node_to_idx)

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

        # Compute beta schedule (use first problem for auto range)
        beta_schedule, beta_range = compute_beta_schedule(
            h[0], J[0], num_sweeps, num_sweeps_per_beta, beta_range, beta_schedule_type, beta_schedule
        )

        self.logger.debug(f"[MetalSA] Beta schedule: {len(beta_schedule)} betas from {beta_schedule[0]:.4f} to {beta_schedule[-1]:.4f}")

        # RNG seed
        if seed is None:
            seed = np.random.randint(0, 2**31)

        # Create Metal buffers for concatenated CSR arrays
        csr_row_ptr_buf = _create_buffer(self.device, all_csr_row_ptr, "csr_row_ptr")
        csr_col_ind_buf = _create_buffer(self.device, all_csr_col_ind, "csr_col_ind")
        csr_J_vals_buf = _create_buffer(self.device, all_csr_J_vals, "csr_J_vals")
        csr_h_vals_buf = _create_buffer(self.device, all_h_vals, "csr_h_vals")
        row_ptr_offsets_buf = _create_buffer(self.device, row_ptr_offsets, "row_ptr_offsets")
        col_ind_offsets_buf = _create_buffer(self.device, col_ind_offsets, "col_ind_offsets")

        beta_schedule_buf = _create_buffer(self.device, beta_schedule, "beta_schedule")

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

            # Unpack and build SampleSet
            sampleset = unpack_metal_results(
                prob_packed, prob_energies, N, num_reads, node_to_idx_list[prob_idx],
                beta_range, beta_schedule_type
            )
            samplesets.append(sampleset)

            self.logger.debug(f"[MetalSA] Problem {prob_idx}: energy range [{prob_energies.min()}, {prob_energies.max()}]")

        return samplesets

    def _fill_batch_values(
        self,
        models: List[IsingModel],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fill J and h value arrays using cached topology structure.

        Uses the precomputed edge position index to place J values
        directly into the correct CSR positions, avoiding the full
        adjacency rebuild that build_csr_from_ising performs.

        Returns:
            (all_row_ptr, all_col_ind, all_J_vals, all_h_vals,
             row_ptr_offsets, col_ind_offsets) — concatenated for
            all problems in the batch.
        """
        N = self._topo_N
        node_to_idx = self._topo_node_to_idx
        row_ptr = self._topo_row_ptr
        col_ind = self._topo_col_ind
        edge_pos = self._topo_edge_pos
        nnz = len(col_ind)
        rp_len = len(row_ptr)

        num_problems = len(models)

        # Pre-allocate concatenated arrays
        all_row_ptr = np.tile(row_ptr, num_problems)
        all_col_ind = np.tile(col_ind, num_problems)
        all_J_vals = np.zeros(num_problems * nnz, dtype=np.int8)
        all_h_vals = np.zeros(num_problems * N, dtype=np.int8)

        row_ptr_offsets = np.arange(
            0, (num_problems + 1) * rp_len, rp_len,
            dtype=np.int32,
        )
        col_ind_offsets = np.arange(
            0, (num_problems + 1) * nnz, nnz,
            dtype=np.int32,
        )

        for prob_idx, model in enumerate(models):
            j_offset = prob_idx * nnz
            h_offset = prob_idx * N

            # Fill h values
            for node, h_val in model.h.items():
                idx = node_to_idx.get(node)
                if idx is not None:
                    all_h_vals[h_offset + idx] = int(h_val)

            # Fill J values using precomputed positions
            for (u, v), j_val in model.J.items():
                idx_u = node_to_idx.get(u)
                idx_v = node_to_idx.get(v)
                if idx_u is None or idx_v is None:
                    continue
                j_int = int(j_val)
                # Symmetric: fill both (u→v) and (v→u)
                pos_uv = edge_pos[idx_u].get(idx_v)
                if pos_uv is not None:
                    all_J_vals[j_offset + pos_uv] = j_int
                pos_vu = edge_pos[idx_v].get(idx_u)
                if pos_vu is not None:
                    all_J_vals[j_offset + pos_vu] = j_int

        return (
            all_row_ptr, all_col_ind, all_J_vals,
            all_h_vals, row_ptr_offsets, col_ind_offsets,
        )

    def _dispatch_batch(
        self,
        models: List[IsingModel],
        *,
        num_reads: int,
        beta_schedule_arr: np.ndarray,
        beta_range: Tuple[float, float],
        beta_schedule_type: str,
        num_sweeps_per_beta: int,
        seed: int,
    ) -> List[dimod.SampleSet]:
        """Dispatch a batch using cached topology structure.

        Fills only J/h values per batch; reuses the precomputed
        CSR structure from prepare_topology().
        """
        num_problems = len(models)
        N = self._topo_N
        node_to_idx = self._topo_node_to_idx

        (
            all_row_ptr, all_col_ind, all_J_vals,
            all_h_vals, row_ptr_offsets, col_ind_offsets,
        ) = self._fill_batch_values(models)

        # Create Metal buffers
        rp_buf = _create_buffer(self.device, all_row_ptr, "rp")
        ci_buf = _create_buffer(self.device, all_col_ind, "ci")
        jv_buf = _create_buffer(self.device, all_J_vals, "jv")
        hv_buf = _create_buffer(self.device, all_h_vals, "hv")
        rpo_buf = _create_buffer(
            self.device, row_ptr_offsets, "rpo",
        )
        cio_buf = _create_buffer(
            self.device, col_ind_offsets, "cio",
        )
        beta_buf = _create_buffer(
            self.device, beta_schedule_arr, "beta",
        )

        # Scalar parameters
        N_bytes = np.int32(N).tobytes()
        num_betas_bytes = np.int32(
            len(beta_schedule_arr),
        ).tobytes()
        spb_bytes = np.int32(num_sweeps_per_beta).tobytes()
        seed_bytes = np.uint32(seed).tobytes()

        num_threads = num_problems * num_reads
        nt_bytes = np.int32(num_threads).tobytes()
        np_bytes = np.int32(num_problems).tobytes()
        nr_bytes = np.int32(num_reads).tobytes()

        packed_size = (N + 7) // 8

        samples_buf = self.device.newBufferWithLength_options_(
            num_threads * packed_size,
            Metal.MTLResourceStorageModeShared,
        )
        energies_buf = self.device.newBufferWithLength_options_(
            num_threads * 4,
            Metal.MTLResourceStorageModeShared,
        )

        # Encode and dispatch
        cmd_buf = self._command_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipeline)

        encoder.setBuffer_offset_atIndex_(rp_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(ci_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(jv_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(rpo_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(cio_buf, 0, 4)

        encoder.setBytes_length_atIndex_(N_bytes, 4, 5)
        encoder.setBytes_length_atIndex_(num_betas_bytes, 4, 6)
        encoder.setBytes_length_atIndex_(spb_bytes, 4, 7)
        encoder.setBytes_length_atIndex_(seed_bytes, 4, 8)

        encoder.setBuffer_offset_atIndex_(beta_buf, 0, 9)
        encoder.setBuffer_offset_atIndex_(samples_buf, 0, 10)
        encoder.setBuffer_offset_atIndex_(energies_buf, 0, 11)

        encoder.setBytes_length_atIndex_(nt_bytes, 4, 12)
        encoder.setBytes_length_atIndex_(np_bytes, 4, 13)
        encoder.setBytes_length_atIndex_(nr_bytes, 4, 14)

        encoder.setBuffer_offset_atIndex_(hv_buf, 0, 15)

        tg = Metal.MTLSize(width=num_problems, height=1, depth=1)
        tpt = Metal.MTLSize(width=num_reads, height=1, depth=1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(tg, tpt)

        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        if cmd_buf.status() != Metal.MTLCommandBufferStatusCompleted:
            error = cmd_buf.error()
            raise RuntimeError(
                f"Metal command buffer failed: {error}",
            )

        # Unpack results
        packed_data = np.frombuffer(
            samples_buf.contents().as_buffer(
                num_threads * packed_size,
            ),
            dtype=np.int8,
        ).reshape(num_threads, packed_size)

        energies_data = np.frombuffer(
            energies_buf.contents().as_buffer(
                num_threads * 4,
            ),
            dtype=np.int32,
        )

        samplesets = []
        for prob_idx in range(num_problems):
            start = prob_idx * num_reads
            end = start + num_reads
            samplesets.append(
                unpack_metal_results(
                    packed_data[start:end],
                    energies_data[start:end],
                    N, num_reads, node_to_idx,
                    beta_range, beta_schedule_type,
                ),
            )

        return samplesets

    def sample_ising_streaming(
        self,
        models: Iterable[IsingModel],
        *,
        num_reads: int,
        num_sweeps: int,
        max_threadgroups: int,
        num_sweeps_per_beta: int = 1,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
        seed: Optional[int] = None,
        **kwargs,
    ) -> Iterator[Tuple[IsingModel, dimod.SampleSet]]:
        """Stream batched results using cached topology structure.

        On first call, prepares the topology CSR structure once.
        Each batch then only fills J/h values into preallocated
        positions — no adjacency sorting or degree counting.

        Args:
            models: Iterable of IsingModel (typically an IsingFeeder).
            num_reads: SA reads per problem.
            num_sweeps: Total sweeps per run.
            max_threadgroups: Max problems per batch dispatch.
            num_sweeps_per_beta: Sweeps per beta value.
            beta_range: Temperature range or None for auto.
            beta_schedule_type: "linear", "geometric", or "custom".
            seed: Base RNG seed (incremented per batch).

        Yields:
            (IsingModel, dimod.SampleSet) for each completed problem.
        """
        self.prepare_topology()

        model_iter = iter(models)
        batch_seed = (
            seed if seed is not None
            else np.random.randint(0, 2**31)
        )

        # Compute beta schedule once (all problems share the
        # same topology so auto-range is topology-dependent,
        # not nonce-dependent). Use first model for auto range.
        first_model = next(model_iter, None)
        if first_model is None:
            return

        beta_arr, beta_range_out = compute_beta_schedule(
            first_model.h, first_model.J,
            num_sweeps, num_sweeps_per_beta,
            beta_range, beta_schedule_type, None,
        )

        # Put the first model back into the batch
        pending = [first_model]

        while True:
            # Fill batch from pending + iterator
            batch_models: List[IsingModel] = list(pending)
            pending.clear()
            while len(batch_models) < max_threadgroups:
                try:
                    batch_models.append(next(model_iter))
                except StopIteration:
                    break
            if not batch_models:
                return

            samplesets = self._dispatch_batch(
                batch_models,
                num_reads=num_reads,
                beta_schedule_arr=beta_arr,
                beta_range=beta_range_out,
                beta_schedule_type=beta_schedule_type,
                num_sweeps_per_beta=num_sweeps_per_beta,
                seed=batch_seed,
            )

            batch_seed = (batch_seed + 1) & 0x7FFFFFFF

            for model, ss in zip(batch_models, samplesets):
                yield (model, ss)

