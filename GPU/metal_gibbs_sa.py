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
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import dimod
import numpy as np

try:
    import Metal
    import objc  # pyobjc — autorelease pool for per-batch Metal objects
except ImportError:  # Apple Metal framework is macOS-only; absent on Linux/CI.
    Metal = None  # type: ignore[assignment]
    objc = None  # type: ignore[assignment]

from GPU.metal_sa import (
    apply_qos_utility,
    _resolve_budget,
    reads_per_buffer_for_budget,
)
from GPU.metal_scheduler import MetalScheduler
from GPU.metal_utils import (
    build_csr_from_ising,
    compute_beta_schedule,
    pooled_buffer,
    pooled_input,
    unpack_metal_results,
)
from shared.ising_model import IsingModel

# Streaming PAUSE poll interval (seconds): how long to idle before re-reading
# the occupancy budget when throttled to a full stop (battery / critical
# thermal).
_PAUSE_POLL_S = 0.5


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
            function_names = list(lib.functionNames())
            raise RuntimeError(f"Failed to find block_gibbs_sampler kernel. Available: {function_names}")

        self._pipeline, err = self.device.newComputePipelineStateWithFunction_error_(self._kernel, None)
        if err or not self._pipeline:
            raise RuntimeError(f"Failed to create sequential pipeline: {err}")

        # Load parallel kernel
        self._kernel_parallel = lib.newFunctionWithName_("block_gibbs_parallel")
        if not self._kernel_parallel:
            function_names = list(lib.functionNames())
            raise RuntimeError(f"Failed to find block_gibbs_parallel kernel. Available: {function_names}")

        self._pipeline_parallel, err = self.device.newComputePipelineStateWithFunction_error_(self._kernel_parallel, None)
        if err or not self._pipeline_parallel:
            raise RuntimeError(f"Failed to create parallel pipeline: {err}")

        self._command_queue = self.device.newCommandQueue()

        # Per-role MTLBuffer pool (grow-on-demand, reused across batches) — see
        # MetalSASampler for rationale. Recreating the ~13 per-batch buffers
        # every call leaks under a long stream; reusing one per role keeps the
        # footprint bounded and constant.
        self._buf_pool: Dict[str, Any] = {}

    def _pooled_buffer(self, role: str, nbytes: int):
        """Reused shared MTLBuffer of at least ``nbytes`` for ``role`` (see
        ``metal_utils.pooled_buffer``)."""
        return pooled_buffer(self.device, self._buf_pool, role, nbytes)

    def _pooled_input(self, role: str, data: np.ndarray):
        """Pooled buffer for ``role`` filled with ``data`` (see
        ``metal_utils.pooled_input``)."""
        return pooled_input(self.device, self._buf_pool, role, data)

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
    ) -> List[dimod.SampleSet]:
        """Block-Gibbs sample one batch inside an ObjC autorelease pool.

        The per-batch Metal command buffer retains the GPU buffers it
        references and is autoreleased; without a pool drain in a streaming
        loop they accumulate until the process OOMs. Draining a pool per batch
        frees them immediately — safe because ``unpack_metal_results`` copies
        all results out before the pool drains.
        """
        with objc.autorelease_pool():
            return self._sample_ising_impl(
                h, J, num_reads=num_reads, num_sweeps=num_sweeps,
                num_sweeps_per_beta=num_sweeps_per_beta, beta_range=beta_range,
                beta_schedule_type=beta_schedule_type,
                beta_schedule=beta_schedule, seed=seed,
            )

    def _sample_ising_impl(
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
        beta_schedule, beta_range = compute_beta_schedule(
            h[0], J[0], num_sweeps, num_sweeps_per_beta, beta_range, beta_schedule_type, beta_schedule
        )

        self.logger.debug(f"[MetalGibbs] Beta schedule: {len(beta_schedule)} betas from {beta_schedule[0]:.4f} to {beta_schedule[-1]:.4f}")

        # RNG seed
        if seed is None:
            seed = np.random.randint(0, 2**31)

        # Create Metal buffers
        csr_row_ptr_buf = self._pooled_input("csr_row_ptr", all_csr_row_ptr)
        csr_col_ind_buf = self._pooled_input("csr_col_ind", all_csr_col_ind)
        csr_J_vals_buf = self._pooled_input("csr_J_vals", all_csr_J_vals)
        row_ptr_offsets_buf = self._pooled_input("row_ptr_offsets", row_ptr_offsets)
        col_ind_offsets_buf = self._pooled_input("col_ind_offsets", col_ind_offsets)
        csr_h_vals_buf = self._pooled_input("csr_h_vals", all_h_vals)
        beta_schedule_buf = self._pooled_input("beta_schedule", beta_schedule)

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
        color_block_starts_buf = self._pooled_input("color_block_starts", self.block_starts)
        color_block_counts_buf = self._pooled_input("color_block_counts", self.block_counts)
        color_node_indices_buf = self._pooled_input("color_node_indices", csr_color_node_indices)

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

        # Output buffers (pooled — reused across batches, kernel overwrites).
        packed_size = (N + 7) // 8

        final_samples_buf = self._pooled_buffer(
            "final_samples", num_threads * packed_size)
        final_energies_buf = self._pooled_buffer(
            "final_energies", num_threads * 4)

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

            # Unpack and build SampleSet
            sampleset = unpack_metal_results(
                prob_packed, prob_energies, N, num_reads, node_to_idx_list[prob_idx],
                beta_range, beta_schedule_type,
                update_mode=self.update_mode_name
            )
            samplesets.append(sampleset)

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
        scheduler: Optional["MetalScheduler"] = None,
        stop_event=None,
        **kwargs,
    ) -> Iterator[Tuple[IsingModel, dimod.SampleSet]]:
        """Stream batched Gibbs results from a feeder of ``IsingModel``s.

        Mirrors ``MetalSASampler.sample_ising_streaming`` and honours the same
        occupancy budget (max concurrent GPU threads per command buffer) from
        ``scheduler.get_thread_budget()``, re-read **before** each batch:

        - ``0`` → PAUSE: idle ``_PAUSE_POLL_S`` and re-check without pulling
          models. Stop-aware so teardown never blocks.
        - ``UNCAPPED`` → one full-batch dispatch (idle/headless throughput).
        - else → keep the full problem batch and split the reads across
          dispatches so ``problems x reads <= budget`` (each read-buffer an
          independent Gibbs run with a distinct seed, concatenated). Total reads
          preserved.

        Args:
            models: Iterable of ``IsingModel`` (typically a ``RandomIsingFeeder``).
            num_reads: Independent Gibbs runs per problem.
            num_sweeps: Total sweeps per run.
            max_threadgroups: Max problems per batch dispatch.
            num_sweeps_per_beta: Sweeps per beta value.
            beta_range: Temperature range or None for auto.
            beta_schedule_type: "linear", "geometric", or "custom".
            seed: Base RNG seed (incremented per batch).
            scheduler: Optional ``MetalScheduler`` — the per-batch occupancy
                budget source. None ⇒ uncapped (full speed).
            stop_event: Optional ``mp.Event`` — checked during PAUSE so a full
                stop never blocks teardown.

        Yields:
            ``(IsingModel, dimod.SampleSet)`` for each completed problem.
        """
        # CPU-side politeness: lower this (producer) thread's QoS so we yield
        # P-cores to the foreground UI. Matches the SA streaming path.
        apply_qos_utility()

        model_iter = iter(models)
        batch_seed = (
            seed if seed is not None else int(np.random.randint(0, 2**31))
        )

        while True:
            budget = _resolve_budget(scheduler)
            # PAUSE: idle and re-check without consuming the feeder.
            while budget == 0:
                if stop_event is not None and stop_event.is_set():
                    return
                time.sleep(_PAUSE_POLL_S)
                budget = _resolve_budget(scheduler)

            # Full problem batch (occupancy is bounded by splitting reads).
            batch_models: List[IsingModel] = []
            while len(batch_models) < max_threadgroups:
                try:
                    batch_models.append(next(model_iter))
                except StopIteration:
                    break
            if not batch_models:
                return

            samplesets = self._sample_read_split(
                batch_models,
                num_reads=num_reads,
                reads_per_buffer=reads_per_buffer_for_budget(
                    budget, len(batch_models), num_reads,
                ),
                num_sweeps=num_sweeps,
                num_sweeps_per_beta=num_sweeps_per_beta,
                beta_range=beta_range,
                beta_schedule_type=beta_schedule_type,
                seed=batch_seed,
            )
            batch_seed = (batch_seed + 1) & 0x7FFFFFFF

            for model, ss in zip(batch_models, samplesets):
                yield (model, ss)

    def _sample_read_split(
        self,
        models: List[IsingModel],
        *,
        num_reads: int,
        reads_per_buffer: int,
        num_sweeps: int,
        num_sweeps_per_beta: int,
        beta_range: Optional[Tuple[float, float]],
        beta_schedule_type: str,
        seed: int,
    ) -> List[dimod.SampleSet]:
        """Dispatch ``num_reads`` in command buffers of ``reads_per_buffer``.

        Caps occupancy (``problems x reads`` per buffer) while preserving the
        full read count. ``reads_per_buffer >= num_reads`` is a single
        dispatch; otherwise each read-buffer is an independent Gibbs run with a
        distinct seed and the per-problem samplesets are concatenated.
        """
        h = [m.h for m in models]
        j = [m.J for m in models]
        common = dict(
            num_sweeps=num_sweeps, num_sweeps_per_beta=num_sweeps_per_beta,
            beta_range=beta_range, beta_schedule_type=beta_schedule_type,
        )
        if reads_per_buffer >= num_reads:
            return self.sample_ising(h=h, J=j, num_reads=num_reads,
                                     seed=seed, **common)

        parts: List[List[dimod.SampleSet]] = [[] for _ in models]
        done = 0
        buf_idx = 0
        while done < num_reads:
            rc = min(reads_per_buffer, num_reads - done)
            buf_seed = (seed + buf_idx * 0x9E3779B1) & 0x7FFFFFFF
            ss_list = self.sample_ising(h=h, J=j, num_reads=rc,
                                        seed=buf_seed, **common)
            for prob_idx, ss in enumerate(ss_list):
                parts[prob_idx].append(ss)
            done += rc
            buf_idx += 1
        return [dimod.concatenate(p) for p in parts]
