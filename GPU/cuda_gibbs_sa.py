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
    build_csr_structure_from_edges,
    build_edge_position_index,
    compute_beta_schedule,
    compute_color_blocks,
    default_ising_beta_range,
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

    GIBBS_NUM_REGIONS = 12

    def __init__(
        self,
        topology=None,
        update_mode: str = "gibbs",
        parallel: bool = True,
        max_sms: int = 0,
        profile: bool = False,
    ):
        """Initialize CUDA Gibbs sampler.

        Args:
            topology: Topology object (default: DEFAULT_TOPOLOGY).
            update_mode: "gibbs" or "metropolis".
            parallel: Use chromatic parallel kernel (True) or
                fully sequential kernel (False).
            max_sms: Maximum SMs to use (0 = all available).
            profile: Compile with PROFILE_REGIONS for clock64()
                instrumentation.
        """
        self.profile = profile
        self.logger = logging.getLogger(__name__)

        if update_mode.lower() not in ("gibbs", "metropolis"):
            raise ValueError(
                f"update_mode must be 'gibbs' or 'metropolis', "
                f"got {update_mode}"
            )
        self.update_mode = 0 if update_mode.lower() == "gibbs" else 1
        self.update_mode_name = update_mode.lower()
        self.parallel = parallel
        self.max_sms = max_sms
        self.sampler_type = "cuda-gibbs"

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

        compile_options = ['--use_fast_math']
        if self.profile:
            compile_options.append('-DPROFILE_REGIONS=1')
        self._module = cp.RawModule(
            code=kernel_code,
            options=tuple(compile_options),
        )
        self._persistent_kernel = self._module.get_function(
            'cuda_gibbs_persistent'
        )
        self._sequential_kernel = self._module.get_function(
            'cuda_block_gibbs_sequential'
        )
        self._self_feeding_kernel = self._module.get_function(
            'cuda_gibbs_self_feeding'
        )

        self._prepared = False
        self._sf_prepared = False

    def prepare(
        self,
        num_reads: int = 256,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        max_nonces: int = 1,
    ) -> None:
        """Pre-allocate GPU buffers for a fixed topology.

        Call once before the mining loop. Subsequent single-problem
        sample_ising() calls skip CSR rebuild, color block
        computation, and buffer allocation.

        Args:
            num_reads: Max reads per job.
            num_sweeps: Max sweeps (determines beta schedule size).
            num_sweeps_per_beta: Sweeps per beta value.
            max_nonces: Max nonces for multi-nonce dispatch.
        """
        # Build CSR structure from topology
        (csr_row_ptr, csr_col_ind, node_to_idx,
         sorted_neighbors, N, nnz) = (
            build_csr_structure_from_edges(self.edges, self.nodes)
        )

        # Edge position index for fast J updates
        edge_positions = build_edge_position_index(
            self.edges, node_to_idx, csr_row_ptr,
            sorted_neighbors,
        )

        self._prep_N = N
        self._prep_nnz = nnz
        self._prep_node_to_idx = node_to_idx
        self._prep_edge_positions = edge_positions
        self._prep_num_reads = num_reads

        # Vectorized index arrays for fast staging fill
        self._pos_ij = np.array(
            [p[0] for p in edge_positions], dtype=np.int32,
        )
        self._pos_ji = np.array(
            [p[1] for p in edge_positions], dtype=np.int32,
        )
        self._h_idx = np.array(
            [node_to_idx[n] for n in sorted(node_to_idx)],
            dtype=np.int32,
        )
        self._prep_max_num_betas = num_sweeps // num_sweeps_per_beta

        max_packed_size = (N + 7) // 8
        self._prep_max_packed_size = max_packed_size

        # Build color blocks (topology-constant, computed once)
        prob_nodes = sorted(node_to_idx.keys())
        starts, counts, color_indices = compute_color_blocks(
            prob_nodes, self.m, self.t
        )
        # Remap to dense CSR indices
        remapped = np.array(
            [node_to_idx[n] for n in color_indices],
            dtype=np.int32,
        )

        # Single-problem metadata arrays
        problem_N = np.array([N], dtype=np.int32)
        problem_rp = np.array([0], dtype=np.int32)
        problem_ci = np.array([0], dtype=np.int32)
        problem_j = np.array([0], dtype=np.int32)
        problem_h = np.array([0], dtype=np.int32)

        # Upload constant GPU buffers (one-time)
        self._d_row_ptr = cp.asarray(csr_row_ptr)
        self._d_col_ind = cp.asarray(csr_col_ind)
        self._d_block_starts = cp.asarray(starts)
        self._d_block_counts = cp.asarray(counts)
        self._d_color_nodes = cp.asarray(remapped)
        self._d_problem_N = cp.asarray(problem_N)
        self._d_problem_rp = cp.asarray(problem_rp)
        self._d_problem_ci = cp.asarray(problem_ci)
        self._d_problem_j = cp.asarray(problem_j)
        self._d_problem_h = cp.asarray(problem_h)

        # Double-buffered mutable GPU arrays (A/B sets)
        max_betas = self._prep_max_num_betas
        total_samples = num_reads
        self._d_J_vals = [
            cp.zeros(nnz, dtype=cp.int8),
            cp.zeros(nnz, dtype=cp.int8),
        ]
        self._d_h_vals = [
            cp.zeros(N, dtype=cp.int8),
            cp.zeros(N, dtype=cp.int8),
        ]
        self._d_beta_sched = [
            cp.zeros(max_betas, dtype=cp.float32),
            cp.zeros(max_betas, dtype=cp.float32),
        ]
        self._d_final_samples = [
            cp.zeros(
                total_samples * max_packed_size, dtype=cp.int8,
            ),
            cp.zeros(
                total_samples * max_packed_size, dtype=cp.int8,
            ),
        ]
        self._d_final_energies = [
            cp.zeros(total_samples, dtype=cp.int32),
            cp.zeros(total_samples, dtype=cp.int32),
        ]
        self._d_queue_counter = [
            cp.zeros(1, dtype=cp.int32),
            cp.zeros(1, dtype=cp.int32),
        ]

        # Host staging buffers
        self._h_J_vals = np.zeros(nnz, dtype=np.int8)
        self._h_h_vals = np.zeros(N, dtype=np.int8)

        # Multi-nonce double-buffered GPU arrays (A/B)
        # Buffer layout per set: concatenated per-nonce J/h,
        # output samples/energies, atomic work queue counter.
        self._max_nonces = max_nonces
        if max_nonces > 1:
            mn_total_reads = max_nonces * num_reads
            self._d_mn_J = [
                cp.zeros(max_nonces * nnz, dtype=cp.int8),
                cp.zeros(max_nonces * nnz, dtype=cp.int8),
            ]
            self._d_mn_h = [
                cp.zeros(max_nonces * N, dtype=cp.int8),
                cp.zeros(max_nonces * N, dtype=cp.int8),
            ]
            self._d_mn_samples = [
                cp.zeros(
                    mn_total_reads * max_packed_size,
                    dtype=cp.int8,
                ),
                cp.zeros(
                    mn_total_reads * max_packed_size,
                    dtype=cp.int8,
                ),
            ]
            self._d_mn_energies = [
                cp.zeros(mn_total_reads, dtype=cp.int32),
                cp.zeros(mn_total_reads, dtype=cp.int32),
            ]
            self._d_mn_queue = [
                cp.zeros(1, dtype=cp.int32),
                cp.zeros(1, dtype=cp.int32),
            ]
            self._d_mn_beta = [
                cp.zeros(
                    self._prep_max_num_betas,
                    dtype=cp.float32,
                ),
                cp.zeros(
                    self._prep_max_num_betas,
                    dtype=cp.float32,
                ),
            ]
            # Per-nonce metadata: shared topology offsets
            self._d_mn_problem_N = cp.full(
                max_nonces, N, dtype=cp.int32,
            )
            self._d_mn_problem_rp = cp.zeros(
                max_nonces, dtype=cp.int32,
            )
            self._d_mn_problem_ci = cp.zeros(
                max_nonces, dtype=cp.int32,
            )
            self._d_mn_problem_j = cp.asarray(
                np.arange(max_nonces, dtype=np.int32) * nnz,
            )
            self._d_mn_problem_h = cp.asarray(
                np.arange(max_nonces, dtype=np.int32) * N,
            )
            # All nonces share the same topology → same color
            # blocks. Tile so kernel can index
            # [model_id * num_colors + c].
            self._d_mn_block_starts = cp.asarray(
                np.tile(starts, max_nonces),
            )
            self._d_mn_block_counts = cp.asarray(
                np.tile(counts, max_nonces),
            )

            # Double host staging buffers (A/B)
            self._h_mn_J = [
                np.zeros(max_nonces * nnz, dtype=np.int8),
                np.zeros(max_nonces * nnz, dtype=np.int8),
            ]
            self._h_mn_h = [
                np.zeros(max_nonces * N, dtype=np.int8),
                np.zeros(max_nonces * N, dtype=np.int8),
            ]

        # CUDA streams for pipeline overlap
        self._stream_compute = cp.cuda.Stream(non_blocking=True)
        self._stream_transfer = cp.cuda.Stream(non_blocking=True)
        self._event_transfer_done = cp.cuda.Event()

        # Pipeline state — single-nonce double buffer
        self._buf_idx = 0
        self._preloaded = False
        self._preload_meta = None

        # Pipeline state — multi-nonce double buffer
        self._mn_buf_idx = 0
        self._mn_preloaded = False
        self._mn_preload_meta = None
        self._mn_pending = None

        # Cache beta schedule (topology-invariant beta range)
        self._cached_beta_range = None
        self._cached_beta_sched = None
        self._cached_beta_key = None

        self._prepared = True
        self.logger.info(
            f"Prepared Gibbs buffers: N={N}, nnz={nnz}, "
            f"num_reads={num_reads}, "
            f"max_betas={max_betas}, "
            f"max_nonces={max_nonces}"
        )

    def close(self) -> None:
        """Synchronize and release CUDA streams/events."""
        if not self._prepared:
            return
        self._stream_compute.synchronize()
        self._stream_transfer.synchronize()
        self._mn_preloaded = False
        self._preloaded = False
        self._prepared = False

    def _get_cached_beta_schedule(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        num_sweeps: int,
        num_sweeps_per_beta: int,
        beta_range: Optional[Tuple[float, float]],
        beta_schedule_type: str,
        beta_schedule: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Return cached beta schedule if params match."""
        key = (
            num_sweeps, num_sweeps_per_beta,
            beta_schedule_type, beta_range,
        )
        if (self._cached_beta_key == key
                and beta_schedule is None):
            return self._cached_beta_sched, self._cached_beta_range

        sched, br = compute_beta_schedule(
            h, J, num_sweeps, num_sweeps_per_beta,
            beta_range, beta_schedule_type, beta_schedule,
        )
        if beta_schedule is None:
            self._cached_beta_key = key
            self._cached_beta_sched = sched
            self._cached_beta_range = br
        return sched, br

    def preload(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        num_reads: int,
        num_sweeps: int,
        num_sweeps_per_beta: int = 1,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
        beta_schedule: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Preload next job's data asynchronously."""
        assert self._prepared, "Must call prepare() first"
        next_idx = 1 - self._buf_idx
        num_reads = min(num_reads, self._prep_num_reads)

        # Vectorized staging fill
        j_vals = np.fromiter(
            J.values(), dtype=np.int8, count=len(J),
        )
        self._h_J_vals[:] = 0
        self._h_J_vals[self._pos_ij] = j_vals
        self._h_J_vals[self._pos_ji] = j_vals

        h_vals = np.fromiter(
            h.values(), dtype=np.int8, count=len(h),
        )
        self._h_h_vals[:] = 0
        self._h_h_vals[self._h_idx] = h_vals

        sched, beta_range = self._get_cached_beta_schedule(
            h, J, num_sweeps, num_sweeps_per_beta,
            beta_range, beta_schedule_type, beta_schedule,
        )
        num_betas = len(sched)

        if seed is None:
            seed = np.random.randint(0, 2**31)

        # Async H2D on transfer stream
        with self._stream_transfer:
            self._d_J_vals[next_idx].set(self._h_J_vals)
            self._d_h_vals[next_idx].set(self._h_h_vals)
            self._d_beta_sched[next_idx][:num_betas].set(sched)
        self._event_transfer_done.record(self._stream_transfer)

        self._preloaded = True
        self._preload_meta = (
            num_reads, num_betas, num_sweeps_per_beta,
            seed, beta_range, beta_schedule_type,
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
        """Sample from Ising model using block Gibbs sampling."""
        num_problems = len(h)
        assert len(J) == num_problems, (
            f"h and J must have same length: "
            f"{num_problems} vs {len(J)}"
        )

        # Fast path: single-problem with matching topology
        if (self._prepared and num_problems == 1
                and len(J[0]) == len(self._prep_edge_positions)):
            return self._sample_prepared(
                h[0], J[0], num_reads, num_sweeps,
                num_sweeps_per_beta, beta_range,
                beta_schedule_type, beta_schedule, seed,
            )

        # Fresh path: build everything from scratch
        return self._sample_fresh(
            h, J, num_reads, num_sweeps,
            num_sweeps_per_beta, beta_range,
            beta_schedule_type, beta_schedule, seed,
        )

    def _sample_prepared(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        num_reads: int,
        num_sweeps: int,
        num_sweeps_per_beta: int,
        beta_range: Optional[Tuple[float, float]],
        beta_schedule_type: str,
        beta_schedule: Optional[np.ndarray],
        seed: Optional[int],
    ) -> List[dimod.SampleSet]:
        """Fast path using pre-allocated double buffers."""
        N = self._prep_N
        node_to_idx = self._prep_node_to_idx
        max_packed_size = self._prep_max_packed_size
        idx = self._buf_idx

        if self._preloaded:
            next_idx = 1 - idx
            self._stream_compute.wait_event(
                self._event_transfer_done,
            )
            idx = next_idx
            self._buf_idx = next_idx
            (num_reads, num_betas, num_sweeps_per_beta,
             seed, beta_range, beta_schedule_type) = (
                self._preload_meta
            )
            self._preloaded = False
        else:
            num_reads = min(num_reads, self._prep_num_reads)

            j_vals = np.fromiter(
                J.values(), dtype=np.int8, count=len(J),
            )
            self._h_J_vals[:] = 0
            self._h_J_vals[self._pos_ij] = j_vals
            self._h_J_vals[self._pos_ji] = j_vals

            h_vals = np.fromiter(
                h.values(), dtype=np.int8, count=len(h),
            )
            self._h_h_vals[:] = 0
            self._h_h_vals[self._h_idx] = h_vals

            sched, beta_range = self._get_cached_beta_schedule(
                h, J, num_sweeps, num_sweeps_per_beta,
                beta_range, beta_schedule_type, beta_schedule,
            )
            num_betas = len(sched)

            if seed is None:
                seed = np.random.randint(0, 2**31)

            with self._stream_compute:
                self._d_J_vals[idx].set(self._h_J_vals)
                self._d_h_vals[idx].set(self._h_h_vals)
                self._d_beta_sched[idx][:num_betas].set(sched)

        # Zero output buffers
        total_samples = num_reads
        with self._stream_compute:
            self._d_final_samples[idx][
                :total_samples * max_packed_size
            ] = 0
            self._d_final_energies[idx][:total_samples] = 0
            self._d_queue_counter[idx][:] = 0

        # Launch persistent kernel on compute stream
        dev = cp.cuda.Device()
        num_sms = dev.attributes['MultiProcessorCount']
        num_blocks = num_sms
        if self.max_sms > 0:
            num_blocks = min(num_blocks, self.max_sms)

        chunks_per_model = num_blocks
        reads_per_chunk = (
            (num_reads + chunks_per_model - 1)
            // chunks_per_model
        )
        total_work_units = chunks_per_model

        grid = (num_blocks,)
        block = (256,)
        kernel_args = (
            self._d_row_ptr, self._d_col_ind,
            self._d_J_vals[idx], self._d_h_vals[idx],
            self._d_problem_N, self._d_problem_rp,
            self._d_problem_ci, self._d_problem_j,
            self._d_problem_h,
            self._d_block_starts, self._d_block_counts,
            self._d_color_nodes,
            np.int32(self.num_colors),
            self._d_beta_sched[idx],
            np.int32(num_betas),
            np.int32(num_sweeps_per_beta),
            self._d_final_samples[idx],
            self._d_final_energies[idx],
            np.int32(num_reads),
            np.int32(N),
            np.int32(max_packed_size),
            np.int32(1),  # num_problems=1
            self._d_queue_counter[idx],
            np.int32(chunks_per_model),
            np.int32(reads_per_chunk),
            np.int32(total_work_units),
            np.uint32(seed),
            np.int32(self.update_mode),
        )
        with self._stream_compute:
            self._persistent_kernel(
                grid, block, kernel_args,
            )
        self._stream_compute.synchronize()

        # Read results
        packed_raw = cp.asnumpy(
            self._d_final_samples[idx][
                :total_samples * max_packed_size
            ]
        )
        energies_raw = cp.asnumpy(
            self._d_final_energies[idx][:total_samples]
        )

        packed_data = packed_raw.reshape(
            total_samples, max_packed_size,
        )

        self.logger.debug(
            f"[CudaGibbs] Problem 0: energy range "
            f"[{energies_raw.min()}, {energies_raw.max()}]"
        )

        return unpack_packed_results(
            packed_data, energies_raw,
            1, num_reads, N,
            [node_to_idx],
            info={
                "beta_range": beta_range,
                "beta_schedule_type": beta_schedule_type,
                "update_mode": self.update_mode_name,
            },
        )

    def _fill_mn_staging(
        self,
        h_list: List[Dict[int, float]],
        J_list: List[Dict[Tuple[int, int], float]],
        buf: int,
    ) -> None:
        """Pack h/J into host staging buffer `buf` (0 or 1)."""
        N = self._prep_N
        nnz = self._prep_nnz
        num_nonces = len(h_list)

        h_mn_J = self._h_mn_J[buf]
        h_mn_h = self._h_mn_h[buf]
        h_mn_J[:num_nonces * nnz] = 0
        h_mn_h[:num_nonces * N] = 0

        for k in range(num_nonces):
            j_vals = np.fromiter(
                J_list[k].values(), dtype=np.int8,
                count=len(J_list[k]),
            )
            j_off = k * nnz
            h_mn_J[j_off + self._pos_ij] = j_vals
            h_mn_J[j_off + self._pos_ji] = j_vals

            h_vals = np.fromiter(
                h_list[k].values(), dtype=np.int8,
                count=len(h_list[k]),
            )
            h_off = k * N
            h_mn_h[h_off + self._h_idx] = h_vals

    def preload_multi_nonce(
        self,
        h_list: List[Dict[int, float]],
        J_list: List[Dict[Tuple[int, int], float]],
        reads_per_nonce: int = 32,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        sms_per_nonce: int = 4,
        seed: Optional[int] = None,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
        beta_schedule: Optional[np.ndarray] = None,
    ) -> None:
        """Async H2D copy of next multi-nonce batch.

        Packs h/J into host staging, then DMA-copies to the
        alternate GPU buffer via transfer stream. The next
        sample_multi_nonce() call will use this data without
        blocking on the transfer.
        """
        assert self._prepared, "Must call prepare() first"
        num_nonces = len(h_list)
        assert num_nonces <= self._max_nonces

        # Target the NEXT buffer (alternate from current)
        next_buf = 1 - self._mn_buf_idx

        # Fill host staging buffer
        self._fill_mn_staging(h_list, J_list, next_buf)

        N = self._prep_N
        nnz = self._prep_nnz

        # Beta schedule
        sched, beta_range = self._get_cached_beta_schedule(
            h_list[0], J_list[0], num_sweeps,
            num_sweeps_per_beta, beta_range,
            beta_schedule_type, beta_schedule,
        )
        num_betas = len(sched)

        if seed is None:
            seed = np.random.randint(0, 2**31)

        # Async H2D on transfer stream
        with self._stream_transfer:
            self._d_mn_J[next_buf][
                :num_nonces * nnz
            ].set(
                self._h_mn_J[next_buf][:num_nonces * nnz],
            )
            self._d_mn_h[next_buf][
                :num_nonces * N
            ].set(
                self._h_mn_h[next_buf][:num_nonces * N],
            )
            self._d_mn_beta[next_buf][:num_betas].set(sched)
            self._event_transfer_done.record(
                self._stream_transfer,
            )

        self._mn_preloaded = True
        self._mn_preload_meta = {
            'num_nonces': num_nonces,
            'reads_per_nonce': reads_per_nonce,
            'num_sweeps_per_beta': num_sweeps_per_beta,
            'sms_per_nonce': sms_per_nonce,
            'num_betas': num_betas,
            'seed': seed,
            'beta_range': beta_range,
            'beta_schedule_type': beta_schedule_type,
            'buf': next_buf,
        }

    def launch_multi_nonce(
        self,
        h_list: List[Dict[int, float]],
        J_list: List[Dict[Tuple[int, int], float]],
        reads_per_nonce: int = 32,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        sms_per_nonce: int = 4,
        seed: Optional[int] = None,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
        beta_schedule: Optional[np.ndarray] = None,
    ) -> None:
        """Launch Gibbs kernel without synchronizing.

        Call harvest_multi_nonce() to sync and get results.
        Do CPU work between launch and harvest for overlap.
        """
        assert self._prepared, "Must call prepare() first"
        num_nonces = len(h_list)
        assert num_nonces == len(J_list)
        assert num_nonces <= self._max_nonces, (
            f"num_nonces={num_nonces} > max_nonces="
            f"{self._max_nonces}"
        )

        N = self._prep_N
        nnz = self._prep_nnz
        max_packed_size = self._prep_max_packed_size
        reads_per_nonce = min(
            reads_per_nonce, self._prep_num_reads,
        )

        if self._mn_preloaded:
            meta = self._mn_preload_meta
            idx = meta['buf']
            num_nonces = meta['num_nonces']
            num_betas = meta['num_betas']
            num_sweeps_per_beta = meta['num_sweeps_per_beta']
            sms_per_nonce = meta['sms_per_nonce']
            reads_per_nonce = min(
                meta['reads_per_nonce'],
                self._prep_num_reads,
            )
            seed = meta['seed']
            beta_range = meta['beta_range']
            beta_schedule_type = meta['beta_schedule_type']

            self._stream_compute.wait_event(
                self._event_transfer_done,
            )
            self._mn_buf_idx = idx
            self._mn_preloaded = False
            self._mn_preload_meta = None
        else:
            idx = self._mn_buf_idx

            sched, beta_range = (
                self._get_cached_beta_schedule(
                    h_list[0], J_list[0], num_sweeps,
                    num_sweeps_per_beta, beta_range,
                    beta_schedule_type, beta_schedule,
                )
            )
            num_betas = len(sched)

            if seed is None:
                seed = np.random.randint(0, 2**31)

            self._fill_mn_staging(h_list, J_list, idx)

            with self._stream_compute:
                self._d_mn_J[idx][
                    :num_nonces * nnz
                ].set(
                    self._h_mn_J[idx][
                        :num_nonces * nnz
                    ],
                )
                self._d_mn_h[idx][
                    :num_nonces * N
                ].set(
                    self._h_mn_h[idx][
                        :num_nonces * N
                    ],
                )
                self._d_mn_beta[idx][:num_betas].set(
                    sched,
                )

        # Launch kernel (no sync)
        chunks_per_model = sms_per_nonce
        reads_per_chunk = (
            (reads_per_nonce + chunks_per_model - 1)
            // chunks_per_model
        )
        total_work_units = num_nonces * chunks_per_model
        num_blocks = (
            min(total_work_units, self.max_sms)
            if self.max_sms > 0
            else total_work_units
        )
        total_samples = num_nonces * reads_per_nonce

        with self._stream_compute:
            self._d_mn_samples[idx][
                :total_samples * max_packed_size
            ] = 0
            self._d_mn_energies[idx][:total_samples] = 0
            self._d_mn_queue[idx][:] = 0

        grid = (num_blocks,)
        block = (256,)
        kernel_args = (
            self._d_row_ptr, self._d_col_ind,
            self._d_mn_J[idx], self._d_mn_h[idx],
            self._d_mn_problem_N[:num_nonces],
            self._d_mn_problem_rp[:num_nonces],
            self._d_mn_problem_ci[:num_nonces],
            self._d_mn_problem_j[:num_nonces],
            self._d_mn_problem_h[:num_nonces],
            self._d_mn_block_starts[
                :num_nonces * self.num_colors
            ],
            self._d_mn_block_counts[
                :num_nonces * self.num_colors
            ],
            self._d_color_nodes,
            np.int32(self.num_colors),
            self._d_mn_beta[idx],
            np.int32(num_betas),
            np.int32(num_sweeps_per_beta),
            self._d_mn_samples[idx],
            self._d_mn_energies[idx],
            np.int32(reads_per_nonce),
            np.int32(N),
            np.int32(max_packed_size),
            np.int32(num_nonces),
            self._d_mn_queue[idx],
            np.int32(chunks_per_model),
            np.int32(reads_per_chunk),
            np.int32(total_work_units),
            np.uint32(seed),
            np.int32(self.update_mode),
        )
        with self._stream_compute:
            self._persistent_kernel(grid, block, kernel_args)

        # Store metadata for harvest (no sync yet)
        self._mn_pending = {
            'idx': idx,
            'num_nonces': num_nonces,
            'reads_per_nonce': reads_per_nonce,
            'beta_range': beta_range,
            'beta_schedule_type': beta_schedule_type,
            'total_samples': total_samples,
        }

    def harvest_sync(self) -> dict:
        """Synchronize GPU and return pending metadata.

        Waits for the compute stream to finish but does NOT
        download results from VRAM. Call download_results()
        with the returned dict to get SampleSets.

        Returns:
            Pending metadata dict for download_results().
        """
        assert self._mn_pending is not None, (
            "No pending launch — call launch_multi_nonce() first"
        )
        pending = self._mn_pending
        self._mn_pending = None
        self._stream_compute.synchronize()
        return pending

    def download_results(
        self, pending: dict,
    ) -> List[dimod.SampleSet]:
        """Download and unpack results from a completed kernel.

        Args:
            pending: Metadata dict from harvest_sync().

        Returns:
            List of dimod.SampleSet, one per nonce.
        """
        N = self._prep_N
        node_to_idx = self._prep_node_to_idx
        max_packed_size = self._prep_max_packed_size

        idx = pending['idx']
        num_nonces = pending['num_nonces']
        reads_per_nonce = pending['reads_per_nonce']
        beta_range = pending['beta_range']
        beta_schedule_type = pending['beta_schedule_type']
        total_samples = pending['total_samples']

        packed_raw = cp.asnumpy(
            self._d_mn_samples[idx][
                :total_samples * max_packed_size
            ],
        )
        energies_raw = cp.asnumpy(
            self._d_mn_energies[idx][:total_samples],
        )

        packed_data = packed_raw.reshape(
            total_samples, max_packed_size,
        )

        return unpack_packed_results(
            packed_data, energies_raw,
            num_nonces, reads_per_nonce, N,
            [node_to_idx] * num_nonces,
            info={
                "beta_range": beta_range,
                "beta_schedule_type": beta_schedule_type,
                "update_mode": self.update_mode_name,
            },
        )

    def harvest_multi_nonce(self) -> List[dimod.SampleSet]:
        """Synchronize GPU and return results from last launch.

        Convenience wrapper: harvest_sync() + download_results().

        Returns:
            List of dimod.SampleSet, one per nonce.
        """
        pending = self.harvest_sync()
        return self.download_results(pending)

    def sample_multi_nonce(
        self,
        h_list: List[Dict[int, float]],
        J_list: List[Dict[Tuple[int, int], float]],
        reads_per_nonce: int = 32,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        sms_per_nonce: int = 4,
        seed: Optional[int] = None,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
        beta_schedule: Optional[np.ndarray] = None,
    ) -> List[dimod.SampleSet]:
        """Launch + harvest (blocking convenience wrapper)."""
        self.launch_multi_nonce(
            h_list, J_list,
            reads_per_nonce=reads_per_nonce,
            num_sweeps=num_sweeps,
            num_sweeps_per_beta=num_sweeps_per_beta,
            sms_per_nonce=sms_per_nonce,
            seed=seed,
            beta_range=beta_range,
            beta_schedule_type=beta_schedule_type,
            beta_schedule=beta_schedule,
        )
        return self.harvest_multi_nonce()

    def _sample_fresh(
        self,
        h: List[Dict[int, float]],
        J: List[Dict[Tuple[int, int], float]],
        num_reads: int,
        num_sweeps: int,
        num_sweeps_per_beta: int,
        beta_range: Optional[Tuple[float, float]],
        beta_schedule_type: str,
        beta_schedule: Optional[np.ndarray],
        seed: Optional[int],
    ) -> List[dimod.SampleSet]:
        """Original path: build CSR and allocate fresh each call."""
        num_problems = len(h)

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

        # j_offsets = ci_offsets for fresh path (concatenated CSR)
        d_problem_N = cp.asarray(problem_N)
        d_problem_rp = cp.asarray(problem_rp_offsets)
        d_problem_ci = cp.asarray(problem_ci_offsets)
        d_problem_j = cp.asarray(problem_ci_offsets)
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
            dev = cp.cuda.Device()
            num_sms = dev.attributes['MultiProcessorCount']
            num_blocks = num_sms
            if self.max_sms > 0:
                num_blocks = min(num_blocks, self.max_sms)

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

            d_queue_counter = cp.zeros(1, dtype=cp.int32)

            self.logger.debug(
                f"[CudaGibbs] persistent: "
                f"{num_blocks} blocks, "
                f"{chunks_per_model} chunks/model, "
                f"{reads_per_chunk} reads/chunk, "
                f"{total_work_units} total units"
            )

            if self.profile:
                d_profile = cp.zeros(
                    total_work_units * self.GIBBS_NUM_REGIONS,
                    dtype=cp.int64,
                )
                self._profile_work_units = total_work_units

            grid = (num_blocks,)
            block = (256,)
            kernel_args = (
                d_row_ptr, d_col_ind,
                d_J_vals, d_h_vals,
                d_problem_N, d_problem_rp,
                d_problem_ci, d_problem_j,
                d_problem_h,
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
            )
            if self.profile:
                kernel_args = kernel_args + (d_profile,)
                self._d_profile = d_profile
            self._persistent_kernel(
                grid, block, kernel_args,
            )
        else:
            d_state = cp.zeros(
                total_samples * max_N, dtype=cp.int8
            )
            seq_args = (
                d_row_ptr, d_col_ind, d_J_vals, d_h_vals,
                d_problem_N, d_problem_rp,
                d_problem_ci, d_problem_j,
                d_problem_h,
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
        """Build flattened color block arrays for all problems."""
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

    # ==============================================================
    # Self-feeding kernel: 3-slot rotating buffers per nonce
    # ==============================================================
    # Control layout per nonce (CTRL_STRIDE=8 ints):
    #   [0..2] slot_state  [3] active_slot  [4] blocks_done
    #   [5] work_queue  [6] exit_now  [7] generation
    CTRL_STRIDE = 8
    SLOT_EMPTY = 0
    SLOT_READY = 1
    SLOT_ACTIVE = 2
    SLOT_COMPLETE = 3

    def prepare_self_feeding(
        self,
        num_nonces: int,
        reads_per_nonce: int = 32,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        sms_per_nonce: int = 4,
    ) -> None:
        """Allocate 3-slot rotating buffers for self-feeding kernel.

        Args:
            num_nonces: Number of concurrent nonce groups.
            reads_per_nonce: Reads per nonce per model.
            num_sweeps: Max sweeps (for beta schedule size).
            num_sweeps_per_beta: Sweeps per beta value.
            sms_per_nonce: CUDA blocks per nonce group.
        """
        assert self._prepared, "Must call prepare() first"

        N = self._prep_N
        nnz = self._prep_nnz
        max_packed_size = self._prep_max_packed_size
        total_slots = num_nonces * 3

        self._sf_num_nonces = num_nonces
        self._sf_reads_per_nonce = reads_per_nonce
        self._sf_sms_per_nonce = sms_per_nonce
        self._sf_num_sweeps_per_beta = num_sweeps_per_beta
        self._sf_max_num_betas = num_sweeps // num_sweeps_per_beta

        # Flat buffer allocations (Option A: stride math)
        self._d_sf_J = cp.zeros(
            total_slots * nnz, dtype=cp.int8,
        )
        self._d_sf_h = cp.zeros(
            total_slots * N, dtype=cp.int8,
        )
        self._d_sf_samples = cp.zeros(
            total_slots * reads_per_nonce * max_packed_size,
            dtype=cp.int8,
        )
        self._d_sf_energies = cp.zeros(
            total_slots * reads_per_nonce, dtype=cp.int32,
        )

        # Per-nonce control array (device memory)
        self._d_sf_ctrl = cp.zeros(
            num_nonces * self.CTRL_STRIDE, dtype=cp.int32,
        )

        # Host staging buffers (one per slot for async fill)
        self._h_sf_J = np.zeros(nnz, dtype=np.int8)
        self._h_sf_h = np.zeros(N, dtype=np.int8)

        # Shared beta schedule buffer
        self._d_sf_beta = cp.zeros(
            self._sf_max_num_betas, dtype=cp.float32,
        )

        # Color blocks tiled for num_nonces
        starts = cp.asnumpy(self._d_block_starts)
        counts = cp.asnumpy(self._d_block_counts)
        self._d_sf_block_starts = cp.asarray(
            np.tile(starts, num_nonces),
        )
        self._d_sf_block_counts = cp.asarray(
            np.tile(counts, num_nonces),
        )

        # Dedicated streams
        self._sf_stream_compute = cp.cuda.Stream(
            non_blocking=True,
        )
        self._sf_stream_transfer = cp.cuda.Stream(
            non_blocking=True,
        )

        # Chunks per model (work distribution)
        self._sf_chunks_per_model = sms_per_nonce
        self._sf_reads_per_chunk = (
            (reads_per_nonce + sms_per_nonce - 1)
            // sms_per_nonce
        )

        self._sf_kernel_running = False
        self._sf_prepared = True

        self.logger.info(
            "Self-feeding prepared: %d nonces × 3 slots, "
            "%d reads/nonce, %d SMs/nonce",
            num_nonces, reads_per_nonce, sms_per_nonce,
        )

    def upload_slot(
        self,
        nonce_id: int,
        slot_id: int,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
    ) -> None:
        """Upload model data to a specific slot, mark READY.

        Performs async H2D copy on the transfer stream, then
        writes SLOT_READY to the control array.

        Args:
            nonce_id: Nonce group index.
            slot_id: Slot within nonce (0, 1, or 2).
            h: Linear biases.
            J: Quadratic biases.
        """
        assert self._sf_prepared, (
            "Must call prepare_self_feeding() first"
        )
        N = self._prep_N
        nnz = self._prep_nnz
        max_packed_size = self._prep_max_packed_size
        reads = self._sf_reads_per_nonce
        slot_idx = nonce_id * 3 + slot_id

        # Fill host staging
        j_vals = np.fromiter(
            J.values(), dtype=np.int8, count=len(J),
        )
        self._h_sf_J[:] = 0
        self._h_sf_J[self._pos_ij] = j_vals
        self._h_sf_J[self._pos_ji] = j_vals

        h_vals = np.fromiter(
            h.values(), dtype=np.int8, count=len(h),
        )
        self._h_sf_h[:] = 0
        self._h_sf_h[self._h_idx] = h_vals

        # Async H2D on transfer stream
        j_start = slot_idx * nnz
        h_start = slot_idx * N
        sample_start = slot_idx * reads * max_packed_size
        energy_start = slot_idx * reads

        with self._sf_stream_transfer:
            self._d_sf_J[j_start:j_start + nnz].set(
                self._h_sf_J,
            )
            self._d_sf_h[h_start:h_start + N].set(
                self._h_sf_h,
            )
            # Zero output buffers for this slot
            self._d_sf_samples[
                sample_start:sample_start
                + reads * max_packed_size
            ] = 0
            self._d_sf_energies[
                energy_start:energy_start + reads
            ] = 0

        # Mark slot READY via DMA on transfer stream
        ctrl_offset = nonce_id * self.CTRL_STRIDE + slot_id
        ready_val = np.array(
            [self.SLOT_READY], dtype=np.int32,
        )
        with self._sf_stream_transfer:
            self._d_sf_ctrl[ctrl_offset:ctrl_offset + 1].set(
                ready_val,
            )
        self._sf_stream_transfer.synchronize()

    def launch_self_feeding(
        self,
        num_betas: int,
        seed: Optional[int] = None,
        active_nonce_count: Optional[int] = None,
    ) -> None:
        """Launch self-feeding kernel (once, stays resident).

        Slots must already have READY state from upload_slot().
        The kernel runs until no READY slots remain or
        signal_exit() is called.

        Args:
            num_betas: Number of beta schedule entries.
            seed: RNG base seed.
            active_nonce_count: Launch blocks for this many
                nonces (default: all prepared nonces).
        """
        assert self._sf_prepared
        assert not self._sf_kernel_running

        if seed is None:
            seed = np.random.randint(0, 2**31)

        N = self._prep_N
        nnz = self._prep_nnz
        max_packed_size = self._prep_max_packed_size
        num_nonces = self._sf_num_nonces
        blocks_per_nonce = self._sf_sms_per_nonce
        active = (
            active_nonce_count
            if active_nonce_count is not None
            else num_nonces
        )
        num_blocks = active * blocks_per_nonce

        grid = (num_blocks,)
        block = (256,)
        kernel_args = (
            self._d_row_ptr,
            self._d_col_ind,
            self._d_sf_block_starts,
            self._d_sf_block_counts,
            self._d_color_nodes,
            np.int32(self.num_colors),
            self._d_sf_beta,
            np.int32(num_betas),
            np.int32(self._sf_num_sweeps_per_beta),
            self._d_sf_J,
            self._d_sf_h,
            self._d_sf_samples,
            self._d_sf_energies,
            self._d_sf_ctrl,
            np.int32(num_nonces),
            np.int32(blocks_per_nonce),
            np.int32(self._sf_reads_per_nonce),
            np.int32(N),
            np.int32(nnz),
            np.int32(max_packed_size),
            np.int32(self._sf_chunks_per_model),
            np.int32(self._sf_reads_per_chunk),
            np.uint32(seed),
            np.int32(self.update_mode),
        )

        with self._sf_stream_compute:
            self._self_feeding_kernel(
                grid, block, kernel_args,
            )

        self._sf_kernel_running = True

    def upload_beta_schedule(
        self,
        h_first: Dict[int, float],
        J_first: Dict[Tuple[int, int], float],
        num_sweeps: int,
        num_sweeps_per_beta: int = 1,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
    ) -> Tuple[int, Tuple[float, float]]:
        """Upload beta schedule to the shared device buffer.

        Returns:
            (num_betas, beta_range) tuple.
        """
        sched, beta_range = self._get_cached_beta_schedule(
            h_first, J_first, num_sweeps,
            num_sweeps_per_beta, beta_range,
            beta_schedule_type, None,
        )
        num_betas = len(sched)
        self._d_sf_beta[:num_betas].set(sched)
        self._sf_beta_range = beta_range
        return num_betas, beta_range

    def poll_completions(
        self,
    ) -> List[Tuple[int, int]]:
        """Check for COMPLETE slots (non-blocking).

        Reads the control array from device and checks each
        slot's state.

        Returns:
            List of (nonce_id, slot_id) for completed slots.
        """
        assert self._sf_prepared
        ctrl_host = cp.asnumpy(self._d_sf_ctrl)
        completed = []
        for n in range(self._sf_num_nonces):
            base = n * self.CTRL_STRIDE
            for s in range(3):
                if ctrl_host[base + s] == self.SLOT_COMPLETE:
                    completed.append((n, s))
        return completed

    def download_slot(
        self,
        nonce_id: int,
        slot_id: int,
    ) -> dimod.SampleSet:
        """Download results from a COMPLETE slot.

        Does NOT change slot state — caller should upload_slot()
        with new data (which resets to READY) or leave as-is.

        Args:
            nonce_id: Nonce group index.
            slot_id: Slot within nonce.

        Returns:
            dimod.SampleSet with the slot's results.
        """
        N = self._prep_N
        node_to_idx = self._prep_node_to_idx
        max_packed_size = self._prep_max_packed_size
        reads = self._sf_reads_per_nonce
        slot_idx = nonce_id * 3 + slot_id

        sample_start = slot_idx * reads * max_packed_size
        energy_start = slot_idx * reads

        packed_raw = cp.asnumpy(
            self._d_sf_samples[
                sample_start:sample_start
                + reads * max_packed_size
            ],
        )
        energies_raw = cp.asnumpy(
            self._d_sf_energies[
                energy_start:energy_start + reads
            ],
        )

        packed_data = packed_raw.reshape(reads, max_packed_size)

        results = unpack_packed_results(
            packed_data, energies_raw,
            1, reads, N,
            [node_to_idx],
            info={
                "beta_range": getattr(
                    self, '_sf_beta_range', None,
                ),
                "update_mode": self.update_mode_name,
            },
        )
        return results[0]

    def mark_slot_empty(
        self, nonce_id: int, slot_id: int,
    ) -> None:
        """Mark a slot as EMPTY (host done with it)."""
        ctrl_offset = (
            nonce_id * self.CTRL_STRIDE + slot_id
        )
        empty_val = np.array(
            [self.SLOT_EMPTY], dtype=np.int32,
        )
        self._d_sf_ctrl[ctrl_offset:ctrl_offset + 1].set(
            empty_val,
        )

    def signal_nonce_exit(self, nonce_id: int) -> None:
        """Signal one nonce group to exit (not all).

        Sets exit_now for just this nonce_id. The kernel blocks
        for this nonce will exit at their next barrier check.
        Does NOT synchronize — caller should wait if needed.

        Args:
            nonce_id: Nonce group index to signal.
        """
        assert self._sf_prepared
        ctrl_offset = (
            nonce_id * self.CTRL_STRIDE + 6  # CTRL_EXIT_NOW
        )
        exit_val = np.array([1], dtype=np.int32)
        self._d_sf_ctrl[
            ctrl_offset:ctrl_offset + 1
        ].set(exit_val)

    def signal_exit(self) -> None:
        """Set exit_now for all nonces, wait for kernel exit."""
        assert self._sf_prepared
        for n in range(self._sf_num_nonces):
            self.signal_nonce_exit(n)

        self._sf_stream_compute.synchronize()
        self._sf_kernel_running = False

    def relaunch_self_feeding(
        self,
        active_nonce_count: int,
        num_betas: int,
        seed: Optional[int] = None,
    ) -> None:
        """Stop kernel, reset ctrl, relaunch with N nonces.

        Precondition: caller has uploaded slots for nonces
        0..active_nonce_count-1 already.

        Args:
            active_nonce_count: Number of active nonce groups.
            num_betas: Beta schedule length.
            seed: RNG base seed (random if None).
        """
        assert self._sf_prepared

        # Wait for kernel exit
        self._sf_stream_compute.synchronize()
        self._sf_kernel_running = False

        # Zero the entire ctrl array
        self._d_sf_ctrl[:] = 0

        # Relaunch with reduced grid
        self.launch_self_feeding(
            num_betas=num_betas,
            seed=seed,
            active_nonce_count=active_nonce_count,
        )

    def is_kernel_running(self) -> bool:
        """Check if the self-feeding kernel is still running."""
        if not self._sf_kernel_running:
            return False
        # Non-blocking query: check if compute stream is done
        done = self._sf_stream_compute.done
        if done:
            self._sf_kernel_running = False
        return self._sf_kernel_running

    def get_profile_data(self) -> np.ndarray:
        """Copy profile counters from GPU and reshape."""
        if not self.profile:
            raise RuntimeError(
                "Profiling not enabled. "
                "Pass profile=True to __init__()."
            )
        if not hasattr(self, '_d_profile'):
            raise RuntimeError(
                "No profile data. Call sample_ising() first."
            )
        raw = cp.asnumpy(self._d_profile)
        return raw.reshape(
            self._profile_work_units,
            self.GIBBS_NUM_REGIONS,
        )
