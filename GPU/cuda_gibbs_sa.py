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
import time
from typing import Dict, List, Optional, Tuple

import cupy as cp
import dimod
import numpy as np

from GPU.base_cuda_sampler import BaseCudaSampler
from GPU.sampler_utils import (
    compute_beta_schedule,
    compute_color_blocks,
    unpack_packed_results,
)


GIBBS_NUM_REGIONS = 12


class CudaGibbsSampler(BaseCudaSampler):
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
        if update_mode.lower() not in ("gibbs", "metropolis"):
            raise ValueError(
                f"update_mode must be 'gibbs' or "
                f"'metropolis', got {update_mode}"
            )
        self.update_mode = (
            0 if update_mode.lower() == "gibbs" else 1
        )
        self.update_mode_name = update_mode.lower()
        self.parallel = parallel

        super().__init__(
            topology=topology,
            max_sms=max_sms,
            profile=profile,
            sampler_type="cuda-gibbs",
        )

        # Extract Zephyr parameters
        topo_shape = self.properties.get(
            'topology', {}
        ).get('shape', [9, 2])
        self.m = topo_shape[0]
        self.t = topo_shape[1]
        self.num_colors = 4

        # Default SMs per nonce (overridable via
        # prepare_self_feeding kwargs)
        self._sf_sms_per_nonce_val = 4

    # -- BaseCudaSampler hooks --

    def _kernel_filename(self) -> str:
        return 'cuda_gibbs.cu'

    def _kernel_function_name(self) -> str:
        return 'cuda_gibbs_self_feeding'

    def _num_profile_regions(self) -> int:
        return GIBBS_NUM_REGIONS

    @property
    def _sms_per_nonce(self) -> int:
        return self._sf_sms_per_nonce_val

    def _extra_download_info(self) -> dict:
        return {"update_mode": self.update_mode_name}

    # -- Gibbs-specific prepare (adds color blocks) --

    def prepare(
        self,
        num_reads: int = 256,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        max_nonces: int = 1,
    ) -> None:
        """Pre-allocate GPU buffers for a fixed topology.

        Extends base prepare() with color block computation
        and double-buffered GPU arrays for the non-self-feeding
        pipeline path.

        Args:
            num_reads: Max reads per job.
            num_sweeps: Max sweeps (determines beta schedule
                size).
            num_sweeps_per_beta: Sweeps per beta value.
            max_nonces: Max nonces for multi-nonce dispatch.
        """
        super().prepare(
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            num_sweeps_per_beta=num_sweeps_per_beta,
        )

        N = self._prep_N
        nnz = self._prep_nnz
        max_packed_size = self._prep_max_packed_size
        node_to_idx = self._prep_node_to_idx

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

        # Upload Gibbs-specific constant GPU buffers
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
                total_samples * max_packed_size,
                dtype=cp.int8,
            ),
            cp.zeros(
                total_samples * max_packed_size,
                dtype=cp.int8,
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

        # Multi-nonce double-buffered GPU arrays (A/B)
        self._max_nonces = max_nonces
        if max_nonces > 1:
            mn_total_reads = max_nonces * num_reads
            self._d_mn_J = [
                cp.zeros(
                    max_nonces * nnz, dtype=cp.int8,
                ),
                cp.zeros(
                    max_nonces * nnz, dtype=cp.int8,
                ),
            ]
            self._d_mn_h = [
                cp.zeros(
                    max_nonces * N, dtype=cp.int8,
                ),
                cp.zeros(
                    max_nonces * N, dtype=cp.int8,
                ),
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
                cp.zeros(
                    mn_total_reads, dtype=cp.int32,
                ),
                cp.zeros(
                    mn_total_reads, dtype=cp.int32,
                ),
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
            # Per-nonce metadata
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
                np.arange(
                    max_nonces, dtype=np.int32,
                ) * nnz,
            )
            self._d_mn_problem_h = cp.asarray(
                np.arange(
                    max_nonces, dtype=np.int32,
                ) * N,
            )
            self._d_mn_block_starts = cp.asarray(
                np.tile(starts, max_nonces),
            )
            self._d_mn_block_counts = cp.asarray(
                np.tile(counts, max_nonces),
            )

            # Double host staging buffers (A/B)
            self._h_mn_J = [
                np.zeros(
                    max_nonces * nnz, dtype=np.int8,
                ),
                np.zeros(
                    max_nonces * nnz, dtype=np.int8,
                ),
            ]
            self._h_mn_h = [
                np.zeros(
                    max_nonces * N, dtype=np.int8,
                ),
                np.zeros(
                    max_nonces * N, dtype=np.int8,
                ),
            ]

        # CUDA streams for pipeline overlap
        self._stream_compute = cp.cuda.Stream(
            non_blocking=True,
        )
        self._stream_transfer = cp.cuda.Stream(
            non_blocking=True,
        )
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

        self.logger.info(
            "Prepared Gibbs buffers: N=%d, nnz=%d, "
            "num_reads=%d, max_betas=%d, max_nonces=%d",
            N, nnz, num_reads,
            self._prep_max_num_betas, max_nonces,
        )

    def _allocate_kernel_buffers(
        self,
        num_nonces: int,
        reads_per_nonce: int,
        num_sweeps: int,
        num_sweeps_per_beta: int,
        sms_per_nonce: int = 4,
    ) -> None:
        """Allocate Gibbs-specific GPU buffers.

        Color blocks tiled for num_nonces, and work
        distribution metadata.
        """
        self._sf_sms_per_nonce_val = sms_per_nonce

        # Color blocks tiled for num_nonces
        starts = cp.asnumpy(self._d_block_starts)
        counts = cp.asnumpy(self._d_block_counts)
        self._d_sf_block_starts = cp.asarray(
            np.tile(starts, num_nonces),
        )
        self._d_sf_block_counts = cp.asarray(
            np.tile(counts, num_nonces),
        )

        # Chunks per model (work distribution)
        self._sf_chunks_per_model = sms_per_nonce
        self._sf_reads_per_chunk = (
            (reads_per_nonce + sms_per_nonce - 1)
            // sms_per_nonce
        )

    def _kernel_launch_args(
        self,
        active: int,
        num_betas: int,
        seed: int,
    ) -> tuple:
        N = self._prep_N
        nnz = self._prep_nnz
        max_packed_size = self._prep_max_packed_size
        num_nonces = self._sf_num_nonces
        blocks_per_nonce = self._sf_sms_per_nonce_val

        return (
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

    # -- Gibbs-specific close (also handles double-buffer
    #    streams) --

    def close(self) -> None:
        """Synchronize and release all CUDA streams."""
        super().close()
        if hasattr(self, '_stream_compute'):
            self._stream_compute.synchronize()
            self._stream_transfer.synchronize()
        self._mn_preloaded = False
        self._preloaded = False

    # -- Gibbs-specific methods --

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
            return (
                self._cached_beta_sched,
                self._cached_beta_range,
            )

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
            self._d_beta_sched[next_idx][
                :num_betas
            ].set(sched)
        self._event_transfer_done.record(
            self._stream_transfer,
        )

        self._preloaded = True
        self._preload_meta = (
            num_reads, num_betas, num_sweeps_per_beta,
            seed, beta_range, beta_schedule_type,
        )

    # -- sample_ising --

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
        """Sample from Ising model using self-feeding kernel.

        Args:
            h: List of linear biases per problem.
            J: List of quadratic biases per problem.
            num_reads: Number of independent samples per
                problem.
            num_sweeps: Total number of sweeps.
            num_sweeps_per_beta: Sweeps per beta value.
            beta_range: (hot_beta, cold_beta) or None for
                auto.
            beta_schedule_type: Schedule type.
            beta_schedule: Custom schedule.
            seed: RNG seed.

        Returns:
            List of dimod.SampleSet, one per problem.
        """
        num_problems = len(h)
        assert len(J) == num_problems, (
            f"h and J must have same length: "
            f"{num_problems} vs {len(J)}"
        )

        # Prepare topology structures if needed
        if not self._prepared:
            self.prepare(
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                num_sweeps_per_beta=num_sweeps_per_beta,
                max_nonces=num_problems,
            )

        # Prepare self-feeding buffers if needed
        if not self._sf_prepared:
            sms_per_nonce = max(1, 4)
            self.prepare_self_feeding(
                num_nonces=num_problems,
                reads_per_nonce=num_reads,
                num_sweeps=num_sweeps,
                num_sweeps_per_beta=num_sweeps_per_beta,
                sms_per_nonce=sms_per_nonce,
            )

        # Reset ctrl array (clears stale EXIT_NOW from
        # previous sample_ising call)
        self._d_sf_ctrl[:] = 0

        # Upload beta schedule
        num_betas, beta_range = self.upload_beta_schedule(
            h[0], J[0], num_sweeps,
            num_sweeps_per_beta, beta_range,
            beta_schedule_type,
        )

        # Upload one model per nonce to slot 0
        for i in range(num_problems):
            self.upload_slot(i, 0, h[i], J[i])

        # Launch kernel
        self._sf_kernel_running = False  # allow re-launch
        self.launch_self_feeding(
            num_betas=num_betas,
            seed=seed,
            active_nonce_count=num_problems,
        )

        # Poll until all nonces complete
        completed = set()
        deadline = time.time() + 300.0
        while len(completed) < num_problems:
            assert time.time() < deadline, (
                "sample_ising timed out waiting for kernel"
            )
            for nonce_id, slot_id in self.poll_completions():
                if nonce_id not in completed:
                    completed.add(nonce_id)
            if len(completed) < num_problems:
                time.sleep(0.001)

        # Download results
        results = []
        for i in range(num_problems):
            ss = self.download_slot(i, 0)
            results.append(ss)

        # Signal exit and wait
        self.signal_exit()

        return results
