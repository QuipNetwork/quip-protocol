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
from itertools import chain
from typing import (
    Any, Dict, Iterable, Iterator, List, Optional, Tuple,
)

import cupy as cp
import dimod
import numpy as np

from GPU.base_cuda_sampler import BaseCudaSampler
from GPU.gpu_scheduler import throttled_stream
from GPU.sampler_utils import compute_color_blocks
from shared.ising_model import IsingModel


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
        sms_per_nonce: int = 4,
    ):
        """Initialize CUDA Gibbs sampler.

        Args:
            topology: Topology object (default: DEFAULT_TOPOLOGY).
            update_mode: "gibbs" or "metropolis".
            parallel: Use chromatic parallel kernel (True) or
                fully sequential kernel (False).
            max_sms: Maximum SMs to use (0 = all available).
            profile: Enable auto-profiling with clock64()
                instrumentation.
            sms_per_nonce: SMs allocated per nonce (default 4).
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

        # SMs per nonce (overridable via
        # prepare_self_feeding kwargs)
        self._sf_sms_per_nonce_val = sms_per_nonce

    # -- BaseCudaSampler hooks --

    def _kernel_filename(self) -> str:
        return 'cuda_gibbs.cu'

    def _kernel_function_name(self) -> str:
        return 'cuda_gibbs_self_feeding'

    def _profiling_mode(self) -> str:
        return "thread_zero"

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
    ) -> None:
        """Pre-allocate GPU buffers for a fixed topology.

        Extends base prepare() with color block computation.

        Args:
            num_reads: Max reads per job.
            num_sweeps: Max sweeps (determines beta schedule
                size).
            num_sweeps_per_beta: Sweeps per beta value.
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

        self.logger.info(
            "Prepared Gibbs buffers: N=%d, nnz=%d, "
            "num_reads=%d, max_betas=%d",
            N, nnz, num_reads, self._prep_max_num_betas,
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

    # -- Gibbs-specific streaming API --

    def sample_ising_streaming(
        self,
        models: Iterable[IsingModel],
        *,
        num_reads: int = 200,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
        seed: Optional[int] = None,
        num_kernels: Optional[int] = None,
        poll_timeout: Optional[float] = None,
        scheduler: Any = None,
    ) -> Iterator[Tuple[IsingModel, dimod.SampleSet]]:
        """Stream Ising model solutions via Gibbs kernel.

        Gibbs-specific prepare logic (sms_per_nonce) +
        base rotation loop.

        Args:
            models: Iterable of IsingModel.
            num_reads: Samples per model.
            num_sweeps: Total sweeps per model.
            num_sweeps_per_beta: Sweeps per beta value.
            beta_range: (hot, cold) or None for auto.
            beta_schedule_type: Schedule type.
            seed: RNG seed.
            num_kernels: Concurrent nonces (default: auto).
            poll_timeout: Seconds before TimeoutError.

        Yields:
            (model, SampleSet) in completion order.
        """
        num_k = num_kernels or max(
            1,
            self.max_sms // self._sms_per_nonce,
        )

        if not self._prepared:
            self.prepare(
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                num_sweeps_per_beta=num_sweeps_per_beta,
            )

        if not self._sf_prepared:
            self.prepare_self_feeding(
                num_nonces=num_k,
                reads_per_nonce=num_reads,
                num_sweeps=num_sweeps,
                num_sweeps_per_beta=num_sweeps_per_beta,
                sms_per_nonce=self._sf_sms_per_nonce_val,
            )

        # Peek first model for beta schedule
        model_iter = iter(models)
        try:
            first = next(model_iter)
        except StopIteration:
            return

        num_betas, _ = self.upload_beta_schedule(
            first.h, first.J, num_sweeps,
            num_sweeps_per_beta, beta_range,
            beta_schedule_type,
        )

        # Throttle before pulling each result (yielding mode): the unified
        # driver path bypasses the old _sample_batch back-off, so honor it here.
        yield from throttled_stream(
            self._run_streaming_loop(
                chain([first], model_iter),
                num_k=num_k,
                num_betas=num_betas,
                seed=seed,
                poll_timeout=poll_timeout,
            ),
            scheduler,
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
            )

        # Prepare self-feeding buffers if needed
        if not self._sf_prepared:
            self.prepare_self_feeding(
                num_nonces=num_problems,
                reads_per_nonce=num_reads,
                num_sweeps=num_sweeps,
                num_sweeps_per_beta=num_sweeps_per_beta,
                sms_per_nonce=self._sf_sms_per_nonce_val,
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
        while len(completed) < num_problems:
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
