# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Abstract base class for CUDA self-feeding kernel samplers.

Captures the shared 3-slot rotating buffer protocol used by
both CudaSASampler and CudaGibbsSampler. Subclasses provide
kernel-specific compilation, buffer allocation, and launch
arguments.

Control layout per nonce (CTRL_STRIDE=8 ints):
    [0..2] slot_state  [3] active_slot  [4] blocks_done
    [5] work_queue  [6] exit_now  [7] generation
"""
from __future__ import annotations

import abc
import dataclasses
import logging
import os
import time
from typing import (
    Dict, Iterable, Iterator, List, Optional, Tuple,
)

import cupy as cp
import dimod
import numpy as np

from shared.ising_model import IsingModel

from GPU.sampler_utils import (
    build_csr_structure_from_edges,
    build_edge_position_index,
    compute_beta_schedule,
    unpack_packed_results,
)

# Minimum CUDA runtime version (runtimeGetVersion()) for each
# GPU architecture. Used for PTX fallback when NVRTC is older
# than the GPU (e.g., CUDA 12.6 with Blackwell sm_120).
_CUDA_ARCH_MIN_VERSION = {
    121: 12090, 120: 12080,
    103: 12090, 101: 12080, 100: 12080,
    90: 12000, 89: 11080, 86: 11010, 80: 11000,
}


def _best_fallback_arch(cuda_ver: int) -> int:
    """Highest GPU arch the given CUDA version supports."""
    return max(
        (a for a, v in _CUDA_ARCH_MIN_VERSION.items()
         if v <= cuda_ver),
        default=80,
    )


@dataclasses.dataclass(slots=True)
class _SlotState:
    """Per-kernel slot assignment for streaming API.

    Tracks which slot holds the active model (being computed),
    which holds the next model (preloaded), and which is free
    for upload.
    """

    active_slot: int
    active_model: Optional[IsingModel]
    next_slot: int
    next_model: Optional[IsingModel]
    free_slot: int


class BaseCudaSampler(abc.ABC):
    """Abstract base for CUDA self-feeding kernel samplers.

    Provides the shared 3-slot rotating buffer protocol:
    prepare topology, allocate self-feeding buffers, upload
    models, launch/poll/download/signal the kernel.

    Subclasses implement kernel-specific hooks:
        _kernel_filename()
        _kernel_function_name()
        _num_profile_regions()
        _extra_compile_options()
        _sms_per_nonce (property)
        _allocate_kernel_buffers()
        _kernel_launch_args()
        _extra_download_info()
    """

    CTRL_STRIDE = 8
    SLOT_EMPTY = 0
    SLOT_READY = 1
    SLOT_ACTIVE = 2
    SLOT_COMPLETE = 3

    def __init__(
        self,
        topology=None,
        max_sms: int = 0,
        profile: bool = False,
        sampler_type: str = "cuda",
    ):
        """Initialize base CUDA sampler.

        Args:
            topology: Topology object (default: DEFAULT_TOPOLOGY).
            max_sms: Maximum SMs to use (0 = all available).
            profile: Enable auto-profiling with clock64()
                instrumentation.
            sampler_type: Identifier string for this sampler.
        """
        self.profile = profile
        self.logger = logging.getLogger(
            type(self).__module__,
        )
        if max_sms == 0:
            max_sms = cp.cuda.Device().attributes[
                'MultiProcessorCount'
            ]
        self.max_sms = max_sms
        self.sampler_type = sampler_type

        from dwave_topologies import DEFAULT_TOPOLOGY
        topology_obj = (
            topology if topology is not None
            else DEFAULT_TOPOLOGY
        )
        topology_graph = topology_obj.graph
        self.nodes = list(topology_graph.nodes())
        self.edges = list(topology_graph.edges())
        self.nodelist = self.nodes
        self.edgelist = self.edges
        self.properties = topology_obj.properties

        # Compile CUDA kernel
        kernel_path = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__),
            ),
            self._kernel_filename(),
        )
        self._profile_manifest = None

        compile_options = ['--use_fast_math']
        compile_options.extend(
            self._extra_compile_options(),
        )

        if self.profile:
            # Auto-instrument: generates profiled temp file
            # and manifest from clean source
            import sys
            _project_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)),
            )
            if _project_root not in sys.path:
                sys.path.insert(0, _project_root)
            from tools.cuda_auto_profiler import (
                auto_instrument,
            )
            profiled_path, manifest = auto_instrument(
                kernel_path,
                self._kernel_function_name(),
                self._profiling_mode(),
            )
            self._profile_manifest = manifest
            with open(profiled_path, 'r') as f:
                kernel_code = f.read()
            self.logger.info(
                "%s auto-profiler: %d regions from %s",
                self.sampler_type,
                manifest["num_regions"],
                kernel_path,
            )
        else:
            with open(kernel_path, 'r') as f:
                kernel_code = f.read()

        self._module = self._compile_module(
            kernel_code, compile_options,
        )
        self._self_feeding_kernel = self._module.get_function(
            self._kernel_function_name(),
        )

        self._prepared = False
        self._sf_prepared = False

    # ----------------------------------------------------------
    # Abstract hooks for subclasses
    # ----------------------------------------------------------

    @abc.abstractmethod
    def _kernel_filename(self) -> str:
        """CUDA source file name (e.g., 'cuda_sa.cu')."""

    @abc.abstractmethod
    def _kernel_function_name(self) -> str:
        """Kernel entry point name."""

    def _num_profile_regions(self) -> int:
        """Number of profiling regions (from manifest)."""
        assert self._profile_manifest is not None, (
            "Profile manifest not loaded. "
            "Pass profile=True to __init__()."
        )
        return self._profile_manifest["num_regions"]

    @abc.abstractmethod
    def _profiling_mode(self) -> str:
        """Profiling mode: 'per_thread' or 'thread_zero'."""

    def _extra_compile_options(self) -> List[str]:
        """Extra nvcc compile options (default: none)."""
        return []

    def _compile_module(
        self,
        kernel_code: str,
        compile_options: List[str],
    ) -> cp.RawModule:
        """Compile CUDA kernel with PTX fallback.

        Normal path: NVRTC auto-detects GPU arch. If the GPU
        is newer than NVRTC supports, falls back to PTX from
        the highest supported arch (driver JIT-compiles it).
        """
        dev = cp.cuda.Device()
        cc = int(dev.compute_capability)
        cuda_ver = cp.cuda.runtime.runtimeGetVersion()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        gpu_name = props.get('name', 'unknown').decode()
        self.logger.info(
            "%s GPU: %s (sm_%d, CUDA runtime %d)",
            self.sampler_type, gpu_name, cc, cuda_ver,
        )

        min_cuda = _CUDA_ARCH_MIN_VERSION.get(cc, 0)
        using_fallback = min_cuda > cuda_ver
        if using_fallback:
            fb = _best_fallback_arch(cuda_ver)
            self.logger.warning(
                "NVRTC (CUDA %d) cannot target sm_%d; "
                "using compute_%d PTX. Upgrade Docker "
                "image for native support.",
                cuda_ver, cc, fb,
            )
            compile_options = list(compile_options) + [
                '--gpu-architecture=compute_%d' % fb,
            ]

        try:
            return cp.RawModule(
                code=kernel_code,
                options=tuple(compile_options),
            )
        except Exception:
            if using_fallback:
                raise
            fb = _best_fallback_arch(cuda_ver)
            self.logger.warning(
                "NVRTC failed for sm_%d; retrying "
                "with compute_%d PTX.",
                cc, fb,
            )
            compile_options = list(compile_options) + [
                '--gpu-architecture=compute_%d' % fb,
            ]
            return cp.RawModule(
                code=kernel_code,
                options=tuple(compile_options),
            )

    @property
    @abc.abstractmethod
    def _sms_per_nonce(self) -> int:
        """CUDA blocks launched per nonce group."""

    @abc.abstractmethod
    def _allocate_kernel_buffers(
        self,
        num_nonces: int,
        reads_per_nonce: int,
        num_sweeps: int,
        num_sweeps_per_beta: int,
    ) -> None:
        """Allocate kernel-specific GPU buffers.

        Called during prepare_self_feeding() after common
        buffers are allocated. Subclass stores its own
        device arrays as instance attributes.
        """

    @abc.abstractmethod
    def _kernel_launch_args(
        self,
        active: int,
        num_betas: int,
        seed: int,
    ) -> tuple:
        """Build kernel launch args tuple.

        Args:
            active: Number of active nonces.
            num_betas: Beta schedule length.
            seed: RNG seed.

        Returns:
            Tuple of kernel arguments.
        """

    def _extra_download_info(self) -> dict:
        """Extra metadata for download_slot SampleSet info.

        Override to add sampler-specific info fields.
        """
        return {}

    # ----------------------------------------------------------
    # Shared prepare (topology CSR)
    # ----------------------------------------------------------

    def prepare(
        self,
        num_reads: int = 256,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
    ) -> None:
        """Build CSR topology structures (one-time).

        Args:
            num_reads: Max reads per job.
            num_sweeps: Max sweeps (for beta schedule size).
            num_sweeps_per_beta: Sweeps per beta value.
        """
        (csr_row_ptr, csr_col_ind, node_to_idx,
         sorted_neighbors, N, nnz) = (
            build_csr_structure_from_edges(
                self.edges, self.nodes,
            )
        )

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
        self._prep_max_num_betas = (
            num_sweeps // num_sweeps_per_beta
        )

        max_packed_size = (N + 7) // 8
        self._prep_max_packed_size = max_packed_size

        # Upload constant GPU buffers
        self._d_row_ptr = cp.asarray(csr_row_ptr)
        self._d_col_ind = cp.asarray(csr_col_ind)

        # Host staging buffers
        self._h_J_vals = np.zeros(nnz, dtype=np.int8)
        self._h_h_vals = np.zeros(N, dtype=np.int8)

        # Beta schedule cache
        self._cached_beta_key = None
        self._cached_beta_sched = None
        self._cached_beta_range = None

        self._prepared = True
        self.logger.info(
            "%s topology prepared: N=%d, nnz=%d",
            self.sampler_type, N, nnz,
        )

    # ----------------------------------------------------------
    # Self-feeding buffer allocation
    # ----------------------------------------------------------

    def prepare_self_feeding(
        self,
        num_nonces: int,
        reads_per_nonce: int = 32,
        num_sweeps: int = 1000,
        num_sweeps_per_beta: int = 1,
        **kwargs,
    ) -> None:
        """Allocate 3-slot rotating buffers for self-feeding.

        Args:
            num_nonces: Number of concurrent nonces.
            reads_per_nonce: Reads per nonce per model.
            num_sweeps: Max sweeps (for beta schedule size).
            num_sweeps_per_beta: Sweeps per beta value.
            **kwargs: Passed to _allocate_kernel_buffers().
        """
        assert self._prepared, "Must call prepare() first"

        N = self._prep_N
        nnz = self._prep_nnz
        max_packed_size = self._prep_max_packed_size
        total_slots = num_nonces * 3

        self._sf_num_nonces = num_nonces
        self._sf_reads_per_nonce = reads_per_nonce
        self._sf_num_sweeps_per_beta = num_sweeps_per_beta
        self._sf_max_num_betas = (
            num_sweeps // num_sweeps_per_beta
        )

        # Flat buffer allocations (3 slots per nonce)
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

        # Per-nonce control array
        self._d_sf_ctrl = cp.zeros(
            num_nonces * self.CTRL_STRIDE, dtype=cp.int32,
        )

        # Shared beta schedule buffer
        self._d_sf_beta = cp.zeros(
            self._sf_max_num_betas, dtype=cp.float32,
        )

        # Dedicated streams
        self._sf_stream_compute = cp.cuda.Stream(
            non_blocking=True,
        )
        self._sf_stream_transfer = cp.cuda.Stream(
            non_blocking=True,
        )

        # Subclass-specific buffers
        self._allocate_kernel_buffers(
            num_nonces, reads_per_nonce,
            num_sweeps, num_sweeps_per_beta,
            **kwargs,
        )

        # Profile buffer (size depends on mode)
        if self.profile:
            num_regions = self._num_profile_regions()
            num_blocks = num_nonces * self._sms_per_nonce
            if self._profiling_mode() == "per_thread":
                # One entry per thread
                max_work_units = num_blocks * 256
            else:
                # One entry per block (thread 0 only)
                max_work_units = num_blocks
            self._d_sf_profile = cp.zeros(
                max_work_units * num_regions,
                dtype=cp.int64,
            )
            self._sf_profile_work_units = max_work_units

        self._sf_kernel_running = False
        self._sf_prepared = True

        self.logger.info(
            "%s self-feeding prepared: %d nonces × 3 slots, "
            "%d reads/nonce",
            self.sampler_type,
            num_nonces, reads_per_nonce,
        )

    # ----------------------------------------------------------
    # Slot upload / download
    # ----------------------------------------------------------

    def upload_slot(
        self,
        nonce_id: int,
        slot_id: int,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
    ) -> None:
        """Upload model data to a specific slot, mark READY.

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
        self._h_J_vals[:] = 0
        self._h_J_vals[self._pos_ij] = j_vals
        self._h_J_vals[self._pos_ji] = j_vals

        h_vals = np.fromiter(
            h.values(), dtype=np.int8, count=len(h),
        )
        self._h_h_vals[:] = 0
        self._h_h_vals[self._h_idx] = h_vals

        # Async H2D on transfer stream
        j_start = slot_idx * nnz
        h_start = slot_idx * N
        sample_start = (
            slot_idx * reads * max_packed_size
        )
        energy_start = slot_idx * reads

        with self._sf_stream_transfer:
            self._d_sf_J[j_start:j_start + nnz].set(
                self._h_J_vals,
            )
            self._d_sf_h[h_start:h_start + N].set(
                self._h_h_vals,
            )
            # Zero output buffers for this slot
            self._d_sf_samples[
                sample_start:sample_start
                + reads * max_packed_size
            ] = 0
            self._d_sf_energies[
                energy_start:energy_start + reads
            ] = 0

        # Mark slot READY
        ctrl_offset = (
            nonce_id * self.CTRL_STRIDE + slot_id
        )
        ready_val = np.array(
            [self.SLOT_READY], dtype=np.int32,
        )
        with self._sf_stream_transfer:
            self._d_sf_ctrl[
                ctrl_offset:ctrl_offset + 1
            ].set(ready_val)
        self._sf_stream_transfer.synchronize()

    def upload_beta_schedule(
        self,
        h_first: Dict[int, float],
        J_first: Dict[Tuple[int, int], float],
        num_sweeps: int,
        num_sweeps_per_beta: int = 1,
        beta_range: Optional[Tuple[float, float]] = None,
        beta_schedule_type: str = "geometric",
    ) -> Tuple[int, Tuple[float, float]]:
        """Upload beta schedule to shared device buffer.

        Returns:
            (num_betas, beta_range) tuple.
        """
        key = (
            num_sweeps, num_sweeps_per_beta,
            beta_schedule_type, beta_range,
        )
        if self._cached_beta_key == key:
            sched = self._cached_beta_sched
            beta_range = self._cached_beta_range
        else:
            sched, beta_range = compute_beta_schedule(
                h_first, J_first, num_sweeps,
                num_sweeps_per_beta, beta_range,
                beta_schedule_type, None,
            )
            self._cached_beta_key = key
            self._cached_beta_sched = sched
            self._cached_beta_range = beta_range

        num_betas = len(sched)
        # Grow beta buffer if schedule exceeds allocation
        if num_betas > len(self._d_sf_beta):
            self._d_sf_beta = cp.zeros(
                num_betas, dtype=cp.float32,
            )
        self._d_sf_beta[:num_betas].set(sched)
        self._sf_beta_range = beta_range
        return num_betas, beta_range

    def download_slot(
        self,
        nonce_id: int,
        slot_id: int,
    ) -> dimod.SampleSet:
        """Download results from a COMPLETE slot.

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

        sample_start = (
            slot_idx * reads * max_packed_size
        )
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

        packed_data = packed_raw.reshape(
            reads, max_packed_size,
        )

        info = {
            "beta_range": getattr(
                self, '_sf_beta_range', None,
            ),
        }
        info.update(self._extra_download_info())

        results = unpack_packed_results(
            packed_data, energies_raw,
            1, reads, N,
            [node_to_idx],
            info=info,
        )
        return results[0]

    def mark_slot_empty(
        self, nonce_id: int, slot_id: int,
    ) -> None:
        """Mark a slot as EMPTY."""
        ctrl_offset = (
            nonce_id * self.CTRL_STRIDE + slot_id
        )
        empty_val = np.array(
            [self.SLOT_EMPTY], dtype=np.int32,
        )
        self._d_sf_ctrl[
            ctrl_offset:ctrl_offset + 1
        ].set(empty_val)

    # ----------------------------------------------------------
    # Kernel launch / control
    # ----------------------------------------------------------

    def launch_self_feeding(
        self,
        num_betas: int,
        seed: Optional[int] = None,
        active_nonce_count: Optional[int] = None,
    ) -> None:
        """Launch self-feeding kernel (once, stays resident).

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

        num_nonces = self._sf_num_nonces
        active = (
            active_nonce_count
            if active_nonce_count is not None
            else num_nonces
        )

        num_blocks = active * self._sms_per_nonce
        grid = (num_blocks,)
        block = (256,)
        kernel_args = self._kernel_launch_args(
            active, num_betas, seed,
        )
        if self.profile:
            kernel_args = kernel_args + (
                self._d_sf_profile,
            )

        with self._sf_stream_compute:
            self._self_feeding_kernel(
                grid, block, kernel_args,
            )

        self._sf_kernel_running = True

    def poll_completions(
        self,
    ) -> List[Tuple[int, int]]:
        """Check for COMPLETE slots (non-blocking).

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

    def signal_nonce_exit(self, nonce_id: int) -> None:
        """Signal one nonce to exit."""
        assert self._sf_prepared
        ctrl_offset = (
            nonce_id * self.CTRL_STRIDE + 6
        )
        exit_val = np.array([1], dtype=np.int32)
        self._d_sf_ctrl[
            ctrl_offset:ctrl_offset + 1
        ].set(exit_val)

    def signal_exit(self, wait: bool = True) -> None:
        """Set exit_now for all nonces.

        Args:
            wait: If True, synchronize the compute stream
                (blocks until kernel exits). If False, signal
                only — caller is responsible for cleanup
                (e.g., process is about to exit).
        """
        assert self._sf_prepared
        for n in range(self._sf_num_nonces):
            self.signal_nonce_exit(n)

        if wait:
            self._sf_stream_compute.synchronize()
        self._sf_kernel_running = False

    def relaunch_self_feeding(
        self,
        active_nonce_count: int,
        num_betas: int,
        seed: Optional[int] = None,
    ) -> None:
        """Stop kernel, reset ctrl, relaunch.

        Args:
            active_nonce_count: Number of active nonces.
            num_betas: Beta schedule length.
            seed: RNG base seed.
        """
        assert self._sf_prepared

        self._sf_stream_compute.synchronize()
        self._sf_kernel_running = False

        # Zero the entire ctrl array
        self._d_sf_ctrl[:] = 0

        self.launch_self_feeding(
            num_betas=num_betas,
            seed=seed,
            active_nonce_count=active_nonce_count,
        )

    def is_kernel_running(self) -> bool:
        """Check if the self-feeding kernel is still running."""
        if not self._sf_kernel_running:
            return False
        done = self._sf_stream_compute.done
        if done:
            self._sf_kernel_running = False
        return self._sf_kernel_running

    def get_profile_data(self) -> np.ndarray:
        """Copy profile counters from GPU and reshape.

        Returns:
            Array of shape (work_units, num_regions).
        """
        if not self.profile:
            raise RuntimeError(
                "Profiling not enabled. "
                "Pass profile=True to __init__()."
            )
        if not hasattr(self, '_d_sf_profile'):
            raise RuntimeError(
                "No profile data. Call sample_ising() first."
            )
        raw = cp.asnumpy(self._d_sf_profile)
        return raw.reshape(
            self._sf_profile_work_units,
            self._num_profile_regions(),
        )

    # ----------------------------------------------------------
    # Streaming rotation loop (mechanical slot management)
    # ----------------------------------------------------------

    def _run_streaming_loop(
        self,
        models: Iterable[IsingModel],
        *,
        num_k: int,
        num_betas: int,
        seed: Optional[int] = None,
        poll_timeout: Optional[float] = None,
    ) -> Iterator[Tuple[IsingModel, dimod.SampleSet]]:
        """Run the 3-slot rotation loop over models.

        Preconditions (caller must ensure):
        - prepare() already called
        - prepare_self_feeding() already called
        - Beta schedule already uploaded (num_betas)

        Args:
            models: Iterable of IsingModel.
            num_k: Number of concurrent nonces.
            num_betas: Beta schedule length.
            seed: RNG seed.
            poll_timeout: Seconds before raising
                TimeoutError. None = block forever.

        Yields:
            (model, SampleSet) in completion order.
        """
        assert self._prepared, (
            "Must call prepare() before _run_streaming_loop"
        )
        assert self._sf_prepared, (
            "Must call prepare_self_feeding() before "
            "_run_streaming_loop"
        )

        # Reset ctrl array
        self._d_sf_ctrl[:] = 0

        model_iter = iter(models)
        _has_try_pop = hasattr(models, 'try_pop')
        _exhausted = False

        def _pull_blocking() -> Optional[IsingModel]:
            """Wait for a model (cold start only).

            Uses IsingFeeder.pop_blocking() when available,
            otherwise falls back to next(model_iter).
            """
            nonlocal _exhausted
            if _exhausted:
                return None
            if hasattr(models, 'pop_blocking'):
                try:
                    return models.pop_blocking()
                except (StopIteration, RuntimeError):
                    _exhausted = True
                    return None
            try:
                return next(model_iter)
            except StopIteration:
                _exhausted = True
                return None

        def _pull_nonblocking() -> Optional[IsingModel]:
            """Return a model if one is ready, else None."""
            nonlocal _exhausted
            if _exhausted:
                return None
            if _has_try_pop:
                m = models.try_pop()
                if m is None:
                    return None
                return m
            return _pull_blocking()

        # Build per-kernel slot state
        # Slots: 0=active, 1=next, 2=free
        slots: list[_SlotState] = []
        for _ in range(num_k):
            slots.append(_SlotState(
                active_slot=0, active_model=None,
                next_slot=1, next_model=None,
                free_slot=2,
            ))

        # Cold start: fill active slots (slot 0) — blocking
        pending_models: list[IsingModel] = []
        for _ in range(num_k):
            m = _pull_blocking()
            if m is None:
                break
            pending_models.append(m)

        if not pending_models:
            return

        for i, m in enumerate(pending_models):
            self.upload_slot(
                i, slots[i].active_slot, m.h, m.J,
            )
            slots[i].active_model = m

        # Fill next slots (slot 1) — blocking to ensure
        # the kernel has work queued when it finishes slot 0
        for i in range(len(pending_models)):
            m = _pull_blocking()
            if m is None:
                break
            self.upload_slot(
                i, slots[i].next_slot, m.h, m.J,
            )
            slots[i].next_model = m

        # Launch kernel
        self._sf_kernel_running = False
        self.launch_self_feeding(
            num_betas=num_betas,
            seed=seed,
            active_nonce_count=len(pending_models),
        )

        try:
            last_completion = time.monotonic()
            while True:
                # Check if any kernels still have work
                any_active = any(
                    s.active_model is not None
                    for s in slots
                )
                if not any_active:
                    break

                # Try to fill any empty next-slots before
                # polling, so the GPU doesn't stall waiting
                for nonce_id, ss in enumerate(slots):
                    if (
                        ss.active_model is not None
                        and ss.next_model is None
                        and ss.free_slot >= 0
                    ):
                        m = _pull_nonblocking()
                        if m is not None:
                            self.upload_slot(
                                nonce_id, ss.free_slot,
                                m.h, m.J,
                            )
                            ss.next_slot = ss.free_slot
                            ss.next_model = m
                            ss.free_slot = -1

                # Single DMA read of ctrl array
                ctrl = cp.asnumpy(self._d_sf_ctrl)
                found = False

                for nonce_id, ss in enumerate(slots):
                    if ss.active_model is None:
                        continue
                    base = nonce_id * self.CTRL_STRIDE
                    state = ctrl[base + ss.active_slot]
                    if state != self.SLOT_COMPLETE:
                        continue

                    found = True
                    last_completion = time.monotonic()

                    # Download completed results
                    result_ss = self.download_slot(
                        nonce_id, ss.active_slot,
                    )
                    completed_model = ss.active_model

                    # Rotate: active -> free,
                    # next -> active
                    ss.free_slot = ss.active_slot
                    ss.active_slot = ss.next_slot
                    ss.active_model = ss.next_model
                    ss.next_slot = -1
                    ss.next_model = None

                    # Non-blocking fill of freed slot
                    m = _pull_nonblocking()
                    if m is not None:
                        self.upload_slot(
                            nonce_id, ss.free_slot,
                            m.h, m.J,
                        )
                        ss.next_slot = ss.free_slot
                        ss.next_model = m
                        ss.free_slot = -1

                    yield (completed_model, result_ss)

                if not found:
                    if (
                        poll_timeout is not None
                        and time.monotonic()
                        - last_completion
                        > poll_timeout
                    ):
                        raise TimeoutError(
                            f"No completion after "
                            f"{poll_timeout}s"
                        )
                    time.sleep(0.001)

        finally:
            self.signal_exit(wait=False)

    def close(self) -> None:
        """Synchronize streams and free GPU buffers.

        Explicitly deletes device arrays so CuPy's memory
        pool can reclaim them immediately, rather than
        waiting for GC. Includes both SA-specific and
        Gibbs-specific attrs — hasattr guards handle
        whichever sampler type is active.
        """
        if not self._sf_prepared:
            return
        self._sf_stream_compute.synchronize()
        self._sf_stream_transfer.synchronize()
        self._sf_kernel_running = False
        self._sf_prepared = False

        for attr in (
            '_d_sf_J', '_d_sf_h', '_d_sf_samples',
            '_d_sf_energies', '_d_sf_ctrl', '_d_sf_beta',
            '_d_sf_profile', '_d_sf_delta_energy',
            '_d_sf_block_starts', '_d_sf_block_counts',
        ):
            if hasattr(self, attr):
                delattr(self, attr)
