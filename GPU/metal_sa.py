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

import ctypes
import logging
import os
import sys
import time
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import dimod
import numpy as np

try:
    import Metal
except ImportError:  # Apple Metal framework is macOS-only; absent on Linux/CI.
    Metal = None  # type: ignore[assignment]

from shared.ising_model import IsingModel
from GPU.metal_scheduler import DutyCycleController, MetalScheduler
from GPU.metal_utils import _create_buffer, compute_beta_schedule, unpack_metal_results

# Default per-command-buffer wall-clock budget (~one 120Hz/60Hz frame). Sizing
# each committed chunk to ~burst_ms gives the macOS compositor a GPU preemption
# seam every frame, which is the real responsiveness lever on Apple Silicon.
DEFAULT_BURST_MS = 8.0

# Betas dispatched in the first (calibration) chunk to seed the per-beta EMA.
_CALIB_BETAS = 2

# Seconds to idle between target re-checks while paused (target_pct == 0).
_PAUSE_POLL_S = 0.5

# EMA smoothing for the per-beta GPU wall-time estimate.
_BETA_EMA_ALPHA = 0.3

# pthread QoS class for the Metal sampler thread. QOS_CLASS_UTILITY (0x11, per
# <sys/qos.h>) deprioritizes us against the foreground UI on the P-cores.
# WARNING: 0x21 is USER_INTERACTIVE — the opposite of intended (a boost).
QOS_CLASS_UTILITY = 0x11

# Applied once per process (the stream-driver child); never raises.
_qos_applied = False
_qos_log = logging.getLogger(__name__)


def apply_qos_utility() -> bool:
    """Lower this thread's QoS to UTILITY (darwin only). Returns True if applied.

    Metal-only CPU-side politeness: reduces P-core contention with the UI. It
    does NOT bound GPU occupancy (a command buffer runs to completion
    regardless of submitter QoS) — it composes with the burst/duty cap. Run in
    the child process (spawn isolation means a parent call wouldn't carry over)
    and invoked from the Metal sampler only — never from the shared QPU stream
    driver. Idempotent and non-fatal.
    """
    global _qos_applied
    if _qos_applied or sys.platform != "darwin":
        return False
    try:
        libc = ctypes.CDLL(None)  # libSystem
        fn = libc.pthread_set_qos_class_self_np
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_int, ctypes.c_int]
        fn(QOS_CLASS_UTILITY, 0)
        _qos_applied = True
        return True
    except Exception as exc:  # noqa: BLE001 — non-fatal CPU-politeness hint
        _qos_log.debug("QoS clamp skipped: %s", exc)
        return False


def compute_betas_per_chunk(
    burst_ms: float,
    ema_per_beta_ms: float,
    total_betas: int,
) -> int:
    """Betas per command buffer to keep each chunk near ``burst_ms``.

    Args:
        burst_ms: Per-command-buffer wall-clock budget (ms).
        ema_per_beta_ms: Smoothed per-beta GPU wall time (ms). ``<= 0`` means
            "not yet calibrated" — dispatch the whole schedule.
        total_betas: Total betas in the schedule (the chunk upper bound).

    Returns:
        Chunk size in [1, total_betas].
    """
    if ema_per_beta_ms <= 0:
        return total_betas
    n = round(burst_ms / ema_per_beta_ms)
    return max(1, min(int(n), total_betas))


def _resolve_target_pct(scheduler, duty_cycle) -> int:
    """Resolve the per-batch cap %: scheduler (adaptive) > duty cycle > 100.

    The scheduler's ``get_target_pct`` is the live, sensor-driven cap. With no
    scheduler, fall back to the duty cycle's static target; with neither, run
    flat-out (100).
    """
    if scheduler is not None and hasattr(scheduler, "get_target_pct"):
        return int(scheduler.get_target_pct())
    if duty_cycle is not None:
        return int(duty_cycle.target_pct)
    return 100


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
        """Sample from Ising model using pure simulated annealing.

        Delegates to _dispatch_batch via IsingModel wrappers.

        Args:
            h: List of linear biases [{node: bias}, ...].
            J: List of quadratic biases [{(n1, n2): coupling}, ...].
            num_reads: Independent SA runs per problem.
            num_sweeps: Total sweeps (default 1000).
            num_sweeps_per_beta: Sweeps per beta value (default 1).
            beta_range: (hot, cold) or None for auto.
            beta_schedule_type: "linear", "geometric", or "custom".
            beta_schedule: Custom schedule (needs type="custom").
            seed: RNG seed.

        Returns:
            List of dimod.SampleSet per problem.
        """
        num_problems = len(h)
        if len(J) != num_problems:
            raise ValueError(
                f"h and J must have same length: "
                f"{num_problems} vs {len(J)}",
            )

        # Wrap raw h/J dicts as IsingModel objects so we can
        # delegate to the single _dispatch_batch code path.
        models = [
            IsingModel(h=h_i, J=J_i, nonce=0, salt=b"")
            for h_i, J_i in zip(h, J)
        ]

        # Ensure topology CSR cache is built
        self.prepare_topology()

        # Compute beta schedule
        beta_arr, beta_range_out = compute_beta_schedule(
            h[0], J[0],
            num_sweeps, num_sweeps_per_beta,
            beta_range, beta_schedule_type, beta_schedule,
        )

        if seed is None:
            seed = np.random.randint(0, 2**31)

        return self._dispatch_batch(
            models,
            num_reads=num_reads,
            beta_schedule_arr=beta_arr,
            beta_range=beta_range_out,
            beta_schedule_type=beta_schedule_type,
            num_sweeps_per_beta=num_sweeps_per_beta,
            seed=seed,
        )

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

        # Persistent buffers for chunked dispatch (kernel always
        # writes these; monolithic dispatch just ignores them)
        persist_state_buf = self.device.newBufferWithLength_options_(
            max(1, num_threads * packed_size),
            Metal.MTLResourceStorageModeShared,
        )
        persist_de_buf = self.device.newBufferWithLength_options_(
            max(1, num_threads * N),
            Metal.MTLResourceStorageModeShared,
        )
        persist_rng_buf = self.device.newBufferWithLength_options_(
            max(1, num_threads * 4),
            Metal.MTLResourceStorageModeShared,
        )
        persist_energy_buf = self.device.newBufferWithLength_options_(
            max(1, num_threads * 4),
            Metal.MTLResourceStorageModeShared,
        )

        total_betas = len(beta_schedule_arr)
        beta_start_bytes = np.int32(0).tobytes()
        beta_count_bytes = np.int32(total_betas).tobytes()

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

        encoder.setBytes_length_atIndex_(beta_start_bytes, 4, 16)
        encoder.setBytes_length_atIndex_(beta_count_bytes, 4, 17)
        encoder.setBuffer_offset_atIndex_(
            persist_state_buf, 0, 18,
        )
        encoder.setBuffer_offset_atIndex_(
            persist_de_buf, 0, 19,
        )
        encoder.setBuffer_offset_atIndex_(
            persist_rng_buf, 0, 20,
        )
        encoder.setBuffer_offset_atIndex_(
            persist_energy_buf, 0, 21,
        )

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

    def _dispatch_batch_chunked(
        self,
        models: List[IsingModel],
        *,
        num_reads: int,
        beta_schedule_arr: np.ndarray,
        beta_range: Tuple[float, float],
        beta_schedule_type: str,
        num_sweeps_per_beta: int,
        seed: int,
        burst_ms: float = DEFAULT_BURST_MS,
        duty_cycle: Optional[DutyCycleController] = None,
        scheduler: Optional['MetalScheduler'] = None,
    ) -> List[dimod.SampleSet]:
        """Dispatch a batch in adaptively-sized beta-schedule chunks.

        The first (calibration) chunk dispatches ``_CALIB_BETAS`` betas to seed
        a per-beta GPU-wall-time EMA; every subsequent chunk is sized via
        ``compute_betas_per_chunk`` so each committed command buffer runs
        ≈ ``burst_ms`` — bounding burst length (the real jank lever) and giving
        the compositor a preemption seam every frame. Between chunks the GPU is
        released and a duty-cycle sleep paces the average occupancy.

        State is persisted between chunks via device buffers, so the result is
        identical to a monolithic dispatch given the same seed (chunk-size
        changes are safe: the kernel takes beta_start/beta_count).
        """
        num_problems = len(models)
        N = self._topo_N
        node_to_idx = self._topo_node_to_idx

        (
            all_row_ptr, all_col_ind, all_J_vals,
            all_h_vals, row_ptr_offsets, col_ind_offsets,
        ) = self._fill_batch_values(models)

        # Topology buffers (shared across all chunks)
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

        # Scalar bytes (shared across chunks)
        N_bytes = np.int32(N).tobytes()
        total_betas = len(beta_schedule_arr)
        num_betas_bytes = np.int32(total_betas).tobytes()
        spb_bytes = np.int32(num_sweeps_per_beta).tobytes()
        seed_bytes = np.uint32(seed).tobytes()

        num_threads = num_problems * num_reads
        nt_bytes = np.int32(num_threads).tobytes()
        np_bytes = np.int32(num_problems).tobytes()
        nr_bytes = np.int32(num_reads).tobytes()

        packed_size = (N + 7) // 8

        # Output buffers (read only after last chunk)
        samples_buf = self.device.newBufferWithLength_options_(
            num_threads * packed_size,
            Metal.MTLResourceStorageModeShared,
        )
        energies_buf = self.device.newBufferWithLength_options_(
            num_threads * 4,
            Metal.MTLResourceStorageModeShared,
        )

        # Persistent state buffers (read/written every chunk)
        persist_state_buf = (
            self.device.newBufferWithLength_options_(
                num_threads * packed_size,
                Metal.MTLResourceStorageModeShared,
            )
        )
        persist_de_buf = (
            self.device.newBufferWithLength_options_(
                num_threads * N,
                Metal.MTLResourceStorageModeShared,
            )
        )
        persist_rng_buf = (
            self.device.newBufferWithLength_options_(
                num_threads * 4,
                Metal.MTLResourceStorageModeShared,
            )
        )
        persist_energy_buf = (
            self.device.newBufferWithLength_options_(
                num_threads * 4,
                Metal.MTLResourceStorageModeShared,
            )
        )

        tg = Metal.MTLSize(width=num_problems, height=1, depth=1)
        tpt = Metal.MTLSize(width=num_reads, height=1, depth=1)

        # Dispatch adaptively-sized beta chunks. The first chunk calibrates the
        # per-beta wall-time EMA; later chunks target burst_ms.
        ema_per_beta_ms = 0.0
        chunk_start = 0
        chunk_idx = 0
        while chunk_start < total_betas:
            if ema_per_beta_ms <= 0:
                chunk_count = min(_CALIB_BETAS, total_betas - chunk_start)
            else:
                chunk_count = min(
                    compute_betas_per_chunk(
                        burst_ms, ema_per_beta_ms, total_betas,
                    ),
                    total_betas - chunk_start,
                )
            bs_bytes = np.int32(chunk_start).tobytes()
            bc_bytes = np.int32(chunk_count).tobytes()

            t0 = time.perf_counter()

            cmd_buf = self._command_queue.commandBuffer()
            encoder = cmd_buf.computeCommandEncoder()
            encoder.setComputePipelineState_(self._pipeline)

            encoder.setBuffer_offset_atIndex_(rp_buf, 0, 0)
            encoder.setBuffer_offset_atIndex_(ci_buf, 0, 1)
            encoder.setBuffer_offset_atIndex_(jv_buf, 0, 2)
            encoder.setBuffer_offset_atIndex_(rpo_buf, 0, 3)
            encoder.setBuffer_offset_atIndex_(cio_buf, 0, 4)

            encoder.setBytes_length_atIndex_(N_bytes, 4, 5)
            encoder.setBytes_length_atIndex_(
                num_betas_bytes, 4, 6,
            )
            encoder.setBytes_length_atIndex_(spb_bytes, 4, 7)
            encoder.setBytes_length_atIndex_(seed_bytes, 4, 8)

            encoder.setBuffer_offset_atIndex_(beta_buf, 0, 9)
            encoder.setBuffer_offset_atIndex_(
                samples_buf, 0, 10,
            )
            encoder.setBuffer_offset_atIndex_(
                energies_buf, 0, 11,
            )

            encoder.setBytes_length_atIndex_(nt_bytes, 4, 12)
            encoder.setBytes_length_atIndex_(np_bytes, 4, 13)
            encoder.setBytes_length_atIndex_(nr_bytes, 4, 14)

            encoder.setBuffer_offset_atIndex_(hv_buf, 0, 15)

            encoder.setBytes_length_atIndex_(bs_bytes, 4, 16)
            encoder.setBytes_length_atIndex_(bc_bytes, 4, 17)
            encoder.setBuffer_offset_atIndex_(
                persist_state_buf, 0, 18,
            )
            encoder.setBuffer_offset_atIndex_(
                persist_de_buf, 0, 19,
            )
            encoder.setBuffer_offset_atIndex_(
                persist_rng_buf, 0, 20,
            )
            encoder.setBuffer_offset_atIndex_(
                persist_energy_buf, 0, 21,
            )

            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                tg, tpt,
            )

            encoder.endEncoding()
            cmd_buf.commit()
            cmd_buf.waitUntilCompleted()

            if cmd_buf.status() != (
                Metal.MTLCommandBufferStatusCompleted
            ):
                error = cmd_buf.error()
                raise RuntimeError(
                    f"Metal command buffer failed "
                    f"(chunk {chunk_start}): {error}",
                )

            # Update the per-beta wall-time EMA from this chunk (drives the
            # next chunk's size toward burst_ms).
            compute_s = time.perf_counter() - t0
            per_beta_ms = (compute_s * 1000.0) / max(1, chunk_count)
            if ema_per_beta_ms <= 0:
                ema_per_beta_ms = per_beta_ms
            else:
                ema_per_beta_ms = (
                    _BETA_EMA_ALPHA * per_beta_ms
                    + (1.0 - _BETA_EMA_ALPHA) * ema_per_beta_ms
                )

            # Duty-cycle sleep between chunks (paces average occupancy).
            if duty_cycle and duty_cycle.enabled:
                sleep_s = duty_cycle.compute_sleep(compute_s)
                self.logger.debug(
                    "[duty-cycle] chunk %d betas=%d "
                    "compute=%.1fms sleep=%.1fms "
                    "per_beta=%.2fms ema_compute=%.1fms mult=%.2f",
                    chunk_idx, chunk_count,
                    compute_s * 1000, sleep_s * 1000,
                    ema_per_beta_ms,
                    duty_cycle._ema_compute_s * 1000,
                    duty_cycle._duty_multiplier,
                )
                time.sleep(sleep_s)

                # Closed-loop trim: nudge the duty multiplier toward target.
                if scheduler is not None:
                    measured = scheduler.get_cached_utilization()
                    duty_cycle.feedback(measured)
                    self.logger.debug(
                        "[duty-cycle] measured=%d%%", measured,
                    )

            chunk_start += chunk_count
            chunk_idx += 1

        # Unpack results from final chunk
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
        duty_cycle: Optional[DutyCycleController] = None,
        scheduler: Optional['MetalScheduler'] = None,
        burst_ms: float = DEFAULT_BURST_MS,
        **kwargs,
    ) -> Iterator[Tuple[IsingModel, dimod.SampleSet]]:
        """Stream batched results using cached topology structure.

        On first call, prepares the topology CSR structure once. Each batch
        then only fills J/h values into preallocated positions. The cap
        ``target_pct`` is re-read **per batch** (from ``scheduler`` when
        present, else the duty cycle's static target) to pick the dispatch
        path: 100 → monolithic (max throughput), 0 → pause (idle and re-check),
        0<target<100 → adaptive chunked dispatch sized to ``burst_ms``.

        Args:
            models: Iterable of IsingModel (typically a RandomIsingFeeder).
            num_reads: SA reads per problem.
            num_sweeps: Total sweeps per run.
            max_threadgroups: Max problems per batch dispatch.
            num_sweeps_per_beta: Sweeps per beta value.
            beta_range: Temperature range or None for auto.
            beta_schedule_type: "linear", "geometric", or "custom".
            seed: Base RNG seed (incremented per batch).
            duty_cycle: Optional controller for GPU duty cycling.
            scheduler: Optional MetalScheduler — the per-batch cap source.
            burst_ms: Per-command-buffer wall-clock budget for chunked dispatch.

        Yields:
            (IsingModel, dimod.SampleSet) for each completed problem.
        """
        # CPU-side politeness: lower this (stream-driver child) thread's QoS so
        # we yield P-cores to the foreground UI. Metal-only; once per process.
        apply_qos_utility()
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
            # Re-read the cap before consuming the feeder so a PAUSE does not
            # burn queued models.
            target = _resolve_target_pct(scheduler, duty_cycle)
            if target <= 0:
                # PAUSE: idle and re-check; do not dispatch or pull models.
                time.sleep(_PAUSE_POLL_S)
                continue

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

            if target >= 100:
                samplesets = self._dispatch_batch(
                    batch_models,
                    num_reads=num_reads,
                    beta_schedule_arr=beta_arr,
                    beta_range=beta_range_out,
                    beta_schedule_type=beta_schedule_type,
                    num_sweeps_per_beta=num_sweeps_per_beta,
                    seed=batch_seed,
                )
                # Minimal yield for WindowServer compositor on
                # Apple Silicon's shared GPU (~2% overhead).
                time.sleep(0.002)
            else:
                # Throttled: adaptive chunked dispatch. Retarget the duty cycle
                # in lock-step (resets EMA + PI integral together) so a tier
                # change doesn't carry windup into the new cap.
                if duty_cycle is not None:
                    duty_cycle.set_target(target)
                samplesets = self._dispatch_batch_chunked(
                    batch_models,
                    num_reads=num_reads,
                    beta_schedule_arr=beta_arr,
                    beta_range=beta_range_out,
                    beta_schedule_type=beta_schedule_type,
                    num_sweeps_per_beta=num_sweeps_per_beta,
                    seed=batch_seed,
                    burst_ms=burst_ms,
                    duty_cycle=duty_cycle,
                    scheduler=scheduler,
                )

            batch_seed = (batch_seed + 1) & 0x7FFFFFFF

            for model, ss in zip(batch_models, samplesets):
                yield (model, ss)

