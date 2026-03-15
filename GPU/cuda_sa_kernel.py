# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Per-job CUDA SA kernel launcher with multi-nonce support.

Single-nonce: build CSR, launch kernel, return results.
Multi-nonce: prepare() pre-allocates double-buffered GPU arrays,
then sample_multi_nonce() launches one block per nonce.
"""

import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import cupy as cp
import dimod
import numpy as np

from GPU.sampler_utils import (
    build_csr_structure_from_edges,
    build_edge_position_index,
    default_ising_beta_range,
)
from shared.beta_schedule import _default_ising_beta_range


class CudaSAKernel:
    """Per-job CUDA simulated annealing kernel.

    Single-nonce: sample_ising() launches a fresh kernel per call.
    Multi-nonce: prepare() + sample_multi_nonce() for batched
    dispatch with double-buffered GPU arrays.

    Args:
        max_N: Maximum problem size (nodes). Default 5000.
    """

    def __init__(self, max_N: int = 5000):
        self.logger = logging.getLogger(__name__)
        self.max_N = max_N

        # Compile kernel
        kernel_path = os.path.join(
            os.path.dirname(__file__), 'cuda_sa.cu'
        )
        with open(kernel_path, 'r') as f:
            kernel_code = f.read()

        self._module = cp.RawModule(
            code=kernel_code,
            options=('--use_fast_math',),
        )
        self._kernel = self._module.get_function('cuda_sa_oneshot')
        self._multi_kernel = self._module.get_function(
            'cuda_sa_multi'
        )
        self._sf_kernel = self._module.get_function(
            'cuda_sa_self_feeding'
        )

        # Query device
        dev = cp.cuda.Device()
        self.device_sms = dev.attributes['MultiProcessorCount']
        self.logger.debug(
            f"CudaSAKernel compiled, device has "
            f"{self.device_sms} SMs"
        )

        self._prepared = False
        self._sf_prepared = False
        self._sf_kernel_running = False

    def prepare(
        self,
        nodes: List[int],
        edges: List[Tuple[int, int]],
        num_reads: int = 32,
        max_num_betas: int = 400,
        max_nonces: int = 1,
    ) -> None:
        """Pre-allocate GPU buffers for multi-nonce SA.

        Args:
            nodes: Topology nodes.
            edges: Topology edges.
            num_reads: Max reads per nonce.
            max_num_betas: Max temperature steps.
            max_nonces: Max nonces per batch.
        """
        # Build CSR structure from topology
        (csr_row_ptr, csr_col_ind, node_to_idx,
         sorted_neighbors, N, nnz) = (
            build_csr_structure_from_edges(edges, nodes)
        )

        # Edge position index for fast J updates
        edge_positions = build_edge_position_index(
            edges, node_to_idx, csr_row_ptr,
            sorted_neighbors,
        )

        self._prep_N = N
        self._prep_nnz = nnz
        self._prep_node_to_idx = node_to_idx
        self._prep_edge_positions = edge_positions
        self._prep_num_reads = num_reads
        self._prep_max_num_betas = max_num_betas
        self._max_nonces = max_nonces

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

        # Upload constant GPU buffers (topology, shared)
        self._d_row_ptr = cp.asarray(csr_row_ptr)
        self._d_col_ind = cp.asarray(csr_col_ind)

        # Per-nonce offset arrays (topology is shared, offsets
        # are just multiples of nnz/N)
        self._d_j_offsets = cp.asarray(
            np.arange(max_nonces, dtype=np.int32) * nnz,
        )
        self._d_h_offsets = cp.asarray(
            np.arange(max_nonces, dtype=np.int32) * N,
        )

        # Double-buffered mutable GPU arrays [A, B]
        total_reads = max_nonces * num_reads
        self._d_J = [
            cp.zeros(max_nonces * nnz, dtype=cp.int8),
            cp.zeros(max_nonces * nnz, dtype=cp.int8),
        ]
        self._d_h = [
            cp.zeros(max_nonces * N, dtype=cp.float32),
            cp.zeros(max_nonces * N, dtype=cp.float32),
        ]
        self._d_beta = [
            cp.zeros(max_num_betas, dtype=cp.float32),
            cp.zeros(max_num_betas, dtype=cp.float32),
        ]
        self._d_delta_ws = [
            cp.zeros(total_reads * N, dtype=cp.int8),
            cp.zeros(total_reads * N, dtype=cp.int8),
        ]
        self._d_samples = [
            cp.zeros(total_reads * N, dtype=cp.float32),
            cp.zeros(total_reads * N, dtype=cp.float32),
        ]
        self._d_energies = [
            cp.zeros(total_reads, dtype=cp.float32),
            cp.zeros(total_reads, dtype=cp.float32),
        ]

        # Host staging buffers [A, B]
        self._h_J = [
            np.zeros(max_nonces * nnz, dtype=np.int8),
            np.zeros(max_nonces * nnz, dtype=np.int8),
        ]
        self._h_h = [
            np.zeros(max_nonces * N, dtype=np.float32),
            np.zeros(max_nonces * N, dtype=np.float32),
        ]

        # CUDA streams for pipeline overlap
        self._stream_compute = cp.cuda.Stream(non_blocking=True)
        self._stream_transfer = cp.cuda.Stream(non_blocking=True)
        self._event_transfer_done = cp.cuda.Event()

        # Pipeline state
        self._buf_idx = 0
        self._preloaded = False
        self._preload_meta = None
        self._pending = None

        # Cache beta schedule
        self._cached_beta_key = None
        self._cached_beta_sched = None
        self._cached_beta_range = None

        self._prepared = True
        self.logger.info(
            f"Prepared SA buffers: N={N}, nnz={nnz}, "
            f"num_reads={num_reads}, "
            f"max_betas={max_num_betas}, "
            f"max_nonces={max_nonces}"
        )

    def close(self) -> None:
        """Synchronize and release CUDA streams/events."""
        if not self._prepared:
            return
        if self._sf_kernel_running:
            self.signal_exit()
        self._stream_compute.synchronize()
        self._stream_transfer.synchronize()
        self._preloaded = False
        self._prepared = False

    # ── Self-feeding kernel interface ────────────────────────
    # Control layout per nonce (CTRL_STRIDE=8 ints):
    #   [0..2] = slot states, [6] = exit_now
    CTRL_STRIDE = 8
    SLOT_EMPTY = 0
    SLOT_READY = 1
    SLOT_COMPLETE = 3

    def prepare_self_feeding(
        self,
        num_nonces: int,
        num_reads: int = 32,
        num_betas: int = 400,
        num_sweeps_per_beta: int = 1,
    ) -> None:
        """Allocate 3-slot rotating buffers for self-feeding.

        Args:
            num_nonces: Number of concurrent nonce groups.
            num_reads: Reads per nonce per model.
            num_betas: Max beta schedule entries.
            num_sweeps_per_beta: Sweeps per beta value.
        """
        assert self._prepared, "Must call prepare() first"

        N = self._prep_N
        nnz = self._prep_nnz
        total_slots = num_nonces * 3

        self._sf_num_nonces = num_nonces
        self._sf_num_reads = min(num_reads, 256)
        self._sf_num_sweeps_per_beta = num_sweeps_per_beta
        self._sf_max_num_betas = num_betas

        # Flat buffer allocations (stride math)
        # J: int8 (coupling values)
        self._d_sf_J = cp.zeros(
            total_slots * nnz, dtype=cp.int8,
        )
        # h: float32 (linear biases — SA uses float h)
        self._d_sf_h = cp.zeros(
            total_slots * N, dtype=cp.float32,
        )
        # Samples: float32 (unpacked spins)
        self._d_sf_samples = cp.zeros(
            total_slots * self._sf_num_reads * N,
            dtype=cp.float32,
        )
        # Energies: float32
        self._d_sf_energies = cp.zeros(
            total_slots * self._sf_num_reads,
            dtype=cp.float32,
        )

        # Per-nonce control array
        self._d_sf_ctrl = cp.zeros(
            num_nonces * self.CTRL_STRIDE, dtype=cp.int32,
        )

        # Delta energy workspace (1 per thread)
        total_threads = num_nonces * self._sf_num_reads
        self._d_sf_delta_ws = cp.zeros(
            total_threads * N, dtype=cp.int8,
        )

        # Host staging buffers (reused per upload)
        self._h_sf_J = np.zeros(nnz, dtype=np.int8)
        self._h_sf_h = np.zeros(N, dtype=np.float32)

        # Shared beta schedule buffer
        self._d_sf_beta = cp.zeros(
            num_betas, dtype=cp.float32,
        )

        # Dedicated streams
        self._sf_stream_compute = cp.cuda.Stream(
            non_blocking=True,
        )
        self._sf_stream_transfer = cp.cuda.Stream(
            non_blocking=True,
        )

        self._sf_kernel_running = False
        self._sf_prepared = True

        self.logger.info(
            "SA self-feeding prepared: %d nonces × 3 slots, "
            "%d reads/nonce",
            num_nonces, self._sf_num_reads,
        )

    def upload_slot(
        self,
        nonce_id: int,
        slot_id: int,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
    ) -> None:
        """Upload model data to a slot, mark READY.

        Args:
            nonce_id: Nonce group index.
            slot_id: Slot within nonce (0, 1, or 2).
            h: Linear biases.
            J: Quadratic biases.
        """
        assert self._sf_prepared
        N = self._prep_N
        nnz = self._prep_nnz
        reads = self._sf_num_reads
        slot_idx = nonce_id * 3 + slot_id

        # Fill host staging — J values
        j_vals = np.fromiter(
            J.values(), dtype=np.int8, count=len(J),
        )
        self._h_sf_J[:] = 0
        self._h_sf_J[self._pos_ij] = j_vals
        self._h_sf_J[self._pos_ji] = j_vals

        # Fill host staging — h values (float32)
        self._h_sf_h[:] = 0.0
        for node, val in h.items():
            idx = self._prep_node_to_idx.get(node)
            if idx is not None:
                self._h_sf_h[idx] = float(val)

        # Async H2D on transfer stream
        j_start = slot_idx * nnz
        h_start = slot_idx * N
        sample_start = slot_idx * reads * N
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
                sample_start:sample_start + reads * N
            ] = 0
            self._d_sf_energies[
                energy_start:energy_start + reads
            ] = 0

        # Mark slot READY
        ctrl_offset = nonce_id * self.CTRL_STRIDE + slot_id
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
        num_betas: int,
        beta_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[int, Tuple[float, float]]:
        """Upload beta schedule to the shared device buffer.

        Returns:
            (num_betas, beta_range) tuple.
        """
        sched, beta_range = self._get_cached_beta_schedule(
            h_first, J_first, num_betas, beta_range,
        )
        actual_betas = len(sched)
        self._d_sf_beta[:actual_betas].set(sched)
        self._sf_beta_range = beta_range
        return actual_betas, beta_range

    def launch_self_feeding(
        self,
        num_betas: int,
        seed: Optional[int] = None,
        active_nonce_count: Optional[int] = None,
    ) -> None:
        """Launch self-feeding kernel (stays resident).

        Args:
            num_betas: Number of beta schedule entries.
            seed: RNG base seed.
            active_nonce_count: Launch blocks for this many
                nonces (default: all prepared nonces).
        """
        assert self._sf_prepared
        assert not self._sf_kernel_running

        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        N = self._prep_N
        nnz = self._prep_nnz
        num_reads = self._sf_num_reads
        num_nonces = self._sf_num_nonces
        active = (
            active_nonce_count
            if active_nonce_count is not None
            else num_nonces
        )

        # 1 block per nonce, num_reads threads per block
        grid = (active,)
        block = (num_reads,)

        kernel_args = (
            self._d_row_ptr,
            self._d_col_ind,
            self._d_sf_J,
            self._d_sf_h,
            self._d_sf_beta,
            np.int32(N),
            np.int32(num_reads),
            np.int32(num_betas),
            np.int32(self._sf_num_sweeps_per_beta),
            np.uint32(seed),
            self._d_sf_delta_ws,
            self._d_sf_samples,
            self._d_sf_energies,
            self._d_sf_ctrl,
            np.int32(num_nonces),
            np.int32(nnz),
        )

        with self._sf_stream_compute:
            self._sf_kernel(grid, block, kernel_args)

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

    def download_slot(
        self,
        nonce_id: int,
        slot_id: int,
    ) -> dimod.SampleSet:
        """Download results from a COMPLETE slot.

        SA outputs are float32 unpacked spins — no bit
        unpacking needed (unlike Gibbs).

        Args:
            nonce_id: Nonce group index.
            slot_id: Slot within nonce.

        Returns:
            dimod.SampleSet with the slot's results.
        """
        N = self._prep_N
        reads = self._sf_num_reads
        slot_idx = nonce_id * 3 + slot_id

        sample_start = slot_idx * reads * N
        energy_start = slot_idx * reads

        samples_raw = cp.asnumpy(
            self._d_sf_samples[
                sample_start:sample_start + reads * N
            ],
        ).reshape(reads, N)

        energies_raw = cp.asnumpy(
            self._d_sf_energies[
                energy_start:energy_start + reads
            ],
        )

        return dimod.SampleSet.from_samples(
            samples_raw.astype(np.int8),
            vartype='SPIN',
            energy=energies_raw,
            info={
                'beta_range': getattr(
                    self, '_sf_beta_range', None,
                ),
                'min_energy': float(energies_raw.min()),
            },
        )

    def signal_nonce_exit(self, nonce_id: int) -> None:
        """Signal one nonce group to exit.

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
        """Signal all nonces to exit, wait for kernel."""
        assert self._sf_prepared
        for n in range(self._sf_num_nonces):
            self.signal_nonce_exit(n)
        self._sf_stream_compute.synchronize()
        self._sf_kernel_running = False

    def is_kernel_running(self) -> bool:
        """Check if the self-feeding kernel is still running."""
        if not self._sf_kernel_running:
            return False
        done = self._sf_stream_compute.done
        if done:
            self._sf_kernel_running = False
        return self._sf_kernel_running

    def _get_cached_beta_schedule(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        num_betas: int,
        beta_range: Optional[Tuple[float, float]],
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Return cached beta schedule if params match."""
        key = (num_betas, beta_range)
        if self._cached_beta_key == key:
            return self._cached_beta_sched, self._cached_beta_range

        if beta_range is None:
            beta_range = default_ising_beta_range(h, J)
        hot_beta, cold_beta = beta_range
        if num_betas == 1:
            sched = np.array([cold_beta], dtype=np.float32)
        else:
            sched = np.geomspace(
                hot_beta, cold_beta, num=num_betas,
                dtype=np.float32,
            )
        self._cached_beta_key = key
        self._cached_beta_sched = sched
        self._cached_beta_range = beta_range
        return sched, beta_range

    def _fill_staging(
        self,
        h_list: List[Dict[int, float]],
        J_list: List[Dict[Tuple[int, int], float]],
        buf: int,
    ) -> None:
        """Pack h/J into host staging buffer `buf` (0 or 1)."""
        N = self._prep_N
        nnz = self._prep_nnz
        num_nonces = len(h_list)

        h_J = self._h_J[buf]
        h_h = self._h_h[buf]
        h_J[:num_nonces * nnz] = 0
        h_h[:num_nonces * N] = 0

        for k in range(num_nonces):
            j_vals = np.fromiter(
                J_list[k].values(), dtype=np.int8,
                count=len(J_list[k]),
            )
            j_off = k * nnz
            h_J[j_off + self._pos_ij] = j_vals
            h_J[j_off + self._pos_ji] = j_vals

            h_vals = np.zeros(N, dtype=np.float32)
            for node, val in h_list[k].items():
                idx = self._prep_node_to_idx.get(node)
                if idx is not None:
                    h_vals[idx] = float(val)
            h_off = k * N
            h_h[h_off:h_off + N] = h_vals

    def preload_multi_nonce(
        self,
        h_list: List[Dict[int, float]],
        J_list: List[Dict[Tuple[int, int], float]],
        num_reads: int = 32,
        num_betas: int = 50,
        num_sweeps_per_beta: int = 1,
        seed: Optional[int] = None,
        beta_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Async H2D copy of next multi-nonce SA batch."""
        assert self._prepared, "Must call prepare() first"
        num_nonces = len(h_list)
        assert num_nonces <= self._max_nonces

        next_buf = 1 - self._buf_idx

        # Fill host staging
        self._fill_staging(h_list, J_list, next_buf)

        N = self._prep_N
        nnz = self._prep_nnz
        num_reads = min(num_reads, self._prep_num_reads)

        # Beta schedule
        sched, beta_range = self._get_cached_beta_schedule(
            h_list[0], J_list[0], num_betas, beta_range,
        )

        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        # Async H2D on transfer stream
        with self._stream_transfer:
            self._d_J[next_buf][
                :num_nonces * nnz
            ].set(
                self._h_J[next_buf][:num_nonces * nnz],
            )
            self._d_h[next_buf][
                :num_nonces * N
            ].set(
                self._h_h[next_buf][:num_nonces * N],
            )
            self._d_beta[next_buf][:num_betas].set(sched)
        self._event_transfer_done.record(self._stream_transfer)

        self._preloaded = True
        self._preload_meta = {
            'num_nonces': num_nonces,
            'num_reads': num_reads,
            'num_betas': num_betas,
            'num_sweeps_per_beta': num_sweeps_per_beta,
            'seed': seed,
            'beta_range': beta_range,
            'buf': next_buf,
        }

    def launch_multi_nonce(
        self,
        h_list: List[Dict[int, float]],
        J_list: List[Dict[Tuple[int, int], float]],
        num_reads: int = 32,
        num_betas: int = 50,
        num_sweeps_per_beta: int = 1,
        seed: Optional[int] = None,
        beta_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Launch SA kernel without synchronizing.

        Call harvest_multi_nonce() to sync and get results.
        Do CPU work between launch and harvest for overlap.
        """
        assert self._prepared, "Must call prepare() first"

        N = self._prep_N
        nnz = self._prep_nnz

        if self._preloaded:
            meta = self._preload_meta
            idx = meta['buf']
            num_nonces = meta['num_nonces']
            num_reads = min(
                meta['num_reads'], self._prep_num_reads,
            )
            num_betas = meta['num_betas']
            num_sweeps_per_beta = meta['num_sweeps_per_beta']
            seed = meta['seed']
            beta_range = meta['beta_range']

            self._stream_compute.wait_event(
                self._event_transfer_done,
            )
            self._buf_idx = idx
            self._preloaded = False
            self._preload_meta = None
        else:
            num_nonces = len(h_list)
            assert num_nonces <= self._max_nonces, (
                f"num_nonces={num_nonces} > max_nonces="
                f"{self._max_nonces}"
            )
            num_reads = min(num_reads, self._prep_num_reads)
            idx = self._buf_idx

            sched, beta_range = self._get_cached_beta_schedule(
                h_list[0], J_list[0], num_betas, beta_range,
            )

            if seed is None:
                seed = random.randint(0, 2**31 - 1)

            self._fill_staging(h_list, J_list, idx)

            with self._stream_compute:
                self._d_J[idx][
                    :num_nonces * nnz
                ].set(
                    self._h_J[idx][:num_nonces * nnz],
                )
                self._d_h[idx][
                    :num_nonces * N
                ].set(
                    self._h_h[idx][:num_nonces * N],
                )
                self._d_beta[idx][:num_betas].set(sched)

        total_reads = num_nonces * num_reads

        # Launch: 1 block per nonce, num_reads threads per block
        grid = (num_nonces,)
        block = (num_reads,)
        kernel_args = (
            self._d_row_ptr, self._d_col_ind,
            self._d_J[idx], self._d_h[idx],
            self._d_j_offsets[:num_nonces],
            self._d_h_offsets[:num_nonces],
            self._d_beta[idx],
            np.int32(N),
            np.int32(num_reads),
            np.int32(num_betas),
            np.int32(num_sweeps_per_beta),
            np.uint32(seed),
            self._d_delta_ws[idx],
            self._d_samples[idx],
            self._d_energies[idx],
        )
        with self._stream_compute:
            self._multi_kernel(grid, block, kernel_args)

        # Store metadata for harvest (no sync yet)
        self._pending = {
            'idx': idx,
            'num_nonces': num_nonces,
            'num_reads': num_reads,
            'num_betas': num_betas,
            'beta_range': beta_range,
            'total_reads': total_reads,
        }

    def harvest_multi_nonce(self) -> List[dimod.SampleSet]:
        """Synchronize GPU and return results from last launch.

        Returns:
            List of dimod.SampleSet, one per nonce.
        """
        assert self._pending is not None, (
            "No pending launch — call launch_multi_nonce() first"
        )
        N = self._prep_N
        p = self._pending
        idx = p['idx']
        num_nonces = p['num_nonces']
        num_reads = p['num_reads']
        num_betas = p['num_betas']
        beta_range = p['beta_range']
        total_reads = p['total_reads']
        self._pending = None

        self._stream_compute.synchronize()

        # Read results
        samples_raw = cp.asnumpy(
            self._d_samples[idx][:total_reads * N],
        ).reshape(total_reads, N)
        energies_raw = cp.asnumpy(
            self._d_energies[idx][:total_reads],
        )

        # Build per-nonce SampleSets
        results = []
        for k in range(num_nonces):
            start = k * num_reads
            end = (k + 1) * num_reads
            samples_k = samples_raw[start:end].astype(np.int8)
            energies_k = energies_raw[start:end]
            ss = dimod.SampleSet.from_samples(
                samples_k,
                vartype='SPIN',
                energy=energies_k,
                info={
                    'beta_range': beta_range,
                    'num_betas': num_betas,
                    'min_energy': float(energies_k.min()),
                },
            )
            results.append(ss)

        return results

    def sample_multi_nonce(
        self,
        h_list: List[Dict[int, float]],
        J_list: List[Dict[Tuple[int, int], float]],
        num_reads: int = 32,
        num_betas: int = 50,
        num_sweeps_per_beta: int = 1,
        seed: Optional[int] = None,
        beta_range: Optional[Tuple[float, float]] = None,
    ) -> List[dimod.SampleSet]:
        """Launch + harvest (blocking convenience wrapper)."""
        self.launch_multi_nonce(
            h_list, J_list,
            num_reads=num_reads,
            num_betas=num_betas,
            num_sweeps_per_beta=num_sweeps_per_beta,
            seed=seed,
            beta_range=beta_range,
        )
        return self.harvest_multi_nonce()

    def sample_ising(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        num_reads: int = 100,
        num_betas: int = 50,
        num_sweeps_per_beta: int = 1,
        seed: Optional[int] = None,
        beta_range: Optional[Tuple[float, float]] = None,
    ) -> dimod.SampleSet:
        """Run SA on a single Ising problem.

        Args:
            h: Linear biases {node: bias}.
            J: Quadratic couplings {(i, j): coupling}.
            num_reads: Number of independent SA runs.
            num_betas: Number of temperature steps.
            num_sweeps_per_beta: Sweeps per temperature step.
            seed: RNG seed.
            beta_range: (hot_beta, cold_beta) or None for auto.

        Returns:
            dimod.SampleSet with num_reads samples.
        """
        num_reads = min(num_reads, 256)

        # Determine N from h and J
        max_node = 0
        if h:
            max_node = max(max_node, max(h.keys()))
        if J:
            max_node = max(
                max_node,
                max(max(i, j) for i, j in J.keys()),
            )
        N = max_node + 1

        # Build symmetric CSR
        csr_row_ptr, csr_col_ind, csr_J_vals = self._build_csr(
            J, N
        )

        # Build h array
        h_arr = np.zeros(N, dtype=np.float32)
        for node, val in h.items():
            if node < N:
                h_arr[node] = float(val)

        # Beta schedule
        if beta_range is None:
            beta_range = _default_ising_beta_range(h, J)
        hot_beta, cold_beta = beta_range
        if num_betas == 1:
            beta_sched = np.array(
                [cold_beta], dtype=np.float32,
            )
        else:
            beta_sched = np.geomspace(
                hot_beta, cold_beta, num=num_betas,
                dtype=np.float32,
            )

        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        # Transfer to GPU
        d_row_ptr = cp.asarray(csr_row_ptr, dtype=cp.int32)
        d_col_ind = cp.asarray(csr_col_ind, dtype=cp.int32)
        d_J_vals = cp.asarray(csr_J_vals, dtype=cp.int8)
        d_h = cp.asarray(h_arr, dtype=cp.float32)
        d_beta = cp.asarray(beta_sched, dtype=cp.float32)

        # Workspace and output buffers
        d_delta_ws = cp.zeros(
            num_reads * N, dtype=cp.int8
        )
        d_samples = cp.zeros(
            num_reads * N, dtype=cp.float32
        )
        d_energies = cp.zeros(num_reads, dtype=cp.float32)

        # Launch: 1 block, num_reads threads
        grid = (1,)
        block = (num_reads,)
        self._kernel(
            grid, block, (
                d_row_ptr, d_col_ind, d_J_vals, d_h,
                d_beta,
                np.int32(N),
                np.int32(num_reads),
                np.int32(num_betas),
                np.int32(num_sweeps_per_beta),
                np.uint32(seed),
                d_delta_ws,
                d_samples,
                d_energies,
            ),
        )
        cp.cuda.Stream.null.synchronize()

        # Read results
        samples = cp.asnumpy(d_samples).reshape(num_reads, N)
        energies = cp.asnumpy(d_energies)

        return dimod.SampleSet.from_samples(
            samples.astype(np.int8),
            vartype='SPIN',
            energy=energies,
            info={
                'beta_range': beta_range,
                'num_betas': num_betas,
                'min_energy': float(energies.min()),
            },
        )

    def _build_csr(
        self,
        J: Dict[Tuple[int, int], float],
        N: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build symmetric CSR from J couplings."""
        rows = [[] for _ in range(N)]
        vals = [[] for _ in range(N)]
        for (i, j), Jij in J.items():
            if i >= N or j >= N:
                continue
            rows[i].append(j)
            vals[i].append(int(Jij))
            rows[j].append(i)
            vals[j].append(int(Jij))

        csr_row_ptr = np.zeros(N + 1, dtype=np.int32)
        nnz = 0
        for i in range(N):
            csr_row_ptr[i] = nnz
            if rows[i]:
                order = np.argsort(rows[i])
                rows[i] = [rows[i][k] for k in order]
                vals[i] = [vals[i][k] for k in order]
            nnz += len(rows[i])
        csr_row_ptr[N] = nnz

        csr_col_ind = np.array(
            [c for row in rows for c in row], dtype=np.int32
        )
        csr_J_vals = np.array(
            [v for row in vals for v in row], dtype=np.int8
        )
        return csr_row_ptr, csr_col_ind, csr_J_vals
