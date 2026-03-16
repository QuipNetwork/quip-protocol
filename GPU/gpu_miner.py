# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Unified GPU miner base class for CUDA SA and Gibbs kernels.

Owns the shared pipeline infrastructure: IsingFeeder for
background model generation, KernelScheduler for SM budget,
SIGTERM cleanup, sparse topology filtering, and the
enqueue/poll/dequeue/re-enqueue mining loop.

Pipeline model (per kernel):
    3 slots: completed | active | next
    Kernel persists until no "next" slot, then exits.
    Host: dequeue completed → enqueue replacement.
    Feeder keeps num_kernels models buffered for burst.

Subclasses implement 6 abstract methods (kernel adapter
protocol) to plug in their specific kernel backend.
"""
from __future__ import annotations

import dataclasses
import signal
import sys
import time
import warnings
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import dimod
import numpy as np

from shared.base_miner import BaseMiner
from shared.block_requirements import BlockRequirements
from shared.ising_feeder import IsingFeeder
from shared.ising_model import IsingModel
from GPU.gpu_scheduler import (
    KernelScheduler,
    configure_mps_thread_limit,
)

try:
    import cupy as cp
except ImportError:
    cp = None


# ----------------------------------------------------------
# Sampler utilities (pure Python/NumPy, no CUDA dependency)
# ----------------------------------------------------------

def default_ising_beta_range(
    h: Dict[int, float],
    J: Dict[tuple, float],
    max_single_qubit_excitation_rate: float = 0.01,
    scale_T_with_N: bool = True
) -> Tuple[float, float]:
    """Determine the starting and ending beta from h, J.

    Exact replica of D-Wave's _default_ising_beta_range function.

    Args:
        h: External field of Ising model (linear bias).
        J: Couplings of Ising model (quadratic biases).
        max_single_qubit_excitation_rate: Targeted single qubit
            excitation rate at final temperature.
        scale_T_with_N: Whether to scale temperature with
            system size.

    Returns:
        (hot_beta, cold_beta) tuple of starting and ending
        inverse temperatures.
    """
    if not 0 < max_single_qubit_excitation_rate < 1:
        raise ValueError(
            'Targeted single qubit excitations rates '
            'must be in range (0,1)'
        )

    sum_abs_bias_dict = defaultdict(
        int, {k: abs(v) for k, v in h.items()}
    )
    if sum_abs_bias_dict:
        min_abs_bias_dict = {
            k: v for k, v in sum_abs_bias_dict.items()
            if v != 0
        }
    else:
        min_abs_bias_dict = {}

    for (k1, k2), v in J.items():
        for k in [k1, k2]:
            sum_abs_bias_dict[k] += abs(v)
            if v != 0:
                if k in min_abs_bias_dict:
                    min_abs_bias_dict[k] = min(
                        abs(v), min_abs_bias_dict[k]
                    )
                else:
                    min_abs_bias_dict[k] = abs(v)

    if not min_abs_bias_dict:
        warn_msg = (
            'All bqm biases are zero (all energies are '
            'zero), this is likely a value error. '
            'Temperature range is set arbitrarily to '
            '[0.1,1]. Metropolis-Hastings update is '
            'non-ergodic.'
        )
        warnings.warn(warn_msg)
        return (0.1, 1.0)

    max_effective_field = max(
        sum_abs_bias_dict.values(), default=0,
    )

    if max_effective_field == 0:
        hot_beta = 1.0
    else:
        hot_beta = np.log(2) / (2 * max_effective_field)

    if len(min_abs_bias_dict) == 0:
        cold_beta = hot_beta
    else:
        values_array = np.array(
            list(min_abs_bias_dict.values()), dtype=float
        )
        min_effective_field = np.min(values_array)
        if scale_T_with_N:
            number_min_gaps = np.sum(
                min_effective_field == values_array
            )
        else:
            number_min_gaps = 1
        cold_beta = (
            np.log(
                number_min_gaps
                / max_single_qubit_excitation_rate
            )
            / (2 * min_effective_field)
        )

    return (hot_beta, cold_beta)


def build_csr_from_ising(
    h_list: List[Dict[int, float]],
    J_list: List[Dict[Tuple[int, int], float]]
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, List[dict], List[int]
]:
    """Build concatenated CSR arrays from Ising problems.

    Constructs compressed sparse row representation for each
    problem and concatenates them with offset arrays for GPU
    dispatch.

    Args:
        h_list: List of linear biases per problem.
        J_list: List of quadratic biases per problem.

    Returns:
        Tuple of (csr_row_ptr, csr_col_ind, csr_J_vals,
        h_vals, row_ptr_offsets, col_ind_offsets,
        node_to_idx_list, N_list).
    """
    num_problems = len(h_list)
    assert len(J_list) == num_problems, (
        f"h and J must have same length: "
        f"{num_problems} vs {len(J_list)}"
    )

    all_csr_row_ptr = []
    all_csr_col_ind = []
    all_csr_J_vals = []
    all_h_vals = []
    row_ptr_offsets = [0]
    col_ind_offsets = [0]
    node_to_idx_list = []
    N_list = []

    for h_prob, J_prob in zip(h_list, J_list):
        all_nodes = set(h_prob.keys()) | set(
            n for edge in J_prob.keys() for n in edge
        )
        N = len(all_nodes)
        N_list.append(N)
        node_list = sorted(all_nodes)
        node_to_idx = {
            node: idx for idx, node in enumerate(node_list)
        }
        node_to_idx_list.append(node_to_idx)

        csr_row_ptr = np.zeros(N + 1, dtype=np.int32)

        h_vals_array = np.zeros(N, dtype=np.int8)
        for node, h_val in h_prob.items():
            if node in node_to_idx:
                h_vals_array[node_to_idx[node]] = int(h_val)

        degree = np.zeros(N, dtype=np.int32)
        for (i, j) in J_prob.keys():
            if i in node_to_idx and j in node_to_idx:
                degree[node_to_idx[i]] += 1
                degree[node_to_idx[j]] += 1

        csr_row_ptr[1:] = np.cumsum(degree)

        adjacency = [[] for _ in range(N)]
        for (i, j), Jij in J_prob.items():
            if i in node_to_idx and j in node_to_idx:
                idx_i = node_to_idx[i]
                idx_j = node_to_idx[j]
                adjacency[idx_i].append((idx_j, Jij))
                adjacency[idx_j].append((idx_i, Jij))

        csr_col_ind = []
        csr_J_vals = []
        for i in range(N):
            adjacency[i].sort()
            for j, Jij in adjacency[i]:
                csr_col_ind.append(j)
                csr_J_vals.append(int(Jij))

        all_csr_row_ptr.extend(csr_row_ptr)
        all_csr_col_ind.extend(csr_col_ind)
        all_csr_J_vals.extend(csr_J_vals)
        all_h_vals.extend(h_vals_array)

        row_ptr_offsets.append(len(all_csr_row_ptr))
        col_ind_offsets.append(len(all_csr_col_ind))

    return (
        np.array(all_csr_row_ptr, dtype=np.int32),
        np.array(all_csr_col_ind, dtype=np.int32),
        np.array(all_csr_J_vals, dtype=np.int8),
        np.array(all_h_vals, dtype=np.int8),
        np.array(row_ptr_offsets, dtype=np.int32),
        np.array(col_ind_offsets, dtype=np.int32),
        node_to_idx_list,
        N_list,
    )


def compute_beta_schedule(
    h_first: Dict[int, float],
    J_first: Dict[tuple, float],
    num_sweeps: int,
    num_sweeps_per_beta: int = 1,
    beta_range: Optional[Tuple[float, float]] = None,
    beta_schedule_type: str = "geometric",
    beta_schedule: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    """Compute the annealing beta (inverse temperature) schedule.

    Args:
        h_first: Linear biases of the first problem.
        J_first: Quadratic biases of the first problem.
        num_sweeps: Total number of sweeps.
        num_sweeps_per_beta: Sweeps per beta value.
        beta_range: (hot_beta, cold_beta) or None for auto.
        beta_schedule_type: "linear", "geometric", or
            "custom".
        beta_schedule: Pre-computed schedule (requires
            type="custom").

    Returns:
        (beta_schedule_array, beta_range) where beta_range
        may have been auto-computed.
    """
    if beta_schedule_type == "custom":
        if beta_schedule is None:
            raise ValueError(
                "'beta_schedule' must be provided for "
                "beta_schedule_type = 'custom'"
            )
        beta_schedule = np.array(
            beta_schedule, dtype=np.float32,
        )
        num_betas = len(beta_schedule)
        if num_sweeps != num_betas * num_sweeps_per_beta:
            raise ValueError(
                f"num_sweeps ({num_sweeps}) must equal "
                f"len(beta_schedule) * num_sweeps_per_beta"
            )
        return beta_schedule, beta_range

    num_betas, rem = divmod(num_sweeps, num_sweeps_per_beta)
    if rem > 0 or num_betas < 0:
        raise ValueError(
            "'num_sweeps' must be divisible by "
            "'num_sweeps_per_beta'"
        )

    if beta_range is None:
        beta_range = default_ising_beta_range(
            h_first, J_first,
        )
    elif len(beta_range) != 2 or min(beta_range) < 0:
        raise ValueError(
            "'beta_range' should be a 2-tuple of "
            "positive numbers"
        )

    if num_betas == 1:
        schedule = np.array(
            [beta_range[-1]], dtype=np.float32,
        )
    elif beta_schedule_type == "linear":
        schedule = np.linspace(
            beta_range[0], beta_range[1],
            num=num_betas, dtype=np.float32,
        )
    elif beta_schedule_type == "geometric":
        if min(beta_range) <= 0:
            raise ValueError(
                "'beta_range' must contain non-zero values "
                "for geometric schedule"
            )
        schedule = np.geomspace(
            beta_range[0], beta_range[1],
            num=num_betas, dtype=np.float32,
        )
    else:
        raise ValueError(
            f"Beta schedule type {beta_schedule_type} "
            f"not implemented"
        )

    return schedule, beta_range


def unpack_packed_results(
    packed_data: np.ndarray,
    energies_data: np.ndarray,
    num_problems: int,
    num_reads: int,
    N: int,
    node_to_idx_list: List[dict],
    info: Optional[dict] = None,
) -> List[dimod.SampleSet]:
    """Unpack bit-packed GPU results into dimod SampleSets.

    Args:
        packed_data: Bit-packed samples, shape
            (total, packed_size).
        energies_data: Energy values, shape (total,).
        num_problems: Number of problems in the batch.
        num_reads: Number of reads per problem.
        N: Max number of variables.
        node_to_idx_list: Per-problem node-to-index mappings.
        info: Extra metadata to include in each SampleSet.

    Returns:
        List of dimod.SampleSet, one per problem.
    """
    samplesets = []
    for prob_idx in range(num_problems):
        start_idx = prob_idx * num_reads
        end_idx = (prob_idx + 1) * num_reads

        prob_packed = packed_data[start_idx:end_idx]
        prob_energies = energies_data[start_idx:end_idx]

        node_to_idx = node_to_idx_list[prob_idx]
        prob_N = len(node_to_idx)

        # Vectorized bit unpack: kernel stores LSB-first
        bits = np.unpackbits(
            prob_packed.view(np.uint8),
            axis=1, bitorder='little',
        )[:, :prob_N]

        # Map 0/1 bits -> +1/-1 spins
        spins = np.where(bits, np.int8(-1), np.int8(1))

        # Variable labels in index order
        labels = sorted(
            node_to_idx, key=node_to_idx.__getitem__,
        )

        sampleset = dimod.SampleSet.from_samples(
            (spins, labels),
            energy=prob_energies.astype(float),
            vartype=dimod.SPIN,
            info=info or {},
        )
        samplesets.append(sampleset)

    return samplesets


def zephyr_four_color_linear(
    linear_idx: int, m: int = 9, t: int = 2
) -> int:
    """Compute 4-color for Zephyr node given linear index.

    Converts linear index to Zephyr coordinates, then applies
    coloring. Based on dwave_networkx.zephyr_four_color
    scheme 0.

    Args:
        linear_idx: Linear node index.
        m: Zephyr m parameter (default 9 for Z(9,2)).
        t: Zephyr t parameter (default 2).

    Returns:
        Color index (0-3).
    """
    M = 2 * m + 1

    r = linear_idx
    r, z = divmod(r, m)
    r, j = divmod(r, 2)
    r, k = divmod(r, t)
    u, w = divmod(r, M)

    return j + ((w + 2 * (z + u) + j) & 2)


def build_csr_structure_from_edges(
    edges: List[Tuple[int, int]],
    nodes: List[int],
) -> Tuple[
    np.ndarray, np.ndarray, Dict[int, int],
    List[List[int]], int, int
]:
    """Build CSR structure from topology edges (no J values).

    Uses dense indexing: nodes are mapped to contiguous
    0..N-1 indices via node_to_idx.

    Args:
        edges: Topology edges [(i, j), ...].
        nodes: Topology nodes.

    Returns:
        Tuple of (csr_row_ptr, csr_col_ind, node_to_idx,
        sorted_neighbors, N, nnz).
    """
    node_list = sorted(nodes)
    N = len(node_list)
    node_to_idx = {
        node: idx for idx, node in enumerate(node_list)
    }

    adjacency: List[List[int]] = [[] for _ in range(N)]
    for i, j in edges:
        idx_i = node_to_idx[i]
        idx_j = node_to_idx[j]
        adjacency[idx_i].append(idx_j)
        adjacency[idx_j].append(idx_i)

    sorted_neighbors: List[List[int]] = [
        sorted(adj) for adj in adjacency
    ]

    csr_row_ptr = np.zeros(N + 1, dtype=np.int32)
    nnz = 0
    for node_idx in range(N):
        csr_row_ptr[node_idx] = nnz
        nnz += len(sorted_neighbors[node_idx])
    csr_row_ptr[N] = nnz

    csr_col_ind = np.array(
        [c for row in sorted_neighbors for c in row],
        dtype=np.int32,
    )

    return (
        csr_row_ptr, csr_col_ind, node_to_idx,
        sorted_neighbors, N, nnz,
    )


def build_edge_position_index(
    edges: List[Tuple[int, int]],
    node_to_idx: Dict[int, int],
    csr_row_ptr: np.ndarray,
    sorted_neighbors: List[List[int]],
) -> List[Tuple[int, int]]:
    """Map each topology edge to its two CSR positions.

    For edge (i, j), returns the CSR offset of j within
    row i and of i within row j. Enables O(1) J-value
    updates.

    Args:
        edges: Topology edges [(i, j), ...].
        node_to_idx: Node ID -> dense index mapping.
        csr_row_ptr: CSR row pointers.
        sorted_neighbors: Per-node sorted neighbor lists.

    Returns:
        List of (pos_ij, pos_ji) per edge.
    """
    positions: List[Tuple[int, int]] = []
    for i, j in edges:
        idx_i = node_to_idx[i]
        idx_j = node_to_idx[j]
        pos_ij = (
            int(csr_row_ptr[idx_i])
            + sorted_neighbors[idx_i].index(idx_j)
        )
        pos_ji = (
            int(csr_row_ptr[idx_j])
            + sorted_neighbors[idx_j].index(idx_i)
        )
        positions.append((pos_ij, pos_ji))
    return positions


def compute_color_blocks(
    nodes: List[int], m: int = 9, t: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute color block partitions for Zephyr topology.

    Partitions nodes by their graph coloring. For Zephyr
    topologies, this produces 4 independent sets where no
    two adjacent nodes share the same color.

    Args:
        nodes: List of node indices.
        m: Zephyr m parameter.
        t: Zephyr t parameter.

    Returns:
        Tuple of (block_starts, block_counts,
        color_node_indices).
    """
    node_colors = {
        node: zephyr_four_color_linear(node, m, t)
        for node in nodes
    }

    color_groups = defaultdict(list)
    for node in nodes:
        color_groups[node_colors[node]].append(node)

    for color in color_groups:
        color_groups[color].sort()

    num_colors = 4
    block_starts = np.zeros(num_colors, dtype=np.int32)
    block_counts = np.zeros(num_colors, dtype=np.int32)

    color_node_indices = []
    current_start = 0
    for color in range(num_colors):
        nodes_in_color = color_groups.get(color, [])
        block_starts[color] = current_start
        block_counts[color] = len(nodes_in_color)
        color_node_indices.extend(nodes_in_color)
        current_start += len(nodes_in_color)

    color_node_indices = np.array(
        color_node_indices, dtype=np.int32,
    )

    return block_starts, block_counts, color_node_indices


# ----------------------------------------------------------
# Pipeline constants
# ----------------------------------------------------------

_POLL_INTERVAL = 0.001  # 1ms between completion polls
_PIPELINE_STALL_TIMEOUT = 30.0


@dataclasses.dataclass(slots=True)
class InFlightModel:
    """Tracks a model currently in the GPU pipeline.

    Attributes:
        job_id: Unique kernel job identifier.
        nonce: Blockchain nonce for proof-of-work.
        salt: Random salt used to derive the nonce.
        enqueue_time: Monotonic timestamp when enqueued.
    """

    job_id: int
    nonce: int
    salt: bytes
    enqueue_time: float


class GPUMiner(BaseMiner):
    """Shared pipeline base for CUDA GPU miners.

    Provides IsingFeeder, KernelScheduler, SIGTERM cleanup,
    the pipeline loop, sparse topology filtering, and
    adaptive parameter calculation.

    Subclasses must implement the kernel adapter protocol:
        _kernel_sms_per_model()
        _kernel_enqueue(model, job_id, num_reads, num_sweeps, **p)
        _kernel_signal_ready()
        _kernel_try_dequeue() -> Optional[Tuple[int, Any]]
        _kernel_harvest(raw_result) -> dimod.SampleSet
        _kernel_stop()
    """

    def __init__(
        self,
        miner_id: str,
        sampler,
        *,
        device: str = "0",
        gpu_utilization: int = 100,
        yielding: bool = False,
        miner_type: str = "GPU-CUDA",
    ):
        if cp is None:
            raise ImportError("cupy not available")

        dev_id = int(device)

        # MPS + device context (idempotent if subclass
        # already called _init_cuda_device)
        if not getattr(self, '_cuda_initialized', False):
            self._init_cuda_device(
                dev_id, gpu_utilization, yielding,
            )

        super().__init__(
            miner_id, sampler, miner_type=miner_type,
        )

        self.device = device

        if not 0 < gpu_utilization <= 100:
            raise ValueError(
                f"gpu_utilization must be 1-100, "
                f"got {gpu_utilization}"
            )
        self.gpu_utilization = gpu_utilization

        device_sms = cp.cuda.Device(
            int(device),
        ).attributes['MultiProcessorCount']
        self._device_sms = device_sms

        self._scheduler = KernelScheduler(
            device_id=int(device),
            device_sms=device_sms,
            gpu_utilization_pct=gpu_utilization,
            yielding=yielding,
        )

        # Sparse topology node indices for filtering
        self._node_indices = np.array(
            sampler.nodes, dtype=np.int32,
        )

        # Pipeline state (reset per mine_block call)
        self._feeder: Optional[IsingFeeder] = None
        self._in_flight: Dict[int, InFlightModel] = {}
        self._next_job_id = 0
        self._cold_start = True

        signal.signal(signal.SIGTERM, self._cleanup_handler)

    def _init_cuda_device(
        self,
        dev_id: int,
        gpu_utilization: int,
        yielding: bool,
    ) -> None:
        """Set MPS thread limit and activate CUDA device.

        Must be called before any CUDA API call. Safe to call
        multiple times — subsequent calls are no-ops.

        Subclasses that need CUDA before super().__init__()
        should call this explicitly in their __init__.
        """
        if getattr(self, '_cuda_initialized', False):
            return
        self._mps_enforced = configure_mps_thread_limit(
            gpu_utilization_pct=gpu_utilization,
            device_id=dev_id,
            yielding=yielding,
        )
        cp.cuda.Device(dev_id).use()
        self._cuda_initialized = True

    # ----------------------------------------------------------
    # Kernel adapter protocol (abstract)
    # ----------------------------------------------------------

    @abstractmethod
    def _kernel_sms_per_model(self) -> int:
        """SMs consumed per in-flight model."""

    @abstractmethod
    def _kernel_enqueue(
        self,
        model: IsingModel,
        job_id: int,
        num_reads: int,
        num_sweeps: int,
        **params,
    ) -> None:
        """Upload one model to a kernel slot."""

    @abstractmethod
    def _kernel_signal_ready(self) -> None:
        """Tell kernel it can start (or that new work exists)."""

    @abstractmethod
    def _kernel_try_dequeue(
        self,
    ) -> Optional[Tuple[int, Any]]:
        """Non-blocking poll. Returns (job_id, raw) or None."""

    @abstractmethod
    def _kernel_harvest(
        self, raw_result: Any,
    ) -> dimod.SampleSet:
        """Convert raw kernel result to dimod.SampleSet."""

    @abstractmethod
    def _kernel_stop(self) -> None:
        """Stop kernel and release GPU resources."""

    # ----------------------------------------------------------
    # Pipeline properties
    # ----------------------------------------------------------

    @property
    def _num_kernels(self) -> int:
        """Number of concurrent kernel instances."""
        budget = self._scheduler.get_sm_budget()
        return max(1, budget // self._kernel_sms_per_model())

    # ----------------------------------------------------------
    # BaseMiner hooks
    # ----------------------------------------------------------

    def _pre_mine_setup(self, *args, **kwargs) -> bool:
        """Set CUDA device and create IsingFeeder."""
        try:
            cp.cuda.Device(int(self.device)).use()
        except Exception as e:
            self.logger.error(
                f"Failed to set device context: {e}",
            )
            return False

        # Extract block context from BaseMiner's positional args
        prev_block = args[0] if len(args) > 0 else None
        node_info = args[1] if len(args) > 1 else None
        if prev_block is None or node_info is None:
            self.logger.error(
                "Missing prev_block or node_info",
            )
            return False

        cur_index = prev_block.header.index + 1
        num_k = self._num_kernels

        self._feeder = IsingFeeder(
            prev_hash=prev_block.hash,
            miner_id=node_info.miner_id,
            cur_index=cur_index,
            nodes=self.sampler.nodes,
            edges=self.sampler.edges,
            buffer_size=num_k * 2,
        )

        self._in_flight.clear()
        self._next_job_id = 0
        self._cold_start = True

        return True

    def _adapt_mining_params(
        self,
        current_requirements: BlockRequirements,
        nodes: List[int],
        edges: List[Tuple[int, int]],
    ) -> dict:
        """Compute adaptive params from difficulty."""
        return self.adapt_parameters(
            current_requirements.difficulty_energy,
            current_requirements.min_diversity,
            current_requirements.min_solutions,
            num_nodes=len(nodes),
            num_edges=len(edges),
        )

    def _sample_batch(
        self,
        prev_hash: bytes,
        miner_id: str,
        cur_index: int,
        nodes: List[int],
        edges: List[Tuple[int, int]],
        *,
        num_reads: int,
        num_sweeps: int,
        **kwargs,
    ) -> Optional[
        List[Tuple[int, bytes, dimod.SampleSet]]
    ]:
        """Pipeline: enqueue→poll→dequeue→re-enqueue.

        Cold start: enqueue num_kernels models, signal kernel.
        Steady state: poll for completions, dequeue + re-enqueue
        each, return harvested results.
        """
        if self._scheduler.should_throttle():
            time.sleep(0.5)

        extra = {
            k: v for k, v in kwargs.items()
            if k not in ('num_reads', 'num_sweeps')
        }

        # Cold start: fill active + next slots per kernel
        if self._cold_start:
            self._cold_start = False
            fill = self._num_kernels * 2
            for _ in range(fill):
                self._enqueue_one(
                    num_reads, num_sweeps, **extra,
                )
            self._kernel_signal_ready()

        # Poll for completions
        deadline = time.monotonic() + _PIPELINE_STALL_TIMEOUT
        while time.monotonic() < deadline:
            pair = self._kernel_try_dequeue()
            if pair is not None:
                job_id, raw_result = pair
                tracked = self._in_flight.pop(
                    job_id, None,
                )

                # Immediately enqueue replacement
                self._enqueue_one(
                    num_reads, num_sweeps, **extra,
                )
                self._kernel_signal_ready()

                sampleset = self._kernel_harvest(
                    raw_result,
                )

                if tracked is not None:
                    return [
                        (tracked.nonce, tracked.salt,
                         sampleset),
                    ]
                return []

            time.sleep(_POLL_INTERVAL)

        self.logger.warning(
            "Pipeline stall: no completions after "
            f"{_PIPELINE_STALL_TIMEOUT}s",
        )
        return None

    def _sample(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        *,
        num_reads: int,
        num_sweeps: int,
        **kwargs,
    ) -> dimod.SampleSet:
        """Single-nonce fallback (synchronous)."""
        extra = {
            k: v for k, v in kwargs.items()
            if k not in ('num_reads', 'num_sweeps')
        }
        model = IsingModel(h=h, J=J, nonce=0, salt=b'')
        job_id = self._alloc_job_id()
        self._kernel_enqueue(
            model, job_id, num_reads, num_sweeps,
            **extra,
        )
        self._kernel_signal_ready()

        deadline = time.monotonic() + 300.0
        while time.monotonic() < deadline:
            pair = self._kernel_try_dequeue()
            if pair is not None:
                _, raw_result = pair
                return self._kernel_harvest(raw_result)
            time.sleep(0.05)

        raise TimeoutError(
            "Kernel did not produce result within 300s",
        )

    def _enqueue_one(
        self,
        num_reads: int,
        num_sweeps: int,
        **extra,
    ) -> None:
        """Pop a model from feeder and enqueue it."""
        assert self._feeder is not None, (
            "_enqueue_one called before _pre_mine_setup"
        )
        model = self._feeder.pop()
        job_id = self._alloc_job_id()
        self._in_flight[job_id] = InFlightModel(
            job_id=job_id,
            nonce=model.nonce,
            salt=model.salt,
            enqueue_time=time.monotonic(),
        )
        self._kernel_enqueue(
            model, job_id, num_reads, num_sweeps,
            **extra,
        )

    def _alloc_job_id(self) -> int:
        """Allocate a monotonically increasing job ID."""
        jid = self._next_job_id
        self._next_job_id += 1
        return jid

    def _post_sample(
        self, sampleset: dimod.SampleSet,
    ) -> dimod.SampleSet:
        """Filter samples for sparse topology."""
        return self._filter_sparse_topology(sampleset)

    def _filter_sparse_topology(
        self, sampleset: dimod.SampleSet,
    ) -> dimod.SampleSet:
        """Extract only active topology nodes from kernel output.

        Kernel returns N=max_node+1 but validation expects
        only active node count.
        """
        samples = sampleset.record.sample
        filtered = samples[:, self._node_indices].astype(
            np.int8,
        )
        return dimod.SampleSet.from_samples(
            filtered,
            vartype='SPIN',
            energy=sampleset.record.energy,
            info=sampleset.info,
        )

    def _post_mine_cleanup(self) -> None:
        """Stop feeder, kernel, and scheduler."""
        if self._feeder is not None:
            self._feeder.stop()
            self._feeder = None
        self._in_flight.clear()
        try:
            self._kernel_stop()
        except Exception:
            pass

    def _cleanup_handler(self, signum, frame):
        """Handle SIGTERM for graceful CUDA cleanup."""
        if hasattr(self, '_scheduler'):
            self._scheduler.stop()

        try:
            self._kernel_stop()
        except Exception:
            pass

        self.logger.info(
            f"{self.miner_type} miner {self.miner_id} "
            f"received SIGTERM, cleaning up...",
        )
        try:
            if cp is not None:
                cp.cuda.Device(int(self.device)).use()
                cp.cuda.Stream.null.synchronize()
                mem = cp.get_default_memory_pool()
                mem.free_all_blocks()
                pin = cp.get_default_pinned_memory_pool()
                pin.free_all_blocks()
        except Exception as e:
            self.logger.error(f"CUDA cleanup error: {e}")
        sys.exit(0)
