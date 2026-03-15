# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Shared sampler utilities for GPU backends (CUDA, Metal).

Pure Python/NumPy functions with no hardware-specific dependencies.
Extracted from metal_sa.py, metal_gibbs_sa.py, cuda_sa.py to eliminate
code duplication across GPU backends.
"""

import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import dimod
import numpy as np


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
        scale_T_with_N: Whether to scale temperature with system size.

    Returns:
        (hot_beta, cold_beta) tuple of starting and ending inverse
        temperatures.
    """
    if not 0 < max_single_qubit_excitation_rate < 1:
        raise ValueError(
            'Targeted single qubit excitations rates must be in range (0,1)'
        )

    sum_abs_bias_dict = defaultdict(
        int, {k: abs(v) for k, v in h.items()}
    )
    if sum_abs_bias_dict:
        min_abs_bias_dict = {
            k: v for k, v in sum_abs_bias_dict.items() if v != 0
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
            'All bqm biases are zero (all energies are zero), this is '
            'likely a value error. Temperature range is set arbitrarily '
            'to [0.1,1]. Metropolis-Hastings update is non-ergodic.'
        )
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
            np.log(number_min_gaps / max_single_qubit_excitation_rate)
            / (2 * min_effective_field)
        )

    return (hot_beta, cold_beta)


# Backward-compatible alias for internal callers
_default_ising_beta_range = default_ising_beta_range


def build_csr_from_ising(
    h_list: List[Dict[int, float]],
    J_list: List[Dict[Tuple[int, int], float]]
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, List[dict], List[int]
]:
    """Build concatenated CSR arrays from a batch of Ising problems.

    Constructs compressed sparse row representation for each problem
    and concatenates them with offset arrays for GPU dispatch.

    Args:
        h_list: List of linear biases [{node: bias}, ...] per problem.
        J_list: List of quadratic biases [{(n1, n2): coupling}, ...]
            per problem.

    Returns:
        Tuple of:
        - csr_row_ptr: Concatenated CSR row pointers (int32)
        - csr_col_ind: Concatenated CSR column indices (int32)
        - csr_J_vals: Concatenated coupling values (int8)
        - h_vals: Concatenated linear biases (int8)
        - row_ptr_offsets: Per-problem offsets into csr_row_ptr (int32)
        - col_ind_offsets: Per-problem offsets into csr_col_ind (int32)
        - node_to_idx_list: Per-problem node-to-index mappings
        - N_list: Per-problem node counts
    """
    num_problems = len(h_list)
    assert len(J_list) == num_problems, (
        f"h and J must have same length: {num_problems} vs {len(J_list)}"
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
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
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
        h_first: Linear biases of the first problem (for auto range).
        J_first: Quadratic biases of the first problem (for auto range).
        num_sweeps: Total number of sweeps.
        num_sweeps_per_beta: Sweeps per beta value.
        beta_range: (hot_beta, cold_beta) or None for auto.
        beta_schedule_type: "linear", "geometric", or "custom".
        beta_schedule: Pre-computed schedule (requires type="custom").

    Returns:
        (beta_schedule_array, beta_range) where beta_range may have
        been auto-computed.
    """
    if beta_schedule_type == "custom":
        if beta_schedule is None:
            raise ValueError(
                "'beta_schedule' must be provided for "
                "beta_schedule_type = 'custom'"
            )
        beta_schedule = np.array(beta_schedule, dtype=np.float32)
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
            "'num_sweeps' must be divisible by 'num_sweeps_per_beta'"
        )

    if beta_range is None:
        beta_range = default_ising_beta_range(h_first, J_first)
    elif len(beta_range) != 2 or min(beta_range) < 0:
        raise ValueError(
            "'beta_range' should be a 2-tuple of positive numbers"
        )

    if num_betas == 1:
        schedule = np.array([beta_range[-1]], dtype=np.float32)
    elif beta_schedule_type == "linear":
        schedule = np.linspace(
            beta_range[0], beta_range[1],
            num=num_betas, dtype=np.float32
        )
    elif beta_schedule_type == "geometric":
        if min(beta_range) <= 0:
            raise ValueError(
                "'beta_range' must contain non-zero values "
                "for geometric schedule"
            )
        schedule = np.geomspace(
            beta_range[0], beta_range[1],
            num=num_betas, dtype=np.float32
        )
    else:
        raise ValueError(
            f"Beta schedule type {beta_schedule_type} not implemented"
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
        packed_data: Bit-packed samples, shape (total, packed_size).
            packed_size may be based on max_N across problems.
        energies_data: Energy values, shape (total,).
        num_problems: Number of problems in the batch.
        num_reads: Number of reads per problem.
        N: Max number of variables (stride for packed_data).
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

        # Map 0/1 bits → +1/-1 spins (0 → +1, 1 → −1)
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

    Converts linear index to Zephyr coordinates, then applies coloring.
    Based on dwave_networkx.zephyr_four_color scheme 0.

    The Zephyr linear index encoding is:
        r = u * M * t * 2 * m + w * t * 2 * m + k * 2 * m + j * m + z
    where M = 2*m + 1

    We reverse this to get (u, w, k, j, z), then apply:
        color = j + ((w + 2*(z+u) + j) & 2)

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

    Uses dense indexing: nodes are mapped to contiguous 0..N-1
    indices via node_to_idx.

    Args:
        edges: Topology edges [(i, j), ...].
        nodes: Topology nodes.

    Returns:
        Tuple of:
        - csr_row_ptr: Row pointers (int32), length N+1.
        - csr_col_ind: Column indices (int32), length nnz.
        - node_to_idx: Node ID -> dense index mapping.
        - sorted_neighbors: Per-node sorted neighbor lists.
        - N: Number of nodes.
        - nnz: Number of non-zeros (2 * len(edges)).
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

    For edge (i, j), returns the CSR offset of j within row i
    and of i within row j. Enables O(1) J-value updates.

    Args:
        edges: Topology edges [(i, j), ...].
        node_to_idx: Node ID -> dense index mapping.
        csr_row_ptr: CSR row pointers.
        sorted_neighbors: Per-node sorted neighbor lists.

    Returns:
        List of (pos_ij, pos_ji) per edge, same order as edges.
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

    Partitions nodes by their graph coloring. For Zephyr topologies,
    this produces 4 independent sets where no two adjacent nodes
    share the same color.

    Args:
        nodes: List of node indices.
        m: Zephyr m parameter.
        t: Zephyr t parameter.

    Returns:
        Tuple of (block_starts, block_counts, color_node_indices):
        - block_starts: [4] start indices into color_node_indices
        - block_counts: [4] number of nodes per color
        - color_node_indices: [N] nodes sorted by color
    """
    node_colors = {
        node: zephyr_four_color_linear(node, m, t) for node in nodes
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

    color_node_indices = np.array(color_node_indices, dtype=np.int32)

    return block_starts, block_counts, color_node_indices
