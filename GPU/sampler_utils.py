# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Pure Python/NumPy utility functions for CUDA samplers.

Shared by CudaSASampler, CudaGibbsSampler, and GPUMiner. No CUDA
dependency — only NumPy and standard library.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import dimod
import numpy as np

from GPU.gpu_csr_beta import build_csr_single, compute_beta_schedule_core


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
        (csr_row_ptr, csr_col_ind, csr_J_vals,
         h_vals_array, node_to_idx, N) = build_csr_single(
            h_prob, J_prob,
        )
        N_list.append(N)
        node_to_idx_list.append(node_to_idx)

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
    return compute_beta_schedule_core(
        h_first, J_first, num_sweeps, num_sweeps_per_beta,
        beta_range, beta_schedule_type, beta_schedule,
        custom_fills_beta_range=False,
    )


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
