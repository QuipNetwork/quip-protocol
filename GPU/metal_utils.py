# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""
Shared utility functions for Metal GPU samplers.

This module contains functions duplicated across metal_sa.py, metal_gibbs_sa.py,
and metal_splash_sa.py for CSR graph construction, beta schedule computation,
Metal buffer creation, and result unpacking.
"""

from typing import Dict, List, Optional, Tuple

import dimod
import Metal
import numpy as np

from shared.beta_schedule import _default_ising_beta_range


def _create_buffer(device, data: np.ndarray, label: str = ""):
    """Create a Metal buffer from numpy array.

    Args:
        device: Metal device
        data: Numpy array to copy to GPU
        label: Optional label for error messages

    Returns:
        Metal buffer
    """
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    byte_data = data.tobytes()
    byte_length = len(byte_data)
    buf = device.newBufferWithBytes_length_options_(
        byte_data, byte_length, Metal.MTLResourceStorageModeShared
    )
    if not buf:
        raise RuntimeError(f"Failed to create buffer: {label}")
    return buf


def compute_beta_schedule(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    num_sweeps: int,
    num_sweeps_per_beta: int,
    beta_range: Optional[Tuple[float, float]],
    beta_schedule_type: str,
    beta_schedule: Optional[np.ndarray]
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Compute beta schedule for annealing.

    Args:
        h: Linear biases for one problem
        J: Quadratic biases for one problem
        num_sweeps: Total number of sweeps
        num_sweeps_per_beta: Sweeps per beta value
        beta_range: (hot_beta, cold_beta) or None for auto
        beta_schedule_type: "linear", "geometric", or "custom"
        beta_schedule: Custom beta schedule (for type="custom")

    Returns:
        Tuple of (beta_schedule array, beta_range tuple)
    """
    if beta_schedule_type == "custom":
        if beta_schedule is None:
            raise ValueError("'beta_schedule' must be provided for beta_schedule_type = 'custom'")
        beta_schedule = np.array(beta_schedule, dtype=np.float32)
        num_betas = len(beta_schedule)
        if num_sweeps != num_betas * num_sweeps_per_beta:
            raise ValueError(f"num_sweeps ({num_sweeps}) must equal len(beta_schedule) * num_sweeps_per_beta")
        # For custom schedule, beta_range is informational only
        if beta_range is None:
            beta_range = (float(beta_schedule[0]), float(beta_schedule[-1]))
    else:
        num_betas, rem = divmod(num_sweeps, num_sweeps_per_beta)
        if rem > 0 or num_betas < 0:
            raise ValueError("'num_sweeps' must be divisible by 'num_sweeps_per_beta'")

        if beta_range is None:
            beta_range = _default_ising_beta_range(h, J)
        elif len(beta_range) != 2 or min(beta_range) < 0:
            raise ValueError("'beta_range' should be a 2-tuple of positive numbers")

        if num_betas == 1:
            beta_schedule = np.array([beta_range[-1]], dtype=np.float32)
        else:
            if beta_schedule_type == "linear":
                beta_schedule = np.linspace(beta_range[0], beta_range[1], num=num_betas, dtype=np.float32)
            elif beta_schedule_type == "geometric":
                if min(beta_range) <= 0:
                    raise ValueError("'beta_range' must contain non-zero values for geometric schedule")
                beta_schedule = np.geomspace(beta_range[0], beta_range[1], num=num_betas, dtype=np.float32)
            else:
                raise ValueError(f"Beta schedule type {beta_schedule_type} not implemented")

    return beta_schedule, beta_range


def build_csr_from_ising(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    use_float: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, int], int]:
    """Build Compressed Sparse Row representation from Ising model.

    Args:
        h: Linear biases {node: bias}
        J: Quadratic biases {(node1, node2): coupling}
        use_float: If True, use float32 for J values; if False, use int8

    Returns:
        Tuple of (csr_row_ptr, csr_col_ind, csr_J_vals, h_vals, node_to_idx, N)
        - csr_row_ptr: Row pointer array (int32)
        - csr_col_ind: Column index array (int32)
        - csr_J_vals: J coupling values (float32 or int8)
        - h_vals: Linear bias values (float32 or int8)
        - node_to_idx: Mapping from node IDs to dense indices
        - N: Number of nodes
    """
    # Get all nodes for this problem
    all_nodes = set(h.keys()) | set(n for edge in J.keys() for n in edge)
    N = len(all_nodes)
    node_list = sorted(all_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Build CSR representation
    csr_row_ptr = np.zeros(N + 1, dtype=np.int32)
    csr_col_ind = []
    csr_J_vals = []

    # Extract h values in node order
    h_dtype = np.float32 if use_float else np.int8
    h_vals_array = np.zeros(N, dtype=h_dtype)
    for node, h_val in h.items():
        if node in node_to_idx:
            h_vals_array[node_to_idx[node]] = float(h_val) if use_float else int(h_val)

    # Count degrees
    degree = np.zeros(N, dtype=np.int32)
    for (i, j) in J.keys():
        if i in node_to_idx and j in node_to_idx:
            degree[node_to_idx[i]] += 1
            degree[node_to_idx[j]] += 1

    # Build CSR
    csr_row_ptr[1:] = np.cumsum(degree)

    adjacency = [[] for _ in range(N)]
    for (i, j), Jij in J.items():
        if i in node_to_idx and j in node_to_idx:
            idx_i = node_to_idx[i]
            idx_j = node_to_idx[j]
            adjacency[idx_i].append((idx_j, Jij))
            adjacency[idx_j].append((idx_i, Jij))

    for i in range(N):
        adjacency[i].sort()  # Ensure deterministic ordering
        for j, Jij in adjacency[i]:
            csr_col_ind.append(j)
            csr_J_vals.append(float(Jij) if use_float else int(Jij))

    csr_col_ind = np.array(csr_col_ind, dtype=np.int32)
    j_dtype = np.float32 if use_float else np.int8
    csr_J_vals = np.array(csr_J_vals, dtype=j_dtype)

    return csr_row_ptr, csr_col_ind, csr_J_vals, h_vals_array, node_to_idx, N


def unpack_metal_results(
    packed_data: np.ndarray,
    energies_data: np.ndarray,
    N: int,
    num_reads: int,
    node_to_idx: Dict[int, int],
    beta_range: Optional[Tuple[float, float]] = None,
    beta_schedule_type: str = "geometric",
    **extra_info
) -> dimod.SampleSet:
    """Unpack bit-packed Metal results and build dimod SampleSet.

    Args:
        packed_data: Bit-packed samples array (num_reads, packed_size)
        energies_data: Energy values (num_reads,)
        N: Number of variables
        num_reads: Number of samples
        node_to_idx: Mapping from node IDs to dense indices
        beta_range: Beta range for info dict
        beta_schedule_type: Beta schedule type for info dict
        **extra_info: Additional fields to add to SampleSet info dict

    Returns:
        dimod.SampleSet with unpacked samples
    """
    # Unpack bit-packed samples
    samples_data = np.zeros((num_reads, N), dtype=np.int8)
    for read_idx in range(num_reads):
        for var in range(N):
            byte_idx = var >> 3  # var / 8
            bit_idx = var & 7    # var % 8
            bit = (packed_data[read_idx, byte_idx] >> bit_idx) & 1
            samples_data[read_idx, var] = -1 if bit else 1

    # Build SampleSet using node_to_idx mapping
    samples_dict = []
    for sample in samples_data:
        samples_dict.append({node: int(sample[idx]) for node, idx in node_to_idx.items()})

    info = {"beta_range": beta_range, "beta_schedule_type": beta_schedule_type}
    info.update(extra_info)

    sampleset = dimod.SampleSet.from_samples(
        samples_dict,
        energy=energies_data.astype(float),
        vartype=dimod.SPIN,
        info=info
    )

    return sampleset
