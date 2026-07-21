# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""Shared CSR-builder and beta-schedule cores for the GPU samplers.

Both the batched CUDA path (``GPU.sampler_utils``) and the single-problem
Metal path (``GPU.metal_utils``) wrap these cores so the per-problem CSR
construction and the annealing beta schedule are defined exactly once.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from shared.beta_schedule import _default_ising_beta_range


def build_csr_single(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    use_float: bool = False,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, int], int
]:
    """Build a single-problem CSR representation from an Ising model.

    Args:
        h: Linear biases ``{node: bias}``.
        J: Quadratic biases ``{(node1, node2): coupling}``.
        use_float: If True, store J/h values as float32; else int8.

    Returns:
        Tuple of ``(csr_row_ptr, csr_col_ind, csr_J_vals, h_vals,
        node_to_idx, N)``.
    """
    all_nodes = set(h.keys()) | set(n for edge in J.keys() for n in edge)
    N = len(all_nodes)
    node_list = sorted(all_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    val_dtype = np.float32 if use_float else np.int8

    def _cast(value):
        return float(value) if use_float else int(value)

    csr_row_ptr = np.zeros(N + 1, dtype=np.int32)

    h_vals_array = np.zeros(N, dtype=val_dtype)
    for node, h_val in h.items():
        if node in node_to_idx:
            h_vals_array[node_to_idx[node]] = _cast(h_val)

    degree = np.zeros(N, dtype=np.int32)
    for (i, j) in J.keys():
        if i in node_to_idx and j in node_to_idx:
            degree[node_to_idx[i]] += 1
            degree[node_to_idx[j]] += 1

    csr_row_ptr[1:] = np.cumsum(degree)

    adjacency: List[list] = [[] for _ in range(N)]
    for (i, j), Jij in J.items():
        if i in node_to_idx and j in node_to_idx:
            idx_i = node_to_idx[i]
            idx_j = node_to_idx[j]
            adjacency[idx_i].append((idx_j, Jij))
            adjacency[idx_j].append((idx_i, Jij))

    csr_col_ind: List[int] = []
    csr_J_vals: List[float] = []
    for i in range(N):
        adjacency[i].sort()  # Deterministic ordering
        for j, Jij in adjacency[i]:
            csr_col_ind.append(j)
            csr_J_vals.append(_cast(Jij))

    return (
        csr_row_ptr,
        np.array(csr_col_ind, dtype=np.int32),
        np.array(csr_J_vals, dtype=val_dtype),
        h_vals_array,
        node_to_idx,
        N,
    )


def compute_beta_schedule_core(
    h: Dict[int, float],
    J: Dict[tuple, float],
    num_sweeps: int,
    num_sweeps_per_beta: int,
    beta_range: Optional[Tuple[float, float]],
    beta_schedule_type: str,
    beta_schedule: Optional[np.ndarray],
    custom_fills_beta_range: bool,
) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    """Compute the annealing beta (inverse temperature) schedule.

    Args:
        h: Linear biases of the (first) problem.
        J: Quadratic biases of the (first) problem.
        num_sweeps: Total number of sweeps.
        num_sweeps_per_beta: Sweeps per beta value.
        beta_range: ``(hot_beta, cold_beta)`` or None for auto.
        beta_schedule_type: "linear", "geometric", or "custom".
        beta_schedule: Pre-computed schedule (requires type="custom").
        custom_fills_beta_range: When True and ``beta_schedule_type`` is
            "custom" with ``beta_range`` None, derive ``beta_range`` from the
            schedule endpoints; when False, return ``beta_range`` unchanged.

    Returns:
        ``(beta_schedule_array, beta_range)`` where ``beta_range`` may have
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
        if custom_fills_beta_range and beta_range is None:
            beta_range = (
                float(beta_schedule[0]),
                float(beta_schedule[-1]),
            )
        return beta_schedule, beta_range

    num_betas, rem = divmod(num_sweeps, num_sweeps_per_beta)
    if rem > 0:
        raise ValueError(
            "'num_sweeps' must be divisible by 'num_sweeps_per_beta'"
        )

    if beta_range is None:
        beta_range = _default_ising_beta_range(h, J)
    elif len(beta_range) != 2 or min(beta_range) < 0:
        raise ValueError(
            "'beta_range' should be a 2-tuple of positive numbers"
        )

    if num_betas == 1:
        schedule = np.array([beta_range[-1]], dtype=np.float32)
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
            f"Beta schedule type {beta_schedule_type} not implemented"
        )

    return schedule, beta_range
