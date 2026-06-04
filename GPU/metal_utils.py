# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""
Shared utility functions for Metal GPU samplers.

This module contains functions duplicated across metal_sa.py, metal_gibbs_sa.py,
and metal_splash_sa.py for CSR graph construction, beta schedule computation,
Metal buffer creation, and result unpacking.
"""

from typing import Any, Dict, Optional, Tuple

import dimod
import numpy as np

try:
    import Metal
except ImportError:  # Apple Metal framework is macOS-only; absent on Linux/CI.
    Metal = None  # type: ignore[assignment]

from GPU.gpu_csr_beta import build_csr_single, compute_beta_schedule_core


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


def pooled_buffer(device, pool: Dict[str, Any], role: str, nbytes: int):
    """Return a reused shared MTLBuffer of at least ``nbytes`` for ``role``.

    Grows on demand and persists in ``pool`` (keyed by ``role``), so a
    streaming loop allocates each role's buffer once at its max size instead of
    every batch. ``role`` namespaces buffers so two same-sized roles never
    alias. Shared by the Metal SA and Gibbs samplers.
    """
    nbytes = max(1, int(nbytes))
    buf = pool.get(role)
    if buf is None or buf.length() < nbytes:
        buf = device.newBufferWithLength_options_(
            nbytes, Metal.MTLResourceStorageModeShared,
        )
        pool[role] = buf
    return buf


def pooled_input(device, pool: Dict[str, Any], role: str, data: np.ndarray):
    """Pooled buffer for ``role`` filled with ``data`` (copied to shared mem).

    Safe across batches because the dispatch is synchronous
    (``waitUntilCompleted``) before the buffer is refilled.
    """
    if not data.flags["C_CONTIGUOUS"]:
        data = np.ascontiguousarray(data)
    byte_data = data.tobytes()
    buf = pooled_buffer(device, pool, role, len(byte_data))
    buf.contents().as_buffer(len(byte_data))[:] = byte_data
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
    return compute_beta_schedule_core(
        h, J, num_sweeps, num_sweeps_per_beta, beta_range,
        beta_schedule_type, beta_schedule,
        custom_fills_beta_range=True,
    )


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
    return build_csr_single(h, J, use_float=use_float)


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
    # Unpack bit-packed samples (kernel stores LSB-first)
    bits = np.unpackbits(
        packed_data.view(np.uint8), axis=1, bitorder='little',
    )[:, :N]
    samples_data = np.where(bits, np.int8(-1), np.int8(1))

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
