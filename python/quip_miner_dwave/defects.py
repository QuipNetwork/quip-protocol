"""Defect-qubit clamping and full-topology reconstruction (v0.2 port).

Offline qubits are fixed to deterministic spins (seeded from a nonce) and their
couplings are absorbed into neighbor biases so the reduced problem stays
energy-consistent with the full topology.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class DefectInfo:
    """Metadata needed to reconstruct a full-topology sample from a reduced one."""

    fixed_spins: Dict[int, int]
    energy_offset: float
    removed_edges: Dict[Tuple[int, int], float]


def live_topology(
    nodes: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    defective_qubits: Sequence[int],
    defective_edges: "set[Tuple[int, int]]",
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Canonical live (non-defective) node/edge orderings."""
    defective_set = set(defective_qubits)
    live_nodes = [n for n in nodes if n not in defective_set]
    live_edges = [
        (u, v)
        for (u, v) in edges
        if u not in defective_set
        and v not in defective_set
        and (u, v) not in defective_edges
    ]
    return live_nodes, live_edges


def clamp_fixed_variables(
    h: Dict[int, float],
    j: Dict[Tuple[int, int], float],
    nonce_seed: Union[int, bytes],
    defective_qubits: Sequence[int],
    defective_edges: "set[Tuple[int, int]]",
) -> Tuple[
    Dict[int, float],
    Dict[Tuple[int, int], float],
    Dict[int, int],
    float,
    Dict[Tuple[int, int], float],
]:
    """Clamp defective qubits; return reduced h/J + reconstruction metadata."""
    defective_set = set(defective_qubits)
    if isinstance(nonce_seed, (bytes, bytearray)):
        nonce_seed = int.from_bytes(nonce_seed, "big")
    rng = np.random.default_rng(nonce_seed)

    fixed_spins: Dict[int, int] = {}
    for qubit in defective_qubits:
        fixed_spins[qubit] = int(2 * rng.integers(2) - 1)

    h_reduced = {k: v for k, v in h.items() if k not in defective_set}
    for (u, v), j_val in j.items():
        if u in defective_set and v not in defective_set:
            h_reduced[v] = h_reduced.get(v, 0.0) + j_val * fixed_spins[u]
        elif v in defective_set and u not in defective_set:
            h_reduced[u] = h_reduced.get(u, 0.0) + j_val * fixed_spins[v]

    j_reduced = {
        (u, v): val
        for (u, v), val in j.items()
        if u not in defective_set
        and v not in defective_set
        and (u, v) not in defective_edges
    }

    energy_offset = 0.0
    for k, s_k in fixed_spins.items():
        energy_offset += h.get(k, 0.0) * s_k
    for (u, v), j_val in j.items():
        if u in defective_set and v in defective_set:
            energy_offset += j_val * fixed_spins[u] * fixed_spins[v]

    removed_edges = {
        (u, v): val
        for (u, v), val in j.items()
        if u not in defective_set
        and v not in defective_set
        and (u, v) in defective_edges
    }
    return h_reduced, j_reduced, fixed_spins, energy_offset, removed_edges


def prepare_problem(
    h: Dict[int, float],
    j: Dict[Tuple[int, int], float],
    *,
    defective_qubits: Sequence[int] = (),
    defective_edges: Optional[set] = None,
    nonce_seed: Union[int, bytes, None] = None,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], Optional[DefectInfo]]:
    """Apply defect clamping when defects and a seed are present."""
    de = defective_edges or set()
    if not (defective_qubits or de) or nonce_seed is None:
        return h, j, None
    h_r, j_r, fixed, offset, removed = clamp_fixed_variables(
        h, j, nonce_seed, defective_qubits, de
    )
    return h_r, j_r, DefectInfo(fixed, offset, removed)


def reconstruct_sample(
    reduced: Dict[int, int],
    reduced_energy: float,
    defect_info: Optional[DefectInfo],
) -> Tuple[Dict[int, int], float]:
    """Reinsert clamped spins and correct energy for the full topology."""
    if defect_info is None:
        return reduced, reduced_energy
    full = dict(reduced)
    full.update(defect_info.fixed_spins)
    energy = reduced_energy + defect_info.energy_offset
    for (u, v), j_val in defect_info.removed_edges.items():
        energy += j_val * full[u] * full[v]
    return full, energy
