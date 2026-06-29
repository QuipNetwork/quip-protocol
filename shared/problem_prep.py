"""SDK-free problem preparation: defect clamping + array reduction for QPU.

Lives in ``shared/`` (not ``QPU/``) on purpose: the PoW feeder runs this inside
its spawn-context generator workers, and importing any ``QPU.*`` submodule would
trigger ``QPU/__init__`` → ``dwave_sampler`` → the full D-Wave SDK
(``dwave.system``) in every worker. This module imports only numpy and shared
types, so the workers stay cheap.

The point is throughput: defect clamping is GIL-held CPU that the single submit
path serializes, and the full ``(h, J)`` dict is a ~739 KB pickle that backs up
the feeder→driver pipe. Doing the clamp in the feeder's parallel workers and
emitting the **reduced problem as dense float64 arrays** (≈368 KB, ≈half) lets
those arrays ride the shared-memory :class:`~shared.ring_views.ProblemView` ring
to the submitter with no per-element pickle overhead.

The submitter rebuilds the (cheap) dimod model from the arrays via
:func:`rebuild_ising` immediately before ``sample_bqm``. Producer and consumer
MUST agree on the live node/edge ordering — both derive it from the SAME
:func:`live_topology`, so an array position maps back to exactly one label and
the consensus-energy reconstruction (``reconstruct_full_matrix``) stays exact.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from shared.ising_model import IsingModel


class DefectInfo:
    """Lightweight metadata for reconstructing clamped QPU results.

    Carried (per nonce) alongside a reduced problem when defective qubits or
    couplers are present. The consumer decides when (and whether) to
    reconstruct — most samples won't meet the energy threshold and never need
    it.

    Attributes:
        fixed_spins: ``{qubit_id: spin_value}`` for clamped qubits.
        energy_offset: Constant energy from clamped qubits (h + J_fixed_fixed).
        removed_edges: J values for defective couplers between live qubits.
    """

    __slots__ = ("fixed_spins", "energy_offset", "removed_edges")

    def __init__(
        self,
        fixed_spins: Dict[int, int],
        energy_offset: float,
        removed_edges: Dict[Tuple[int, int], float],
    ):
        self.fixed_spins = fixed_spins
        self.energy_offset = energy_offset
        self.removed_edges = removed_edges


def live_topology(
    nodes: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    defective_qubits: Sequence[int],
    defective_edges: "set[Tuple[int, int]]",
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Canonical live (non-defective) node and edge orderings.

    The single source of truth for the reduced-problem layout. The feeder
    (producer) and the submitter (consumer) both call this with the same
    ``(nodes, edges, defective_qubits, defective_edges)`` — fixed for a QPU
    session — so array position ``i`` means ``live_nodes[i]`` on both sides.

    Live edges are exactly those the clamp keeps in ``J_reduced``: both
    endpoints live, and the coupler not itself defective. Node/edge order
    follows the input ``nodes``/``edges`` order (not sorted) — reconstruction
    maps labels explicitly, so any consistent order is correct.

    Args:
        nodes: Full topology node list.
        edges: Full topology edge list.
        defective_qubits: Offline qubit ids.
        defective_edges: Offline couplers between two live qubits.

    Returns:
        ``(live_nodes, live_edges)`` — the reduced problem's column labels.
    """
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
    J: Dict[Tuple[int, int], float],
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
    """Clamp defective qubits to deterministic spins and adjust neighbors.

    For each offline qubit k, assigns a fixed spin s_k (deterministic from
    ``nonce_seed``) and absorbs its coupling energy into neighbors' h-fields:
    ``h'[j] += J[k,j] * s_k`` for all neighbors j of k. This preserves the
    clamped qubit's energy contribution in the reduced problem so the QPU
    optimizes the remaining variables correctly.

    Args:
        h: Linear biases for all nodes (full topology).
        J: Quadratic biases for all edges (full topology).
        nonce_seed: Seed for deterministic spin assignment. Accepts either the
            32-byte block nonce (post-MR-!20 wire shape) or a legacy int seed.
        defective_qubits: Offline qubit ids to clamp.
        defective_edges: Couplers between two live qubits that are offline on the
            hardware; excluded from ``J_reduced`` and from the offset, and
            returned in ``removed_edges`` for per-sample reconstruction.

    Returns:
        5-tuple ``(h_reduced, J_reduced, fixed_spins, energy_offset,
        removed_edges)`` — see :class:`DefectInfo` for the reconstruction
        contract.
    """
    defective_set = set(defective_qubits)
    # numpy's SeedSequence rejects bytes ("expects int or sequence of ints for
    # entropy"). Post-MR-!20, `derive_nonce` returns 32 bytes rather than an
    # int, so the seed needs explicit conversion. big-endian matches the U256
    # wire encoding.
    if isinstance(nonce_seed, (bytes, bytearray)):
        nonce_seed = int.from_bytes(nonce_seed, "big")
    rng = np.random.default_rng(nonce_seed)

    # Assign deterministic ±1 spins to defective qubits.
    fixed_spins: Dict[int, int] = {}
    for qubit in defective_qubits:
        fixed_spins[qubit] = int(2 * rng.integers(2) - 1)

    # Copy h, remove defective qubits, adjust neighbors.
    h_reduced = {k: v for k, v in h.items() if k not in defective_set}

    for (u, v), j_val in J.items():
        if u in defective_set and v not in defective_set:
            h_reduced[v] = h_reduced.get(v, 0.0) + j_val * fixed_spins[u]
        elif v in defective_set and u not in defective_set:
            h_reduced[u] = h_reduced.get(u, 0.0) + j_val * fixed_spins[v]
        # If both are defective, energy is constant — handled in the offset.

    # Remove edges involving defective qubits AND defective couplers (between two
    # live qubits whose coupler is offline — the QPU would reject them).
    J_reduced = {
        (u, v): val
        for (u, v), val in J.items()
        if u not in defective_set
        and v not in defective_set
        and (u, v) not in defective_edges
    }

    # Constant energy offset from clamped qubits: their h-fields plus J between
    # two clamped qubits. Constant for this nonce — added once per QPU energy.
    energy_offset = 0.0
    for k, s_k in fixed_spins.items():
        energy_offset += h.get(k, 0.0) * s_k
    for (u, v), j_val in J.items():
        if u in defective_set and v in defective_set:
            energy_offset += j_val * fixed_spins[u] * fixed_spins[v]

    # Defective edges (two live qubits, coupler offline) are NOT in the offset —
    # their contribution depends on the QPU solution and is computed per-sample
    # during reconstruction.
    removed_edges = {
        (u, v): val
        for (u, v), val in J.items()
        if u not in defective_set
        and v not in defective_set
        and (u, v) in defective_edges
    }

    return h_reduced, J_reduced, fixed_spins, energy_offset, removed_edges


def reduce_to_arrays(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    nonce_seed: Union[int, bytes],
    defective_qubits: Sequence[int],
    defective_edges: "set[Tuple[int, int]]",
    live_nodes: Sequence[int],
    live_edges: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, "DefectInfo | None"]:
    """Reduce a full ``(h, J)`` to dense float64 arrays for ``ProblemView``.

    When defects exist, clamps them (deterministic from ``nonce_seed``) and lays
    the reduced biases/couplings out in ``live_nodes``/``live_edges`` order.
    Otherwise lays out the full problem in the same order (which then equals the
    full topology). The returned arrays are exactly the ``ProblemView.write``
    contract: ``h_vec`` shape ``(len(live_nodes),)``, ``j_vec`` shape
    ``(len(live_edges),)``.

    Args:
        h: Full-topology linear biases.
        J: Full-topology quadratic couplings.
        nonce_seed: Deterministic clamp seed (32-byte nonce or int).
        defective_qubits: Offline qubit ids.
        defective_edges: Offline couplers between live qubits.
        live_nodes: Canonical live-node order from :func:`live_topology`.
        live_edges: Canonical live-edge order from :func:`live_topology`.

    Returns:
        ``(h_vec, j_vec, defect_info)`` — ``defect_info`` is ``None`` when there
        are no defects.
    """
    if defective_qubits or defective_edges:
        h_red, J_red, fixed, offset, removed = clamp_fixed_variables(
            h, J, nonce_seed, defective_qubits, defective_edges
        )
        defect_info: "DefectInfo | None" = DefectInfo(fixed, offset, removed)
        h_src, J_src = h_red, J_red
    else:
        defect_info = None
        h_src, J_src = h, J

    h_vec = np.fromiter(
        (h_src.get(n, 0.0) for n in live_nodes), dtype=np.float64,
        count=len(live_nodes),
    )
    j_vec = np.fromiter(
        (J_src[e] for e in live_edges), dtype=np.float64, count=len(live_edges),
    )
    return h_vec, j_vec, defect_info


def rebuild_ising(
    h_vec: np.ndarray,
    j_vec: np.ndarray,
    live_nodes: Sequence[int],
    live_edges: Sequence[Tuple[int, int]],
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float]]:
    """Rebuild ``(h, J)`` dicts from ``ProblemView`` arrays (submitter side).

    Inverse of :func:`reduce_to_arrays`'s layout step. Cheap (dict comprehension)
    and done in the submitter immediately before ``from_ising`` + ``sample_bqm``.

    Args:
        h_vec: float64 ``(len(live_nodes),)`` linear biases.
        j_vec: float64 ``(len(live_edges),)`` quadratic couplings.
        live_nodes: Canonical live-node order from :func:`live_topology`.
        live_edges: Canonical live-edge order from :func:`live_topology`.

    Returns:
        ``(h_dict, J_dict)`` ready for ``dimod.BinaryQuadraticModel.from_ising``.
    """
    h_dict = {int(n): float(h_vec[i]) for i, n in enumerate(live_nodes)}
    J_dict = {
        (int(u), int(v)): float(j_vec[k])
        for k, (u, v) in enumerate(live_edges)
    }
    return h_dict, J_dict


@dataclasses.dataclass(frozen=True, slots=True)
class ReducedProblem:
    """A defect-reduced problem produced by the feeder workers.

    Carries dense float64 arrays (the :class:`~shared.ring_views.ProblemView`
    contract) plus per-nonce reconstruction metadata and provenance. The
    submitter rebuilds the dimod model from the arrays via :func:`rebuild_ising`
    immediately before ``sample_bqm``; the downstream descriptor reads only
    ``nonce``/``salt``/``defect_info``.

    Attributes:
        h_vec: float64 ``(len(live_nodes),)`` linear biases (reduced order).
        j_vec: float64 ``(len(live_edges),)`` quadratic couplings (reduced order).
        defect_info: Reconstruction metadata, or ``None`` when no defects.
        nonce: 32-byte derivation nonce (provenance).
        salt: 32-byte salt (provenance).
    """

    h_vec: Any  # np.ndarray — Any keeps the slot picklable-light
    j_vec: Any  # np.ndarray
    defect_info: Optional[DefectInfo]
    nonce: bytes
    salt: bytes
    # Round generation. The feeder worker can't know it (it predates the round
    # tag), so it defaults to 0; the producer-side driver stamps the real value
    # (via dataclasses.replace) when it hands the problem to the submitter, and
    # the submitter tags each result with it for the consumer's staleness filter.
    generation: int = 0


def prepare_reduced(
    model: IsingModel,
    defective_qubits: Sequence[int],
    defective_edges: "set[Tuple[int, int]]",
    live_nodes: Sequence[int],
    live_edges: Sequence[Tuple[int, int]],
) -> ReducedProblem:
    """Feeder-worker entry point: reduce a raw Ising model to a ReducedProblem.

    Runs off the submit path, in the feeder's parallel generator workers. Picks
    up the model's own nonce as the deterministic clamp seed.

    Args:
        model: Freshly derived Ising model from the feeder.
        defective_qubits: Offline qubit ids (empty when the QPU has none).
        defective_edges: Offline couplers between live qubits (empty when none).
        live_nodes: Canonical live-node order from :func:`live_topology`.
        live_edges: Canonical live-edge order from :func:`live_topology`.

    Returns:
        A :class:`ReducedProblem` ready for the ``ProblemView`` transport.
    """
    h_vec, j_vec, defect_info = reduce_to_arrays(
        model.h, model.J, model.nonce,
        defective_qubits, defective_edges, live_nodes, live_edges,
    )
    return ReducedProblem(
        h_vec=h_vec,
        j_vec=j_vec,
        defect_info=defect_info,
        nonce=model.nonce,
        salt=model.salt,
    )


__all__ = [
    "DefectInfo",
    "ReducedProblem",
    "clamp_fixed_variables",
    "live_topology",
    "prepare_reduced",
    "reduce_to_arrays",
    "rebuild_ising",
]
