"""Quantum proof-of-work primitives + sampleset evaluation.

Two distinct surfaces live in this module:

  - PoW primitives (``derive_nonce``, ``generate_ising_model_from_nonce``).
    These mirror ``quantum_validation::{derive_nonce, generate_ising_model}``
    in ``quip-protocol-rs`` and are cross-language-deterministic: identical
    inputs must produce identical outputs in both languages.
  - Sampleset evaluation utilities (``evaluate_sampleset``, ``validate_solution``,
    ``select_diverse_solutions``, ``calculate_diversity``). These are
    Python-only helpers used by miners to turn a dimod sampleset into a
    :class:`shared.miner_types.MiningResult`.

Post-MR-!20 wire shape:
  - ``derive_nonce`` returns the full 256-bit BLAKE3 digest as 32 raw bytes.
    Inputs are fixed 32-byte buffers (``parent_hash``, ``miner``, ``salt``).
  - ``generate_ising_model_from_nonce`` seeds ChaCha8Rng from those 32 bytes
    directly via ``from_seed`` — no PCG32 u64 expansion. h and j are sampled
    via :class:`shared.allowed_value_spec.AllowedValueSpec`.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Tuple, Dict, Optional, List, Union

from blake3 import blake3
import numpy as np

from shared.allowed_value_spec import (
    AllowedValueSet,
    AllowedValueSpec,
    MILLI_SCALE,
    sample as _sample_spec,
)
from shared.chacha8 import ChaCha8Rng
from shared.logging_config import get_logger
from shared.miner_types import MiningResult
from dwave_topologies import DEFAULT_TOPOLOGY

logger = get_logger('quantum_proof_of_work')


# Default sampling specs (mirror the on-chain v0.2 defaults: ternary h,
# binary j and spin). Milli-precision integers; multiply by 1/MILLI_SCALE
# to read as float.
DEFAULT_ALLOWED_H: AllowedValueSpec = AllowedValueSet((-MILLI_SCALE, 0, MILLI_SCALE))
DEFAULT_ALLOWED_J: AllowedValueSpec = AllowedValueSet((-MILLI_SCALE, MILLI_SCALE))
DEFAULT_ALLOWED_SPIN: AllowedValueSpec = AllowedValueSet((-MILLI_SCALE, MILLI_SCALE))


def _to_nonce_bytes(nonce: Union[int, bytes]) -> bytes:
    """Coerce a nonce input to its canonical 32-byte big-endian representation.

    The chain-integration path always passes ``bytes`` (the full 256-bit
    digest from ``derive_nonce``). Tools and tests sometimes pass a small
    integer for clarity; accept those and big-endian-encode to 32 bytes.
    Negative integers and oversize bytes raise.
    """
    if isinstance(nonce, (bytes, bytearray)):
        b = bytes(nonce)
        if len(b) != 32:
            raise ValueError(
                f"nonce must be 32 bytes when supplied as bytes, got {len(b)}"
            )
        return b
    if not isinstance(nonce, int):
        raise TypeError(
            f"nonce must be int or bytes, got {type(nonce).__name__}"
        )
    if nonce < 0 or nonce >= (1 << 256):
        raise ValueError("nonce int must fit in 256 bits (0..2^256-1)")
    return nonce.to_bytes(32, "big")


def derive_nonce(
    last_proof_block_hash: bytes,
    miner: bytes,
    salt: bytes,
) -> bytes:
    """Derive the canonical 32-byte PoW nonce.

    Mirrors ``quantum_validation::derive_nonce`` in ``quip-protocol-rs``.
    Inputs are three fixed-size 32-byte buffers so the PoW search space is
    statically known and identical across every call:

    - ``last_proof_block_hash`` — ``block_hash(LastProofBlock)``, the header
      hash of the most recent winning block. Stable across the entire
      round (only changes on the next win), so miner submissions don't
      race the txpool / executing-block-number.
    - ``miner`` — 32-byte canonical miner identity (typically
      ``blake2_256(SCALE(account_id))``)
    - ``salt`` — the only freely-chosen miner input, 32 bytes

    Returns the full 256-bit BLAKE3 digest as raw bytes. No truncation.
    """
    if len(last_proof_block_hash) != 32:
        raise ValueError(
            f"last_proof_block_hash must be 32 bytes, got {len(last_proof_block_hash)}"
        )
    if len(miner) != 32:
        raise ValueError(f"miner must be 32 bytes, got {len(miner)}")
    if len(salt) != 32:
        raise ValueError(f"salt must be 32 bytes, got {len(salt)}")
    hasher = blake3()
    hasher.update(last_proof_block_hash)
    hasher.update(miner)
    hasher.update(salt)
    return hasher.digest()


def generate_ising_model_from_nonce(
    nonce: Union[int, bytes],
    nodes: List[int],
    edges: List[Tuple[int, int]],
    allowed_h: Optional[AllowedValueSpec] = None,
    allowed_j: Optional[AllowedValueSpec] = None,
    *,
    h_values: Optional[List[float]] = None,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float]]:
    """Generate (h, J) Ising parameters deterministically from ``nonce``.

    Mirrors ``quantum_validation::generate_ising_model`` in
    ``quip-protocol-rs`` (post-MR-!20):

    - Seeds :class:`shared.chacha8.ChaCha8Rng` from the full 32-byte nonce
      via ``from_seed`` (not the legacy ``seed_from_u64`` PCG32 expansion).
    - Samples one h per node from ``allowed_h`` first, then one j per edge
      from ``allowed_j``. Both flow through
      :func:`shared.allowed_value_spec.sample`.

    ``allowed_h`` and ``allowed_j`` default to the chain's v0.2 specs
    (ternary h, binary j). The legacy ``h_values`` keyword accepts a list
    of float values for backwards compatibility with diagnostic tools;
    each float is converted to its milli-precision i32 representation
    inside an :class:`shared.allowed_value_spec.AllowedValueSet`.

    Returned dictionaries hold floats (``milli / MILLI_SCALE``) to match the
    rest of the Python miner stack, which works in physical units.
    """
    if not nodes:
        raise ValueError("nodes must be non-empty for Ising model generation")

    if h_values is not None:
        if allowed_h is not None:
            raise ValueError(
                "pass either `allowed_h` or legacy `h_values`, not both"
            )
        allowed_h = AllowedValueSet(
            tuple(int(round(float(v) * MILLI_SCALE)) for v in h_values)
        )

    if allowed_h is None:
        allowed_h = DEFAULT_ALLOWED_H
    if allowed_j is None:
        allowed_j = DEFAULT_ALLOWED_J

    seed = _to_nonce_bytes(nonce)
    rng = ChaCha8Rng.from_seed(seed)

    h: Dict[int, float] = {}
    for node_id in nodes:
        h[int(node_id)] = _sample_spec(allowed_h, rng) / MILLI_SCALE

    j: Dict[Tuple[int, int], float] = {}
    for (u, v) in edges:
        j[(int(u), int(v))] = _sample_spec(allowed_j, rng) / MILLI_SCALE

    return h, j


def energy_of_solution(solution: List[int], h: Dict[int, float], J: Dict[Tuple[int, int], float], nodes: List[int]) -> float:
    """Compute Ising energy for a solution vector respecting node order.

    - solution values are mapped to spins in {-1,+1}
    - h, J dictionaries are keyed by node ids and node-id pairs respectively
    - nodes defines the variable ordering used in the sampler
    """
    # Map values to spins in {-1, +1}
    spins = [1 if v > 0 else -1 for v in solution]
    e = 0.0
    # Map node id -> position
    node_pos = {int(node_id): pos for pos, node_id in enumerate(nodes)}
    # Local fields
    for pos, node_id in enumerate(nodes[:len(spins)]):
        e += float(h.get(int(node_id), 0.0)) * spins[pos]
    # Couplers
    for (u, v), Jij in J.items():
        pu = node_pos.get(int(u))
        pv = node_pos.get(int(v))
        if pu is not None and pv is not None and pu < len(spins) and pv < len(spins):
            e += float(Jij) * spins[pu] * spins[pv]
    return float(e)


def energies_for_solutions(solutions: List[List[int]], h: Dict[int, float], J: Dict[Tuple[int, int], float], nodes: List[int]) -> List[float]:
    """Compute Ising energies for multiple solutions using vectorized numpy.

    Converts h and J to arrays and computes all energies in one pass.
    ~10x faster than calling energy_of_solution() in a loop for large
    solution counts.

    Accepts either a list of spin lists or a 2D numpy array (n × n_nodes).
    """
    if len(solutions) == 0:
        return []

    n = len(nodes)
    node_pos = {int(nid): pos for pos, nid in enumerate(nodes)}

    # Build h_arr: shape (n,)
    h_arr = np.zeros(n, dtype=np.float64)
    for nid, val in h.items():
        pos = node_pos.get(int(nid))
        if pos is not None:
            h_arr[pos] = val

    # Build J arrays: edge endpoints + values
    edge_u = []
    edge_v = []
    j_vals = []
    for (u, v), val in J.items():
        pu = node_pos.get(int(u))
        pv = node_pos.get(int(v))
        if pu is not None and pv is not None:
            edge_u.append(pu)
            edge_v.append(pv)
            j_vals.append(val)
    edge_u = np.array(edge_u, dtype=np.intp)
    edge_v = np.array(edge_v, dtype=np.intp)
    j_arr = np.array(j_vals, dtype=np.float64)

    # Build spin matrix: shape (n_solutions, n)
    # Fall back to per-solution if lengths are inconsistent
    try:
        spin_matrix = np.array(solutions, dtype=np.float64)
    except ValueError:
        return [energy_of_solution(sol, h, J, nodes) for sol in solutions]
    # Map to {-1, +1}
    spin_matrix = np.where(spin_matrix > 0, 1.0, -1.0)

    # h contribution: sum(h_i * s_i) for each solution
    h_energies = spin_matrix @ h_arr  # (n_solutions,)

    # J contribution: sum(J_ij * s_i * s_j) for each solution
    # Vectorized: s_u * s_v for all edges, then dot with J values
    s_u = spin_matrix[:, edge_u]  # (n_solutions, n_edges)
    s_v = spin_matrix[:, edge_v]  # (n_solutions, n_edges)
    j_energies = (s_u * s_v) @ j_arr  # (n_solutions,)

    return (h_energies + j_energies).tolist()

def calculate_hamming_distance(s1: List[int], s2: List[int]) -> int:
    """Calculate symmetric Hamming distance between two spin arrays.

    For Ising spin variables {-1, +1}, symmetric distance accounts for
    global spin flip symmetry: distance(s, -s) = 0.

    Uses numpy for vectorized operations - much faster than Python loops.
    """
    a1 = np.asarray(s1, dtype=np.int8)
    a2 = np.asarray(s2, dtype=np.int8)

    # Count mismatches (where spins differ)
    distance = np.count_nonzero(a1 != a2)

    # Symmetric: also check inverted (global spin flip)
    distance_inverted = np.count_nonzero(a1 != -a2)

    return min(distance, distance_inverted)


def calculate_diversity(solutions: List[List[int]]) -> float:
    """Average normalized flip-invariant Hamming distance over all pairs.

    Routes through the BLAS distance matrix (one GEMM) rather than a
    Python pairwise loop: at the full mining pool size (~112 solutions ×
    ~4578 spins) the loop converted two 4578-element Python lists to
    arrays per pair across ~6200 pairs, costing ~1.1s/attempt — the
    dominant consumer cost per mining iteration. The GEMM is ~1ms and
    yields identical values (``calculate_hamming_distance`` and the
    matrix both use the flip-invariant metric).
    """
    if len(solutions) < 2:
        return 0.0
    n_features = len(solutions[0])
    if n_features == 0:
        return 0.0

    dist = _compute_distance_matrix_vectorized(solutions)
    iu = np.triu_indices(dist.shape[0], k=1)
    return float(dist[iu].mean() / n_features)


def pack_spins_hex(solution: List[int]) -> str:
    """Compact-encode a {-1, +1} spin vector as hex of packed bits.

    1 bit per spin (1 = +1, 0 = -1). 4578-node topology compresses to
    ~573 bytes / 1146 hex chars per solution — small enough to archive
    top-5 per stored attempt without blowing up disk usage.
    """
    bits = np.array(
        [(s + 1) // 2 for s in solution], dtype=np.uint8,
    )
    return np.packbits(bits).tobytes().hex()


def _unique_rows(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dedup the rows of a 2D spin matrix via a byte (void) view.

    Equivalent to ``np.unique(samples, axis=0, return_index=True,
    return_inverse=True)`` but ~150x faster on wide spin matrices
    (112×4578): viewing each contiguous row as a single opaque ``void``
    scalar turns the per-axis lexsort into a plain 1D sort.

    Returns ``(uniq, first_index, inverse)`` where ``uniq`` are the unique
    rows (byte-sorted order), ``first_index[j]`` is the index of the first
    occurrence of ``uniq[j]`` in ``samples``, and ``uniq[inverse[i]] ==
    samples[i]`` for every row ``i``.
    """
    contiguous = np.ascontiguousarray(samples)
    void_dtype = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    view = contiguous.view(void_dtype).ravel()
    _, first_index, inverse = np.unique(
        view, return_index=True, return_inverse=True,
    )
    return contiguous[first_index], first_index, np.asarray(inverse).ravel()


def gauge_canonicalize(samples: np.ndarray) -> np.ndarray:
    """Collapse spin-flip (Z2) twins by fixing a gauge: anchor spin = +1.

    With zero local field (``h = 0``) the Ising energy is exactly
    flip-invariant, ``E(s) == E(-s)``, so every solution has an equal-energy
    twin ``-s``. The raw row dedup in :func:`_unique_rows` treats ``s`` and
    ``-s`` as distinct rows and double-counts the pair, inflating
    ``n_unique_*`` by up to 2x. Negating each row whose anchor spin (column 0)
    is ``-1`` maps both members of a twin pair to the same canonical
    representative, so the unique-row set built downstream is flip-invariant.

    Applied *before* dedup, this changes the set of unique rows — not just a
    count: both ``n_unique_*`` (the intended fix) and the top-5 rows feeding
    ``top_5_diversity`` then reflect distinct *physical* solutions. The
    diversity metric is already flip-invariant
    (:func:`calculate_hamming_distance`), so at ``h != 0`` — where twins are
    vanishingly rare — canonicalization is a near-no-op; its effect is confined
    to the ``h = 0`` class where twins actually appear. Sampler spins are
    ``{-1, +1}`` so the anchor column is never ``0`` and the convention needs
    no tie-break.

    Args:
        samples: 2D ``(n_reads, n_nodes)`` spin matrix of ``{-1, +1}`` values.

    Returns:
        A new array with each twin pair mapped to its anchor-``+1`` member.
        Rows already in canonical form are copied unchanged.
    """
    contiguous = np.ascontiguousarray(samples)
    if contiguous.size == 0:
        return contiguous
    # +1 keeps the row; -1 negates it. anchor==0 can't occur for ±1 spins,
    # but treating it as "keep" is harmless and avoids a zero-multiply.
    signs = np.where(contiguous[:, 0] < 0, -1, 1).astype(contiguous.dtype)
    return contiguous * signs[:, None]


def compute_solution_meta(
    sampleset, threshold: float, gauge_fix: bool = False,
) -> Tuple[Dict[str, Any], List[List[int]], List[float]]:
    """Solution metadata + top-5 captures for one sampleset.

    Returns ``(meta, top_5_solutions, top_5_energies)`` where:

    - ``meta``: scalar fields safe to embed in the attempts JSONL —
      ``n_unique_total``, ``n_unique_below_threshold``,
      ``top_5_diversity``, ``top_5_energy_ceiling``.
    - ``top_5_solutions``: up to 5 spin vectors sorted by ascending
      energy (i.e., best-energy first). Caller decides whether to
      archive — production writes them via ``SolutionStore`` only for
      ``stored`` / ``submitted`` attempts; canary writes on the same
      criterion.
    - ``top_5_energies``: matching energies, same order.

    Diversity is mean pairwise Hamming distance over the top-5 unique
    samples — the same K that the chain's ``min_solutions`` gate
    typically uses, so this measurement directly reflects whether the
    sampler is producing diverse enough below-target candidates.

    Args:
        sampleset: dimod-style sampleset exposing ``record.energy`` /
            ``record.sample``.
        threshold: Energy gate; ``n_unique_below_threshold`` counts unique
            rows with minimum energy below this.
        gauge_fix: When ``True``, gauge-canonicalize spin rows
            (:func:`gauge_canonicalize`) before dedup so spin-flip twins
            collapse — required for a flip-invariant count gate on zero-field
            (``h = 0``) instances. Defaults to ``False``, leaving the raw-row
            count unchanged. At ``h != 0`` twins are vanishingly rare, so the
            flag is a near-no-op there (raw ≈ flip-invariant); its effect is
            confined to the ``h = 0`` class, where it corrects both
            ``n_unique_below_threshold`` and the top-5 used for diversity.
    """
    try:
        record = sampleset.record
        energies = np.asarray(record.energy, dtype=np.float64)
    except AttributeError:
        return {}, [], []
    if energies.size == 0:
        return {}, [], []

    # Dedup spin rows at C speed and keep the minimum energy per unique
    # row. The previous pure-Python version materialized one ~N-element
    # tuple per read (N≈4578 on full-topology Advantage2) and hashed it,
    # costing ~1s/attempt on CPU-starved nodes — paid on *every* attempt,
    # which defeated the streaming layer's reconstruction-skip. The byte-
    # view dedup runs in ~0.1ms at 112×4578.
    samples = np.asarray(record.sample)
    if gauge_fix:
        samples = gauge_canonicalize(samples)
    uniq, _, inverse = _unique_rows(samples)
    min_energy = np.full(uniq.shape[0], np.inf, dtype=np.float64)
    np.minimum.at(min_energy, inverse, energies)

    # Unique rows by ascending (minimum) energy; best-energy first.
    order = np.argsort(min_energy, kind="stable")
    top_5_arr = uniq[order[:5]]
    top_5_es = [float(e) for e in min_energy[order[:5]]]
    # Only the top-5 rows are materialized as Python lists — diversity
    # reuses the existing flip-invariant metric (≤10 pairs, negligible).
    top_5 = [row.tolist() for row in top_5_arr]

    meta = {
        "n_unique_total": int(uniq.shape[0]),
        "n_unique_below_threshold": int(np.count_nonzero(min_energy < threshold)),
        "top_5_diversity": (
            calculate_diversity(top_5) if len(top_5) >= 2 else 0.0
        ),
        "top_5_energy_ceiling": top_5_es[-1] if top_5_es else None,
    }
    return meta, top_5, top_5_es


def _calculate_set_diversity(indices: List[int], dist_matrix: np.ndarray) -> float:
    """Calculate average pairwise distance for a set of solutions."""
    if len(indices) < 2:
        return 0.0

    sub = dist_matrix[np.ix_(indices, indices)]
    iu = np.triu_indices(sub.shape[0], k=1)
    return float(sub[iu].mean())


def _compute_distance_matrix_vectorized(solutions: List[List[int]]) -> np.ndarray:
    """Symmetric (flip-invariant) Hamming distance matrix via one GEMM.

    For spins in {-1, +1}, ``Sᵢ·Sⱼ = N − 2·hamming(i, j)``, so the
    flip-invariant distance ``min(hamming, N − hamming)`` is::

        D = (N − |S · Sᵀ|) / 2

    This is a single BLAS matrix multiply with O(n²) memory — milliseconds
    at n≈112, N≈4578 — versus materializing an n×n×N tensor and counting.
    float32 is exact here: every product is ±1 and the sums stay integers
    well under 2²⁴.
    """
    arr = np.asarray(solutions, dtype=np.float32)  # n × N, values ±1
    n_features = arr.shape[1]
    gram = arr @ arr.T                              # n × n, one GEMM
    dist = (n_features - np.abs(gram)) / 2.0
    # Defensive no-op: the diagonal is exactly 0 by construction
    # (row·row = N), set explicitly for clarity.
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float64)


def select_diverse_solutions(solutions: List[List[int]], target_count: int) -> List[int]:
    """Filter solutions to maintain maximum diversity using farthest point sampling.

    Uses farthest point sampling with local search refinement.
    This method provides better diversity than pure greedy selection.
    """
    if len(solutions) <= target_count:
        return list(range(0, len(solutions)))

    n_solutions = len(solutions)

    # Pre-compute distance matrix using vectorized operations (MUCH faster)
    dist_matrix = _compute_distance_matrix_vectorized(solutions)

    # Farthest Point Sampling
    # Start with the two most distant points (use numpy to find max)
    # Only look at upper triangle to avoid duplicates
    upper_tri = np.triu(dist_matrix, k=1)
    max_idx = np.unravel_index(np.argmax(upper_tri), upper_tri.shape)
    selected_indices = list(max_idx)

    # Iteratively add the farthest point from the current set
    while len(selected_indices) < target_count:
        # Get distances to all selected points
        selected_arr = np.array(selected_indices)
        min_dists = np.min(dist_matrix[:, selected_arr], axis=1)

        # Mask already selected
        min_dists[selected_arr] = -1

        # Find point with maximum minimum distance
        best_idx = np.argmax(min_dists)
        selected_indices.append(best_idx)

    # Optional: Local search refinement (limited iterations for performance)
    # Try swapping elements to improve total diversity
    improved = True
    iterations = 0
    max_iterations = 5  # Reduced from 10 for performance

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        current_div = _calculate_set_diversity(selected_indices, dist_matrix)

        for i in range(len(selected_indices)):
            # Check a subset of candidates (not all) for performance
            # Sample up to 50 random candidates if n_solutions is large
            if n_solutions > 100:
                candidates = np.random.choice(n_solutions, min(50, n_solutions), replace=False)
            else:
                candidates = range(n_solutions)

            for cand_idx in candidates:
                if cand_idx in selected_indices:
                    continue

                # Try swapping
                test_indices = selected_indices.copy()
                test_indices[i] = cand_idx
                test_div = _calculate_set_diversity(test_indices, dist_matrix)

                if test_div > current_div:
                    selected_indices[i] = cand_idx
                    current_div = test_div
                    improved = True
                    break

            if improved:
                break

    return selected_indices


def _validate_topology_consistency(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    nodes: List[int],
    edges: Optional[List[Tuple[int, int]]] = None,
    allowed_h_values: Optional[List[float]] = None
) -> List[str]:
    """Validate that h, J parameters match expected topology and constraints.

    Args:
        h: Field parameters dictionary
        J: Coupling parameters dictionary
        nodes: List of node indices for the topology
        edges: List of edges in the topology (if None, uses DEFAULT_TOPOLOGY edges)
        allowed_h_values: List of valid h values (default: any float)
                         Set to validate h values are in allowed set

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Use provided nodes
    expected_nodes = set(nodes)

    # Use provided edges or fall back to DEFAULT_TOPOLOGY
    if edges is not None:
        expected_edges = set(edges)
    else:
        expected_edges = set(DEFAULT_TOPOLOGY.graph.edges())

    # 2. Validate h parameters
    h_nodes = set(h.keys())

    # Validate h values are in allowed set (if specified)
    if allowed_h_values is not None:
        allowed_set = set(allowed_h_values)
        for node_id in h_nodes:
            if node_id not in expected_nodes:
                errors.append(f"h parameter for invalid node: {node_id}")
            elif h[node_id] not in allowed_set:
                errors.append(
                    f"Invalid h[{node_id}] = {h[node_id]}, "
                    f"expected one of {allowed_h_values}"
                )
    else:
        # No allowed_h_values specified, just check nodes are valid
        for node_id in h_nodes:
            if node_id not in expected_nodes:
                errors.append(f"h parameter for invalid node: {node_id}")

    # Check for missing h parameters
    missing_h = expected_nodes - h_nodes
    if missing_h:
        errors.append(f"Missing h parameters for nodes: {sorted(missing_h)}")
    
    # 3. Validate J parameters (couplings)
    j_edges = set()
    for (u, v) in J.keys():
        # Normalize edge order for comparison
        edge = (min(u, v), max(u, v))
        j_edges.add(edge)
        
        # Check edge exists in topology
        if edge not in expected_edges and (edge[1], edge[0]) not in expected_edges:
            errors.append(f"J parameter for invalid edge: ({u}, {v})")
        
        # Check J values are ±1
        j_val = J[(u, v)]
        if j_val not in [-1.0, 1.0]:
            errors.append(f"Invalid J value J[({u}, {v})] = {j_val} (expected ±1.0)")
    
    # Normalize expected edges for comparison  
    normalized_expected = set()
    for (u, v) in expected_edges:
        normalized_expected.add((min(u, v), max(u, v)))
    
    # Check for missing J parameters
    missing_j = normalized_expected - j_edges
    if missing_j:
        errors.append(f"Missing J parameters for edges: {sorted(missing_j)}")
    
    return errors


def validate_solution(spins: List[int], h: Dict[int, float], J: Dict[Tuple[int, int], float], nodes: List[int], edges: Optional[List[Tuple[int, int]]] = None) -> Dict[str, Any]:
    """Validate an Ising model solution for correctness.

    Args:
        spins: Spin configuration as list of {-1, +1} values
        h: Field parameters dictionary
        J: Coupling parameters dictionary
        nodes: List of node indices for the topology
        edges: List of edges in the topology (optional, for validation)

    Returns:
        Dictionary with validation results including validity status and energy
    """
    n = len(nodes)
    node_to_pos = {node_id: pos for pos, node_id in enumerate(nodes)}
    
    result = {
        "valid": True,
        "errors": [],
        "energy": 0.0,
        "satisfaction_rate": 0.0
    }
    
    # 1. Basic format validation
    if len(spins) != n:
        result["valid"] = False
        result["errors"].append(f"Wrong solution length: {len(spins)} != {n}")
        return result
    
    # 2. Check values are {-1, +1}
    unique_values = set(spins)
    if not unique_values.issubset({-1, 1}):
        invalid_values = unique_values - {-1, 1}
        result["valid"] = False  
        result["errors"].append(f"Invalid spin values: {invalid_values} (must be -1 or +1)")
        return result
    
    # 3. Validate topology consistency
    topology_errors = _validate_topology_consistency(h, J, nodes, edges)
    if topology_errors:
        result["valid"] = False
        result["errors"].extend(topology_errors)
        return result
    
    # Calculate energy using existing function
    result["energy"] = energy_of_solution(spins, h, J, nodes)
    
    # Calculate coupling satisfaction rate
    satisfied_couplings = 0
    total_couplings = len(J)
    
    for (node_i, node_j), val in J.items():
        pos_i = node_to_pos.get(int(node_i))
        pos_j = node_to_pos.get(int(node_j))
        
        if pos_i is not None and pos_j is not None:
            spin_i = spins[pos_i]
            spin_j = spins[pos_j]
            coupling_energy = val * spin_i * spin_j
            
            if coupling_energy < 0:  # Satisfied coupling
                satisfied_couplings += 1
    
    result["satisfaction_rate"] = satisfied_couplings / total_couplings if total_couplings > 0 else 0
    
    return result


def _energy_stratified_selection(
    solutions: List[List[int]],
    energies: List[float],
    target_count: int,
) -> Optional[List[int]]:
    """Select solutions from different energy strata for diversity.

    When all valid solutions cluster in one energy basin (low diversity),
    picking from different energy levels pulls in solutions from different
    spin basins. Solutions at slightly worse energies are structurally
    different from the best — they represent alternative local minima.

    Divides the energy range into target_count equal bands and picks
    one solution per band (the most central in each band). Falls back
    to evenly-spaced indices if any stratum is empty.

    Args:
        solutions: Valid solutions (all below energy threshold).
        energies: Corresponding energies (same order).
        target_count: Number of solutions to select.

    Returns:
        List of selected indices, or None if fewer than target_count
        solutions are available.
    """
    if len(solutions) < target_count:
        return None

    # Sort by energy (best = most negative first)
    order = sorted(range(len(energies)), key=lambda i: energies[i])
    sorted_energies = [energies[i] for i in order]

    # Try stratified: divide energy range into equal bands
    e_best = sorted_energies[0]
    e_worst = sorted_energies[-1]
    e_range = e_worst - e_best

    if e_range > 0:
        band_size = e_range / target_count
        selected = []
        for band in range(target_count):
            band_lo = e_best + band * band_size
            band_hi = band_lo + band_size
            # Find solutions in this band
            candidates = [
                order[i] for i, e in enumerate(sorted_energies)
                if (band_lo <= e < band_hi or (band == target_count - 1 and e <= band_hi))
                and order[i] not in selected
            ]
            if candidates:
                # Pick the one closest to the band center
                band_mid = (band_lo + band_hi) / 2
                best_cand = min(candidates, key=lambda i: abs(energies[i] - band_mid))
                selected.append(best_cand)

        if len(selected) >= target_count:
            return selected[:target_count]

    # Fallback: evenly-spaced indices through the energy-sorted list
    n = len(order)
    step = max(1, n // target_count)
    selected = [order[i * step] for i in range(min(target_count, n))]

    # Fill remaining if step didn't give enough
    if len(selected) < target_count:
        remaining = [idx for idx in order if idx not in set(selected)]
        selected.extend(remaining[:target_count - len(selected)])

    return selected[:target_count] if len(selected) >= target_count else None


def _ising_from_requirements(
    requirements,
    nonce: Union[int, bytes],
    nodes: List[int],
    edges: List[Tuple[int, int]],
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float]]:
    """Generate (h, J) from requirements, respecting per-block allowed value specs."""
    allowed_h = getattr(requirements, "allowed_h_values", DEFAULT_ALLOWED_H)
    allowed_j = getattr(requirements, "allowed_j_values", DEFAULT_ALLOWED_J)
    return generate_ising_model_from_nonce(
        nonce, nodes, edges, allowed_h=allowed_h, allowed_j=allowed_j,
    )


def _dedup_and_validate(
    sample_arr: np.ndarray,
    energy_arr: np.ndarray,
    skip_validation: bool,
    requirements,
    nonce: Union[int, bytes],
    nodes: List[int],
    edges: List[Tuple[int, int]],
    h: Optional[Dict[int, float]],
    J: Optional[Dict[Tuple[int, int], float]],
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[int, float]],
           Optional[Dict[Tuple[int, int], float]]]:
    """Dedup the full batch into the constraint-valid set of unique rows.

    Returns the unique solutions (int8 ndarray), their first-occurrence
    energies, and the (possibly regenerated) ``h``/``J`` Ising model so the
    slow path's regeneration propagates back to the caller.
    """
    if skip_validation:
        # FAST PATH: Trust sampler output, skip per-solution validation
        # (safe during mining — we control the sampler). _unique_rows
        # dedups rows at C speed; first_index + a stable argsort reproduce
        # the legacy dict loop exactly: unique rows in first-seen order,
        # each carrying its first occurrence's energy.
        uniq, first_idx, _ = _unique_rows(sample_arr)
        first_seen = np.argsort(first_idx, kind="stable")
        return uniq[first_seen], energy_arr[first_idx[first_seen]], h, J

    # SLOW PATH: Full validation for untrusted sources (block
    # validation). Per-solution validate_solution needs Python lists;
    # this path is not the mining hot loop, so it stays list-based
    # and is converted to an ndarray pool at the end for the shared
    # downstream. Use pre-computed Ising model if provided, otherwise
    # regenerate. Requirements may carry `allowed_h_values` /
    # `allowed_j_values` (post-MR-!20) for non-default distributions.
    if h is None or J is None:
        h, J = _ising_from_requirements(requirements, nonce, nodes, edges)

    seen: set = set()
    valid_rows: List[List[int]] = []
    valid_row_energies: List[float] = []
    invalid_solutions = []
    for idx in range(len(energy_arr)):
        solution = tuple(sample_arr[idx])
        if solution not in seen:
            seen.add(solution)
            solution_list = list(solution)

            # Validate solution format and correctness
            validation_result = validate_solution(solution_list, h, J, nodes, edges)
            if validation_result["valid"]:
                valid_rows.append(solution_list)
                valid_row_energies.append(float(energy_arr[idx]))
            else:
                invalid_solutions.append({
                    "solution": solution_list,
                    "errors": validation_result["errors"]
                })

    # Log any invalid solutions found
    if invalid_solutions:
        local_logger = logging.getLogger(__name__)
        local_logger.warning(f"Found {len(invalid_solutions)} invalid solutions with errors: {[s['errors'] for s in invalid_solutions[:3]]}")

    full_unique_solutions = (
        np.asarray(valid_rows, dtype=sample_arr.dtype)
        if valid_rows
        else np.empty((0, sample_arr.shape[1]), dtype=sample_arr.dtype)
    )
    full_unique_energies = np.asarray(valid_row_energies, dtype=np.float64)
    return full_unique_solutions, full_unique_energies, h, J


def _select_pool_indices(
    full_unique_solutions: np.ndarray,
    full_unique_energies: np.ndarray,
    difficulty_energy: float,
    min_solutions: int,
    strict_energy: bool,
    live_threshold_energy: Optional[float],
) -> np.ndarray:
    """Select the diverse-K candidate pool indices. Three modes:

    1. Strict (legacy / mempool): pool = below snapshot target.
       (best-vs-difficulty already enforced by the caller.)

    2. Lenient + live threshold (substrate ratchet, common case):
       tighten pool to below-target *when* it has enough samples
       to satisfy min_solutions, so the diverse-K's recomputed
       floor stays under the live target and the submit gate
       fires. When the subset is too thin (e.g. right after a
       chain re-snapshot), fall back to the full constraint-valid
       set — the iter can't submit now but it lands in the top-K
       stash for visibility and future submission once
       ``BlockDecayInterval`` raises the live target past its
       floor.

    3. Lenient + no live (tests / legacy): use full pool.
    """
    if strict_energy:
        return np.flatnonzero(full_unique_energies < difficulty_energy)
    if live_threshold_energy is not None:
        below_target = np.flatnonzero(
            full_unique_energies < live_threshold_energy
        )
        if len(below_target) >= min_solutions:
            return below_target
        return np.arange(len(full_unique_solutions))
    return np.arange(len(full_unique_solutions))


def _select_diverse_with_fallback(
    valid_solutions: np.ndarray,
    valid_energies: np.ndarray,
    min_solutions: int,
    min_diversity: float,
    best_energy: float,
) -> Tuple[np.ndarray, float, float]:
    """Select diverse solutions, falling back to energy-stratified selection.

    Tries farthest-point selection first; if its diversity is below
    ``min_diversity`` (and there's slack), retries with energy-stratified
    selection. Returns the filtered solutions, their diversity, and the
    best (minimum) energy among the selection. ``best_energy`` carries the
    caller's current value so the (unreachable in the success path) no-op
    branches preserve it unchanged.
    """
    filtered_solutions = valid_solutions
    diversity = 0.0
    if len(valid_solutions) >= min_solutions:
        selected_indices = select_diverse_solutions(
            valid_solutions, min_solutions,
        )
        filtered_solutions = valid_solutions[selected_indices]
        diversity = calculate_diversity(filtered_solutions)
        best_energy = float(valid_energies[selected_indices].min())

        # Fallback: if farthest-point selection doesn't meet diversity,
        # try energy-stratified selection. Solutions at different energy
        # levels are more likely to be in different spin basins.
        if diversity < min_diversity and len(valid_solutions) > min_solutions:
            stratified = _energy_stratified_selection(
                valid_solutions, valid_energies, min_solutions,
            )
            if stratified is not None:
                strat_div = calculate_diversity(
                    valid_solutions[stratified]
                )
                if strat_div >= min_diversity:
                    selected_indices = stratified
                    filtered_solutions = valid_solutions[selected_indices]
                    diversity = strat_div
                    best_energy = float(
                        valid_energies[selected_indices].min()
                    )
    elif len(valid_energies):
        best_energy = float(valid_energies.min())

    return filtered_solutions, diversity, best_energy


def evaluate_sampleset(sampleset, requirements, nodes: List[int], edges: List[Tuple[int, int]],
                      nonce: Union[int, bytes], salt: bytes, prev_timestamp: int, start_time: float,
                      miner_id: str, miner_type: str,
                      h: Optional[Dict[int, float]] = None,
                      J: Optional[Dict[Tuple[int, int], float]] = None,
                      skip_validation: bool = True,
                      strict_energy: bool = True,
                      live_threshold_energy: Optional[float] = None):
    """Convert a sample set into a mining result if it meets requirements, otherwise return None.

    Args:
        sampleset: dimod.SampleSet from the sampler
        requirements: BlockRequirements object with difficulty settings
        nodes: List of node indices for the topology
        edges: List of edge tuples for the topology
        nonce: Nonce used for this mining attempt
        salt: Salt bytes used for this mining attempt
        prev_timestamp: Timestamp from previous block
        start_time: Start time of mining attempt
        miner_id: ID of the miner
        miner_type: Type of the miner (CPU, GPU, QPU)
        h: Optional pre-computed field parameters (avoids regeneration)
        J: Optional pre-computed coupling parameters (avoids regeneration)
        skip_validation: If True, skip per-solution validation (faster for mining).
                        Set to False for block validation from other miners.
        strict_energy: If True (default), require best_energy <= difficulty_energy
                        and discard the sample otherwise. If False (substrate
                        ratchet path), the energy gate is dropped — every
                        sample considered, MiningResult returned whenever
                        diversity + min_solutions still pass. Caller decides
                        whether the returned best_energy is currently
                        eligible for chain submission against the *live*
                        decayed threshold.

    Returns:
        MiningResult if successful, None if requirements not met
    """
    difficulty_energy = requirements.difficulty_energy
    min_diversity = requirements.min_diversity
    min_solutions = requirements.min_solutions
    best_energy = float('inf')
    # num_valid = count of unique solutions that meet the energy gate
    # (sampler energies in snapshot mode; recomputed energies against the
    # live decayed threshold in ratchet mode). This is the only count
    # that matters for chain acceptance — "did we produce ≥ min_solutions
    # below-target samples?". The pre-dedup read count is implied by
    # the configured num_reads, not reported here.
    num_valid = 0
    diversity = 0.0
    result = None

    try:
        all_energies = sampleset.record.energy
        if len(all_energies) == 0:
            raise ValueError("No samples in sampleset")

        best_energy = float(np.min(all_energies))

        # Strict mode bail-fast: nothing below-target in the snapshot
        # means nothing useful in this iter. (Lenient mode falls through
        # to pool-fallback below, where the iter can still stash.)
        if strict_energy and best_energy > difficulty_energy:
            raise ValueError(
                f"Best energy {best_energy} exceeds difficulty energy {difficulty_energy}"
            )

        # Dedup the FULL batch into the constraint-valid set, BEFORE any
        # target filter. The dedup pool feeds both ``num_valid``
        # (count of below-target solutions) and the diverse-K selection
        # used to compute diversity. The pool is kept as an int8 ndarray
        # end-to-end: the diversity GEMM and farthest-point selection take
        # ndarrays directly, so the hot loop never pays the list<->array
        # conversion that dominated per-attempt cost (~1s on full-topology
        # samplesets) before.
        sample_arr = np.asarray(sampleset.record.sample)
        energy_arr = np.asarray(all_energies, dtype=np.float64)

        # Dedup (fast path) or dedup+validate (slow path) into the
        # constraint-valid pool. The slow path may regenerate ``h``/``J``;
        # capture them back so the submit-floor recompute below reuses them.
        full_unique_solutions, full_unique_energies, h, J = _dedup_and_validate(
            sample_arr, energy_arr, skip_validation,
            requirements, nonce, nodes, edges, h, J,
        )

        # Set num_valid here so it reflects actual sampleset shape even
        # when later validation raises. Without this, the finally-block
        # log line would print "Valid: 0" on any "insufficient valid
        # solutions" rejection, making it impossible to tell whether the
        # sampler produced 0 below-target solutions or 4 (just shy of
        # min_solutions=5). In ratchet mode this gets recomputed against
        # the live decayed threshold using chain-recomputed energies; see
        # the post-success-path block below.
        num_valid = int(np.count_nonzero(full_unique_energies < difficulty_energy))

        # Pool selection for diverse-K (see _select_pool_indices for the
        # three strict/ratchet/lenient modes).
        pool_indices = _select_pool_indices(
            full_unique_solutions, full_unique_energies,
            difficulty_energy, min_solutions, strict_energy,
            live_threshold_energy,
        )

        valid_solutions = full_unique_solutions[pool_indices]
        valid_energies = full_unique_energies[pool_indices]

        if len(valid_solutions) < min_solutions:
            # Diversity snapshot for the rejection log only (the `finally`
            # below prints it). Without it the log shows diversity=0.000
            # (the init value), hiding diverse-but-too-few from clustered.
            # The success path doesn't need it — it overwrites `diversity`
            # via farthest-point selection — so this full-pool GEMM stays
            # off the per-attempt hot path.
            if len(valid_solutions) >= 2:
                diversity = calculate_diversity(valid_solutions)
            raise ValueError(f"Insufficient valid solutions: {len(valid_solutions)} < {min_solutions}")

        # Select diverse solutions — try farthest-point first, then
        # fall back to energy-stratified selection if diversity is too low.
        filtered_solutions, diversity, best_energy = _select_diverse_with_fallback(
            valid_solutions, valid_energies, min_solutions,
            min_diversity, best_energy,
        )

        if diversity < min_diversity:
            raise ValueError(f"Insufficient diversity: {diversity} < {min_diversity}")

        # Independently recompute energies for the SELECTED solutions and
        # take the worst-case (max) as the submit floor. The chain's
        # ``validate_proof`` filters each submitted solution with strict
        # ``energy < max_energy_milli`` before checking
        # ``valid_solution_count >= min_solutions`` (see
        # ``pallets/quantum-pow/src/lib.rs:858``), so submission passes
        # only when EVERY solution clears the threshold. Sampler-reported
        # energies (``valid_energies``) can drift from chain-computed
        # ones at the milli boundary — at the tail of a long mining
        # round the headline best is only a few milli below the threshold
        # while other diverse-selected solutions sit even closer, and
        # those silently fail chain validation. The recompute uses the
        # canonical Ising formula, matching the chain.
        if h is None or J is None:
            h_for_floor, J_for_floor = _ising_from_requirements(requirements, nonce, nodes, edges)
        else:
            h_for_floor, J_for_floor = h, J
        selected_energies = energies_for_solutions(
            filtered_solutions, h_for_floor, J_for_floor, nodes,
        )
        submit_floor_energy = max(selected_energies) if selected_energies else best_energy

        # In ratchet mode, upgrade num_valid to the chain-recomputed
        # count against the live decayed threshold. Sampler-reported
        # energies drift sub-milli from chain-computed ones on most
        # backends but cross the boundary often enough at the round
        # tail to be misleading, so we eat the extra matmul to keep
        # the diagnostic honest. Skipped for mempool / snapshot-only
        # paths — num_valid stays as the sampler-energy count set
        # right after dedup.
        if live_threshold_energy is not None and len(full_unique_solutions):
            full_recomputed = energies_for_solutions(
                full_unique_solutions, h_for_floor, J_for_floor, nodes,
            )
            num_valid = int(
                sum(1 for e in full_recomputed if e < live_threshold_energy)
            )

        # Create mining result for this attempt
        mining_time = time.time() - start_time

        # Create result for this attempt
        result = MiningResult(
            miner_id=miner_id,
            miner_type=miner_type,
            nonce=_to_nonce_bytes(nonce),
            salt=salt,
            timestamp=int(time.time()),
            prev_timestamp=prev_timestamp,
            # Materialize the ≤min_solutions selected rows as Python lists
            # for the result contract (the only list conversion left, and
            # it's tiny — the hot pool stayed an ndarray throughout).
            solutions=[row.tolist() for row in filtered_solutions],
            energy=best_energy,
            diversity=diversity,
            num_valid=num_valid,
            mining_time=int(mining_time),
            node_list=nodes,
            edge_list=edges,
            variable_order=nodes,
            submit_floor_energy=submit_floor_energy,
        )
    except ValueError as e:
        # Use module logger for consistency
        logger.debug(f"Failed to meet requirements: {e}")
    finally:
        # Log every mining attempt (successful or not) for analysis
        logger.info(f"[{miner_id}] Mining attempt - Energy: {best_energy:.0f}, Valid: {num_valid} (best {min_solutions} diversity: {diversity:.3f}) (requirements: energy<={difficulty_energy:.0f}, valid>={min_solutions}, diversity>={min_diversity:.3f})")
    return result
