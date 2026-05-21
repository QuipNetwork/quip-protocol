"""Fast greedy descent pre-filter for nonce scoring.

Provides a cheap CPU-based estimate of Ising landscape quality
for a given nonce, enabling batch scoring and filtering before
expensive SA runs.

Uses scipy sparse matrix-vector multiply for vectorized local field
computation. Generates h/J values directly as numpy arrays (same ChaCha8Rng
logic as generate_ising_model_from_nonce) to avoid Python dict overhead.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from shared.allowed_value_spec import (
    AllowedValueSpec,
    MILLI_SCALE,
    sample as _sample_spec,
)
from shared.chacha8 import ChaCha8Rng
from shared.quantum_proof_of_work import (
    DEFAULT_ALLOWED_H,
    DEFAULT_ALLOWED_J,
    derive_nonce,
    generate_ising_model_from_nonce,
    _to_nonce_bytes,
)


class IsingTopologyCache:
    """Pre-computed CSR structure for a D-Wave topology.

    Building the CSR index arrays once costs ~10ms. After that,
    scoring a nonce via greedy descent costs ~1ms.
    """

    def __init__(
        self,
        nodes: List[int],
        edges: List[Tuple[int, int]],
    ) -> None:
        self.nodes = nodes
        self.n = len(nodes)
        self.n_edges = len(edges)
        self.node_to_pos = {nid: i for i, nid in enumerate(nodes)}

        # Store edge list for generate_ising_model_from_nonce compatibility
        self._edges = edges

        # Build COO indices (symmetric: each edge -> 2 entries)
        nnz = 2 * self.n_edges
        coo_rows = np.empty(nnz, dtype=np.int32)
        coo_cols = np.empty(nnz, dtype=np.int32)

        for i, (u, v) in enumerate(edges):
            pu = self.node_to_pos[u]
            pv = self.node_to_pos[v]
            coo_rows[2 * i] = pu
            coo_cols[2 * i] = pv
            coo_rows[2 * i + 1] = pv
            coo_cols[2 * i + 1] = pu

        self._nnz = nnz

        # Build template CSR to get COO->CSR permutation
        coo_ids = np.arange(nnz, dtype=np.float64)
        template = csr_matrix(
            (coo_ids, (coo_rows, coo_cols)),
            shape=(self.n, self.n),
        )

        # Map: for each edge index i, the CSR data positions
        # where J[edge_i] values should go
        self._edge_csr_pos_fwd = np.empty(self.n_edges, dtype=np.int32)
        self._edge_csr_pos_rev = np.empty(self.n_edges, dtype=np.int32)
        coo_to_csr = np.empty(nnz, dtype=np.int32)
        for csr_pos in range(nnz):
            coo_idx = int(template.data[csr_pos])
            coo_to_csr[coo_idx] = csr_pos
        for i in range(self.n_edges):
            self._edge_csr_pos_fwd[i] = coo_to_csr[2 * i]
            self._edge_csr_pos_rev[i] = coo_to_csr[2 * i + 1]

        # Store fixed CSR structure
        self._indptr = template.indptr.copy()
        self._indices = template.indices.copy()

    def greedy_descent_fast(
        self,
        nonce,
        num_passes: int = 3,
        num_starts: int = 4,
        allowed_h: AllowedValueSpec = DEFAULT_ALLOWED_H,
        allowed_j: AllowedValueSpec = DEFAULT_ALLOWED_J,
    ) -> float:
        """Run greedy descent using array-based Ising generation.

        Reproduces the same RNG logic as
        :func:`shared.quantum_proof_of_work.generate_ising_model_from_nonce`
        (ChaCha8Rng seeded from the 32-byte nonce, h first then J) but
        outputs numpy arrays directly to avoid Python dict overhead.

        ``nonce`` may be either a 32-byte buffer (canonical) or an int that
        fits in 256 bits (convenience for tools / benchmarks that don't go
        through :func:`shared.quantum_proof_of_work.derive_nonce`).
        """
        n = self.n
        rng = ChaCha8Rng.from_seed(_to_nonce_bytes(nonce))

        # Generate h values FIRST (matches generate_ising_model_from_nonce)
        h_arr = np.empty(n, dtype=np.float64)
        for i in range(n):
            h_arr[i] = _sample_spec(allowed_h, rng) / MILLI_SCALE

        # Generate J values SECOND
        j_vals = np.empty(self.n_edges, dtype=np.float64)
        for i in range(self.n_edges):
            j_vals[i] = _sample_spec(allowed_j, rng) / MILLI_SCALE

        # Fill CSR data using vectorized indexing
        csr_data = np.empty(self._nnz, dtype=np.float64)
        csr_data[self._edge_csr_pos_fwd] = j_vals
        csr_data[self._edge_csr_pos_rev] = j_vals

        J_mat = csr_matrix(
            (csr_data, self._indices, self._indptr),
            shape=(n, n),
            copy=False,
        )

        inner_rng = np.random.default_rng()
        best_energy = np.inf

        for _ in range(num_starts):
            spins = (
                2 * inner_rng.integers(0, 2, size=n) - 1
            ).astype(np.float64)

            for _ in range(num_passes):
                local_field = h_arr + J_mat.dot(spins)
                new_spins = -np.sign(local_field)
                zero_mask = new_spins == 0
                new_spins[zero_mask] = spins[zero_mask]
                spins = new_spins

            energy = float(
                np.dot(h_arr, spins)
                + 0.5 * spins.dot(J_mat.dot(spins)),
            )
            if energy < best_energy:
                best_energy = energy

        return float(best_energy)

    def greedy_descent(
        self,
        h: Dict[int, float],
        J: Dict[Tuple[int, int], float],
        num_passes: int = 3,
        num_starts: int = 4,
    ) -> float:
        """Run greedy descent from pre-built h/J dicts (slower path).

        Use greedy_descent_fast() when you have the nonce directly.
        """
        n = self.n

        h_arr = np.empty(n, dtype=np.float64)
        for i, nid in enumerate(self.nodes):
            h_arr[i] = h.get(nid, 0.0)

        j_vals = np.empty(self.n_edges, dtype=np.float64)
        for i, (u, v) in enumerate(self._edges):
            j_vals[i] = J[(u, v)]

        csr_data = np.empty(self._nnz, dtype=np.float64)
        csr_data[self._edge_csr_pos_fwd] = j_vals
        csr_data[self._edge_csr_pos_rev] = j_vals

        J_mat = csr_matrix(
            (csr_data, self._indices, self._indptr),
            shape=(n, n),
            copy=False,
        )

        rng = np.random.default_rng()
        best_energy = np.inf

        for _ in range(num_starts):
            spins = (2 * rng.integers(0, 2, size=n) - 1).astype(np.float64)

            for _ in range(num_passes):
                local_field = h_arr + J_mat.dot(spins)
                new_spins = -np.sign(local_field)
                zero_mask = new_spins == 0
                new_spins[zero_mask] = spins[zero_mask]
                spins = new_spins

            energy = float(
                np.dot(h_arr, spins)
                + 0.5 * spins.dot(J_mat.dot(spins)),
            )
            if energy < best_energy:
                best_energy = energy

        return float(best_energy)


# Module-level cache (lazily initialized)
_topology_cache: Dict[int, IsingTopologyCache] = {}


def _get_cache(
    nodes: List[int],
    edges: List[Tuple[int, int]],
) -> IsingTopologyCache:
    """Get or create topology cache (keyed by node count)."""
    key = len(nodes)
    if key not in _topology_cache:
        _topology_cache[key] = IsingTopologyCache(nodes, edges)
    return _topology_cache[key]


def greedy_descent_energy(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    nodes: List[int],
    edges: List[Tuple[int, int]],
    num_passes: int = 3,
    num_starts: int = 4,
) -> float:
    """Estimate Ising ground-state energy via greedy descent.

    Uses cached CSR structure. For batch scoring, prefer
    batch_score_nonces() which uses the faster nonce-based path.

    Args:
        h: Local field dict (node_id -> value).
        J: Coupling dict ((u, v) -> value).
        nodes: Node IDs in topology order.
        edges: Edge tuples in topology.
        num_passes: Greedy alignment iterations per start.
        num_starts: Number of random initializations.

    Returns:
        Best (lowest) energy found across all starts.
    """
    cache = _get_cache(nodes, edges)
    return cache.greedy_descent(h, J, num_passes, num_starts)


def batch_score_nonces(
    last_winning_hash: bytes,
    miner_bytes: bytes,
    nodes: List[int],
    edges: List[Tuple[int, int]],
    batch_size: int = 16,
    keep: int = 4,
) -> List[Tuple[bytes, bytes, dict, dict, float]]:
    """Generate and score a batch of nonces using greedy descent.

    Scores all nonces via the fast array path, then builds h/J dicts
    only for the top ``keep`` candidates (avoiding expensive dict
    construction for rejected nonces).

    ``last_winning_hash`` and ``miner_bytes`` must each be 32 bytes —
    the canonical fixed-width inputs to
    :func:`shared.quantum_proof_of_work.derive_nonce`.

    Returns:
        Sorted list of ``(salt, nonce_bytes, h, J, greedy_energy)``,
        lowest energy first. Length = ``min(keep, batch_size)``.
    """
    cache = _get_cache(nodes, edges)

    # Phase 1: fast scoring (array path, no dicts)
    scored: List[Tuple[bytes, bytes, float]] = []
    for _ in range(batch_size):
        salt = random.randbytes(32)
        nonce = derive_nonce(last_winning_hash, miner_bytes, salt)
        energy = cache.greedy_descent_fast(nonce)
        scored.append((salt, nonce, energy))

    scored.sort(key=lambda c: c[2])

    # Phase 2: build dicts only for top candidates
    top = scored[:keep]
    candidates = []
    for salt, nonce, energy in top:
        h, J = generate_ising_model_from_nonce(nonce, nodes, edges)
        candidates.append((salt, nonce, h, J, energy))

    return candidates
