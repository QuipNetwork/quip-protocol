"""Graphic Matroid Infrastructure for Bonin-de Mier Formula.

This module provides:
1. GraphicMatroid - Cycle matroid of a graph (ground set = edges, rank via forest size)
2. FlatLattice - Lattice of flats with Mobius function and characteristic polynomial
3. Flat enumeration with free Hasse diagram construction
4. Batch Mobius computation
5. Cyclic flat detection

For graphic matroids, flats correspond to sets of edges where each connected
component of the vertex set (restricted to those edges) has all internal edges
included. Equivalently, a flat is a union of edge sets of complete subgraphs
induced by some vertex partition into connected blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from collections import defaultdict

from ..graph import Graph
from ..polynomial import TuttePolynomial


# Type alias for edges
Edge = Tuple[int, int]


# =============================================================================
# GRAPHIC MATROID
# =============================================================================

class GraphicMatroid:
    """Cycle matroid of a graph.

    Ground set = edges of the graph.
    Independent sets = forests (acyclic edge subsets).
    Rank of edge subset A = |V(A)| - components(A) where V(A) = vertices touched by A.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self._ground_set = frozenset(graph.edges)
        self._nodes = frozenset(graph.nodes)
        # Build edge-to-endpoints mapping
        self._edge_endpoints: Dict[Edge, Tuple[int, int]] = {}
        for u, v in graph.edges:
            self._edge_endpoints[(u, v)] = (u, v)
        # Build node-to-edges mapping
        self._node_edges: Dict[int, Set[Edge]] = defaultdict(set)
        for u, v in graph.edges:
            self._node_edges[u].add((u, v))
            self._node_edges[v].add((u, v))

    @property
    def ground_set(self) -> FrozenSet[Edge]:
        return self._ground_set

    def rank(self, edge_subset: FrozenSet[Edge] = None) -> int:
        """Compute rank of edge subset = |V(A)| - components(A).

        If edge_subset is None, compute rank of full ground set.
        """
        if edge_subset is None:
            edge_subset = self._ground_set
        if not edge_subset:
            return 0

        # Find vertices touched by these edges
        vertices = set()
        for u, v in edge_subset:
            vertices.add(u)
            vertices.add(v)

        # Count components using union-find
        parent = {v: v for v in vertices}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
                return True
            return False

        components = len(vertices)
        for u, v in edge_subset:
            if union(u, v):
                components -= 1

        return len(vertices) - components

    def closure(self, edge_subset: FrozenSet[Edge]) -> FrozenSet[Edge]:
        """Compute closure of edge subset.

        For graphic matroids, cl(A) = all edges whose endpoints are in the
        same connected component of the subgraph induced by A.
        """
        if not edge_subset:
            # Closure of empty set = empty set (no vertices connected)
            return frozenset()

        # Find connected components of edge_subset
        vertices = set()
        for u, v in edge_subset:
            vertices.add(u)
            vertices.add(v)

        parent = {v: v for v in self._nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for u, v in edge_subset:
            union(u, v)

        # Add all edges from ground set whose endpoints are in same component
        result = set()
        for u, v in self._ground_set:
            if u in vertices or v in vertices:
                if find(u) == find(v):
                    result.add((u, v))

        return frozenset(result)

    def is_flat(self, edge_subset: FrozenSet[Edge]) -> bool:
        """Check if edge_subset is a flat (closed set)."""
        return self.closure(edge_subset) == edge_subset

    def contract(self, flat: FrozenSet[Edge]) -> 'GraphicMatroid':
        """Contract a flat: merge endpoints of all edges in the flat.

        Returns a new GraphicMatroid on the contracted graph.
        """
        if not flat:
            return self

        # Find connected components induced by flat edges
        vertices = set()
        for u, v in flat:
            vertices.add(u)
            vertices.add(v)

        parent = {v: v for v in self._nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                # Keep the smaller label as representative
                if rx < ry:
                    parent[ry] = rx
                else:
                    parent[rx] = ry

        for u, v in flat:
            union(u, v)

        # Build contracted graph
        # Map each node to its component representative
        node_map = {n: find(n) for n in self._nodes}
        new_nodes = frozenset(node_map.values())

        # Remaining edges (not in flat), remapped
        new_edges = set()
        for u, v in self._ground_set:
            if (u, v) not in flat:
                nu, nv = node_map[u], node_map[v]
                if nu != nv:  # Skip loops
                    edge = (min(nu, nv), max(nu, nv))
                    new_edges.add(edge)

        new_graph = Graph(nodes=new_nodes, edges=frozenset(new_edges))
        return GraphicMatroid(new_graph)

    def contract_to_graph(self, flat: FrozenSet[Edge]) -> Graph:
        """Contract a flat and return the resulting Graph directly."""
        if not flat:
            return self.graph

        parent = {v: v for v in self._nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                if rx < ry:
                    parent[ry] = rx
                else:
                    parent[rx] = ry

        for u, v in flat:
            union(u, v)

        node_map = {n: find(n) for n in self._nodes}
        new_nodes = frozenset(node_map.values())

        new_edges = set()
        for u, v in self._ground_set:
            if (u, v) not in flat:
                nu, nv = node_map[u], node_map[v]
                if nu != nv:
                    edge = (min(nu, nv), max(nu, nv))
                    new_edges.add(edge)

        return Graph(nodes=new_nodes, edges=frozenset(new_edges))


# =============================================================================
# FLAT ENUMERATION (with free Hasse diagram)
# =============================================================================

def enumerate_flats(matroid: GraphicMatroid) -> List[FrozenSet[Edge]]:
    """Enumerate all flats of a graphic matroid using bottom-up closure.

    Algorithm:
    1. Start with the empty flat (rank 0)
    2. For each known flat F and each element e not in F,
       compute cl(F union {e}) to discover new flats
    3. Collect all unique flats

    This is the standard matroid flat enumeration algorithm.
    """
    if not matroid.ground_set:
        return [frozenset()]

    # Level 0: empty flat
    all_flats: Set[FrozenSet[Edge]] = set()
    empty = frozenset()
    # The closure of empty set is empty for graphic matroids
    all_flats.add(empty)

    # Also add cl(empty) in case it's nonempty
    cl_empty = matroid.closure(empty)
    all_flats.add(cl_empty)

    # BFS by rank level
    current_level = {empty}
    if cl_empty != empty:
        current_level = {cl_empty}

    elements = sorted(matroid.ground_set)

    while current_level:
        next_level: Set[FrozenSet[Edge]] = set()
        for flat in current_level:
            for e in elements:
                if e in flat:
                    continue
                # Compute closure of flat + e
                new_flat = matroid.closure(flat | {e})
                if new_flat not in all_flats:
                    all_flats.add(new_flat)
                    next_level.add(new_flat)
        current_level = next_level

    return sorted(all_flats, key=lambda f: (len(f), sorted(f)))


def enumerate_flats_with_hasse(
    matroid: GraphicMatroid,
) -> Tuple[List[FrozenSet[Edge]], List[int], Dict[int, List[int]]]:
    """Enumerate flats via BFS and build Hasse diagram for FREE.

    Key insight: when computing cl(F union {e}) and result has rank(F)+1,
    the result COVERS F. This comes for free during BFS.
    No separate O(n^2) covering-relation construction step.

    Returns:
        flats: List of flats sorted by rank then size
        ranks: ranks[i] = rank of flat i
        upper_covers: upper_covers[i] = list of flat indices covering flat i
    """
    if not matroid.ground_set:
        return [frozenset()], [0], {}

    # Track flats and their indices
    flat_to_idx: Dict[FrozenSet[Edge], int] = {}
    flats: List[FrozenSet[Edge]] = []
    ranks: List[int] = []
    upper_covers: Dict[int, List[int]] = defaultdict(list)

    def register_flat(f: FrozenSet[Edge], r: int) -> int:
        """Register a flat and return its index."""
        if f in flat_to_idx:
            return flat_to_idx[f]
        idx = len(flats)
        flat_to_idx[f] = idx
        flats.append(f)
        ranks.append(r)
        return idx

    # Start with empty flat (rank 0)
    empty = frozenset()
    cl_empty = matroid.closure(empty)
    r_empty = matroid.rank(cl_empty) if cl_empty else 0
    start_flat = cl_empty if cl_empty else empty
    start_idx = register_flat(start_flat, r_empty)

    # Also register the empty flat if different from cl_empty
    if start_flat != empty:
        register_flat(empty, 0)

    elements = sorted(matroid.ground_set)

    # BFS by rank level
    current_level_indices = {start_idx}

    while current_level_indices:
        next_level_indices: Set[int] = set()

        for parent_idx in current_level_indices:
            parent_flat = flats[parent_idx]
            parent_rank = ranks[parent_idx]

            for e in elements:
                if e in parent_flat:
                    continue

                new_flat = matroid.closure(parent_flat | {e})
                new_rank = matroid.rank(new_flat)

                child_idx = register_flat(new_flat, new_rank)

                # If rank increased by exactly 1, this is a covering relation
                if new_rank == parent_rank + 1:
                    # Avoid duplicate covers
                    if child_idx not in upper_covers[parent_idx]:
                        upper_covers[parent_idx].append(child_idx)

                # Only BFS into genuinely new flats
                if child_idx >= len(flats) - 1 and new_flat not in flat_to_idx or child_idx in next_level_indices:
                    # Already tracked via next_level or already known
                    pass
                if flat_to_idx.get(new_flat) == child_idx and child_idx not in current_level_indices:
                    next_level_indices.add(child_idx)

        # Only process flats we haven't processed before
        current_level_indices = {
            idx for idx in next_level_indices
            if idx not in current_level_indices
        }

    # Re-sort by rank for consistency
    order = sorted(range(len(flats)), key=lambda i: (ranks[i], len(flats[i]), sorted(flats[i])))
    old_to_new = {old: new for new, old in enumerate(order)}

    sorted_flats = [flats[i] for i in order]
    sorted_ranks = [ranks[i] for i in order]
    sorted_covers: Dict[int, List[int]] = defaultdict(list)
    for old_idx, covers in upper_covers.items():
        new_idx = old_to_new[old_idx]
        for old_cover in covers:
            sorted_covers[new_idx].append(old_to_new[old_cover])

    return sorted_flats, sorted_ranks, dict(sorted_covers)


# =============================================================================
# CYCLIC FLAT DETECTION
# =============================================================================

def is_cyclic_flat(matroid: GraphicMatroid, edge_subset: FrozenSet[Edge]) -> bool:
    """A flat is cyclic if every edge is in a cycle within the subgraph.

    For graphic matroids: no bridges in the edge-induced subgraph.
    The empty set is considered cyclic (vacuously true).
    """
    if not edge_subset:
        return True

    if not matroid.is_flat(edge_subset):
        return False

    # Check for bridges: an edge is a bridge if removing it disconnects a component
    # Build adjacency for the edge-induced subgraph
    adj: Dict[int, Set[int]] = defaultdict(set)
    vertices = set()
    for u, v in edge_subset:
        adj[u].add(v)
        adj[v].add(u)
        vertices.add(u)
        vertices.add(v)

    if not vertices:
        return True

    # Use DFS bridge detection
    visited = set()
    disc = {}
    low = {}
    parent = {}
    time = [0]
    has_bridge = [False]

    def dfs(u: int):
        visited.add(u)
        disc[u] = low[u] = time[0]
        time[0] += 1
        for v in adj[u]:
            if v not in visited:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    has_bridge[0] = True
                    return
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    for start in vertices:
        if start not in visited:
            dfs(start)
            if has_bridge[0]:
                return False

    return True


def enumerate_cyclic_flats(matroid: GraphicMatroid) -> List[FrozenSet[Edge]]:
    """Filter enumerate_flats() to cyclic flats only.

    Useful for validation: T(N) via cyclic flats must match T(N) via SP.
    """
    all_flats = enumerate_flats(matroid)
    return [f for f in all_flats if is_cyclic_flat(matroid, f)]


# =============================================================================
# FLAT LATTICE WITH MOBIUS FUNCTION
# =============================================================================

# Maximum number of flats for which we build a full lattice
MAX_FLATS_FOR_LATTICE = 50000


class FlatLattice:
    """Lattice of flats with Mobius function and characteristic polynomial.

    The lattice is ordered by inclusion. The bottom element is the empty flat,
    the top element is the full ground set.

    Supports two construction modes:
    1. From matroid only (backward-compatible): enumerates flats and builds Hasse
    2. From pre-built data: accepts flats, ranks, upper_covers from
       enumerate_flats_with_hasse() for O(n) construction
    """

    def __init__(
        self,
        matroid: GraphicMatroid,
        flats: Optional[List[FrozenSet[Edge]]] = None,
        ranks: Optional[List[int]] = None,
        upper_covers: Optional[Dict[int, List[int]]] = None,
    ):
        self.matroid = matroid

        if flats is not None and ranks is not None:
            # Use pre-built data (from enumerate_flats_with_hasse)
            self._flats = flats
            self._ranks = ranks
        else:
            # Backward-compatible: enumerate flats ourselves
            self._flats = enumerate_flats(matroid)
            self._ranks = [matroid.rank(f) for f in self._flats]

        # Build flat index for quick lookup
        self._flat_index: Dict[FrozenSet[Edge], int] = {
            f: i for i, f in enumerate(self._flats)
        }

        # Group flats by rank for efficient interval queries
        self._flats_by_rank: Dict[int, List[int]] = defaultdict(list)
        for i, r in enumerate(self._ranks):
            self._flats_by_rank[r].append(i)

        if upper_covers is not None:
            # Use pre-built Hasse diagram
            self._upper_covers = defaultdict(list, upper_covers)
        else:
            # Build covering relation O(|flats_r| * |flats_{r+1}|) per rank
            self._upper_covers: Dict[int, List[int]] = defaultdict(list)
            for r in sorted(self._flats_by_rank.keys()):
                if r + 1 not in self._flats_by_rank:
                    continue
                for i in self._flats_by_rank[r]:
                    for j in self._flats_by_rank[r + 1]:
                        if self._flats[i].issubset(self._flats[j]):
                            self._upper_covers[i].append(j)

        # Build lower covers (inverse of upper_covers) for batch Mobius
        self._lower_covers: Dict[int, List[int]] = defaultdict(list)
        for i, covers in self._upper_covers.items():
            for j in covers:
                self._lower_covers[j].append(i)

        # Mobius cache (computed lazily)
        self._mobius_cache: Dict[Tuple[int, int], int] = {}
        # Batch mobius from bottom (computed via precompute_all_mobius_from_bottom)
        self._mobius_from_bottom: Optional[Dict[int, int]] = None

    @property
    def num_flats(self) -> int:
        return len(self._flats)

    def flats(self) -> List[FrozenSet[Edge]]:
        """Return all flats in order of increasing rank/size."""
        return list(self._flats)

    def bottom(self) -> FrozenSet[Edge]:
        """Return the bottom flat (empty set)."""
        return self._flats[0]

    def top(self) -> FrozenSet[Edge]:
        """Return the top flat (full ground set)."""
        return self._flats[-1]

    def flat_rank(self, flat: FrozenSet[Edge]) -> int:
        """Get rank of a flat (cached)."""
        idx = self._flat_index.get(flat)
        if idx is not None:
            return self._ranks[idx]
        return self.matroid.rank(flat)

    def flat_rank_by_idx(self, idx: int) -> int:
        """Get rank of a flat by index."""
        return self._ranks[idx]

    def flat_by_idx(self, idx: int) -> FrozenSet[Edge]:
        """Get flat by index."""
        return self._flats[idx]

    def flat_idx(self, flat: FrozenSet[Edge]) -> Optional[int]:
        """Get index of a flat."""
        return self._flat_index.get(flat)

    def flats_above(self, W: FrozenSet[Edge]) -> List[FrozenSet[Edge]]:
        """Return all flats Z >= W (including W itself)."""
        w_idx = self._flat_index.get(W)
        if w_idx is None:
            return []
        w_rank = self._ranks[w_idx]
        result = [W]
        max_rank = max(self._flats_by_rank.keys()) if self._flats_by_rank else 0
        for r in range(w_rank + 1, max_rank + 1):
            for j in self._flats_by_rank.get(r, []):
                if W.issubset(self._flats[j]):
                    result.append(self._flats[j])
        return result

    def flats_above_idx(self, w_idx: int) -> List[int]:
        """Return indices of all flats Z >= W (including W itself)."""
        w_rank = self._ranks[w_idx]
        W = self._flats[w_idx]
        result = [w_idx]
        max_rank = max(self._flats_by_rank.keys()) if self._flats_by_rank else 0
        for r in range(w_rank + 1, max_rank + 1):
            for j in self._flats_by_rank.get(r, []):
                if W.issubset(self._flats[j]):
                    result.append(j)
        return result

    # -----------------------------------------------------------------
    # Batch Mobius from bottom
    # -----------------------------------------------------------------

    def precompute_all_mobius_from_bottom(self) -> Dict[int, int]:
        """Compute mu(bottom, F) for ALL flats F in one bottom-up pass.

        Uses the standard recursion: mu(0,F) = -sum_{0 <= G < F} mu(0,G)
        Processing rank-by-rank ensures all predecessors are computed first.
        O(|flats|^2) worst case but fast for lattices with small intervals.

        Returns:
            Dict mapping flat index -> mu(bottom, flat)
        """
        if self._mobius_from_bottom is not None:
            return self._mobius_from_bottom

        mu: Dict[int, int] = {}
        bottom_flat = self._flats[0]

        # Bottom element: mu(0, 0) = 1
        mu[0] = 1

        # Process rank by rank (bottom-up)
        max_rank = max(self._flats_by_rank.keys()) if self._flats_by_rank else 0
        for r in range(1, max_rank + 1):
            for j in self._flats_by_rank.get(r, []):
                # mu(0, j) = -sum_{0 <= k < j, bottom <= flats[k] <= flats[j]} mu(0, k)
                total = 0
                for r2 in range(0, r):
                    for k in self._flats_by_rank.get(r2, []):
                        if bottom_flat.issubset(self._flats[k]) and self._flats[k].issubset(self._flats[j]):
                            total += mu.get(k, 0)
                mu[j] = -total

        self._mobius_from_bottom = mu

        # Also populate the pairwise cache
        for j, val in mu.items():
            self._mobius_cache[(0, j)] = val

        return mu

    # -----------------------------------------------------------------
    # Pairwise Mobius (general intervals)
    # -----------------------------------------------------------------

    def _compute_mobius(self, i: int, j: int) -> int:
        """Compute mu(i, j) using interval recursion.

        mu(i, i) = 1
        mu(i, j) = -sum_{k in [i,j), flats[i] <= flats[k] <= flats[j]} mu(i, k)
        """
        if (i, j) in self._mobius_cache:
            return self._mobius_cache[(i, j)]

        if i == j:
            self._mobius_cache[(i, j)] = 1
            return 1

        if not self._flats[i].issubset(self._flats[j]):
            self._mobius_cache[(i, j)] = 0
            return 0

        r_i = self._ranks[i]
        r_j = self._ranks[j]
        total = 0
        for r in range(r_i, r_j):
            for k in self._flats_by_rank.get(r, []):
                if k != j and self._flats[i].issubset(self._flats[k]) and self._flats[k].issubset(self._flats[j]):
                    total += self._compute_mobius(i, k)

        result = -total
        self._mobius_cache[(i, j)] = result
        return result

    def mobius(self, W: FrozenSet[Edge], Z: FrozenSet[Edge]) -> int:
        """Compute Mobius function mu(W, Z)."""
        i = self._flat_index.get(W)
        j = self._flat_index.get(Z)
        if i is None or j is None:
            return 0
        return self._compute_mobius(i, j)

    def mobius_from_bottom(self, Z: FrozenSet[Edge]) -> int:
        """Get mu(bottom, Z) using batch-precomputed values."""
        if self._mobius_from_bottom is None:
            self.precompute_all_mobius_from_bottom()
        j = self._flat_index.get(Z)
        if j is None:
            return 0
        return self._mobius_from_bottom.get(j, 0)

    def precompute_mobius_from(self, W: FrozenSet[Edge]):
        """Precompute mu(W, Z) for all Z >= W.

        This is more efficient than computing each mu(W, Z) individually
        when we need all of them.
        """
        w_idx = self._flat_index.get(W)
        if w_idx is None:
            return

        w_rank = self._ranks[w_idx]
        max_rank = max(self._flats_by_rank.keys()) if self._flats_by_rank else 0

        # Process rank by rank (bottom-up)
        self._mobius_cache[(w_idx, w_idx)] = 1

        for r in range(w_rank + 1, max_rank + 1):
            for j in self._flats_by_rank.get(r, []):
                if not self._flats[w_idx].issubset(self._flats[j]):
                    continue
                # mu(w, j) = -sum_{w<=k<j} mu(w, k)
                total = 0
                for r2 in range(w_rank, r):
                    for k in self._flats_by_rank.get(r2, []):
                        if (self._flats[w_idx].issubset(self._flats[k]) and
                                self._flats[k].issubset(self._flats[j])):
                            total += self._mobius_cache.get((w_idx, k), 0)
                self._mobius_cache[(w_idx, j)] = -total

    def characteristic_poly_coeffs(
        self, contraction_flat: FrozenSet[Edge] = None
    ) -> Dict[int, int]:
        """Compute characteristic polynomial chi(M/W; q) as {power: coeff}.

        chi(M; q) = sum_{F flat} mu(bottom, F) * q^{r(M) - r(F)}

        If contraction_flat is provided, computes chi(M/W; q) by
        working in the interval [W, top] of the lattice.
        """
        if contraction_flat is None:
            contraction_flat = self.bottom()

        w_idx = self._flat_index.get(contraction_flat)
        if w_idx is None:
            return {0: 1}

        # Use batch Mobius if computing from bottom
        if w_idx == 0 and self._mobius_from_bottom is None:
            self.precompute_all_mobius_from_bottom()

        if w_idx != 0:
            # Precompute all Mobius values from this starting flat
            self.precompute_mobius_from(contraction_flat)

        r_total = self._ranks[-1]  # rank of top
        r_w = self._ranks[w_idx]
        contracted_rank = r_total - r_w

        coeffs: Dict[int, int] = defaultdict(int)

        for j in range(len(self._flats)):
            if not contraction_flat.issubset(self._flats[j]):
                continue
            if w_idx == 0 and self._mobius_from_bottom is not None:
                mu_val = self._mobius_from_bottom.get(j, 0)
            else:
                mu_val = self._mobius_cache.get((w_idx, j), 0)
            if mu_val == 0:
                continue
            r_j = self._ranks[j]
            power = contracted_rank - (r_j - r_w)
            coeffs[power] += mu_val

        return dict(coeffs)
