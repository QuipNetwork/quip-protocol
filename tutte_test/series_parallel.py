"""Series-Parallel Graph Recognition and Decomposition.

A 2-terminal series-parallel graph can be reduced to a single edge using:
- Series reduction: Remove degree-2 vertex, connect its neighbors
- Parallel reduction: Merge multiple edges between same endpoints

Series-parallel graphs have treewidth ≤ 2.

This module also provides:
- SPNode: Decomposition tree for series-parallel graphs
- decompose_series_parallel: Build the decomposition tree during reduction
- compute_sp_tutte: Compute Tutte polynomial from decomposition tree in O(n) time
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple, Set
from .graph import Graph
from .polynomial import TuttePolynomial


# =============================================================================
# SERIES-PARALLEL DECOMPOSITION TREE
# =============================================================================

@dataclass
class SPNode:
    """Node in series-parallel decomposition tree.

    The tree structure represents how the graph was built:
    - EDGE: A single edge (leaf node)
    - SERIES: Series composition of children (cut vertex)
    - PARALLEL: Parallel composition of children (multi-edges)

    The decomposition is computed during the reduction process by tracking
    which operations were performed.
    """
    type: str  # "EDGE", "SERIES", or "PARALLEL"
    children: List['SPNode'] = field(default_factory=list)
    edge: Optional[Tuple[int, int]] = None  # For EDGE nodes: original edge

    def __repr__(self) -> str:
        if self.type == "EDGE":
            return f"E{self.edge}"
        elif self.type == "SERIES":
            return f"S({', '.join(repr(c) for c in self.children)})"
        else:  # PARALLEL
            return f"P({', '.join(repr(c) for c in self.children)})"

    def edge_count(self) -> int:
        """Count total edges in this subtree."""
        if self.type == "EDGE":
            return 1
        return sum(c.edge_count() for c in self.children)

    def depth(self) -> int:
        """Maximum depth of the tree."""
        if self.type == "EDGE":
            return 0
        return 1 + max(c.depth() for c in self.children)


def is_series_parallel(graph: Graph) -> bool:
    """Check if graph is series-parallel via reduction algorithm. O(n+m)."""
    if graph.edge_count() <= 1:
        return True

    # Adjacency with edge multiplicities
    adj: Dict[int, Dict[int, int]] = {n: {} for n in graph.nodes}
    for u, v in graph.edges:
        adj[u][v] = adj[u].get(v, 0) + 1
        adj[v][u] = adj[v].get(u, 0) + 1

    n_edges = graph.edge_count()

    while n_edges > 1:
        reduced = False

        # Parallel reduction: multiple edges between same pair
        for u in adj:
            for v, count in adj[u].items():
                if count > 1:
                    adj[u][v] = adj[v][u] = 1
                    n_edges -= count - 1
                    reduced = True
                    break
            if reduced:
                break

        if not reduced:
            # Series reduction: degree-2 vertex
            for u in list(adj.keys()):
                neighbors = list(adj[u].keys())
                if len(neighbors) == 2:
                    v, w = neighbors
                    del adj[u]
                    del adj[v][u]
                    del adj[w][u]
                    adj[v][w] = adj[v].get(w, 0) + 1
                    adj[w][v] = adj[w].get(v, 0) + 1
                    n_edges -= 1
                    reduced = True
                    break

        if not reduced:
            return False

    return True


# =============================================================================
# DECOMPOSITION ALGORITHM
# =============================================================================

def decompose_series_parallel(graph: Graph) -> Optional[SPNode]:
    """Build SP decomposition tree during reduction, or None if not SP.

    The algorithm tracks reduction operations and builds the tree bottom-up:
    1. Start with EDGE nodes for each original edge
    2. When we reduce parallel edges, create PARALLEL node
    3. When we reduce a series vertex, create SERIES node
    4. Return the final tree root

    Args:
        graph: Graph to decompose

    Returns:
        SPNode root of decomposition tree, or None if graph is not SP
    """
    if graph.edge_count() == 0:
        # Empty graph - no decomposition
        return None

    if graph.edge_count() == 1:
        # Single edge
        edge = next(iter(graph.edges))
        return SPNode(type="EDGE", edge=edge)

    # Track decomposition nodes for each edge pair
    # edge_trees[(u,v)] = list of SPNode for edges between u and v
    edge_trees: Dict[Tuple[int, int], List[SPNode]] = {}

    # Initialize with EDGE nodes for each original edge
    for u, v in graph.edges:
        key = (min(u, v), max(u, v))
        if key not in edge_trees:
            edge_trees[key] = []
        edge_trees[key].append(SPNode(type="EDGE", edge=(u, v)))

    # Build adjacency with multiplicities
    adj: Dict[int, Dict[int, int]] = {n: {} for n in graph.nodes}
    for u, v in graph.edges:
        adj[u][v] = adj[u].get(v, 0) + 1
        adj[v][u] = adj[v].get(u, 0) + 1

    n_edges = graph.edge_count()

    while n_edges > 1:
        reduced = False

        # Parallel reduction: multiple edges between same pair
        for u in list(adj.keys()):
            if u not in adj:
                continue
            for v, count in list(adj[u].items()):
                if count > 1:
                    # Create PARALLEL node from all edges between u,v
                    key = (min(u, v), max(u, v))
                    children = edge_trees.pop(key)
                    parallel_node = SPNode(type="PARALLEL", children=children)
                    edge_trees[key] = [parallel_node]

                    adj[u][v] = adj[v][u] = 1
                    n_edges -= count - 1
                    reduced = True
                    break
            if reduced:
                break

        if not reduced:
            # Series reduction: degree-2 vertex
            for u in list(adj.keys()):
                if u not in adj:
                    continue
                neighbors = list(adj[u].keys())
                if len(neighbors) == 2:
                    v, w = neighbors

                    # Get decomposition trees for edges (u,v) and (u,w)
                    key_uv = (min(u, v), max(u, v))
                    key_uw = (min(u, w), max(u, w))

                    trees_uv = edge_trees.pop(key_uv)
                    trees_uw = edge_trees.pop(key_uw)

                    # Create SERIES node (composition at cut vertex u)
                    # Each edge contributes one tree
                    series_children = trees_uv + trees_uw
                    series_node = SPNode(type="SERIES", children=series_children)

                    # Remove u from adjacency
                    del adj[u]
                    del adj[v][u]
                    del adj[w][u]

                    # Add new edge (v,w) or increment if exists
                    adj[v][w] = adj[v].get(w, 0) + 1
                    adj[w][v] = adj[w].get(v, 0) + 1

                    # Add the series node as a tree for the new edge
                    key_vw = (min(v, w), max(v, w))
                    if key_vw not in edge_trees:
                        edge_trees[key_vw] = []
                    edge_trees[key_vw].append(series_node)

                    n_edges -= 1
                    reduced = True
                    break

        if not reduced:
            return None  # Not series-parallel

    # Should have exactly one edge with one tree
    if len(edge_trees) != 1:
        return None

    final_trees = list(edge_trees.values())[0]

    if len(final_trees) == 1:
        return final_trees[0]
    else:
        # Multiple trees for final edge -> parallel composition
        return SPNode(type="PARALLEL", children=final_trees)


# =============================================================================
# TUTTE POLYNOMIAL COMPUTATION FROM DECOMPOSITION
# =============================================================================

def compute_sp_tutte(tree: SPNode) -> TuttePolynomial:
    """Compute Tutte polynomial from SP decomposition tree in O(n) time.

    Formulas:
    - EDGE: T = x (single edge is a bridge)
    - SERIES (cut vertex): T(G1 · G2) = T(G1) × T(G2)
    - PARALLEL (k simple edges): T = x + y + y² + ... + y^(k-1)
    - PARALLEL (complex children): Process children edge-by-edge

    Args:
        tree: SPNode decomposition tree

    Returns:
        TuttePolynomial for the graph represented by this tree
    """
    if tree.type == "EDGE":
        return TuttePolynomial.x()

    elif tree.type == "SERIES":
        # Cut vertex factorization: T(G1 · G2) = T(G1) × T(G2)
        result = TuttePolynomial.one()
        for child in tree.children:
            result = result * compute_sp_tutte(child)
        return result

    elif tree.type == "PARALLEL":
        # All-edge children: T(k parallel) = x + y + y² + ... + y^(k-1)
        if all(c.type == "EDGE" for c in tree.children):
            k = len(tree.children)
            if k == 1:
                return TuttePolynomial.x()
            coeffs = {(1, 0): 1}  # x
            for i in range(1, k):
                coeffs[(0, i)] = 1  # + y^i
            return TuttePolynomial.from_coefficients(coeffs)

        # Complex children: process edge by edge using chord formula
        children_by_edges = sorted(tree.children, key=lambda c: -c.edge_count())

        # Start with largest child as base
        base_child = children_by_edges[0]
        result = compute_sp_tutte(base_child)
        accumulated_contracted = _compute_contracted_poly(base_child)

        # Add remaining children in parallel
        for child in children_by_edges[1:]:
            result, accumulated_contracted = _add_child_edges_to_base(
                child, result, accumulated_contracted
            )

        return result

    else:
        raise ValueError(f"Unknown node type: {tree.type}")


def _add_child_edges_to_base(
    child: SPNode,
    base_poly: TuttePolynomial,
    base_contracted: TuttePolynomial
) -> Tuple[TuttePolynomial, TuttePolynomial]:
    """Add a child structure in parallel to the base.

    Handles complex children by processing them edge by edge.

    - EDGE: direct chord, T(G + e) = T(G) + T(G/{s,t})
    - SERIES: recursively process each sub-child
    - PARALLEL: recursively process each grandchild

    Args:
        child: Child SP node to add in parallel
        base_poly: Current polynomial T(base)
        base_contracted: T(base/{s,t})

    Returns:
        (new_poly, new_contracted) tuple
    """
    if child.type == "EDGE":
        # T(G + e) = T(G) + T(G/{s,t})
        new_poly = base_poly + base_contracted
        # (G + e)/{s,t} = G/{s,t} + loop -> multiply by y
        new_contracted = base_contracted * TuttePolynomial.y()
        return (new_poly, new_contracted)

    elif child.type == "SERIES":
        # Recursively process each sub-child in sequence
        poly = base_poly
        contracted = base_contracted
        for subchild in child.children:
            poly, contracted = _add_child_edges_to_base(subchild, poly, contracted)
        return (poly, contracted)

    elif child.type == "PARALLEL":
        # Recursively process each grandchild
        poly = base_poly
        contracted = base_contracted
        for grandchild in child.children:
            poly, contracted = _add_child_edges_to_base(grandchild, poly, contracted)
        return (poly, contracted)

    else:
        raise ValueError(f"Unknown child type: {child.type}")


def _compute_contracted_poly(tree: SPNode) -> TuttePolynomial:
    """Compute T(tree with terminals merged).

    When terminals s,t of a 2-terminal SP graph are merged:
    - EDGE (s,t) -> loop: T = y
    - SERIES (path with n edges) -> cycle C_n: T = x^(n-1) + ... + x + y
    - PARALLEL -> product of contracted children (bouquet at merged vertex)
    """
    if tree.type == "EDGE":
        return TuttePolynomial.y()

    elif tree.type == "SERIES":
        # Path with n edges, merging endpoints creates cycle C_n
        # T(C_n) = x^(n-1) + x^(n-2) + ... + x + y
        n_edges = tree.edge_count()
        if n_edges == 1:
            return TuttePolynomial.y()

        coeffs: Dict[Tuple[int, int], int] = {}
        for i in range(1, n_edges):
            coeffs[(i, 0)] = 1  # x^i terms
        coeffs[(0, 1)] = 1  # y term
        return TuttePolynomial.from_coefficients(coeffs)

    elif tree.type == "PARALLEL":
        # Each child becomes structure at merged vertex (bouquet/cut vertex)
        # T(bouquet) = product of T(each contracted structure)
        if all(c.type == "EDGE" for c in tree.children):
            # k parallel edges -> k loops -> y^k
            return TuttePolynomial.y(len(tree.children))

        result = TuttePolynomial.one()
        for child in tree.children:
            result = result * _compute_contracted_poly(child)
        return result

    else:
        raise ValueError(f"Unknown node type: {tree.type}")


# =============================================================================
# CONVENIENCE FUNCTION FOR SYNTHESIS
# =============================================================================

def compute_sp_tutte_if_applicable(graph: Graph) -> Optional[TuttePolynomial]:
    """Try to compute Tutte polynomial using SP decomposition.

    This is the main entry point for synthesis integration.
    Returns None if the graph is not series-parallel.

    Args:
        graph: Graph to compute polynomial for

    Returns:
        TuttePolynomial or None if graph is not SP
    """
    if graph.edge_count() == 0:
        return TuttePolynomial.one()

    tree = decompose_series_parallel(graph)
    if tree is None:
        return None

    return compute_sp_tutte(tree)


# =============================================================================
# CHARACTERISTIC POLYNOMIAL VIA SP DECOMPOSITION
# =============================================================================

def compute_sp_chi_coeffs(graph: Graph) -> Optional[Dict[int, int]]:
    """Compute chi(M(G); q) = (-1)^r * T(1-q, 0) for SP graphs.

    Returns {power: coefficient} or None if graph is not SP.
    Uses compute_sp_tutte() for T, then evaluates at (1-q, 0).
    O(n) time via SP decomposition.
    """
    tutte = compute_sp_tutte_if_applicable(graph)
    if tutte is None:
        return None

    return _chi_from_tutte(tutte, graph)


def _chi_from_tutte(tutte: TuttePolynomial, graph: Graph) -> Dict[int, int]:
    """Compute chi(M; q) from T(M; x, y) via chi(q) = (-1)^r * T(1-q, 0).

    T(1-q, 0) means substitute x = 1-q, y = 0 in T(x, y).
    Only terms with y_power == 0 survive.

    For a term c * x^i * y^j:
      If j > 0: contributes 0 (since y=0)
      If j == 0: contributes c * (1-q)^i

    Then multiply by (-1)^r.
    """
    # Compute rank
    n = graph.node_count()
    # For connected graph, rank = n - 1
    # For general graph, rank = n - components
    # We compute via edge count and y-degree: rank = x_degree of T
    # Actually, for Tutte polynomial T(x,y), the x-degree = n - k where k = components
    # Simpler: rank = number of edges in spanning forest
    # For connected: rank = n-1, for disconnected: rank = n - num_components
    # We can read it from T: rank = max x-power in T
    rank = tutte.x_degree()

    # Collect terms with y^0 (only x terms survive y=0)
    # T(1-q, 0) = sum_{i} c_i * (1-q)^i  where c_i = coefficient(x^i, y^0)
    x_coeffs: Dict[int, int] = {}
    for (i, j), c in tutte.to_coefficients().items():
        if j == 0:
            x_coeffs[i] = c

    if not x_coeffs:
        return {0: 0}

    # Expand sum c_i * (1-q)^i using binomial theorem
    # (1-q)^i = sum_{k=0}^{i} C(i,k) * (-q)^k = sum_{k} C(i,k) * (-1)^k * q^k
    from math import comb

    chi_coeffs: Dict[int, int] = {}
    for i, c_i in x_coeffs.items():
        for k in range(i + 1):
            binom = comb(i, k)
            sign = (-1) ** k
            contribution = c_i * binom * sign
            chi_coeffs[k] = chi_coeffs.get(k, 0) + contribution

    # Multiply by (-1)^r
    sign_r = (-1) ** rank
    result = {k: sign_r * v for k, v in chi_coeffs.items() if v != 0}

    return result


def compute_contraction_chi(
    graph: Graph, flat: FrozenSet[Tuple[int, int]]
) -> Optional[Dict[int, int]]:
    """Compute chi(M(G)/W; q) by contracting flat W in graph G.

    1. Contract flat edges (merge endpoints)
    2. Try SP decomposition on contracted graph -> O(n) chi
    3. If not SP, return None (caller should fall back to lattice chi)
    """
    if not flat:
        return compute_sp_chi_coeffs(graph)

    # Contract flat edges: merge endpoints
    # Build union-find for contraction
    all_nodes = set(graph.nodes)
    parent = {v: v for v in all_nodes}

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

    # Build contracted graph
    node_map = {n: find(n) for n in all_nodes}
    new_nodes = frozenset(node_map.values())

    new_edges = set()
    for u, v in graph.edges:
        if (u, v) not in flat:
            nu, nv = node_map[u], node_map[v]
            if nu != nv:
                edge = (min(nu, nv), max(nu, nv))
                new_edges.add(edge)

    if not new_edges:
        # Contracted to a single vertex or no remaining edges
        # chi(point; q) = 1
        return {0: 1}

    contracted_graph = Graph(nodes=new_nodes, edges=frozenset(new_edges))

    # Try SP decomposition first (O(n))
    return compute_sp_chi_coeffs(contracted_graph)
