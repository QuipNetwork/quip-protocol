"""
General Graph Composition Rules for Tutte Polynomials.

This module extends beyond series-parallel to explore:
1. Cut vertex decomposition (1-separation)
2. 2-separation / 2-sum operations
3. Clique-sum operations (k-sum)
4. Vertex/edge identification
5. Graph products (Cartesian, tensor)

Key theoretical results:
- Cut vertex: T(G) = T(G₁) × T(G₂) when G₁, G₂ share only a cut vertex
- 2-sum: More complex formula involving matroid operations
- k-clique sum: Glue on k-clique, formula depends on structure
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Callable
from enum import Enum
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tutte_test.tutte_to_ising import (
    TuttePolynomial,
    GraphBuilder,
    compute_tutte_polynomial,
)


class CompositionOp(Enum):
    """Types of graph composition operations."""
    DISJOINT_UNION = "disjoint_union"      # G₁ ∪ G₂ (no shared vertices)
    CUT_VERTEX = "cut_vertex"               # Share single vertex (1-sum)
    TWO_SUM = "two_sum"                     # Share edge, then delete it
    CLIQUE_SUM = "clique_sum"               # Share k-clique
    CARTESIAN = "cartesian"                 # Cartesian product G □ H
    TENSOR = "tensor"                       # Tensor product G × H
    EDGE_SUBDIVISION = "subdivision"        # Subdivide edges
    VERTEX_IDENTIFICATION = "identify"      # Identify specific vertices


@dataclass
class CompositionResult:
    """Result of a composition operation."""
    graph: GraphBuilder
    tutte_formula: Optional[TuttePolynomial]  # From formula if available
    tutte_computed: TuttePolynomial           # From direct computation
    formula_matches: bool
    operation: CompositionOp
    components: List['CompositionResult'] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


def disjoint_union(g1: GraphBuilder, g2: GraphBuilder) -> GraphBuilder:
    """
    Disjoint union of two graphs.

    T(G₁ ∪ G₂) = T(G₁) × T(G₂)
    """
    result = GraphBuilder()

    # Add all nodes and edges from g1
    node_map1 = {}
    for node in g1.nodes:
        node_map1[node] = result.add_node()
    for edge_id, (u, v) in g1.edges.items():
        result.add_edge(node_map1[u], node_map1[v])

    # Add all nodes and edges from g2
    node_map2 = {}
    for node in g2.nodes:
        node_map2[node] = result.add_node()
    for edge_id, (u, v) in g2.edges.items():
        result.add_edge(node_map2[u], node_map2[v])

    return result


def cut_vertex_join(g1: GraphBuilder, v1: int, g2: GraphBuilder, v2: int) -> GraphBuilder:
    """
    Join two graphs at a cut vertex (1-sum).

    Identifies vertex v1 in g1 with vertex v2 in g2.

    T(G) = T(G₁) × T(G₂) (when the shared vertex is a cut vertex)

    Note: This formula holds because the graphs share no edges,
    only a single vertex.
    """
    result = GraphBuilder()

    # Add all nodes from g1
    node_map1 = {}
    for node in g1.nodes:
        node_map1[node] = result.add_node()

    # Add all edges from g1
    for edge_id, (u, v) in g1.edges.items():
        result.add_edge(node_map1[u], node_map1[v])

    # Add nodes from g2, mapping v2 to v1's image
    node_map2 = {v2: node_map1[v1]}  # Identify the vertices
    for node in g2.nodes:
        if node != v2:
            node_map2[node] = result.add_node()

    # Add all edges from g2
    for edge_id, (u, v) in g2.edges.items():
        result.add_edge(node_map2[u], node_map2[v])

    return result


def two_sum(g1: GraphBuilder, e1: int, g2: GraphBuilder, e2: int) -> GraphBuilder:
    """
    2-sum of two graphs along edges.

    Identifies edges e1 and e2 (and their endpoints), then DELETES the shared edge.

    This is more complex for Tutte polynomials - the formula involves
    the parallel connection of matroids.

    For matroids M₁, M₂ with distinguished elements e₁, e₂:
    The 2-sum M₁ ⊕₂ M₂ deletes the identified element.
    """
    if e1 not in g1.edges or e2 not in g2.edges:
        raise ValueError("Invalid edge IDs")

    u1, v1 = g1.edges[e1]
    u2, v2 = g2.edges[e2]

    result = GraphBuilder()

    # Add nodes from g1
    node_map1 = {}
    for node in g1.nodes:
        node_map1[node] = result.add_node()

    # Add edges from g1, EXCEPT e1
    for edge_id, (u, v) in g1.edges.items():
        if edge_id != e1:
            result.add_edge(node_map1[u], node_map1[v])

    # Add nodes from g2, identifying endpoints of e2 with endpoints of e1
    node_map2 = {
        u2: node_map1[u1],
        v2: node_map1[v1]
    }
    for node in g2.nodes:
        if node not in node_map2:
            node_map2[node] = result.add_node()

    # Add edges from g2, EXCEPT e2
    for edge_id, (u, v) in g2.edges.items():
        if edge_id != e2:
            result.add_edge(node_map2[u], node_map2[v])

    return result


def parallel_connection(g1: GraphBuilder, e1: int, g2: GraphBuilder, e2: int) -> GraphBuilder:
    """
    Parallel connection of two graphs along edges.

    Identifies edges e1 and e2 (and their endpoints), KEEPS the edge (as one edge).
    This is different from 2-sum which deletes the edge.
    """
    if e1 not in g1.edges or e2 not in g2.edges:
        raise ValueError("Invalid edge IDs")

    u1, v1 = g1.edges[e1]
    u2, v2 = g2.edges[e2]

    result = GraphBuilder()

    # Add nodes from g1
    node_map1 = {}
    for node in g1.nodes:
        node_map1[node] = result.add_node()

    # Add edges from g1
    for edge_id, (u, v) in g1.edges.items():
        result.add_edge(node_map1[u], node_map1[v])

    # Add nodes from g2, identifying endpoints
    node_map2 = {
        u2: node_map1[u1],
        v2: node_map1[v1]
    }
    for node in g2.nodes:
        if node not in node_map2:
            node_map2[node] = result.add_node()

    # Add edges from g2, EXCEPT e2 (already have e1)
    for edge_id, (u, v) in g2.edges.items():
        if edge_id != e2:
            result.add_edge(node_map2[u], node_map2[v])

    return result


def series_connection(g1: GraphBuilder, v1: int, g2: GraphBuilder, v2: int) -> GraphBuilder:
    """
    Series connection: identify a vertex from each graph.

    Same as cut_vertex_join - the identified vertex becomes a cut vertex.
    """
    return cut_vertex_join(g1, v1, g2, v2)


def clique_sum(g1: GraphBuilder, clique1: List[int],
               g2: GraphBuilder, clique2: List[int],
               delete_clique_edges: bool = True) -> GraphBuilder:
    """
    k-clique sum of two graphs.

    Identifies k-cliques from each graph, optionally deleting clique edges.

    Args:
        g1, g2: Input graphs
        clique1, clique2: Lists of k vertices forming cliques (same size)
        delete_clique_edges: If True, delete edges within the identified clique
    """
    if len(clique1) != len(clique2):
        raise ValueError("Cliques must have same size")

    k = len(clique1)
    result = GraphBuilder()

    # Add nodes from g1
    node_map1 = {}
    for node in g1.nodes:
        node_map1[node] = result.add_node()

    # Add edges from g1
    clique1_set = set(clique1)
    for edge_id, (u, v) in g1.edges.items():
        if delete_clique_edges and u in clique1_set and v in clique1_set:
            continue  # Skip clique edges
        result.add_edge(node_map1[u], node_map1[v])

    # Map clique2 vertices to clique1's images
    node_map2 = {clique2[i]: node_map1[clique1[i]] for i in range(k)}

    # Add remaining nodes from g2
    for node in g2.nodes:
        if node not in node_map2:
            node_map2[node] = result.add_node()

    # Add edges from g2
    clique2_set = set(clique2)
    for edge_id, (u, v) in g2.edges.items():
        if delete_clique_edges and u in clique2_set and v in clique2_set:
            continue  # Skip clique edges
        result.add_edge(node_map2[u], node_map2[v])

    return result


def edge_subdivision(g: GraphBuilder, edge_id: int, k: int = 1) -> GraphBuilder:
    """
    Subdivide an edge into k+1 edges (adding k new vertices).

    For Tutte polynomial: subdividing doesn't have a simple formula,
    but if the edge is a bridge, T(subdivided) still has x factor.
    """
    if edge_id not in g.edges:
        raise ValueError("Invalid edge ID")

    u, v = g.edges[edge_id]

    result = GraphBuilder()

    # Copy all nodes
    node_map = {}
    for node in g.nodes:
        node_map[node] = result.add_node()

    # Copy all edges except the one being subdivided
    for eid, (a, b) in g.edges.items():
        if eid != edge_id:
            result.add_edge(node_map[a], node_map[b])

    # Add subdivision: u -- new1 -- new2 -- ... -- newk -- v
    prev = node_map[u]
    for i in range(k):
        new_node = result.add_node()
        result.add_edge(prev, new_node)
        prev = new_node
    result.add_edge(prev, node_map[v])

    return result


def cartesian_product(g1: GraphBuilder, g2: GraphBuilder) -> GraphBuilder:
    """
    Cartesian product G □ H.

    Vertices: V(G) × V(H)
    Edges: (u,v)-(u',v') if (u=u' and vv'∈E(H)) or (v=v' and uu'∈E(G))

    No simple Tutte polynomial formula, but useful for building grids, hypercubes.
    """
    result = GraphBuilder()

    nodes1 = sorted(g1.nodes)
    nodes2 = sorted(g2.nodes)

    # Create product vertices
    node_map = {}
    for u in nodes1:
        for v in nodes2:
            node_map[(u, v)] = result.add_node()

    # Add horizontal edges (from g1)
    for eid, (u, u_prime) in g1.edges.items():
        for v in nodes2:
            result.add_edge(node_map[(u, v)], node_map[(u_prime, v)])

    # Add vertical edges (from g2)
    for eid, (v, v_prime) in g2.edges.items():
        for u in nodes1:
            result.add_edge(node_map[(u, v)], node_map[(u, v_prime)])

    return result


def tensor_product(g1: GraphBuilder, g2: GraphBuilder) -> GraphBuilder:
    """
    Tensor product G × H (also called categorical or direct product).

    Vertices: V(G) × V(H)
    Edges: (u,v)-(u',v') if uu'∈E(G) AND vv'∈E(H)
    """
    result = GraphBuilder()

    nodes1 = sorted(g1.nodes)
    nodes2 = sorted(g2.nodes)

    # Create product vertices
    node_map = {}
    for u in nodes1:
        for v in nodes2:
            node_map[(u, v)] = result.add_node()

    # Add edges where both components have edges
    for e1_id, (u, u_prime) in g1.edges.items():
        for e2_id, (v, v_prime) in g2.edges.items():
            # Edge in both directions due to undirected
            result.add_edge(node_map[(u, v)], node_map[(u_prime, v_prime)])
            result.add_edge(node_map[(u, v_prime)], node_map[(u_prime, v)])

    return result


# ============================================================================
# Helper functions to create basic graphs
# ============================================================================

def create_edge() -> GraphBuilder:
    """Single edge (K_2)."""
    g = GraphBuilder()
    u, v = g.add_node(), g.add_node()
    g.add_edge(u, v)
    return g


def create_path(n: int) -> GraphBuilder:
    """Path with n vertices."""
    g = GraphBuilder()
    if n < 1:
        return g
    prev = g.add_node()
    for _ in range(n - 1):
        curr = g.add_node()
        g.add_edge(prev, curr)
        prev = curr
    return g


def create_cycle(n: int) -> GraphBuilder:
    """Cycle with n vertices."""
    if n < 3:
        return create_path(n)
    g = GraphBuilder()
    first = g.add_node()
    prev = first
    for _ in range(n - 1):
        curr = g.add_node()
        g.add_edge(prev, curr)
        prev = curr
    g.add_edge(prev, first)
    return g


def create_complete(n: int) -> GraphBuilder:
    """Complete graph K_n."""
    g = GraphBuilder()
    nodes = [g.add_node() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(nodes[i], nodes[j])
    return g


def create_star(n: int) -> GraphBuilder:
    """Star graph S_n (center + n leaves)."""
    g = GraphBuilder()
    center = g.add_node()
    for _ in range(n):
        leaf = g.add_node()
        g.add_edge(center, leaf)
    return g


# ============================================================================
# Tutte polynomial formulas for compositions
# ============================================================================

def tutte_disjoint_union(t1: TuttePolynomial, t2: TuttePolynomial) -> TuttePolynomial:
    """T(G₁ ∪ G₂) = T(G₁) × T(G₂)"""
    return t1 * t2


def tutte_cut_vertex(t1: TuttePolynomial, t2: TuttePolynomial) -> TuttePolynomial:
    """
    T(G₁ ·₁ G₂) = T(G₁) × T(G₂) when joined at cut vertex.

    This works because the cut vertex creates independent subproblems.
    """
    return t1 * t2


def tutte_bridge_extension(t: TuttePolynomial) -> TuttePolynomial:
    """Adding a bridge (pendant edge) multiplies by x."""
    return t * TuttePolynomial.x()


# ============================================================================
# Analysis and verification
# ============================================================================

def find_cut_vertices(g: GraphBuilder) -> List[int]:
    """Find all cut vertices (articulation points) in the graph."""
    if g.num_nodes() <= 1:
        return []

    cut_vertices = []
    nodes = sorted(g.nodes)

    for v in nodes:
        # Remove v and check if graph becomes disconnected
        remaining = [n for n in nodes if n != v]
        if not remaining:
            continue

        # BFS from first remaining node
        start = remaining[0]
        visited = {start}
        stack = [start]

        while stack:
            curr = stack.pop()
            for eid, (a, b) in g.edges.items():
                if a == v or b == v:
                    continue  # Skip edges incident to v
                neighbor = None
                if a == curr and b not in visited:
                    neighbor = b
                elif b == curr and a not in visited:
                    neighbor = a
                if neighbor is not None:
                    visited.add(neighbor)
                    stack.append(neighbor)

        if len(visited) < len(remaining):
            cut_vertices.append(v)

    return cut_vertices


def find_bridges(g: GraphBuilder) -> List[int]:
    """Find all bridges (cut edges) in the graph."""
    from tutte_test.tutte_utils import is_bridge
    return [eid for eid in g.edges if is_bridge(g, eid)]


def decompose_at_cut_vertex(g: GraphBuilder, v: int) -> List[GraphBuilder]:
    """
    Decompose graph at a cut vertex into components.

    Returns list of subgraphs, each containing v and one component.
    """
    if v not in g.nodes:
        raise ValueError("Vertex not in graph")

    # Find connected components after removing v
    other_nodes = [n for n in g.nodes if n != v]
    if not other_nodes:
        return [g]

    # Build adjacency for non-v nodes
    adj = {n: set() for n in other_nodes}
    for eid, (a, b) in g.edges.items():
        if a != v and b != v:
            adj[a].add(b)
            adj[b].add(a)

    # Find components
    visited = set()
    components = []

    for start in other_nodes:
        if start in visited:
            continue

        component = {start}
        stack = [start]
        while stack:
            curr = stack.pop()
            for neighbor in adj[curr]:
                if neighbor not in component:
                    component.add(neighbor)
                    stack.append(neighbor)

        visited.update(component)
        components.append(component)

    if len(components) <= 1:
        return [g]  # v is not a cut vertex

    # Build subgraphs
    subgraphs = []
    for component in components:
        sub = GraphBuilder()
        node_map = {v: sub.add_node()}  # Always include v
        for n in component:
            node_map[n] = sub.add_node()

        for eid, (a, b) in g.edges.items():
            if a in node_map and b in node_map:
                sub.add_edge(node_map[a], node_map[b])

        subgraphs.append(sub)

    return subgraphs


def verify_composition(op: CompositionOp, g1: GraphBuilder, g2: GraphBuilder,
                       result: GraphBuilder, **kwargs) -> CompositionResult:
    """
    Verify a composition operation by comparing formula vs computation.
    """
    t1 = compute_tutte_polynomial(g1)
    t2 = compute_tutte_polynomial(g2)
    t_result = compute_tutte_polynomial(result)

    # Compute formula if available
    t_formula = None
    if op == CompositionOp.DISJOINT_UNION:
        t_formula = tutte_disjoint_union(t1, t2)
    elif op == CompositionOp.CUT_VERTEX:
        t_formula = tutte_cut_vertex(t1, t2)

    return CompositionResult(
        graph=result,
        tutte_formula=t_formula,
        tutte_computed=t_result,
        formula_matches=(t_formula == t_result) if t_formula else False,
        operation=op,
        metadata={
            'g1_nodes': g1.num_nodes(),
            'g1_edges': g1.num_edges(),
            'g2_nodes': g2.num_nodes(),
            'g2_edges': g2.num_edges(),
            'result_nodes': result.num_nodes(),
            'result_edges': result.num_edges(),
            't1': str(t1),
            't2': str(t2),
        }
    )


# ============================================================================
# Demo
# ============================================================================

def demo_compositions():
    """Demonstrate various composition operations."""
    print("=" * 70)
    print("GENERAL GRAPH COMPOSITION RULES")
    print("=" * 70)

    # 1. Disjoint Union
    print("\n--- 1. Disjoint Union: T(G₁ ∪ G₂) = T(G₁) × T(G₂) ---")
    for (name1, g1), (name2, g2) in [
        (("K3", create_complete(3)), ("K3", create_complete(3))),
        (("P3", create_path(3)), ("C4", create_cycle(4))),
        (("K4", create_complete(4)), ("edge", create_edge())),
    ]:
        result = disjoint_union(g1, g2)
        cr = verify_composition(CompositionOp.DISJOINT_UNION, g1, g2, result)
        print(f"{name1} ∪ {name2}:")
        print(f"  T({name1}) = {cr.metadata['t1']}")
        print(f"  T({name2}) = {cr.metadata['t2']}")
        print(f"  Formula:  {cr.tutte_formula}")
        print(f"  Computed: {cr.tutte_computed}")
        print(f"  Match: {cr.formula_matches}")

    # 2. Cut Vertex Join
    print("\n--- 2. Cut Vertex Join (1-sum): T(G₁ ·₁ G₂) = T(G₁) × T(G₂) ---")
    for (name1, g1, v1), (name2, g2, v2) in [
        (("K3", create_complete(3), 0), ("K3", create_complete(3), 0)),
        (("C4", create_cycle(4), 0), ("P3", create_path(3), 0)),
        (("star3", create_star(3), 0), ("K4", create_complete(4), 0)),
    ]:
        result = cut_vertex_join(g1, v1, g2, v2)
        cr = verify_composition(CompositionOp.CUT_VERTEX, g1, g2, result)
        print(f"{name1} ·₁ {name2} (at vertices {v1}, {v2}):")
        print(f"  T({name1}) = {cr.metadata['t1']}")
        print(f"  T({name2}) = {cr.metadata['t2']}")
        print(f"  Formula:  {cr.tutte_formula}")
        print(f"  Computed: {cr.tutte_computed}")
        print(f"  Match: {cr.formula_matches}")

    # 3. 2-sum
    print("\n--- 3. Two-Sum (glue edges, delete shared edge) ---")
    g1 = create_cycle(4)  # C4
    g2 = create_cycle(4)  # C4
    result = two_sum(g1, 0, g2, 0)  # Use first edge of each
    t1 = compute_tutte_polynomial(g1)
    t2 = compute_tutte_polynomial(g2)
    t_result = compute_tutte_polynomial(result)
    print(f"C4 ⊕₂ C4:")
    print(f"  T(C4) = {t1}")
    print(f"  Result: {result.num_nodes()} nodes, {result.num_edges()} edges")
    print(f"  T(result) = {t_result}")
    print(f"  Note: No simple formula - depends on matroid structure")

    # 4. Parallel Connection
    print("\n--- 4. Parallel Connection (glue edges, keep shared edge) ---")
    g1 = create_cycle(3)  # K3
    g2 = create_cycle(3)  # K3
    result = parallel_connection(g1, 0, g2, 0)
    t_result = compute_tutte_polynomial(result)
    print(f"K3 ∥ K3:")
    print(f"  Result: {result.num_nodes()} nodes, {result.num_edges()} edges")
    print(f"  T(result) = {t_result}")

    # 5. Clique Sum
    print("\n--- 5. Clique Sum (k-sum) ---")
    g1 = create_complete(4)  # K4
    g2 = create_complete(4)  # K4
    # 2-clique sum: identify an edge (2 vertices)
    result = clique_sum(g1, [0, 1], g2, [0, 1], delete_clique_edges=True)
    t_result = compute_tutte_polynomial(result)
    print(f"K4 ⊕₂ K4 (2-clique sum, delete shared edge):")
    print(f"  Result: {result.num_nodes()} nodes, {result.num_edges()} edges")
    print(f"  T(result) = {t_result}")

    # 3-clique sum
    result = clique_sum(g1, [0, 1, 2], g2, [0, 1, 2], delete_clique_edges=True)
    t_result = compute_tutte_polynomial(result)
    print(f"K4 ⊕₃ K4 (3-clique sum, delete shared triangle):")
    print(f"  Result: {result.num_nodes()} nodes, {result.num_edges()} edges")
    print(f"  T(result) = {t_result}")

    # 6. Edge Subdivision
    print("\n--- 6. Edge Subdivision ---")
    g = create_complete(3)
    for k in [1, 2, 3]:
        result = edge_subdivision(g, 0, k)
        t_orig = compute_tutte_polynomial(g)
        t_result = compute_tutte_polynomial(result)
        print(f"K3 with edge subdivided {k}x:")
        print(f"  Original: {g.num_nodes()} nodes, {g.num_edges()} edges, T = {t_orig}")
        print(f"  Result: {result.num_nodes()} nodes, {result.num_edges()} edges, T = {t_result}")

    # 7. Graph Products
    print("\n--- 7. Cartesian Product G □ H ---")
    edge = create_edge()  # K2
    path3 = create_path(3)

    # K2 □ K2 = C4
    result = cartesian_product(edge, edge)
    t_result = compute_tutte_polynomial(result)
    print(f"K2 □ K2 (should be C4):")
    print(f"  Result: {result.num_nodes()} nodes, {result.num_edges()} edges")
    print(f"  T = {t_result}")

    # K2 □ P3 = Ladder
    result = cartesian_product(edge, path3)
    t_result = compute_tutte_polynomial(result)
    print(f"K2 □ P3 (ladder graph):")
    print(f"  Result: {result.num_nodes()} nodes, {result.num_edges()} edges")
    print(f"  T = {t_result}")

    # P3 □ P3 = 3x3 grid
    result = cartesian_product(path3, path3)
    t_result = compute_tutte_polynomial(result)
    print(f"P3 □ P3 (3×3 grid):")
    print(f"  Result: {result.num_nodes()} nodes, {result.num_edges()} edges")
    print(f"  T = {t_result}")

    # 8. Cut Vertex Analysis
    print("\n--- 8. Cut Vertex Decomposition ---")
    # Build a graph with a cut vertex: K3 - v - K3
    g1 = create_complete(3)
    g2 = create_complete(3)
    barbell = cut_vertex_join(g1, 0, g2, 0)

    cuts = find_cut_vertices(barbell)
    bridges = find_bridges(barbell)
    t = compute_tutte_polynomial(barbell)

    print(f"Barbell graph (K3 - v - K3):")
    print(f"  Nodes: {barbell.num_nodes()}, Edges: {barbell.num_edges()}")
    print(f"  Cut vertices: {cuts}")
    print(f"  Bridges: {bridges}")
    print(f"  T = {t}")

    if cuts:
        print(f"  Decomposing at cut vertex {cuts[0]}:")
        subgraphs = decompose_at_cut_vertex(barbell, cuts[0])
        product = TuttePolynomial.one()
        for i, sub in enumerate(subgraphs):
            t_sub = compute_tutte_polynomial(sub)
            product = product * t_sub
            print(f"    Component {i+1}: {sub.num_nodes()} nodes, T = {t_sub}")
        print(f"  Product of components: {product}")
        print(f"  Matches original: {product == t}")

    # 9. Summary
    print("\n" + "=" * 70)
    print("SUMMARY: COMPOSITION RULES FOR TUTTE POLYNOMIALS")
    print("=" * 70)
    print("""
EXACT FORMULAS (multiplicative):
  1. Disjoint Union:  T(G₁ ∪ G₂) = T(G₁) × T(G₂)
  2. Cut Vertex Join: T(G₁ ·₁ G₂) = T(G₁) × T(G₂)  [when v is cut vertex]
  3. Bridge Addition: T(G + bridge) = x × T(G)

NO SIMPLE FORMULA (require direct computation):
  4. 2-sum:              Complex matroid formula
  5. Parallel Connection: Complex formula
  6. k-Clique Sum:        Depends on clique structure
  7. Edge Subdivision:    Changes polynomial structure
  8. Cartesian Product:   No known formula
  9. Tensor Product:      No known formula

KEY INSIGHT FOR ZEPHYR GRAPHS:
  Zephyr topologies are highly connected (no cut vertices) and have
  no simple decomposition. They require direct computation or
  approximation methods.
""")


if __name__ == "__main__":
    demo_compositions()
