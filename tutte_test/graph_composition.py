"""
Graph Composition and Synthesis for Tutte Polynomials.

This module provides:
1. Graph composition operations (disjoint union, cut vertex, 2-sum, clique-sum, products)
2. Graph synthesis from minors (building graphs with desired Tutte polynomial properties)
3. Analysis utilities (cut vertices, bridges, connectivity)

Key theoretical results:
- Cut vertex: T(G) = T(G₁) × T(G₂) when G₁, G₂ share only a cut vertex
- Disjoint union: T(G₁ ∪ G₂) = T(G₁) × T(G₂)
- 2-sum: More complex formula involving matroid operations
- k-clique sum: Glue on k-clique, formula depends on structure
"""

import itertools
import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tutte_test.tutte_utils import (
    GraphBuilder,
    TuttePolynomial,
    compute_tutte_polynomial,
    create_complete_graph,
    create_cycle_graph,
    create_path_graph,
)

# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

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


@dataclass
class SynthesizedGraph:
    """A graph built from minor composition."""
    graph: GraphBuilder
    tutte: TuttePolynomial
    recipe: List[str]  # Description of how it was built
    connectivity: int = 0

    def __post_init__(self):
        if self.connectivity == 0:
            self.connectivity = self._compute_connectivity()

    def _compute_connectivity(self) -> int:
        """Estimate vertex connectivity."""
        cuts = find_cut_vertices(self.graph)
        if cuts:
            return 1  # Has cut vertices
        # No cut vertices means at least 2-connected
        return 2


# =============================================================================
# GRAPH COMPOSITION OPERATIONS
# =============================================================================

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


def identify_vertices(g: GraphBuilder, v1: int, v2: int) -> GraphBuilder:
    """
    Identify (merge) two vertices in a graph.

    All edges incident to v2 are redirected to v1, then v2 is removed.
    """
    if v1 == v2:
        return g

    result = GraphBuilder()

    # Map all nodes except v2
    node_map = {}
    for node in g.nodes:
        if node == v2:
            node_map[node] = None  # Will map to v1's image
        else:
            node_map[node] = result.add_node()

    # v2 maps to v1's image
    node_map[v2] = node_map[v1]

    # Add edges, avoiding duplicates
    added_edges = set()
    for eid, (a, b) in g.edges.items():
        new_a = node_map[a]
        new_b = node_map[b]
        if new_a == new_b:
            continue  # Skip self-loops from identification
        edge_key = (min(new_a, new_b), max(new_a, new_b))
        if edge_key not in added_edges:
            result.add_edge(new_a, new_b)
            added_edges.add(edge_key)

    return result


# =============================================================================
# HELPER FUNCTIONS TO CREATE BASIC GRAPHS
# =============================================================================

def create_edge() -> GraphBuilder:
    """Single edge (K_2)."""
    g = GraphBuilder()
    u, v = g.add_node(), g.add_node()
    g.add_edge(u, v)
    return g


def create_path(n: int) -> GraphBuilder:
    """Path with n vertices."""
    return create_path_graph(n)


def create_cycle(n: int) -> GraphBuilder:
    """Cycle with n vertices."""
    return create_cycle_graph(n)


def create_complete(n: int) -> GraphBuilder:
    """Complete graph K_n."""
    return create_complete_graph(n)


def create_star(n: int) -> GraphBuilder:
    """Star graph S_n (center + n leaves)."""
    g = GraphBuilder()
    center = g.add_node()
    for _ in range(n):
        leaf = g.add_node()
        g.add_edge(center, leaf)
    return g


def create_k3() -> GraphBuilder:
    """Create K3 (triangle)."""
    return create_complete(3)


def create_k4() -> GraphBuilder:
    """Create K4."""
    return create_complete(4)


def create_diamond() -> GraphBuilder:
    """Create diamond graph (K4 minus one edge)."""
    g = GraphBuilder()
    nodes = [g.add_node() for _ in range(4)]
    # Add 5 edges (K4 has 6, we skip one)
    g.add_edge(nodes[0], nodes[1])
    g.add_edge(nodes[0], nodes[2])
    g.add_edge(nodes[0], nodes[3])
    g.add_edge(nodes[1], nodes[2])
    g.add_edge(nodes[2], nodes[3])
    # Skip edge (1,3) to make diamond
    return g


# =============================================================================
# TUTTE POLYNOMIAL FORMULAS FOR COMPOSITIONS
# =============================================================================

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


# =============================================================================
# ANALYSIS AND VERIFICATION
# =============================================================================

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


def analyze_k_cuts(g: GraphBuilder, max_k: int = 3) -> Dict[int, List[Tuple]]:
    """
    Analyze what k-cuts exist in a graph.

    A k-cut is a set of k edges whose removal disconnects the graph.
    Also identifies cut vertices (1-vertex cuts).

    Args:
        g: Graph to analyze
        max_k: Maximum k to analyze (default 3)

    Returns:
        Dict mapping k -> list of k-cuts found.
        Each cut is a tuple like ('edge', edge_id, endpoints) or ('vertex', v).
    """
    cuts = {k: [] for k in range(1, max_k + 1)}

    # 1-cuts: bridges (edge cuts) and cut vertices
    bridges = find_bridges(g)
    for edge_id in bridges:
        cuts[1].append(('edge', edge_id, g.edges[edge_id]))

    cut_verts = find_cut_vertices(g)
    for v in cut_verts:
        cuts[1].append(('vertex', v))

    if max_k < 2:
        return cuts

    # 2-cuts: pairs of edges whose removal disconnects
    edges = list(g.edges.keys())
    for i, e1 in enumerate(edges):
        for e2 in edges[i+1:]:
            if _is_k_edge_cut(g, [e1, e2]):
                cuts[2].append(('edges', e1, e2))

    if max_k < 3:
        return cuts

    # 3-cuts: only compute for small graphs (expensive)
    if g.num_edges() <= 15:
        for i, e1 in enumerate(edges):
            for j, e2 in enumerate(edges[i+1:], i+1):
                for e3 in edges[j+1:]:
                    if _is_k_edge_cut(g, [e1, e2, e3]):
                        cuts[3].append(('edges', e1, e2, e3))

    return cuts


def _is_k_edge_cut(g: GraphBuilder, edge_ids: List[int]) -> bool:
    """Check if removing the given edges disconnects the graph."""
    if g.num_nodes() <= 2:
        return False

    edge_set = set(edge_ids)
    remaining_edges = {k: v for k, v in g.edges.items() if k not in edge_set}

    if not remaining_edges:
        return True  # No edges left

    # BFS to check connectivity
    nodes = set(g.nodes)
    start = next(iter(nodes))
    visited = {start}
    stack = [start]

    while stack:
        curr = stack.pop()
        for eid, (a, b) in remaining_edges.items():
            if a == curr and b not in visited:
                visited.add(b)
                stack.append(b)
            elif b == curr and a not in visited:
                visited.add(a)
                stack.append(a)

    return len(visited) < len(nodes)


def get_edge_connectivity(g: GraphBuilder) -> int:
    """
    Compute the edge connectivity of the graph.

    Edge connectivity is the minimum number of edges whose removal
    disconnects the graph.

    Returns:
        Edge connectivity (0 if graph is already disconnected)
    """
    if g.num_edges() == 0:
        return 0

    # Check if already disconnected
    if not _is_connected(g):
        return 0

    # Check for bridges (1-connected)
    if find_bridges(g):
        return 1

    # Check for 2-cuts
    edges = list(g.edges.keys())
    for i, e1 in enumerate(edges):
        for e2 in edges[i+1:]:
            if _is_k_edge_cut(g, [e1, e2]):
                return 2

    # Check for 3-cuts (only for small graphs)
    if g.num_edges() <= 20:
        for i, e1 in enumerate(edges):
            for j, e2 in enumerate(edges[i+1:], i+1):
                for e3 in edges[j+1:]:
                    if _is_k_edge_cut(g, [e1, e2, e3]):
                        return 3

    # If no small cuts found, return a lower bound
    return 3  # At least 3-edge-connected


def _is_connected(g: GraphBuilder) -> bool:
    """Check if graph is connected."""
    if g.num_nodes() <= 1:
        return True

    nodes = set(g.nodes)
    start = next(iter(nodes))
    visited = {start}
    stack = [start]

    while stack:
        curr = stack.pop()
        for eid, (a, b) in g.edges.items():
            if a == curr and b not in visited:
                visited.add(b)
                stack.append(b)
            elif b == curr and a not in visited:
                visited.add(a)
                stack.append(a)

    return len(visited) == len(nodes)


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


# =============================================================================
# GRAPH SYNTHESIS FUNCTIONS
# =============================================================================

def glue_on_edge(g1: GraphBuilder, e1: int, g2: GraphBuilder, e2: int,
                  keep_edge: bool = True) -> GraphBuilder:
    """
    Glue two graphs along an edge.

    Args:
        keep_edge: If True, keep the shared edge (parallel connection)
                   If False, delete it (2-sum)
    """
    if keep_edge:
        return parallel_connection(g1, e1, g2, e2)
    else:
        return two_sum(g1, e1, g2, e2)


def glue_on_triangle(g1: GraphBuilder, tri1: List[int],
                     g2: GraphBuilder, tri2: List[int],
                     keep_triangle: bool = False) -> GraphBuilder:
    """
    Glue two graphs along a triangle (3-clique sum).
    """
    return clique_sum(g1, tri1, g2, tri2, delete_clique_edges=not keep_triangle)


def build_from_triangles(n_triangles: int) -> SynthesizedGraph:
    """
    Build a graph by gluing triangles together on edges.

    This creates a "ring of triangles" structure.
    """
    if n_triangles < 1:
        raise ValueError("Need at least 1 triangle")

    recipe = [f"Start with {n_triangles} triangles"]

    if n_triangles == 1:
        g = create_k3()
        t = compute_tutte_polynomial(g)
        return SynthesizedGraph(g, t, recipe)

    # Start with first triangle
    result = create_k3()

    # Glue remaining triangles
    for i in range(1, n_triangles):
        tri = create_k3()
        # Find an edge in result to glue on
        edge_id = list(result.edges.keys())[0]
        result = parallel_connection(result, edge_id, tri, 0)
        recipe.append(f"Glue triangle {i+1} via parallel connection")

    t = compute_tutte_polynomial(result)
    return SynthesizedGraph(result, t, recipe)


def build_wheel_variant(spokes: int, rim_edges: int = None) -> SynthesizedGraph:
    """
    Build a wheel-like graph with custom structure.

    Args:
        spokes: Number of spokes from center
        rim_edges: Number of rim edges (default: same as spokes for regular wheel)
    """
    if rim_edges is None:
        rim_edges = spokes

    g = GraphBuilder()
    center = g.add_node()

    # Add spoke endpoints
    rim_nodes = [g.add_node() for _ in range(spokes)]

    # Add spokes
    for node in rim_nodes:
        g.add_edge(center, node)

    # Add rim edges (cycle through rim nodes)
    for i in range(rim_edges):
        g.add_edge(rim_nodes[i % spokes], rim_nodes[(i + 1) % spokes])

    t = compute_tutte_polynomial(g)
    recipe = [f"Wheel variant: {spokes} spokes, {rim_edges} rim edges"]

    return SynthesizedGraph(g, t, recipe)


def build_prism(n: int) -> SynthesizedGraph:
    """
    Build a prism graph (two n-cycles connected by matching).

    Prism_n has 2n vertices, 3n edges.
    """
    g = GraphBuilder()

    # Two rings of n nodes
    ring1 = [g.add_node() for _ in range(n)]
    ring2 = [g.add_node() for _ in range(n)]

    # Connect each ring as a cycle
    for i in range(n):
        g.add_edge(ring1[i], ring1[(i + 1) % n])
        g.add_edge(ring2[i], ring2[(i + 1) % n])

    # Connect corresponding nodes between rings
    for i in range(n):
        g.add_edge(ring1[i], ring2[i])

    t = compute_tutte_polynomial(g)
    recipe = [f"Prism graph with n={n}"]

    return SynthesizedGraph(g, t, recipe)


def build_augmented_prism(n: int, extra_connections: List[Tuple[int, int]]) -> SynthesizedGraph:
    """
    Build a prism with additional cross-connections for higher connectivity.
    """
    g = GraphBuilder()

    ring1 = [g.add_node() for _ in range(n)]
    ring2 = [g.add_node() for _ in range(n)]

    # Base prism structure
    for i in range(n):
        g.add_edge(ring1[i], ring1[(i + 1) % n])
        g.add_edge(ring2[i], ring2[(i + 1) % n])
        g.add_edge(ring1[i], ring2[i])

    # Add extra connections
    for i, j in extra_connections:
        if i < n and j < n:
            g.add_edge(ring1[i], ring2[j])

    t = compute_tutte_polynomial(g)
    recipe = [f"Augmented prism n={n}", f"Extra connections: {extra_connections}"]

    return SynthesizedGraph(g, t, recipe)


def build_multi_clique_chain(clique_sizes: List[int], overlap: int = 2) -> SynthesizedGraph:
    """
    Build a chain of cliques connected by overlapping vertices.

    Args:
        clique_sizes: List of clique sizes
        overlap: Number of vertices shared between adjacent cliques
    """
    if not clique_sizes:
        raise ValueError("Need at least one clique")

    recipe = [f"Chain of cliques {clique_sizes} with overlap {overlap}"]

    # Start with first clique
    result = create_complete(clique_sizes[0])
    current_nodes = sorted(result.nodes)

    for i, size in enumerate(clique_sizes[1:], 1):
        # Create next clique
        next_clique = create_complete(size)
        next_nodes = sorted(next_clique.nodes)

        # Glue with overlap
        overlap_from_current = current_nodes[-overlap:]
        overlap_from_next = next_nodes[:overlap]

        result = clique_sum(result, overlap_from_current,
                           next_clique, overlap_from_next,
                           delete_clique_edges=False)  # Keep shared edges

        # Update current nodes (need to track through the merge)
        current_nodes = sorted(result.nodes)
        recipe.append(f"Added K_{size}")

    t = compute_tutte_polynomial(result)
    return SynthesizedGraph(result, t, recipe)


def build_zephyr_like(unit_cells: int = 2) -> SynthesizedGraph:
    """
    Attempt to build a Zephyr-like structure from minors.

    Zephyr has:
    - Multiple 8-cycles interconnected
    - High connectivity (3+ for Z(1,1))
    - Specific degree pattern

    We'll try to approximate this structure.
    """
    recipe = [f"Zephyr-like construction with {unit_cells} unit cells"]

    # Start with an 8-cycle (common Zephyr minor)
    g = GraphBuilder()
    nodes = [g.add_node() for _ in range(8)]
    for i in range(8):
        g.add_edge(nodes[i], nodes[(i + 1) % 8])

    # Add cross edges to increase connectivity (like Zephyr internal structure)
    g.add_edge(nodes[0], nodes[4])
    g.add_edge(nodes[2], nodes[6])

    recipe.append("Base: 8-cycle with 2 crossing chords")

    # Add more unit cells by gluing
    for cell in range(1, unit_cells):
        # Create another 8-cycle with chords
        cell_g = GraphBuilder()
        cell_nodes = [cell_g.add_node() for _ in range(8)]
        for i in range(8):
            cell_g.add_edge(cell_nodes[i], cell_nodes[(i + 1) % 8])
        cell_g.add_edge(cell_nodes[0], cell_nodes[4])
        cell_g.add_edge(cell_nodes[2], cell_nodes[6])

        # Glue via a 4-clique (share 4 adjacent vertices)
        # This mimics Zephyr's unit cell coupling
        share_from_g = [sorted(g.nodes)[-4 + i] for i in range(4)]
        share_from_cell = [sorted(cell_g.nodes)[i] for i in range(4)]

        g = clique_sum(g, share_from_g, cell_g, share_from_cell,
                       delete_clique_edges=False)
        recipe.append(f"Added unit cell {cell + 1} via 4-vertex overlap")

    t = compute_tutte_polynomial(g)
    return SynthesizedGraph(g, t, recipe)


def synthesize_zephyr_like(target_nodes: int = 12, target_edges: int = 22,
                           seed: int = None) -> SynthesizedGraph:
    """
    Synthesize a graph with Zephyr-like properties.

    Creates a graph matching Z(1,1)'s degree sequence [3,3,3,3,3,3,3,3,5,5,5,5]
    using NetworkX's random degree sequence generator. Different seeds produce
    different graphs with different Tutte polynomials.

    Args:
        target_nodes: Number of nodes (default 12 for Z(1,1))
        target_edges: Number of edges (default 22 for Z(1,1))
        seed: Random seed for reproducibility

    Returns:
        SynthesizedGraph with the constructed graph and its Tutte polynomial
    """
    import networkx as nx

    # Z(1,1) degree sequence: 8 nodes of degree 3, 4 nodes of degree 5
    degree_seq = [3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5]

    recipe = [f"Random graph with Z(1,1) degree sequence (seed={seed})"]

    # Try to generate connected graph
    max_attempts = 100
    for attempt in range(max_attempts):
        try:
            G = nx.random_degree_sequence_graph(degree_seq, seed=seed + attempt if seed else None)
            if nx.is_connected(G):
                break
        except:
            pass
    else:
        # Fallback to deterministic construction
        return _synthesize_deterministic()

    # Convert to GraphBuilder
    g = GraphBuilder()
    node_map = {n: g.add_node() for n in G.nodes()}
    for u, v in G.edges():
        g.add_edge(node_map[u], node_map[v])

    recipe.append(f"Generated connected graph on attempt {attempt + 1}")

    t = compute_tutte_polynomial(g)
    return SynthesizedGraph(g, t, recipe)


def _synthesize_deterministic() -> SynthesizedGraph:
    """Fallback deterministic synthesis."""
    g = GraphBuilder()
    recipe = ["Deterministic K4 + peripherals construction"]

    # Core: K4
    core = [g.add_node() for _ in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(core[i], core[j])

    # 8 peripheral nodes in 4 pairs
    peripheral = [g.add_node() for _ in range(8)]
    for i, (a, b) in enumerate([(0, 1), (2, 3), (4, 5), (6, 7)]):
        g.add_edge(peripheral[a], peripheral[b])
        g.add_edge(peripheral[a], core[i])
        g.add_edge(peripheral[b], core[i])

    # Fixed cross-connections
    for a, b in [(0, 2), (1, 3), (4, 6), (5, 7)]:
        g.add_edge(peripheral[a], peripheral[b])

    t = compute_tutte_polynomial(g)
    return SynthesizedGraph(g, t, recipe)


def generate_diverse_instances(n_instances: int = 10) -> List[SynthesizedGraph]:
    """
    Generate diverse proof-of-work instances with Zephyr-like properties.

    Each instance has the same degree sequence but different Tutte polynomial,
    providing variety for proof-of-work challenges.
    """
    instances = []
    seen_st = set()

    seed = 0
    while len(instances) < n_instances and seed < n_instances * 10:
        sg = synthesize_zephyr_like(seed=seed)
        st = sg.tutte.num_spanning_trees()
        if st not in seen_st:
            seen_st.add(st)
            instances.append(sg)
        seed += 1

    return instances


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_synthesized(sg: SynthesizedGraph, name: str = "Graph"):
    """Analyze properties of a synthesized graph."""
    print(f"\n{name}:")
    print(f"  Recipe: {' -> '.join(sg.recipe)}")
    print(f"  Nodes: {sg.graph.num_nodes()}, Edges: {sg.graph.num_edges()}")

    cuts = find_cut_vertices(sg.graph)
    bridges = find_bridges(sg.graph)
    print(f"  Cut vertices: {len(cuts)}")
    print(f"  Bridges: {len(bridges)}")

    # Degree distribution
    degree_count = {}
    for node in sg.graph.nodes:
        deg = sum(1 for eid, (a, b) in sg.graph.edges.items() if a == node or b == node)
        degree_count[deg] = degree_count.get(deg, 0) + 1
    print(f"  Degree distribution: {dict(sorted(degree_count.items()))}")

    print(f"  Tutte polynomial: {sg.tutte}")
    print(f"  Spanning trees T(1,1): {sg.tutte.num_spanning_trees()}")


def compare_to_zephyr():
    """Compare synthesized graphs to actual Zephyr Z(1,1)."""
    print("=" * 70)
    print("COMPARING SYNTHESIZED GRAPHS TO ZEPHYR Z(1,1)")
    print("=" * 70)

    # Load Z(1,1) properties from rainbow table
    try:
        table_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table.json')
        with open(table_path) as f:
            table = json.load(f)

        # Find Z(1,1)
        z11_entry = None
        for key, entry in table.get('graphs', {}).items():
            if entry.get('name') == 'Z(1,1)':
                z11_entry = entry
                break

        if z11_entry:
            print("\nZ(1,1) Reference:")
            print(f"  Nodes: {z11_entry['nodes']}, Edges: {z11_entry['edges']}")
            print(f"  Spanning trees: {z11_entry['spanning_trees']}")
            print(f"  Polynomial terms: {z11_entry['num_terms']}")
        else:
            print("\nZ(1,1) not found in rainbow table")
            z11_entry = {'nodes': 12, 'edges': 22, 'spanning_trees': 69360}

    except Exception as e:
        print(f"\nCouldn't load rainbow table: {e}")
        z11_entry = {'nodes': 12, 'edges': 22, 'spanning_trees': 69360}

    print("\n" + "-" * 70)
    print("SYNTHESIZED GRAPHS:")
    print("-" * 70)

    # Try various constructions
    constructions = [
        ("Triangle chain (4)", lambda: build_from_triangles(4)),
        ("Prism (4)", lambda: build_prism(4)),
        ("Prism (6)", lambda: build_prism(6)),
        ("Augmented Prism (4)", lambda: build_augmented_prism(4, [(0,2), (1,3)])),
        ("Wheel (6 spokes)", lambda: build_wheel_variant(6)),
        ("Clique chain [K4,K4,K4]", lambda: build_multi_clique_chain([4,4,4], overlap=2)),
        ("Zephyr-like (2 cells)", lambda: build_zephyr_like(2)),
    ]

    results = []
    for name, builder in constructions:
        try:
            sg = builder()
            analyze_synthesized(sg, name)
            results.append({
                'name': name,
                'nodes': sg.graph.num_nodes(),
                'edges': sg.graph.num_edges(),
                'spanning_trees': sg.tutte.num_spanning_trees(),
                'has_cut_vertices': len(find_cut_vertices(sg.graph)) > 0,
            })
        except Exception as e:
            print(f"\n{name}: ERROR - {e}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON TO Z(1,1)")
    print("=" * 70)
    print(f"{'Construction':<30} {'Nodes':>6} {'Edges':>6} {'Span.Trees':>12} {'Cut-V?':>6}")
    print("-" * 70)
    print(f"{'Z(1,1) Target':<30} {z11_entry['nodes']:>6} {z11_entry['edges']:>6} {z11_entry['spanning_trees']:>12} {'No':>6}")
    print("-" * 70)
    for r in results:
        cut_v = "Yes" if r['has_cut_vertices'] else "No"
        print(f"{r['name']:<30} {r['nodes']:>6} {r['edges']:>6} {r['spanning_trees']:>12} {cut_v:>6}")


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

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

    print("\n" + "=" * 70)
    print("GRAPH SYNTHESIS DEMO")
    print("=" * 70)

    compare_to_zephyr()

    print("\n" + "=" * 70)
    print("DIVERSE INSTANCE GENERATION")
    print("=" * 70)
    instances = generate_diverse_instances(5)
    for i, sg in enumerate(instances):
        print(f"\nInstance {i+1}:")
        print(f"  Spanning trees: {sg.tutte.num_spanning_trees()}")
        print(f"  Recipe: {sg.recipe[-1]}")
