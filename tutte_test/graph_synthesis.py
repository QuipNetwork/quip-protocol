"""
Graph Synthesis from Tutte Polynomial Motifs.

GOAL: Build graphs with Zephyr-like connectivity by composing
rainbow table motifs, using REVERSE composition operations.

Instead of: Graph → decompose → polynomials
We want:    Motifs (known polynomials) → compose → Graph with desired structure

Key operations (used in reverse):
- Disjoint union → Connect components
- Cut vertex → Create higher connectivity by adding edges
- 2-sum → Glue on shared edge
- Clique sum → Glue on shared clique
- Parallel connection → Add parallel paths
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tutte_test.tutte_to_ising import (
    TuttePolynomial,
    GraphBuilder,
    compute_tutte_polynomial,
)
from tutte_test.graph_composition import (
    cut_vertex_join,
    two_sum,
    clique_sum,
    parallel_connection,
    disjoint_union,
    create_complete,
    create_cycle,
    create_path,
    find_cut_vertices,
    find_bridges,
)


@dataclass
class SynthesizedGraph:
    """A graph built from motif composition."""
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
        # For exact connectivity would need more analysis
        return 2


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


def add_edge_between(g: GraphBuilder, u: int, v: int) -> GraphBuilder:
    """Add an edge between existing vertices."""
    g.add_edge(u, v)
    return g


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
        # This is approximate - actual node tracking is complex
        current_nodes = sorted(result.nodes)
        recipe.append(f"Added K_{size}")

    t = compute_tutte_polynomial(result)
    return SynthesizedGraph(result, t, recipe)


def build_zephyr_like(unit_cells: int = 2) -> SynthesizedGraph:
    """
    Attempt to build a Zephyr-like structure from motifs.

    Zephyr has:
    - Multiple 8-cycles interconnected
    - High connectivity (3+ for Z(1,1))
    - Specific degree pattern

    We'll try to approximate this structure.
    """
    recipe = [f"Zephyr-like construction with {unit_cells} unit cells"]

    # Start with an 8-cycle (common Zephyr motif)
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

    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS")
    print("=" * 70)
    print("""
To build a graph MATCHING Z(1,1)'s Tutte polynomial, we need:
1. Same number of nodes (12) and edges (22)
2. Same spanning tree count (69,360)
3. No cut vertices (for 2+ connectivity)

The challenge: Simple compositions tend to create cut vertices,
reducing connectivity below Zephyr's level.

NEXT STEPS:
1. Identify motifs that preserve high connectivity when combined
2. Use edge additions after composition to eliminate cut vertices
3. Search for composition recipes that match target polynomials
""")


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


if __name__ == "__main__":
    compare_to_zephyr()

    print("\n" + "=" * 70)
    print("DIVERSE INSTANCE GENERATION")
    print("=" * 70)
    instances = generate_diverse_instances(5)
    for i, sg in enumerate(instances):
        print(f"\nInstance {i+1}:")
        print(f"  Spanning trees: {sg.tutte.num_spanning_trees()}")
        print(f"  Recipe: {sg.recipe[-1]}")
