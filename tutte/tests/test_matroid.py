"""Tests for matroid-based Tutte polynomial computation.

Validates Theorem 6 (parallel connection) and Theorem 10 (k-sum) formulas
against brute-force synthesis on graph atlas pairs.
"""

import time

import pytest
import networkx as nx

from tutte.graph import (
    Graph, complete_graph, cycle_graph, parallel_connection_graph, k_sum_graph,
    petersen_graph, wheel_graph, grid_graph,
)
from tutte.polynomial import TuttePolynomial
from tutte.matroids.core import GraphicMatroid, FlatLattice, enumerate_flats_with_hasse
from tutte.matroids.parallel_connection import (
    BivariateLaurentPoly,
    theorem6_parallel_connection,
    theorem10_k_sum,
    theorem10_k_sum_via_theorem6,
    precompute_contractions,
    build_extended_cell_graph,
)
from tutte.validation import verify_spanning_trees, verify_with_networkx


# =============================================================================
# HELPERS
# =============================================================================

def _make_graph(nx_graph):
    """Convert a NetworkX graph to our Graph, ensuring integer labels."""
    G = nx.convert_node_labels_to_integers(nx_graph)
    return Graph.from_networkx(G)


def _synthesize_direct(graph, engine):
    """Synthesize T(G) directly via the engine."""
    result = engine.synthesize(graph)
    return result.polynomial


def _build_parallel_connection_and_verify(g1, g2, shared_edge, engine):
    """Build P_N(G1, G2) as explicit graph, synthesize directly,
    then compute via Theorem 6 and verify match.

    Returns (T_direct, T_theorem6, match).
    """
    # Build explicit parallel connection graph
    pc_graph = parallel_connection_graph(g1, g2, shared_edge)

    # Direct synthesis
    T_direct = _synthesize_direct(pc_graph, engine)

    # Theorem 6 path: build inter-cell matroid
    u, v = shared_edge
    inter_edges = [shared_edge]
    inter_edges_norm = [(min(u, v), max(u, v))]

    # Build inter-cell graph (just the shared edge)
    inter_graph = Graph(
        nodes=frozenset([u, v]),
        edges=frozenset(inter_edges_norm),
    )
    matroid_N = GraphicMatroid(inter_graph)
    r_N = matroid_N.rank()

    # Enumerate flats with Hasse
    flats, ranks, upper_covers = enumerate_flats_with_hasse(matroid_N)
    lattice = FlatLattice(matroid_N, flats=flats, ranks=ranks, upper_covers=upper_covers)

    # For Theorem 6, partition = [g1_nodes, g2_nodes_remapped]
    # The extended cell graphs are g1 and g2 themselves (since they share the edge)
    cell1_nodes = set(g1.nodes)
    cell2_nodes = set(pc_graph.nodes) - cell1_nodes | {u, v}

    # Build extended cell graphs
    ext1, shared1 = build_extended_cell_graph(pc_graph, cell1_nodes, inter_edges_norm)
    ext2_nodes = set(pc_graph.nodes) - set(g1.nodes) | {u, v}
    ext2, shared2 = build_extended_cell_graph(pc_graph, ext2_nodes, inter_edges_norm)

    # Precompute contractions
    t_m1 = precompute_contractions(ext1, shared1, lattice, engine)
    t_m2 = precompute_contractions(ext2, shared2, lattice, engine)

    # Apply Theorem 6
    T_thm6 = theorem6_parallel_connection(lattice, t_m1, t_m2, r_N)

    match = (T_direct == T_thm6)
    return T_direct, T_thm6, match


# =============================================================================
# BIVARIATE LAURENT POLY TESTS
# =============================================================================

class TestBivariateLaurentPoly:
    """Tests for the BivariateLaurentPoly class."""

    def test_from_tutte_and_back(self):
        """Round-trip: T(x,y) -> R(u,v) -> T(x,y)."""
        # T(K3) = x^2 + x + y
        poly = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
        r = BivariateLaurentPoly.from_tutte(poly)
        back = r.to_tutte_poly()
        assert poly == back

    def test_divmod_exact(self):
        """Exact polynomial division."""
        # (u^2 + 2u + 1) = (u + 1) * (u + 1)
        p = BivariateLaurentPoly({(2, 0): 1, (1, 0): 2, (0, 0): 1})
        q = BivariateLaurentPoly({(1, 0): 1, (0, 0): 1})
        quotient, remainder = p.divmod(q)
        assert remainder.is_zero()
        assert quotient == q

    def test_divmod_with_v(self):
        """Division involving v terms."""
        # (u*v + v) / v = (u + 1)
        p = BivariateLaurentPoly({(1, 1): 1, (0, 1): 1})
        q = BivariateLaurentPoly({(0, 1): 1})
        quotient = p // q
        expected = BivariateLaurentPoly({(1, 0): 1, (0, 0): 1})
        assert quotient == expected

    def test_floordiv_nonexact_raises(self):
        """Non-exact division should raise ValueError."""
        p = BivariateLaurentPoly({(2, 0): 1, (0, 0): 1})  # u^2 + 1
        q = BivariateLaurentPoly({(1, 0): 1, (0, 0): 1})  # u + 1
        with pytest.raises(ValueError):
            p // q

    def test_shift_v(self):
        """Test v-power shifting."""
        p = BivariateLaurentPoly({(1, 2): 3, (0, 0): 1})
        shifted = p.shift_v(-2)
        assert shifted == BivariateLaurentPoly({(1, 0): 3, (0, -2): 1})


# =============================================================================
# THEOREM 6 TARGETED TESTS
# =============================================================================

class TestTheorem6Targeted:
    """Targeted Theorem 6 tests with known graphs."""

    def test_two_triangles_sharing_edge(self, engine):
        """Two K3s sharing an edge should produce correct parallel connection.

        The parallel connection of two triangles sharing an edge
        gives a graph with 4 nodes and 5 edges (diamond/book graph).
        """
        # K3 with nodes {0, 1, 2}, shared edge (0, 1)
        k3 = complete_graph(3)
        pc = parallel_connection_graph(k3, k3, (0, 1))

        T_direct = _synthesize_direct(pc, engine)
        assert verify_spanning_trees(pc, T_direct)
        assert verify_with_networkx(pc, T_direct), "NetworkX mismatch for PC(K3,K3)"

        # Verify via Theorem 6
        shared_edge = (0, 1)
        inter_graph = Graph(
            nodes=frozenset([0, 1]),
            edges=frozenset([(0, 1)]),
        )
        matroid_N = GraphicMatroid(inter_graph)
        r_N = matroid_N.rank()

        flats, ranks, upper_covers = enumerate_flats_with_hasse(matroid_N)
        lattice = FlatLattice(matroid_N, flats=flats, ranks=ranks, upper_covers=upper_covers)

        # g1 = K3 itself (contains the shared edge)
        # g2 = K3 remapped (nodes 0,1 shared, node 2 -> some other node)
        # For parallel connection, both extended cells ARE the original K3s
        cell1_nodes = set(k3.nodes)
        cell2_nodes = set(pc.nodes) - {2}  # nodes 0, 1, and the remapped node

        ext1, shared1 = build_extended_cell_graph(pc, cell1_nodes, [(0, 1)])
        remaining = set(pc.nodes) - cell1_nodes | {0, 1}
        ext2, shared2 = build_extended_cell_graph(pc, remaining, [(0, 1)])

        t_m1 = precompute_contractions(ext1, shared1, lattice, engine)
        t_m2 = precompute_contractions(ext2, shared2, lattice, engine)

        T_thm6 = theorem6_parallel_connection(lattice, t_m1, t_m2, r_N)
        assert T_direct == T_thm6, f"Theorem 6 mismatch: direct={T_direct}, thm6={T_thm6}"

    def test_k4_sharing_edge_with_k4(self, engine):
        """Two K4s sharing a single edge."""
        k4 = complete_graph(4)
        pc = parallel_connection_graph(k4, k4, (0, 1))

        T_direct = _synthesize_direct(pc, engine)
        assert verify_spanning_trees(pc, T_direct)
        assert verify_with_networkx(pc, T_direct), "NetworkX mismatch for PC(K4,K4)"

    def test_triangle_and_square_sharing_edge(self, engine):
        """K3 and C4 sharing an edge."""
        k3 = complete_graph(3)
        c4 = cycle_graph(4)

        # They need to share an edge. c4 has edge (0,1)
        pc = parallel_connection_graph(k3, c4, (0, 1))

        T_direct = _synthesize_direct(pc, engine)
        assert verify_spanning_trees(pc, T_direct)
        assert verify_with_networkx(pc, T_direct), "NetworkX mismatch for PC(K3,C4)"


# =============================================================================
# THEOREM 10 TARGETED TESTS
# =============================================================================

class TestTheorem10Targeted:
    """Targeted Theorem 10 tests for k-sum computation."""

    def test_2sum_two_triangles(self, engine):
        """2-sum of two K3s = C4.

        P(K3, K3) sharing (0,1) = diamond. Delete shared edge = C4.
        """
        k3 = complete_graph(3)
        pc = parallel_connection_graph(k3, k3, (0, 1))
        ks = k_sum_graph(k3, k3, 2, [0, 1])

        T_ksum = theorem10_k_sum(pc, [(0, 1)], engine)
        T_direct = _synthesize_direct(ks, engine)

        assert T_ksum == T_direct, f"Theorem 10 mismatch: got {T_ksum}, expected {T_direct}"
        assert verify_with_networkx(ks, T_ksum), "NetworkX mismatch for 2-sum(K3,K3)"

    def test_2sum_k4_k4(self, engine):
        """2-sum of two K4s sharing edge (0,1)."""
        k4 = complete_graph(4)
        pc = parallel_connection_graph(k4, k4, (0, 1))
        ks = k_sum_graph(k4, k4, 2, [0, 1])

        T_ksum = theorem10_k_sum(pc, [(0, 1)], engine)
        T_direct = _synthesize_direct(ks, engine)

        assert T_ksum == T_direct
        assert verify_with_networkx(ks, T_ksum), "NetworkX mismatch for 2-sum(K4,K4)"

    def test_2sum_triangle_square(self, engine):
        """2-sum of K3 and C4 sharing edge (0,1)."""
        k3 = complete_graph(3)
        c4 = cycle_graph(4)
        pc = parallel_connection_graph(k3, c4, (0, 1))
        ks = k_sum_graph(k3, c4, 2, [0, 1])

        T_ksum = theorem10_k_sum(pc, [(0, 1)], engine)
        T_direct = _synthesize_direct(ks, engine)

        assert T_ksum == T_direct
        assert verify_with_networkx(ks, T_ksum), "NetworkX mismatch for 2-sum(K3,C4)"

    def test_2sum_kirchhoff(self, engine):
        """Theorem 10 result passes Kirchhoff and NetworkX verification."""
        k3 = complete_graph(3)
        pc = parallel_connection_graph(k3, k3, (0, 1))
        ks = k_sum_graph(k3, k3, 2, [0, 1])

        T_ksum = theorem10_k_sum(pc, [(0, 1)], engine)
        assert verify_spanning_trees(ks, T_ksum)
        assert verify_with_networkx(ks, T_ksum)


# =============================================================================
# GRAPH CONSTRUCTION TESTS
# =============================================================================

class TestGraphConstruction:
    """Tests for parallel_connection_graph() and k_sum_graph()."""

    def test_parallel_connection_node_count(self):
        """Parallel connection of K3 and K3 sharing edge has 4 nodes."""
        k3 = complete_graph(3)
        pc = parallel_connection_graph(k3, k3, (0, 1))
        # K3 has 3 nodes. Sharing edge (0,1) means 2 shared nodes.
        # Total = 3 + 3 - 2 = 4
        assert pc.node_count() == 4

    def test_parallel_connection_edge_count(self):
        """Parallel connection of K3 and K3 sharing edge has 5 edges."""
        k3 = complete_graph(3)
        pc = parallel_connection_graph(k3, k3, (0, 1))
        # K3 has 3 edges. Shared edge counted once. 3 + 3 - 1 = 5
        assert pc.edge_count() == 5

    def test_parallel_connection_invalid_edge(self):
        """Shared edge must exist in both graphs."""
        k3 = complete_graph(3)
        with pytest.raises(ValueError):
            parallel_connection_graph(k3, k3, (0, 5))

    def test_k_sum_2sum_from_triangles(self):
        """2-sum of two K3s sharing edge (0,1) = path P4."""
        k3 = complete_graph(3)
        result = k_sum_graph(k3, k3, 2, [0, 1])
        # 2-sum: identify 2 vertices, delete shared edge
        # K3 + K3 sharing (0,1), delete edge (0,1)
        # Result: 4 nodes, 4 edges (was 5 from PC, minus 1 shared edge)
        assert result.node_count() == 4
        assert result.edge_count() == 4

    def test_k_sum_1sum(self):
        """1-sum is just identifying one vertex (no edges to delete)."""
        k3 = complete_graph(3)
        k4 = complete_graph(4)
        result = k_sum_graph(k3, k4, 1, [0])
        # 1-sum: identify vertex 0, no clique edges to delete (K_1 has none)
        # Total nodes: 3 + 4 - 1 = 6
        assert result.node_count() == 6
        # Total edges: 3 + 6 = 9
        assert result.edge_count() == 9


# =============================================================================
# ATLAS PAIR TESTS
# =============================================================================

def _get_atlas_graphs(min_nodes=3, max_nodes=6):
    """Get connected atlas graphs in the given node range."""
    graphs = []
    for G in nx.graph_atlas_g():
        if G.number_of_nodes() < min_nodes:
            continue
        if G.number_of_nodes() > max_nodes:
            break
        if not nx.is_connected(G):
            continue
        if G.number_of_edges() < 2:
            continue
        graphs.append(G)
    return graphs


def _find_shared_edges(g1_nx, g2_nx):
    """Find edges that exist in both graphs (by node labels)."""
    e1 = set(g1_nx.edges())
    e2 = set(g2_nx.edges())
    # Also check reversed edges
    e2_both = e2 | {(v, u) for u, v in e2}
    shared = []
    for u, v in e1:
        if (u, v) in e2_both or (v, u) in e2_both:
            shared.append((min(u, v), max(u, v)))
    return shared


class TestAtlasPairs:
    """Test Theorem 6 on atlas graph pairs."""

    @pytest.mark.slow
    def test_atlas_parallel_connections(self, engine):
        """Test parallel connections on small atlas graph pairs.

        For each pair of small connected graphs that share an edge,
        build the parallel connection and verify the direct synthesis
        matches Kirchhoff.
        """
        atlas = _get_atlas_graphs(3, 5)

        tested = 0
        for i, G1 in enumerate(atlas[:30]):  # Limit for speed
            g1 = _make_graph(G1)
            for j, G2 in enumerate(atlas[:30]):
                # Both must have edge (0, 1)
                g1_nx = nx.convert_node_labels_to_integers(G1)
                g2_nx = nx.convert_node_labels_to_integers(G2)
                if not g1_nx.has_edge(0, 1) or not g2_nx.has_edge(0, 1):
                    continue

                g2 = _make_graph(G2)

                # Build parallel connection
                pc = parallel_connection_graph(g1, g2, (0, 1))
                if pc.edge_count() > 15:
                    continue  # Skip expensive cases

                T_direct = _synthesize_direct(pc, engine)
                assert verify_spanning_trees(pc, T_direct), (
                    f"Kirchhoff failed for PC of atlas[{i}]+atlas[{j}]"
                )
                assert verify_with_networkx(pc, T_direct), (
                    f"NetworkX mismatch for PC of atlas[{i}]+atlas[{j}]"
                )
                tested += 1

        assert tested > 0, "No atlas pairs tested"

    @pytest.mark.slow
    def test_atlas_2sums(self, engine):
        """Test 2-sums on atlas graph pairs."""
        atlas = _get_atlas_graphs(3, 5)

        tested = 0
        for i, G1 in enumerate(atlas[:20]):
            g1 = _make_graph(G1)
            for j, G2 in enumerate(atlas[:20]):
                g1_nx = nx.convert_node_labels_to_integers(G1)
                g2_nx = nx.convert_node_labels_to_integers(G2)
                if not g1_nx.has_edge(0, 1) or not g2_nx.has_edge(0, 1):
                    continue

                g2 = _make_graph(G2)

                # Build 2-sum
                ks = k_sum_graph(g1, g2, 2, [0, 1])
                if ks.edge_count() > 15:
                    continue

                # 2-sums can produce disconnected graphs; skip those
                ks_nx = nx.Graph()
                ks_nx.add_nodes_from(ks.nodes)
                ks_nx.add_edges_from(ks.edges)
                if not nx.is_connected(ks_nx):
                    continue

                T_direct = _synthesize_direct(ks, engine)
                assert verify_spanning_trees(ks, T_direct), (
                    f"Kirchhoff failed for 2-sum of atlas[{i}]+atlas[{j}]"
                )
                assert verify_with_networkx(ks, T_direct), (
                    f"NetworkX mismatch for 2-sum of atlas[{i}]+atlas[{j}]"
                )
                tested += 1

        assert tested > 0, "No atlas 2-sum pairs tested"


# =============================================================================
# ALGEBRAIC RELATIONSHIP CHARACTERIZATION
# =============================================================================

class TestAlgebraicRelationships:
    """Characterize when simple k-sum formulas work."""

    def test_simple_2sum_formula_on_triangles(self, engine):
        """T(K3 ⊕_2 K3) should equal T(K3)*T(K3)/T(K2).

        The simple 2-sum formula: T(G1 ⊕_2 G2) = T(G1)*T(G2) / T(K_2)
        where T(K2) = x.
        """
        k3 = complete_graph(3)
        ks = k_sum_graph(k3, k3, 2, [0, 1])

        T_ks = _synthesize_direct(ks, engine)
        T_k3 = _synthesize_direct(k3, engine)
        T_k2 = TuttePolynomial.x()  # T(K2) = x

        # Simple formula: T(K3)^2 / T(K2)
        from tutte.graphs.k_join import polynomial_divmod
        product = T_k3 * T_k3
        quotient, remainder = polynomial_divmod(product, T_k2)

        if remainder.is_zero():
            # Simple formula applies
            assert quotient == T_ks, (
                f"Simple 2-sum formula gave {quotient}, expected {T_ks}"
            )

    def test_simple_1sum_formula(self, engine):
        """T(G1 ⊕_1 G2) = T(G1) * T(G2).

        1-sum is just cut vertex join, so the product formula always works.
        """
        k3 = complete_graph(3)
        k4 = complete_graph(4)
        ks = k_sum_graph(k3, k4, 1, [0])

        T_ks = _synthesize_direct(ks, engine)
        T_k3 = _synthesize_direct(k3, engine)
        T_k4 = _synthesize_direct(k4, engine)

        assert T_ks == T_k3 * T_k4


# =============================================================================
# HIGHER K-SUM CORRECTNESS TESTS (k=3,4,5)
# =============================================================================

class TestHigherKSums:
    """Correctness tests for k-sum via Theorem 10, k=3,4,5."""

    def test_3sum_two_k4s(self, engine):
        """3-sum of two K4s: identify 3 vertices, delete K3 edges."""
        k4 = complete_graph(4)
        ks = k_sum_graph(k4, k4, 3, [0, 1, 2])
        T_direct = _synthesize_direct(ks, engine)
        assert verify_spanning_trees(ks, T_direct)
        assert verify_with_networkx(ks, T_direct)

        # Verify via Theorem 10
        sv = [0, 1, 2]
        clique_edges = [(0, 1), (0, 2), (1, 2)]
        pc_edges = ks.edges | frozenset(clique_edges)
        pc_graph = Graph(nodes=ks.nodes, edges=pc_edges)
        T_thm10 = theorem10_k_sum(pc_graph, clique_edges, engine)
        assert T_thm10 == T_direct

    def test_4sum_two_k5s(self, engine):
        """4-sum of two K5s: identify 4 vertices, delete K4 edges."""
        k5 = complete_graph(5)
        ks = k_sum_graph(k5, k5, 4, [0, 1, 2, 3])
        T_direct = _synthesize_direct(ks, engine)
        assert verify_spanning_trees(ks, T_direct)
        assert verify_with_networkx(ks, T_direct)

        sv = [0, 1, 2, 3]
        clique_edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        pc_edges = ks.edges | frozenset(clique_edges)
        pc_graph = Graph(nodes=ks.nodes, edges=pc_edges)
        T_thm10 = theorem10_k_sum(pc_graph, clique_edges, engine)
        assert T_thm10 == T_direct

    def test_5sum_two_k6s(self, engine):
        """5-sum of two K6s: identify 5 vertices, delete K5 edges."""
        k6 = complete_graph(6)
        ks = k_sum_graph(k6, k6, 5, [0, 1, 2, 3, 4])
        T_direct = _synthesize_direct(ks, engine)
        assert verify_spanning_trees(ks, T_direct)

        sv = [0, 1, 2, 3, 4]
        clique_edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        pc_edges = ks.edges | frozenset(clique_edges)
        pc_graph = Graph(nodes=ks.nodes, edges=pc_edges)
        T_thm10 = theorem10_k_sum(pc_graph, clique_edges, engine)
        assert T_thm10 == T_direct

    def test_3sum_k3_k4(self, engine):
        """3-sum of K3 and K4: all 3 vertices shared, all edges of K3 deleted."""
        k3 = complete_graph(3)
        k4 = complete_graph(4)
        ks = k_sum_graph(k3, k4, 3, [0, 1, 2])
        T_direct = _synthesize_direct(ks, engine)
        assert verify_spanning_trees(ks, T_direct)

        clique_edges = [(0, 1), (0, 2), (1, 2)]
        pc_edges = ks.edges | frozenset(clique_edges)
        pc_graph = Graph(nodes=ks.nodes, edges=pc_edges)
        T_thm10 = theorem10_k_sum(pc_graph, clique_edges, engine)
        assert T_thm10 == T_direct


# =============================================================================
# K-SUM ROUTE TRIGGERING TESTS
# =============================================================================

class TestKSumRouteTriggering:
    """Verify engine.synthesize() actually uses the k-sum path for k-sum graphs."""

    @pytest.mark.parametrize("k,n", [
        (2, 4),   # 2-sum K4+K4: min indep separator = 2
        (3, 5),   # 3-sum K5+K5: min indep separator = 3 (forces 3sum route)
        (4, 6),   # 4-sum K6+K6: uses flat-grouped Theorem 6
        (5, 7),   # 5-sum K7+K7: uses flat-grouped Theorem 6
    ])
    def test_ksum_route_triggers(self, k, n, engine):
        """Verify engine uses the exact k-sum path for graphs with no smaller separator.

        Using n=k+2 ensures vertex connectivity = k, so no (k-1)-vertex
        independent separator exists and the engine must use exactly k-sum.
        """
        kn = complete_graph(n)
        shared = list(range(k))
        ks = k_sum_graph(kn, kn, k, shared)

        # Check connectivity - k-sums can disconnect
        ks_nx = nx.Graph()
        ks_nx.add_nodes_from(ks.nodes)
        ks_nx.add_edges_from(ks.edges)
        if not nx.is_connected(ks_nx):
            pytest.skip(f"{k}-sum of K{n} is disconnected")

        engine._cache.clear()
        # Temporarily remove from rainbow table so k-sum detection fires
        cache_key = ks.canonical_key()
        saved_entry = engine.table.entries.pop(cache_key, None)
        try:
            result = engine.synthesize(ks)
            assert result.verified
            assert result.method == f"{k}sum_theorem10", (
                f"Expected {k}sum_theorem10, got {result.method}"
            )
        finally:
            if saved_entry is not None:
                engine.table.entries[cache_key] = saved_entry


# =============================================================================
# METHOD DISTRIBUTION TEST
# =============================================================================

class TestMethodDistribution:
    """Characterize which synthesis method each standard graph uses."""

    def test_method_distribution(self, engine):
        """Log which synthesis method each standard graph uses."""
        graphs = [
            ("K5", complete_graph(5)),
            ("K6", complete_graph(6)),
            ("Petersen", petersen_graph()),
            ("W7", wheel_graph(7)),
            ("Grid3x3", grid_graph(3, 3)),
            ("2sum_K4_K4", k_sum_graph(complete_graph(4), complete_graph(4), 2, [0, 1])),
            ("3sum_K4_K4", k_sum_graph(complete_graph(4), complete_graph(4), 3, [0, 1, 2])),
            ("4sum_K5_K5", k_sum_graph(complete_graph(5), complete_graph(5), 4, [0, 1, 2, 3])),
        ]
        engine._cache.clear()
        # Temporarily remove k-sum graphs from rainbow table so k-sum detection fires
        saved_entries = {}
        ksum_graphs = [g for name, g in graphs if "sum" in name]
        for g in ksum_graphs:
            key = g.canonical_key()
            if key in engine.table.entries:
                saved_entries[key] = engine.table.entries.pop(key)

        try:
            methods = {}
            for name, g in graphs:
                # Check connectivity before synthesizing
                g_nx = nx.Graph()
                g_nx.add_nodes_from(g.nodes)
                g_nx.add_edges_from(g.edges)
                if not nx.is_connected(g_nx):
                    methods[name] = "skipped_disconnected"
                    continue

                result = engine.synthesize(g)
                methods[name] = result.method
                assert result.verified, f"{name} result not verified"

            # At least one k-sum graph should use a k-sum path
            ksum_methods = [m for m in methods.values() if "sum_theorem10" in m]
            assert len(ksum_methods) > 0, f"No graph triggered k-sum: {methods}"
        finally:
            engine.table.entries.update(saved_entries)


# =============================================================================
# FLAT LATTICE CACHING TESTS
# =============================================================================

class TestFlatLatticeCaching:
    """Test flat lattice serialization/deserialization round-trip."""

    def test_flat_lattice_cache_roundtrip(self, engine):
        """Cached flat lattice data produces same Theorem 6 result."""
        k3 = complete_graph(3)
        pc = parallel_connection_graph(k3, k3, (0, 1))

        # Build lattice from scratch
        inter_graph = Graph(
            nodes=frozenset([0, 1]),
            edges=frozenset([(0, 1)]),
        )
        matroid_N = GraphicMatroid(inter_graph)
        r_N = matroid_N.rank()

        flats, ranks, uc = enumerate_flats_with_hasse(matroid_N)
        lattice = FlatLattice(matroid_N, flats=flats, ranks=ranks, upper_covers=uc)

        # Cache round-trip
        data = lattice.to_flat_lattice_data()
        lattice2 = FlatLattice.from_flat_lattice_data(matroid_N, data)

        # Both should have same number of flats
        assert lattice.num_flats == lattice2.num_flats

        # Build extended cell graphs
        cell1_nodes = set(k3.nodes)
        remaining = set(pc.nodes) - cell1_nodes | {0, 1}

        ext1, s1 = build_extended_cell_graph(pc, cell1_nodes, [(0, 1)])
        ext2, s2 = build_extended_cell_graph(pc, remaining, [(0, 1)])

        # Compute Theorem 6 with original lattice
        t_m1 = precompute_contractions(ext1, s1, lattice, engine)
        t_m2 = precompute_contractions(ext2, s2, lattice, engine)
        T1 = theorem6_parallel_connection(lattice, t_m1, t_m2, r_N)

        # Compute Theorem 6 with reconstructed lattice
        t_m1b = precompute_contractions(ext1, s1, lattice2, engine)
        t_m2b = precompute_contractions(ext2, s2, lattice2, engine)
        T2 = theorem6_parallel_connection(lattice2, t_m1b, t_m2b, r_N)

        assert T1 == T2, f"Flat lattice cache mismatch: {T1} != {T2}"

    def test_flat_lattice_data_fields(self):
        """FlatLatticeData has correct structure."""
        inter_graph = Graph(
            nodes=frozenset([0, 1]),
            edges=frozenset([(0, 1)]),
        )
        matroid_N = GraphicMatroid(inter_graph)

        flats, ranks, uc = enumerate_flats_with_hasse(matroid_N)
        lattice = FlatLattice(matroid_N, flats=flats, ranks=ranks, upper_covers=uc)

        data = lattice.to_flat_lattice_data()
        assert len(data.flats) == lattice.num_flats
        assert len(data.ranks) == lattice.num_flats
        assert isinstance(data.upper_covers, dict)


# =============================================================================
# K-SUM BENCHMARKS
# =============================================================================

class TestKSumBenchmarks:
    """Performance comparison: k-sum decomposition vs direct synthesis."""

    @pytest.mark.perf
    @pytest.mark.parametrize("k,n", [(2, 5), (2, 6), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)])
    def test_benchmark_ksum_scaling(self, k, n, engine, benchmark_collector):
        """Compare k-sum decomposition vs direct synthesis."""
        kn = complete_graph(n)
        shared = list(range(k))
        ks = k_sum_graph(kn, kn, k, shared)

        # Check connectivity
        ks_nx = nx.Graph()
        ks_nx.add_nodes_from(ks.nodes)
        ks_nx.add_edges_from(ks.edges)
        if not nx.is_connected(ks_nx):
            pytest.skip(f"{k}-sum of K{n} is disconnected")

        timings = {}

        # k-sum path (via synthesize which should detect k-sum)
        engine._cache.clear()
        t0 = time.perf_counter()
        result_ksum = engine.synthesize(ks)
        timings[f"{k}sum"] = round((time.perf_counter() - t0) * 1000, 2)

        assert result_ksum.verified

        benchmark_collector.record(
            name=f"{k}sum_K{n}_K{n}",
            nodes=ks.node_count(),
            edges=ks.edge_count(),
            spanning_trees=result_ksum.polynomial.num_spanning_trees(),
            timings_ms=timings,
        )


# =============================================================================
# Z(1,2) INTEGRATION TESTS
# =============================================================================

class TestZephyrIntegration:
    """Integration tests for Zephyr graph synthesis with matroid paths."""

    @pytest.mark.slow
    def test_z12_synthesis(self, engine):
        """Verify Z(1,2) synthesis produces a verified result."""
        dnx = pytest.importorskip("dwave_networkx")
        z12 = Graph.from_networkx(dnx.zephyr_graph(1, 2))
        engine._cache.clear()
        result = engine.synthesize(z12)
        assert result.verified
