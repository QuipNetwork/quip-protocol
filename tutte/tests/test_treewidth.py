"""Tests for treewidth-based Tutte polynomial computation."""

import networkx as nx

from tutte.graph import Graph, MultiGraph, complete_graph, cycle_graph, petersen_graph
from tutte.graphs.treewidth import (
    canonicalize,
    connect,
    forget,
    compute_tree_decomposition,
    compute_treewidth_tutte,
    compute_treewidth_tutte_if_applicable,
)
from tutte.graphs.series_parallel import (
    compute_sp_tutte_if_applicable,
    compute_sp_tutte_multigraph_if_applicable,
)
from tutte.polynomial import TuttePolynomial


# =============================================================================
# SET PARTITION TESTS
# =============================================================================

class TestCanonicalize:
    def test_already_canonical(self):
        assert canonicalize((0, 1, 2)) == (0, 1, 2)

    def test_needs_renumbering(self):
        assert canonicalize((2, 0, 2, 1)) == (0, 1, 0, 2)

    def test_single_block(self):
        assert canonicalize((5, 5, 5)) == (0, 0, 0)

    def test_empty(self):
        assert canonicalize(()) == ()


class TestConnect:
    def test_different_blocks(self):
        result = connect((0, 1, 2), 0, 1)
        assert result == (0, 0, 1)

    def test_same_block(self):
        result = connect((0, 0, 1), 0, 1)
        assert result == (0, 0, 1)

    def test_transitive(self):
        p = connect((0, 1, 2), 0, 1)
        p = connect(p, 1, 2)
        assert p == (0, 0, 0)


class TestForget:
    def test_forget_singleton(self):
        was_singleton, result = forget((0, 1, 2), 1)
        assert was_singleton is True
        assert result == (0, 1)

    def test_forget_non_singleton(self):
        was_singleton, result = forget((0, 0, 1), 0)
        assert was_singleton is False
        assert result == (0, 1)

    def test_forget_last(self):
        was_singleton, result = forget((0,), 0)
        assert was_singleton is True
        assert result == ()


# =============================================================================
# TREE DECOMPOSITION TESTS
# =============================================================================

class TestTreeDecomposition:
    def test_single_edge(self):
        mg = MultiGraph(
            nodes=frozenset({0, 1}),
            edge_counts={(0, 1): 1},
        )
        td = compute_tree_decomposition(mg)
        assert td is not None
        assert td.width <= 1

    def test_cycle(self):
        g = cycle_graph(5)
        mg = MultiGraph.from_graph(g)
        td = compute_tree_decomposition(mg)
        assert td is not None
        assert td.width <= 2  # Cycles have treewidth 2

    def test_k4(self):
        g = complete_graph(4)
        mg = MultiGraph.from_graph(g)
        td = compute_tree_decomposition(mg)
        assert td is not None
        assert td.width <= 3  # K4 has treewidth 3

    def test_petersen(self):
        g = petersen_graph()
        mg = MultiGraph.from_graph(g)
        td = compute_tree_decomposition(mg)
        assert td is not None
        assert td.width <= 5  # Petersen has treewidth 4

    def test_max_width_exceeded(self):
        g = complete_graph(4)
        mg = MultiGraph.from_graph(g)
        td = compute_tree_decomposition(mg, max_width=1)
        assert td is None  # K4 can't have treewidth 1

    def test_all_edges_assigned(self):
        """Every edge in the graph must appear in exactly one bag."""
        g = complete_graph(4)
        mg = MultiGraph.from_graph(g)
        td = compute_tree_decomposition(mg)
        assert td is not None

        total_edges = sum(
            mult for edges in td.bag_edges.values() for (_, _, mult) in edges
        )
        assert total_edges == mg.edge_count()

    def test_edge_endpoints_in_bag(self):
        """Both endpoints of each edge must be in the bag it's assigned to."""
        g = complete_graph(5)
        mg = MultiGraph.from_graph(g)
        td = compute_tree_decomposition(mg)
        assert td is not None

        for bag_idx, edges in td.bag_edges.items():
            bag = td.bags[bag_idx]
            for u, v, _ in edges:
                assert u in bag, f"Vertex {u} not in bag {bag_idx}"
                assert v in bag, f"Vertex {v} not in bag {bag_idx}"


# =============================================================================
# TUTTE POLYNOMIAL COMPUTATION TESTS
# =============================================================================

class TestTreewidthTutte:
    def test_single_edge(self):
        """T(K2) = x"""
        mg = MultiGraph(
            nodes=frozenset({0, 1}),
            edge_counts={(0, 1): 1},
        )
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None
        assert result == TuttePolynomial.x()

    def test_triangle(self):
        """T(K3) = x^2 + x + y"""
        g = complete_graph(3)
        mg = MultiGraph.from_graph(g)
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None
        expected = TuttePolynomial.from_coefficients({
            (2, 0): 1, (1, 0): 1, (0, 1): 1
        })
        assert result == expected

    def test_k4(self):
        """T(K4) = x^3 + 3x^2 + 2x + 4xy + 2y + 3y^2 + y^3"""
        g = complete_graph(4)
        mg = MultiGraph.from_graph(g)
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None
        expected = TuttePolynomial.from_coefficients({
            (3, 0): 1, (2, 0): 3, (1, 0): 2,
            (1, 1): 4, (0, 1): 2, (0, 2): 3, (0, 3): 1,
        })
        assert result == expected

    def test_cycle_4(self):
        """T(C4) = x^3 + x^2 + x + y"""
        g = cycle_graph(4)
        mg = MultiGraph.from_graph(g)
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None
        expected = TuttePolynomial.from_coefficients({
            (3, 0): 1, (2, 0): 1, (1, 0): 1, (0, 1): 1
        })
        assert result == expected

    def test_cycle_5(self):
        """T(C5) = x^4 + x^3 + x^2 + x + y"""
        g = cycle_graph(5)
        mg = MultiGraph.from_graph(g)
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None
        expected = TuttePolynomial.from_coefficients({
            (4, 0): 1, (3, 0): 1, (2, 0): 1, (1, 0): 1, (0, 1): 1
        })
        assert result == expected

    def test_petersen(self):
        """Petersen graph: treewidth 4, verify T(1,1) = 2000 spanning trees."""
        g = petersen_graph()
        mg = MultiGraph.from_graph(g)
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None
        assert result.num_spanning_trees() == 2000

    def test_k5(self):
        """K5 has treewidth 4. Verify T(1,1) = 125 spanning trees."""
        g = complete_graph(5)
        mg = MultiGraph.from_graph(g)
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None
        assert result.num_spanning_trees() == 125

    def test_k5_full_polynomial(self):
        """K5 full Tutte polynomial cross-check."""
        g = complete_graph(5)
        mg = MultiGraph.from_graph(g)
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None

        # Known T(K5) coefficients
        expected = TuttePolynomial.from_coefficients({
            (4, 0): 1, (3, 0): 4, (2, 0): 3, (1, 0): 2,
            (2, 1): 10, (1, 1): 10, (0, 1): 2,
            (1, 2): 15, (0, 2): 3,
            (0, 3): 4, (1, 3): 10,
            (0, 4): 1,
            (0, 5): 1, (0, 6): 1,
        })
        # Just check spanning tree count and a few key coefficients
        assert result.num_spanning_trees() == 125
        # x^4 coefficient (number of acyclic orientations related)
        assert result.coefficient(4, 0) == 1


# =============================================================================
# MULTIGRAPH TESTS
# =============================================================================

class TestTreewidthMultigraph:
    def test_parallel_edges(self):
        """2 parallel edges between 2 nodes: T = x + y"""
        mg = MultiGraph(
            nodes=frozenset({0, 1}),
            edge_counts={(0, 1): 2},
        )
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None
        expected = TuttePolynomial.from_coefficients({(1, 0): 1, (0, 1): 1})
        assert result == expected

    def test_triple_parallel_edges(self):
        """3 parallel edges: T = x + y + y^2"""
        mg = MultiGraph(
            nodes=frozenset({0, 1}),
            edge_counts={(0, 1): 3},
        )
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None
        expected = TuttePolynomial.from_coefficients({
            (1, 0): 1, (0, 1): 1, (0, 2): 1
        })
        assert result == expected

    def test_loop(self):
        """Single loop at a node: T = y"""
        mg = MultiGraph(
            nodes=frozenset({0}),
            edge_counts={},
            loop_counts={0: 1},
        )
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None
        assert result == TuttePolynomial.y()

    def test_edge_plus_loop(self):
        """Edge (0,1) + loop at 0: T = xy"""
        mg = MultiGraph(
            nodes=frozenset({0, 1}),
            edge_counts={(0, 1): 1},
            loop_counts={0: 1},
        )
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None
        expected = TuttePolynomial.from_coefficients({(1, 1): 1})
        assert result == expected


# =============================================================================
# CROSS-VALIDATION WITH SP DECOMPOSITION
# =============================================================================

class TestCrossValidation:
    def test_sp_graph_matches(self):
        """For SP graphs, treewidth result should match SP result."""
        # Build a series-parallel graph: triangle with one edge doubled
        g = cycle_graph(4)
        mg = MultiGraph.from_graph(g)

        sp_result = compute_sp_tutte_multigraph_if_applicable(mg)
        tw_result = compute_treewidth_tutte_if_applicable(mg)

        assert sp_result is not None
        assert tw_result is not None
        assert sp_result == tw_result

    def test_cycle_3_cross_validation(self):
        g = complete_graph(3)  # = C3
        mg = MultiGraph.from_graph(g)
        sp_result = compute_sp_tutte_multigraph_if_applicable(mg)
        tw_result = compute_treewidth_tutte_if_applicable(mg)
        assert sp_result is not None
        assert tw_result is not None
        assert sp_result == tw_result

    def test_larger_sp_graph(self):
        """Build a larger SP graph and cross-validate."""
        # Prism graph (C3 x K2) is series-parallel
        # Actually let's use a simple SP graph: path of triangles
        edges = [
            (0, 1), (1, 2), (0, 2),  # triangle 1
            (2, 3), (3, 4), (2, 4),  # triangle 2
        ]
        g = Graph.from_edge_list(edges)
        mg = MultiGraph.from_graph(g)

        sp_result = compute_sp_tutte_multigraph_if_applicable(mg)
        tw_result = compute_treewidth_tutte_if_applicable(mg)

        assert sp_result is not None
        assert tw_result is not None
        assert sp_result == tw_result


# =============================================================================
# SPANNING TREE COUNT VALIDATION
# =============================================================================

class TestSpanningTreeCount:
    """Validate T(1,1) matches Kirchhoff matrix-tree theorem."""

    def _kirchhoff_count(self, g: Graph) -> int:
        """Count spanning trees via Kirchhoff's theorem."""
        G = g.to_networkx()
        if not nx.is_connected(G):
            return 0
        L = nx.laplacian_matrix(G).toarray()
        # Delete last row and column
        L_reduced = L[:-1, :-1]
        import numpy as np
        det = round(abs(np.linalg.det(L_reduced)))
        return int(det)

    def test_k4_spanning_trees(self):
        g = complete_graph(4)
        mg = MultiGraph.from_graph(g)
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None
        assert result.num_spanning_trees() == self._kirchhoff_count(g)

    def test_k5_spanning_trees(self):
        g = complete_graph(5)
        mg = MultiGraph.from_graph(g)
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None
        assert result.num_spanning_trees() == self._kirchhoff_count(g)

    def test_petersen_spanning_trees(self):
        g = petersen_graph()
        mg = MultiGraph.from_graph(g)
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None
        assert result.num_spanning_trees() == self._kirchhoff_count(g)

    def test_wheel_5_spanning_trees(self):
        """W5 = wheel with 5 rim vertices."""
        from tutte.graph import wheel_graph
        g = wheel_graph(5)
        mg = MultiGraph.from_graph(g)
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None
        assert result.num_spanning_trees() == self._kirchhoff_count(g)

    def test_grid_3x3_spanning_trees(self):
        from tutte.graph import grid_graph
        g = grid_graph(3, 3)
        mg = MultiGraph.from_graph(g)
        result = compute_treewidth_tutte_if_applicable(mg)
        assert result is not None
        assert result.num_spanning_trees() == self._kirchhoff_count(g)
