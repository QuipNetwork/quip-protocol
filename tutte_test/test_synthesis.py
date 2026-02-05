"""Test Suite for New Synthesis Engine.

This module tests:
1. Polynomial bitstring encoding/decoding
2. Graph operations and conversions
3. Rainbow table lookups
4. Synthesis engine correctness
5. Spanning tree invariant verification
6. Composition formula verification

Run with: python -m pytest tutte_test/test_synthesis.py -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from tutte_test.covering import (compute_fringe, find_disjoint_cover,
                                 find_subgraph_isomorphisms)
from tutte_test.graph import (Graph, MutableGraph, MultiGraph, complete_graph,
                              cut_vertex_join, cycle_graph, disjoint_union,
                              path_graph, star_graph, wheel_graph)
from tutte_test.polynomial import (TuttePolynomial, cycle_polynomial,
                                   decode_varsint, decode_varuint,
                                   encode_varsint, encode_varuint,
                                   path_polynomial)
from tutte_test.rainbow_table import RainbowTable, load_default_table
from tutte_test.synthesis import (SynthesisEngine, SynthesisResult,
                                  compute_from_mutable,
                                  compute_tutte_polynomial, synthesize)
from tutte_test.validation import (count_spanning_trees_kirchhoff,
                                   verify_spanning_trees, verify_with_networkx)

# =============================================================================
# POLYNOMIAL ENCODING TESTS
# =============================================================================

class TestVarintEncoding(unittest.TestCase):
    """Test varint encoding/decoding."""

    def test_encode_decode_small(self):
        """Test encoding/decoding small numbers."""
        for n in range(256):
            encoded = encode_varuint(n)
            decoded, offset = decode_varuint(encoded)
            self.assertEqual(decoded, n)

    def test_encode_decode_large(self):
        """Test encoding/decoding large numbers."""
        test_values = [127, 128, 255, 256, 16383, 16384, 2**20, 2**28]
        for n in test_values:
            encoded = encode_varuint(n)
            decoded, offset = decode_varuint(encoded)
            self.assertEqual(decoded, n)

    def test_signed_varint(self):
        """Test signed varint encoding."""
        test_values = [-100, -1, 0, 1, 100, -1000, 1000]
        for n in test_values:
            encoded = encode_varsint(n)
            decoded, offset = decode_varsint(encoded)
            self.assertEqual(decoded, n)


class TestPolynomialEncoding(unittest.TestCase):
    """Test polynomial bitstring encoding/decoding."""

    def test_roundtrip_dense(self):
        """Test roundtrip for dense polynomial."""
        coeffs = {(2, 0): 1, (1, 0): 1, (0, 1): 1}  # K_3
        poly = TuttePolynomial.from_coefficients(coeffs)
        restored = TuttePolynomial.from_bytes(poly.to_bytes())
        self.assertEqual(poly, restored)

    def test_roundtrip_sparse(self):
        """Test roundtrip for sparse polynomial."""
        coeffs = {(10, 0): 1, (0, 10): 1}  # Very sparse
        poly = TuttePolynomial.from_coefficients(coeffs)
        restored = TuttePolynomial.from_bytes(poly.to_bytes())
        self.assertEqual(poly, restored)

    def test_roundtrip_k4(self):
        """Test roundtrip for K_4 polynomial."""
        coeffs = {
            (3, 0): 1, (2, 0): 3, (1, 1): 4, (1, 0): 2,
            (0, 1): 2, (0, 2): 3, (0, 3): 1
        }
        poly = TuttePolynomial.from_coefficients(coeffs)
        restored = TuttePolynomial.from_bytes(poly.to_bytes())
        self.assertEqual(poly, restored)

    def test_zero_polynomial(self):
        """Test zero polynomial encoding."""
        poly = TuttePolynomial.zero()
        restored = TuttePolynomial.from_bytes(poly.to_bytes())
        self.assertEqual(poly, restored)
        self.assertTrue(poly.is_zero())

    def test_one_polynomial(self):
        """Test constant 1 polynomial."""
        poly = TuttePolynomial.one()
        self.assertEqual(poly.evaluate(5, 7), 1)
        self.assertEqual(poly.num_spanning_trees(), 1)


class TestPolynomialArithmetic(unittest.TestCase):
    """Test polynomial arithmetic operations."""

    def test_addition(self):
        """Test polynomial addition."""
        p1 = TuttePolynomial.x(2)  # x^2
        p2 = TuttePolynomial.x()   # x
        p3 = TuttePolynomial.y()   # y

        result = p1 + p2 + p3
        expected = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
        self.assertEqual(result, expected)

    def test_multiplication(self):
        """Test polynomial multiplication."""
        p1 = TuttePolynomial.x()  # x
        p2 = TuttePolynomial.y()  # y

        result = p1 * p2
        expected = TuttePolynomial.from_coefficients({(1, 1): 1})  # xy
        self.assertEqual(result, expected)

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        p = TuttePolynomial.x()
        result = 3 * p
        expected = TuttePolynomial.from_coefficients({(1, 0): 3})
        self.assertEqual(result, expected)

    def test_subtraction(self):
        """Test polynomial subtraction."""
        p1 = TuttePolynomial.from_coefficients({(2, 0): 3, (1, 0): 2})
        p2 = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1})

        result = p1 - p2
        expected = TuttePolynomial.from_coefficients({(2, 0): 2, (1, 0): 1})
        self.assertEqual(result, expected)


# =============================================================================
# GRAPH TESTS
# =============================================================================

class TestGraph(unittest.TestCase):
    """Test Graph class operations."""

    def test_complete_graph(self):
        """Test complete graph construction."""
        k4 = complete_graph(4)
        self.assertEqual(k4.node_count(), 4)
        self.assertEqual(k4.edge_count(), 6)  # C(4,2) = 6

    def test_cycle_graph(self):
        """Test cycle graph construction."""
        c5 = cycle_graph(5)
        self.assertEqual(c5.node_count(), 5)
        self.assertEqual(c5.edge_count(), 5)

    def test_path_graph(self):
        """Test path graph construction."""
        p4 = path_graph(4)
        self.assertEqual(p4.node_count(), 4)
        self.assertEqual(p4.edge_count(), 3)

    def test_canonical_key(self):
        """Test canonical key generation."""
        k3 = complete_graph(3)
        key = k3.canonical_key()
        self.assertEqual(len(key), 64)  # SHA256 hex

    def test_is_connected(self):
        """Test connectivity check."""
        k3 = complete_graph(3)
        self.assertTrue(k3.is_connected())

        # Create disconnected graph
        g1 = complete_graph(2)
        g2 = complete_graph(2)
        union = disjoint_union(g1, g2)
        self.assertFalse(union.is_connected())

    def test_disjoint_union(self):
        """Test disjoint union operation."""
        k2_1 = complete_graph(2)
        k2_2 = complete_graph(2)

        union = disjoint_union(k2_1, k2_2)
        self.assertEqual(union.node_count(), 4)
        self.assertEqual(union.edge_count(), 2)

    def test_cut_vertex_join(self):
        """Test cut vertex join operation."""
        k3_1 = complete_graph(3)
        k3_2 = complete_graph(3)

        joined = cut_vertex_join(k3_1, 0, k3_2, 0)
        self.assertEqual(joined.node_count(), 5)  # 3 + 3 - 1
        self.assertEqual(joined.edge_count(), 6)  # 3 + 3


class TestMutableGraph(unittest.TestCase):
    """Test MutableGraph class operations."""

    def test_add_nodes_edges(self):
        """Test adding nodes and edges."""
        g = MutableGraph()
        n1 = g.add_node()
        n2 = g.add_node()
        e = g.add_edge(n1, n2)

        self.assertEqual(g.num_nodes(), 2)
        self.assertEqual(g.num_edges(), 1)

    def test_freeze(self):
        """Test freezing to immutable Graph."""
        g = MutableGraph()
        nodes = g.add_nodes(3)
        g.add_edge(nodes[0], nodes[1])
        g.add_edge(nodes[1], nodes[2])

        frozen = g.freeze()
        self.assertEqual(frozen.node_count(), 3)
        self.assertEqual(frozen.edge_count(), 2)

    def test_contract_edge(self):
        """Test edge contraction."""
        g = MutableGraph()
        nodes = g.add_nodes(3)
        g.add_edge(nodes[0], nodes[1])
        g.add_edge(nodes[1], nodes[2])

        # Contract edge between nodes[0] and nodes[1]
        edge_id = 0
        g.contract_edge(edge_id)

        self.assertEqual(g.num_nodes(), 2)


# =============================================================================
# RAINBOW TABLE TESTS
# =============================================================================

class TestRainbowTable(unittest.TestCase):
    """Test rainbow table operations."""

    @classmethod
    def setUpClass(cls):
        """Load rainbow table once."""
        cls.table = load_default_table()

    def test_load_table(self):
        """Test table loads successfully."""
        self.assertGreater(len(self.table), 0)

    def test_lookup_k3(self):
        """Test looking up K_3 polynomial."""
        k3_poly = self.table.lookup_by_name('K_3')
        self.assertIsNotNone(k3_poly)

        expected = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
        self.assertEqual(k3_poly, expected)

    def test_lookup_by_graph(self):
        """Test looking up by graph canonical key."""
        k3 = complete_graph(3)
        poly = self.table.lookup(k3)
        self.assertIsNotNone(poly)

    def test_get_entry(self):
        """Test getting full entry."""
        entry = self.table.get_entry('K_4')
        self.assertIsNotNone(entry)
        self.assertEqual(entry.node_count, 4)
        self.assertEqual(entry.edge_count, 6)
        self.assertEqual(entry.spanning_trees, 16)

    def test_find_minors(self):
        """Test finding minors of a graph."""
        k4 = complete_graph(4)
        minors = self.table.find_minors_of(k4)

        # Should find K_3, K_2, etc.
        minor_names = {m.name for m in minors}
        self.assertIn('K_3', minor_names)
        self.assertIn('K_2', minor_names)


# =============================================================================
# SYNTHESIS TESTS
# =============================================================================

class TestSynthesis(unittest.TestCase):
    """Test synthesis engine."""

    def setUp(self):
        """Set up synthesis engine."""
        self.engine = SynthesisEngine(verbose=False)

    def test_synthesize_k2(self):
        """Test synthesizing K_2."""
        k2 = complete_graph(2)
        result = self.engine.synthesize(k2)

        self.assertEqual(result.polynomial, TuttePolynomial.x())
        self.assertTrue(result.verified)

    def test_synthesize_k3(self):
        """Test synthesizing K_3."""
        k3 = complete_graph(3)
        result = self.engine.synthesize(k3)

        expected = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
        self.assertEqual(result.polynomial, expected)

    def test_synthesize_k4(self):
        """Test synthesizing K_4."""
        k4 = complete_graph(4)
        result = self.engine.synthesize(k4)

        # Verify spanning trees
        self.assertEqual(result.polynomial.num_spanning_trees(), 16)

    def test_synthesize_path(self):
        """Test synthesizing path graph."""
        p4 = path_graph(4)
        result = self.engine.synthesize(p4)

        # P_4 has 3 edges, all bridges -> T = x^3
        expected = TuttePolynomial.x(3)
        self.assertEqual(result.polynomial, expected)

    def test_synthesize_cycle(self):
        """Test synthesizing cycle graph."""
        c5 = cycle_graph(5)
        result = self.engine.synthesize(c5)

        # Verify spanning trees (C_5 has 5 spanning trees)
        self.assertEqual(result.polynomial.num_spanning_trees(), 5)

    def test_synthesize_disjoint_union(self):
        """Test synthesizing disjoint union."""
        k3_1 = complete_graph(3)
        k3_2 = complete_graph(3)
        union = disjoint_union(k3_1, k3_2)

        result = self.engine.synthesize(union)

        # T(K_3 ∪ K_3) = T(K_3) × T(K_3)
        k3_poly = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
        expected = k3_poly * k3_poly

        self.assertEqual(result.polynomial, expected)


class TestSpanningTreeVerification(unittest.TestCase):
    """Verify spanning tree counts match Kirchhoff."""

    def test_k3_spanning_trees(self):
        """K_3 has 3 spanning trees."""
        k3 = complete_graph(3)
        poly = compute_tutte_polynomial(k3)
        self.assertEqual(poly.num_spanning_trees(), 3)
        self.assertTrue(verify_spanning_trees(k3, poly))

    def test_k4_spanning_trees(self):
        """K_4 has 16 spanning trees."""
        k4 = complete_graph(4)
        poly = compute_tutte_polynomial(k4)
        self.assertEqual(poly.num_spanning_trees(), 16)
        self.assertTrue(verify_spanning_trees(k4, poly))

    def test_c5_spanning_trees(self):
        """C_5 has 5 spanning trees."""
        c5 = cycle_graph(5)
        poly = compute_tutte_polynomial(c5)
        self.assertEqual(poly.num_spanning_trees(), 5)
        self.assertTrue(verify_spanning_trees(c5, poly))

    def test_kirchhoff_comparison(self):
        """Compare with Kirchhoff's theorem for various graphs."""
        test_graphs = [
            ("K_3", complete_graph(3), 3),
            ("K_4", complete_graph(4), 16),
            ("C_4", cycle_graph(4), 4),
            ("C_6", cycle_graph(6), 6),
            ("P_5", path_graph(5), 1),  # Paths have 1 spanning tree
        ]

        for name, graph, expected in test_graphs:
            with self.subTest(graph=name):
                poly = compute_tutte_polynomial(graph)
                tutte_trees = poly.num_spanning_trees()
                kirchhoff_trees = count_spanning_trees_kirchhoff(graph)

                self.assertEqual(tutte_trees, expected, f"{name}: Tutte mismatch")
                self.assertEqual(tutte_trees, kirchhoff_trees, f"{name}: Tutte != Kirchhoff")


class TestCompositionFormulas(unittest.TestCase):
    """Test that composition formulas hold."""

    def test_disjoint_union_formula(self):
        """T(G₁ ∪ G₂) = T(G₁) × T(G₂)"""
        g1 = complete_graph(3)
        g2 = cycle_graph(4)

        t1 = compute_tutte_polynomial(g1)
        t2 = compute_tutte_polynomial(g2)

        union = disjoint_union(g1, g2)
        t_union = compute_tutte_polynomial(union)

        expected = t1 * t2
        self.assertEqual(t_union, expected)

    def test_cut_vertex_formula(self):
        """T(G₁ ·₁ G₂) = T(G₁) × T(G₂) for cut vertex join."""
        g1 = complete_graph(3)
        g2 = complete_graph(3)

        t1 = compute_tutte_polynomial(g1)
        t2 = compute_tutte_polynomial(g2)

        joined = cut_vertex_join(g1, 0, g2, 0)
        t_joined = compute_tutte_polynomial(joined)

        expected = t1 * t2
        self.assertEqual(t_joined, expected)


# =============================================================================
# COVERING TESTS
# =============================================================================

class TestCovering(unittest.TestCase):
    """Test covering algorithms."""

    def test_find_subgraph_isomorphisms(self):
        """Test finding subgraph isomorphisms."""
        k4 = complete_graph(4)
        k3 = complete_graph(3)

        # K_4 contains 4 induced K_3 subgraphs, each with 3! = 6 automorphisms
        # VF2 returns all isomorphisms, so 4 * 6 = 24 total
        matches = find_subgraph_isomorphisms(k4, k3)
        self.assertEqual(len(matches), 24)

    def test_disjoint_cover(self):
        """Test finding disjoint cover."""
        table = load_default_table()
        minor = table.get_entry('K_3')

        if minor is None:
            self.skipTest("K_3 not in table")

        # K_6 should be coverable by 2 disjoint K_3
        k6 = complete_graph(6)
        cover = find_disjoint_cover(k6, minor, table)

        # Should find at least 1 tile
        self.assertGreater(len(cover.tiles), 0)


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidation(unittest.TestCase):
    """Test validation utilities."""

    def test_verify_with_networkx(self):
        """Test verification against NetworkX."""
        k3 = complete_graph(3)
        poly = compute_tutte_polynomial(k3)

        # Should verify successfully (or skip if sympy unavailable)
        result = verify_with_networkx(k3, poly)
        self.assertTrue(result)

    def test_verify_spanning_trees(self):
        """Test spanning tree verification."""
        k4 = complete_graph(4)
        poly = compute_tutte_polynomial(k4)

        self.assertTrue(verify_spanning_trees(k4, poly))


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests with old codebase."""

    def test_compatible_with_old_tutte_utils(self):
        """Test that results match old tutte_utils implementation."""
        try:
            from tutte_test.tutte_utils import \
                TuttePolynomial as OldTuttePolynomial
            from tutte_test.tutte_utils import \
                compute_tutte_polynomial as old_compute
            from tutte_test.tutte_utils import \
                create_complete_graph as old_complete_graph
        except ImportError:
            self.skipTest("Old tutte_utils not available")

        # Compare K_4 polynomials
        old_k4 = old_complete_graph(4)
        old_poly = old_compute(old_k4)

        new_k4 = complete_graph(4)
        new_poly = compute_tutte_polynomial(new_k4)

        # Compare spanning trees
        self.assertEqual(old_poly.num_spanning_trees(), new_poly.num_spanning_trees())

        # Compare coefficients
        old_coeffs = old_poly.coefficients
        new_coeffs = new_poly.to_coefficients()
        self.assertEqual(old_coeffs, new_coeffs)


# =============================================================================
# MULTIGRAPH TESTS
# =============================================================================

class TestMultiGraph(unittest.TestCase):
    """Test MultiGraph class operations."""

    def test_from_graph(self):
        """Test converting simple Graph to MultiGraph."""
        k3 = complete_graph(3)
        mg = MultiGraph.from_graph(k3)
        self.assertEqual(mg.node_count(), 3)
        self.assertEqual(mg.edge_count(), 3)
        self.assertTrue(mg.is_simple())

    def test_merge_nodes_creates_parallel(self):
        """Test that merging nodes creates parallel edges."""
        k3 = complete_graph(3)
        merged = k3.merge_nodes(0, 1)
        # Should have 2 parallel edges (0-2 and 1-2 become both 0-2)
        self.assertEqual(merged.edge_multiplicity(0, 2), 2)
        # Should have 1 loop (from edge 0-1)
        self.assertEqual(merged.loop_counts.get(0, 0), 1)
        self.assertFalse(merged.is_simple())

    def test_is_just_parallel_edges(self):
        """Test detection of simple parallel edge graphs."""
        mg_parallel = MultiGraph(
            nodes=frozenset([0, 1]),
            edge_counts={(0, 1): 3},
            loop_counts={}
        )
        self.assertTrue(mg_parallel.is_just_parallel_edges())
        self.assertEqual(mg_parallel.parallel_edge_count(), 3)

        # With a loop, not just parallel edges
        mg_with_loop = MultiGraph(
            nodes=frozenset([0, 1]),
            edge_counts={(0, 1): 2},
            loop_counts={0: 1}
        )
        self.assertFalse(mg_with_loop.is_just_parallel_edges())

    def test_multigraph_cut_vertex(self):
        """Test cut vertex detection in multigraph."""
        # Create multigraph with cut vertex: triangle connected to another triangle
        # First triangle: 0-1-2, second triangle: 2-3-4, connected at vertex 2
        mg = MultiGraph(
            nodes=frozenset([0, 1, 2, 3, 4]),
            edge_counts={
                (0, 1): 1, (1, 2): 1, (0, 2): 1,  # First triangle
                (2, 3): 1, (3, 4): 1, (2, 4): 1,  # Second triangle
            },
            loop_counts={}
        )
        cut = mg.has_cut_vertex()
        # Vertex 2 is the cut vertex connecting the two triangles
        self.assertIsNotNone(cut)
        self.assertEqual(cut, 2)

    def test_multigraph_split(self):
        """Test splitting multigraph at cut vertex."""
        # Two triangles joined at vertex 2
        mg = MultiGraph(
            nodes=frozenset([0, 1, 2, 3, 4]),
            edge_counts={
                (0, 1): 1, (1, 2): 1, (0, 2): 1,  # First triangle
                (2, 3): 1, (3, 4): 1, (2, 4): 1,  # Second triangle
            },
            loop_counts={}
        )
        parts = mg.split_at_cut_vertex(2)
        self.assertEqual(len(parts), 2)
        # Each part should contain vertex 2
        for part in parts:
            self.assertIn(2, part.nodes)
        # Each part should be a triangle (3 nodes, 3 edges)
        for part in parts:
            self.assertEqual(part.node_count(), 3)
            self.assertEqual(part.edge_count(), 3)


class TestGraphCutVertex(unittest.TestCase):
    """Test cut vertex operations on Graph."""

    def test_has_cut_vertex_bowtie(self):
        """Test cut vertex detection on bowtie (two triangles sharing vertex)."""
        k3a = complete_graph(3)
        k3b = complete_graph(3)
        bowtie = cut_vertex_join(k3a, 0, k3b, 0)
        cut = bowtie.has_cut_vertex()
        self.assertIsNotNone(cut)
        self.assertEqual(cut, 0)

    def test_has_cut_vertex_cycle(self):
        """Cycles have no cut vertices."""
        c5 = cycle_graph(5)
        self.assertIsNone(c5.has_cut_vertex())

    def test_has_cut_vertex_complete(self):
        """Complete graphs with n >= 3 have no cut vertices."""
        k4 = complete_graph(4)
        self.assertIsNone(k4.has_cut_vertex())

    def test_split_at_cut_vertex(self):
        """Test splitting bowtie at cut vertex."""
        k3a = complete_graph(3)
        k3b = complete_graph(3)
        bowtie = cut_vertex_join(k3a, 0, k3b, 0)
        parts = bowtie.split_at_cut_vertex(0)
        self.assertEqual(len(parts), 2)
        for part in parts:
            self.assertEqual(part.node_count(), 3)
            self.assertEqual(part.edge_count(), 3)


class TestParallelEdges(unittest.TestCase):
    """Test parallel edge polynomial formulas."""

    def setUp(self):
        """Set up synthesis engine."""
        self.engine = SynthesisEngine(verbose=False)

    def test_single_edge(self):
        """T(1 edge) = x"""
        poly = self.engine._parallel_edges_formula(1)
        self.assertEqual(poly, TuttePolynomial.x())

    def test_two_parallel(self):
        """T(2 parallel edges) = x + y"""
        poly = self.engine._parallel_edges_formula(2)
        expected = TuttePolynomial.from_coefficients({(1, 0): 1, (0, 1): 1})
        self.assertEqual(poly, expected)
        self.assertEqual(poly.num_spanning_trees(), 2)

    def test_three_parallel(self):
        """T(3 parallel edges) = x + y + y^2"""
        poly = self.engine._parallel_edges_formula(3)
        expected = TuttePolynomial.from_coefficients({(1, 0): 1, (0, 1): 1, (0, 2): 1})
        self.assertEqual(poly, expected)
        self.assertEqual(poly.num_spanning_trees(), 3)

    def test_four_parallel(self):
        """T(4 parallel edges) = x + y + y^2 + y^3"""
        poly = self.engine._parallel_edges_formula(4)
        expected = TuttePolynomial.from_coefficients({
            (1, 0): 1, (0, 1): 1, (0, 2): 1, (0, 3): 1
        })
        self.assertEqual(poly, expected)
        self.assertEqual(poly.num_spanning_trees(), 4)

    def test_synthesize_parallel_multigraph(self):
        """Test synthesizing multigraph with parallel edges."""
        mg = MultiGraph(
            nodes=frozenset([0, 1]),
            edge_counts={(0, 1): 3},
            loop_counts={}
        )
        poly = self.engine._synthesize_multigraph(mg)
        expected = TuttePolynomial.from_coefficients({(1, 0): 1, (0, 1): 1, (0, 2): 1})
        self.assertEqual(poly, expected)


class TestEdgeAddition(unittest.TestCase):
    """Test edge addition formula: T(G + e) = T(G) + T(G/{u,v})"""

    def setUp(self):
        """Set up synthesis engine."""
        self.engine = SynthesisEngine(verbose=False)

    def test_p3_to_c3(self):
        """P_3 + edge(0,2) = C_3"""
        p3 = path_graph(3)
        p3_poly = self.engine.synthesize(p3).polynomial

        # Add edge using formula
        c3_via_addition = self.engine._add_edges_to_graph(p3, p3_poly, [(0, 2)])

        # Compute C_3 directly
        c3 = cycle_graph(3)
        c3_direct = self.engine.synthesize(c3).polynomial

        self.assertEqual(c3_via_addition, c3_direct)

    def test_c4_to_k4_minus_edge(self):
        """C_4 + diagonal = K_4 minus one edge"""
        c4 = cycle_graph(4)
        c4_poly = self.engine.synthesize(c4).polynomial

        # Add one diagonal
        result = self.engine._add_edges_to_graph(c4, c4_poly, [(0, 2)])

        # K_4 minus edge has 8 spanning trees
        self.assertEqual(result.num_spanning_trees(), 8)

    def test_multiple_edge_addition(self):
        """Add multiple edges sequentially."""
        p4 = path_graph(4)  # 0-1-2-3
        p4_poly = self.engine.synthesize(p4).polynomial

        # Add edges to make it a cycle
        result = self.engine._add_edges_to_graph(p4, p4_poly, [(0, 3)])

        # Should equal C_4
        c4 = cycle_graph(4)
        c4_poly = self.engine.synthesize(c4).polynomial

        self.assertEqual(result, c4_poly)


class TestLoopHandling(unittest.TestCase):
    """Test loop handling in multigraph synthesis."""

    def setUp(self):
        """Set up synthesis engine."""
        self.engine = SynthesisEngine(verbose=False)

    def test_single_loop(self):
        """T(single loop) = y"""
        mg = MultiGraph(
            nodes=frozenset([0]),
            edge_counts={},
            loop_counts={0: 1}
        )
        poly = self.engine._synthesize_multigraph(mg)
        self.assertEqual(poly, TuttePolynomial.y())

    def test_multiple_loops(self):
        """T(k loops) = y^k"""
        mg = MultiGraph(
            nodes=frozenset([0]),
            edge_counts={},
            loop_counts={0: 3}
        )
        poly = self.engine._synthesize_multigraph(mg)
        self.assertEqual(poly, TuttePolynomial.y(3))

    def test_edge_plus_loop(self):
        """T(edge + loop) = xy"""
        mg = MultiGraph(
            nodes=frozenset([0, 1]),
            edge_counts={(0, 1): 1},
            loop_counts={0: 1}
        )
        poly = self.engine._synthesize_multigraph(mg)
        expected = TuttePolynomial.from_coefficients({(1, 1): 1})  # xy
        self.assertEqual(poly, expected)


class TestCutVertexFactorization(unittest.TestCase):
    """Test cut vertex factorization: T(G1 · G2) = T(G1) × T(G2)."""

    def setUp(self):
        """Set up synthesis engine."""
        self.engine = SynthesisEngine(verbose=False)

    def test_bowtie_factorization(self):
        """Bowtie = K3 · K3 at cut vertex, so T = T(K3)^2."""
        k3a = complete_graph(3)
        k3b = complete_graph(3)
        bowtie = cut_vertex_join(k3a, 0, k3b, 0)

        # Synthesize bowtie
        bowtie_poly = self.engine.synthesize(bowtie).polynomial

        # Compute expected: T(K3)^2
        k3_poly = self.engine.synthesize(k3a).polynomial
        expected = k3_poly * k3_poly

        self.assertEqual(bowtie_poly, expected)

    def test_multigraph_cut_vertex_factorization(self):
        """Test factorization for multigraph with cut vertex."""
        # Create a multigraph: two pairs of parallel edges connected at vertex 2
        mg = MultiGraph(
            nodes=frozenset([0, 1, 2, 3]),
            edge_counts={
                (0, 1): 2,  # 2 parallel edges
                (0, 2): 1,  # Connect to cut vertex
                (2, 3): 1,  # Connect from cut vertex
            },
            loop_counts={}
        )
        # This should factor because vertex 2 is a cut vertex... but actually
        # it's not because 0 connects to both 1 and 2. Let me create a proper one.

        # Proper cut vertex graph
        mg2 = MultiGraph(
            nodes=frozenset([0, 1, 2, 3, 4]),
            edge_counts={
                (0, 2): 1, (1, 2): 1,  # First component: 0-2-1 with cut at 2
                (2, 3): 1, (2, 4): 1,  # Second component: 3-2-4 with cut at 2
            },
            loop_counts={}
        )
        poly = self.engine._synthesize_multigraph(mg2)
        # Both components are P3 with center removed, which are 2 K2s each = x^2 each
        # So result should be x^4
        self.assertEqual(poly.num_spanning_trees(), 1)  # Tree, so 1 spanning tree


# =============================================================================
# KNOWN POLYNOMIAL TESTS
# =============================================================================

class TestKnownPolynomials(unittest.TestCase):
    """Test known polynomial formulas."""

    def test_cycle_polynomial_formula(self):
        """Test cycle polynomial formula: x^{n-1} + ... + x + y"""
        for n in [3, 4, 5, 6]:
            with self.subTest(n=n):
                poly = cycle_polynomial(n)

                # Should have n terms: x^{n-1}, x^{n-2}, ..., x, y
                self.assertEqual(poly.num_terms(), n)

                # Check spanning trees (C_n has n spanning trees)
                self.assertEqual(poly.num_spanning_trees(), n)

    def test_path_polynomial_formula(self):
        """Test path polynomial formula: x^{n-1}"""
        for n in [2, 3, 4, 5]:
            with self.subTest(n=n):
                poly = path_polynomial(n)

                # Should be x^{n-1}
                self.assertEqual(poly, TuttePolynomial.x(n - 1))

                # Paths have 1 spanning tree
                self.assertEqual(poly.num_spanning_trees(), 1)


# =============================================================================
# RUN TESTS
# =============================================================================

def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestVarintEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestPolynomialEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestPolynomialArithmetic))
    suite.addTests(loader.loadTestsFromTestCase(TestGraph))
    suite.addTests(loader.loadTestsFromTestCase(TestMutableGraph))
    suite.addTests(loader.loadTestsFromTestCase(TestRainbowTable))
    suite.addTests(loader.loadTestsFromTestCase(TestSynthesis))
    suite.addTests(loader.loadTestsFromTestCase(TestSpanningTreeVerification))
    suite.addTests(loader.loadTestsFromTestCase(TestCompositionFormulas))
    suite.addTests(loader.loadTestsFromTestCase(TestCovering))
    suite.addTests(loader.loadTestsFromTestCase(TestValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiGraph))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphCutVertex))
    suite.addTests(loader.loadTestsFromTestCase(TestParallelEdges))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeAddition))
    suite.addTests(loader.loadTestsFromTestCase(TestLoopHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestCutVertexFactorization))
    suite.addTests(loader.loadTestsFromTestCase(TestKnownPolynomials))

    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    run_tests()
