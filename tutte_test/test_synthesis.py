"""Consolidated Test Suite for Tutte Polynomial Synthesis.

This module tests:
1. Polynomial bitstring encoding/decoding
2. Graph operations and conversions
3. Rainbow table lookups and binary encoding
4. Synthesis engine correctness (tiling, hybrid, algebraic)
5. Spanning tree invariant verification (Tutte vs Kirchhoff)
6. Composition formula verification
7. Zephyr Z(m,t) synthesis with basic primitives
8. Z(1,1) synthesis with empty rainbow table

Run with: python -m pytest tutte_test/test_synthesis.py -v
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from tutte_test.covering import (compute_fringe, find_disjoint_cover,
                                 find_subgraph_isomorphisms)
from tutte_test.graph import (Graph, MultiGraph, complete_graph,
                              cut_vertex_join, cycle_graph, disjoint_union,
                              path_graph, star_graph, wheel_graph)
from tutte_test.polynomial import (TuttePolynomial, cycle_polynomial,
                                   decode_varsint, decode_varuint,
                                   encode_varsint, encode_varuint,
                                   path_polynomial)
from tutte_test.rainbow_table import (RainbowTable, load_default_table,
                                      build_basic_table,
                                      encode_rainbow_table_binary,
                                      create_minimal_json,
                                      analyze_binary_breakdown)
from tutte_test.synthesis import (SynthesisEngine, SynthesisResult,
                                  compute_tutte_polynomial, synthesize)
from tutte_test.validation import (count_spanning_trees_kirchhoff,
                                   verify_spanning_trees, verify_with_networkx)


# =============================================================================
# HELPERS
# =============================================================================

def get_zephyr_graph(m: int, t: int) -> nx.Graph:
    """Get Zephyr graph Z(m,t) using dwave_networkx."""
    try:
        import dwave_networkx as dnx
        return dnx.zephyr_graph(m, t)
    except ImportError:
        raise ImportError(
            "dwave_networkx required for Zephyr graphs. "
            "Install with: pip install dwave-networkx"
        )


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

    def test_k5_networkx_verification(self):
        """K_5 polynomial matches NetworkX."""
        k5 = complete_graph(5)
        poly = compute_tutte_polynomial(k5)
        self.assertTrue(verify_with_networkx(k5, poly))

    def test_c5_networkx_verification(self):
        """C_5 polynomial matches NetworkX."""
        c5 = cycle_graph(5)
        poly = compute_tutte_polynomial(c5)
        self.assertTrue(verify_with_networkx(c5, poly))

    def test_petersen_networkx_verification(self):
        """Petersen graph polynomial matches NetworkX."""
        g_pet = Graph.from_networkx(nx.petersen_graph())
        poly = compute_tutte_polynomial(g_pet)
        self.assertTrue(verify_with_networkx(g_pet, poly))

    def test_k5_rainbow_table_consistency(self):
        """K_5 rainbow table entry matches computed polynomial."""
        entry = self.table.get_entry('K_5')
        if entry is None:
            self.skipTest("K_5 not in rainbow table")
        k5 = complete_graph(5)
        computed = compute_tutte_polynomial(k5)
        self.assertEqual(computed.num_spanning_trees(), entry.spanning_trees)

    def test_k6_rainbow_table_consistency(self):
        """K_6 rainbow table entry matches Kirchhoff."""
        entry = self.table.get_entry('K_6')
        if entry is None:
            self.skipTest("K_6 not in rainbow table")
        k6 = complete_graph(6)
        kirchhoff = count_spanning_trees_kirchhoff(k6)
        self.assertEqual(entry.spanning_trees, kirchhoff)

    def test_petersen_rainbow_table_consistency(self):
        """Petersen rainbow table entry matches Kirchhoff."""
        entry = self.table.get_entry('Petersen')
        if entry is None:
            self.skipTest("Petersen not in rainbow table")
        g_pet = Graph.from_networkx(nx.petersen_graph())
        kirchhoff = count_spanning_trees_kirchhoff(g_pet)
        self.assertEqual(entry.spanning_trees, kirchhoff)

    def test_wheel_rainbow_table_consistency(self):
        """Wheel graph rainbow table entries match Kirchhoff."""
        for n in [5, 6]:
            with self.subTest(n=n):
                entry = self.table.get_entry(f'W_{n}')
                if entry is None:
                    self.skipTest(f"W_{n} not in rainbow table")
                g = Graph.from_networkx(nx.wheel_graph(n))
                kirchhoff = count_spanning_trees_kirchhoff(g)
                self.assertEqual(entry.spanning_trees, kirchhoff)


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

    def test_k5_two_algorithm_check(self):
        """K_5 spanning trees via Tutte and Kirchhoff agree."""
        k5 = complete_graph(5)
        poly = compute_tutte_polynomial(k5)
        self.assertEqual(poly.num_spanning_trees(), 125)
        self.assertTrue(verify_spanning_trees(k5, poly))

    def test_k33_two_algorithm_check(self):
        """K_3,3 spanning trees via Tutte and Kirchhoff agree."""
        G_nx = nx.complete_bipartite_graph(3, 3)
        g = Graph.from_networkx(G_nx)
        poly = compute_tutte_polynomial(g)
        kirchhoff = count_spanning_trees_kirchhoff(g)
        self.assertEqual(poly.num_spanning_trees(), kirchhoff)
        self.assertEqual(kirchhoff, 81)


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

    def test_disjoint_union_k3_k4(self):
        """T(K_3 ∪ K_4) = T(K_3) × T(K_4)."""
        g1 = complete_graph(3)
        g2 = complete_graph(4)

        t1 = compute_tutte_polynomial(g1)
        t2 = compute_tutte_polynomial(g2)

        union = disjoint_union(g1, g2)
        t_union = compute_tutte_polynomial(union)

        expected = t1 * t2
        self.assertEqual(t_union, expected)

    def test_cut_vertex_join_cycles(self):
        """T(C_4 · C_5) = T(C_4) × T(C_5) for cut vertex join."""
        g1 = cycle_graph(4)
        g2 = cycle_graph(5)

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

        matches = find_subgraph_isomorphisms(k4, k3)
        self.assertEqual(len(matches), 24)

    def test_disjoint_cover(self):
        """Test finding disjoint cover."""
        table = load_default_table()
        minor = table.get_entry('K_3')

        if minor is None:
            self.skipTest("K_3 not in table")

        k6 = complete_graph(6)
        cover = find_disjoint_cover(k6, minor, table)

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

        result = verify_with_networkx(k3, poly)
        self.assertTrue(result)

    def test_verify_spanning_trees(self):
        """Test spanning tree verification."""
        k4 = complete_graph(4)
        poly = compute_tutte_polynomial(k4)

        self.assertTrue(verify_spanning_trees(k4, poly))


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
        self.assertEqual(merged.edge_multiplicity(0, 2), 2)
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

        mg_with_loop = MultiGraph(
            nodes=frozenset([0, 1]),
            edge_counts={(0, 1): 2},
            loop_counts={0: 1}
        )
        self.assertFalse(mg_with_loop.is_just_parallel_edges())

    def test_multigraph_cut_vertex(self):
        """Test cut vertex detection in multigraph."""
        mg = MultiGraph(
            nodes=frozenset([0, 1, 2, 3, 4]),
            edge_counts={
                (0, 1): 1, (1, 2): 1, (0, 2): 1,
                (2, 3): 1, (3, 4): 1, (2, 4): 1,
            },
            loop_counts={}
        )
        cut = mg.has_cut_vertex()
        self.assertIsNotNone(cut)
        self.assertEqual(cut, 2)

    def test_multigraph_split(self):
        """Test splitting multigraph at cut vertex."""
        mg = MultiGraph(
            nodes=frozenset([0, 1, 2, 3, 4]),
            edge_counts={
                (0, 1): 1, (1, 2): 1, (0, 2): 1,
                (2, 3): 1, (3, 4): 1, (2, 4): 1,
            },
            loop_counts={}
        )
        parts = mg.split_at_cut_vertex(2)
        self.assertEqual(len(parts), 2)
        for part in parts:
            self.assertIn(2, part.nodes)
        for part in parts:
            self.assertEqual(part.node_count(), 3)
            self.assertEqual(part.edge_count(), 3)


class TestGraphCutVertex(unittest.TestCase):
    """Test cut vertex operations on Graph."""

    def test_has_cut_vertex_bowtie(self):
        """Test cut vertex detection on bowtie."""
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
        self.engine = SynthesisEngine(verbose=False)

    def test_p3_to_c3(self):
        """P_3 + edge(0,2) = C_3"""
        p3 = path_graph(3)
        p3_poly = self.engine.synthesize(p3).polynomial
        c3_via_addition = self.engine._add_edges_to_graph(p3, p3_poly, [(0, 2)])
        c3 = cycle_graph(3)
        c3_direct = self.engine.synthesize(c3).polynomial
        self.assertEqual(c3_via_addition, c3_direct)

    def test_c4_to_k4_minus_edge(self):
        """C_4 + diagonal = K_4 minus one edge"""
        c4 = cycle_graph(4)
        c4_poly = self.engine.synthesize(c4).polynomial
        result = self.engine._add_edges_to_graph(c4, c4_poly, [(0, 2)])
        self.assertEqual(result.num_spanning_trees(), 8)

    def test_multiple_edge_addition(self):
        """Add multiple edges sequentially."""
        p4 = path_graph(4)
        p4_poly = self.engine.synthesize(p4).polynomial
        result = self.engine._add_edges_to_graph(p4, p4_poly, [(0, 3)])
        c4 = cycle_graph(4)
        c4_poly = self.engine.synthesize(c4).polynomial
        self.assertEqual(result, c4_poly)


class TestLoopHandling(unittest.TestCase):
    """Test loop handling in multigraph synthesis."""

    def setUp(self):
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
        expected = TuttePolynomial.from_coefficients({(1, 1): 1})
        self.assertEqual(poly, expected)


class TestCutVertexFactorization(unittest.TestCase):
    """Test cut vertex factorization: T(G1 · G2) = T(G1) × T(G2)."""

    def setUp(self):
        self.engine = SynthesisEngine(verbose=False)

    def test_bowtie_factorization(self):
        """Bowtie = K3 · K3 at cut vertex, so T = T(K3)^2."""
        k3a = complete_graph(3)
        k3b = complete_graph(3)
        bowtie = cut_vertex_join(k3a, 0, k3b, 0)
        bowtie_poly = self.engine.synthesize(bowtie).polynomial
        k3_poly = self.engine.synthesize(k3a).polynomial
        expected = k3_poly * k3_poly
        self.assertEqual(bowtie_poly, expected)

    def test_multigraph_cut_vertex_factorization(self):
        """Test factorization for multigraph with cut vertex."""
        mg2 = MultiGraph(
            nodes=frozenset([0, 1, 2, 3, 4]),
            edge_counts={
                (0, 2): 1, (1, 2): 1,
                (2, 3): 1, (2, 4): 1,
            },
            loop_counts={}
        )
        poly = self.engine._synthesize_multigraph(mg2)
        self.assertEqual(poly.num_spanning_trees(), 1)


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
                self.assertEqual(poly.num_terms(), n)
                self.assertEqual(poly.num_spanning_trees(), n)

    def test_path_polynomial_formula(self):
        """Test path polynomial formula: x^{n-1}"""
        for n in [2, 3, 4, 5]:
            with self.subTest(n=n):
                poly = path_polynomial(n)
                self.assertEqual(poly, TuttePolynomial.x(n - 1))
                self.assertEqual(poly.num_spanning_trees(), 1)


# =============================================================================
# BINARY ENCODING TESTS (migrated from benchmark_synthesis)
# =============================================================================

class TestBinaryEncoding(unittest.TestCase):
    """Test binary rainbow table encoding and roundtrip."""

    @classmethod
    def setUpClass(cls):
        cls.table = load_default_table()

    def test_binary_smaller_than_json(self):
        """Binary table is smaller than JSON."""
        table_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table.json')
        if not os.path.exists(table_path):
            self.skipTest("No JSON table to compare")
        json_size = os.path.getsize(table_path)
        binary_data = encode_rainbow_table_binary(self.table)
        self.assertLess(len(binary_data), json_size)

    def test_binary_roundtrip_header(self):
        """Binary encoding has correct magic header."""
        binary_data = encode_rainbow_table_binary(self.table)
        self.assertTrue(binary_data.startswith(b"RTBL"))
        self.assertEqual(binary_data[4], 1)  # version

    def test_minimal_json_smaller_than_full(self):
        """Minimal JSON is smaller than full JSON."""
        table_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table.json')
        if not os.path.exists(table_path):
            self.skipTest("No JSON table to compare")
        json_size = os.path.getsize(table_path)
        minimal = create_minimal_json(self.table)
        minimal_str = json.dumps(minimal, separators=(',', ':'))
        self.assertLess(len(minimal_str), json_size)

    def test_binary_breakdown_sums_correctly(self):
        """Binary breakdown components sum to total."""
        breakdown = analyze_binary_breakdown(self.table)
        expected_total = (breakdown['header'] + breakdown['names'] +
                         breakdown['metadata'] + breakdown['polynomials'])
        self.assertEqual(breakdown['total'], expected_total)


# =============================================================================
# BENCHMARK-AS-TEST: SYNTHESIS CORRECTNESS
# =============================================================================

class TestBenchmarkSynthesis(unittest.TestCase):
    """Run synthesis on graph families up to 22 edges, verify via Kirchhoff."""

    def test_complete_graphs_kirchhoff(self):
        """Complete graphs K_3 through K_6 match Kirchhoff."""
        for n in [3, 4, 5, 6]:
            with self.subTest(graph=f"K_{n}"):
                g = complete_graph(n)
                result = synthesize(g)
                kirchhoff = count_spanning_trees_kirchhoff(g)
                self.assertEqual(result.polynomial.num_spanning_trees(), kirchhoff)

    def test_cycle_graphs_kirchhoff(self):
        """Cycle graphs C_4 through C_10 match Kirchhoff."""
        for n in [4, 5, 6, 8, 10]:
            with self.subTest(graph=f"C_{n}"):
                g = cycle_graph(n)
                result = synthesize(g)
                kirchhoff = count_spanning_trees_kirchhoff(g)
                self.assertEqual(result.polynomial.num_spanning_trees(), kirchhoff)

    def test_wheel_graphs_kirchhoff(self):
        """Wheel graphs W_4 through W_6 match Kirchhoff."""
        for n in [4, 5, 6]:
            with self.subTest(graph=f"W_{n}"):
                g = Graph.from_networkx(nx.wheel_graph(n))
                result = synthesize(g)
                kirchhoff = count_spanning_trees_kirchhoff(g)
                self.assertEqual(result.polynomial.num_spanning_trees(), kirchhoff)

    def test_petersen_kirchhoff(self):
        """Petersen graph matches Kirchhoff (15 edges)."""
        g = Graph.from_networkx(nx.petersen_graph())
        result = synthesize(g)
        kirchhoff = count_spanning_trees_kirchhoff(g)
        self.assertEqual(result.polynomial.num_spanning_trees(), kirchhoff)

    def test_grid_3x3_kirchhoff(self):
        """Grid 3x3 matches Kirchhoff (12 edges)."""
        G_nx = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))
        g = Graph.from_networkx(G_nx)
        result = synthesize(g)
        kirchhoff = count_spanning_trees_kirchhoff(g)
        self.assertEqual(result.polynomial.num_spanning_trees(), kirchhoff)


# =============================================================================
# Z(1,1) EMPTY TABLE TEST
# =============================================================================

class TestZ11EmptyTable(unittest.TestCase):
    """Test Z(1,1) synthesis with empty rainbow table (no pre-cached entries).

    This verifies that the tiling and hybrid engines can compute Z(1,1)
    without relying on Zephyr-specific decomposition, and that no D-C
    fallback is needed (dc_stats == 0).
    """

    @classmethod
    def setUpClass(cls):
        """Build basic table and Z(1,1) graph."""
        cls.basic_table = build_basic_table()

        # Build Z(1,1) graph from NetworkX definition
        G = nx.Graph()
        for i in range(4):
            for j in range(4, 8):
                G.add_edge(i, j)
        G.add_edge(0, 1)
        G.add_edge(2, 3)
        G.add_edge(4, 5)
        G.add_edge(6, 7)
        G.add_edge(0, 2)
        G.add_edge(1, 3)
        cls.z11_nx = G
        cls.z11_graph = Graph.from_networkx(G)
        cls.kirchhoff = int(round(nx.number_of_spanning_trees(G)))

    def test_z11_via_tiling_engine(self):
        """Z(1,1) via SynthesisEngine (tiling) matches Kirchhoff."""
        engine = SynthesisEngine(table=self.basic_table, verbose=False)
        result = engine.synthesize(self.z11_graph)
        self.assertEqual(result.polynomial.num_spanning_trees(), self.kirchhoff)

    def test_z11_via_hybrid_engine(self):
        """Z(1,1) via HybridSynthesisEngine matches Kirchhoff, dc_stats == 0."""
        from tutte_test.hybrid_synthesis import HybridSynthesisEngine

        engine = HybridSynthesisEngine(table=self.basic_table, verbose=False)
        engine.reset_stats()
        result = engine.synthesize(self.z11_graph)

        self.assertEqual(result.polynomial.num_spanning_trees(), self.kirchhoff)
        self.assertTrue(result.verified)

        stats = engine.get_stats()
        self.assertEqual(stats['dc'], 0,
                        f"D-C fallback was used {stats['dc']} times; expected 0")


# =============================================================================
# ZEPHYR Z(m,t) SYNTHESIS TESTS (migrated from test_zephyr_synthesis)
# =============================================================================

class TestZephyrSynthesis(unittest.TestCase):
    """Test Zephyr graph synthesis with bootstrapped Z(1,1) tile.

    Z(m,t) graphs are built from Z(1,1) unit cells. By synthesizing
    Z(1,1) first and adding it to the table, larger graphs can tile
    with it and only need edge addition for inter-cell connections.
    """

    @classmethod
    def setUpClass(cls):
        """Build table and bootstrap Z(1,1) as a reusable tile."""
        cls.table = build_basic_table()
        cls.engine = SynthesisEngine(table=cls.table, verbose=False)

        # Bootstrap: synthesize Z(1,1) and add to table
        try:
            G_z11 = get_zephyr_graph(1, 1)
            z11_graph = Graph.from_networkx(G_z11)
            z11_result = cls.engine.synthesize(z11_graph)
            cls.table.add(z11_graph, "Z_1_1", z11_result.polynomial)
            cls.z11_available = True
        except ImportError:
            cls.z11_available = False

    def _test_zephyr(self, m: int, t: int):
        """Test synthesis of Z(m,t) and verify via Kirchhoff."""
        if not self.z11_available:
            self.skipTest("dwave_networkx required for Zephyr graphs")
            return

        G = get_zephyr_graph(m, t)
        graph = Graph.from_networkx(G)
        self.engine._cache.clear()

        result = self.engine.synthesize(graph)
        kirchhoff = int(round(nx.number_of_spanning_trees(G)))
        tutte_trees = result.polynomial.num_spanning_trees()

        self.assertEqual(tutte_trees, kirchhoff,
                        f"Z({m},{t}) spanning tree mismatch: {tutte_trees} != {kirchhoff}")

    def test_z11(self):
        """Test Z(1,1) - smallest Zephyr unit cell (12 nodes, 22 edges)."""
        self._test_zephyr(1, 1)

    @unittest.skip("VF2 isomorphism too slow for 36+ node graphs; requires tiling optimization")
    def test_z12(self):
        """Test Z(1,2) - two Z(1,1) cells."""
        self._test_zephyr(1, 2)

    @unittest.skip("VF2 isomorphism too slow for 36+ node graphs; requires tiling optimization")
    def test_z13(self):
        """Test Z(1,3) - three Z(1,1) cells."""
        self._test_zephyr(1, 3)

    @unittest.skip("VF2 isomorphism too slow for 36+ node graphs; requires tiling optimization")
    def test_z14(self):
        """Test Z(1,4) - four Z(1,1) cells."""
        self._test_zephyr(1, 4)

    @unittest.skip("VF2 isomorphism too slow for 36+ node graphs; requires tiling optimization")
    def test_z21(self):
        """Test Z(2,1)."""
        self._test_zephyr(2, 1)

    @unittest.skip("VF2 isomorphism too slow for 36+ node graphs; requires tiling optimization")
    def test_z22(self):
        """Test Z(2,2)."""
        self._test_zephyr(2, 2)


class TestLargeZephyr(unittest.TestCase):
    """Test larger Zephyr graphs - uses Z(1,1) tiling."""

    @classmethod
    def setUpClass(cls):
        cls.table = build_basic_table()
        cls.engine = SynthesisEngine(table=cls.table, verbose=False)

        # Bootstrap Z(1,1)
        try:
            G_z11 = get_zephyr_graph(1, 1)
            z11_graph = Graph.from_networkx(G_z11)
            z11_result = cls.engine.synthesize(z11_graph)
            cls.table.add(z11_graph, "Z_1_1", z11_result.polynomial)
            cls.z11_available = True
        except ImportError:
            cls.z11_available = False

    def _test_zephyr(self, m: int, t: int):
        """Test synthesis of Z(m,t)."""
        if not self.z11_available:
            self.skipTest("dwave_networkx required for Zephyr graphs")
            return

        G = get_zephyr_graph(m, t)
        graph = Graph.from_networkx(G)
        self.engine._cache.clear()

        result = self.engine.synthesize(graph)
        kirchhoff = int(round(nx.number_of_spanning_trees(G)))
        tutte_trees = result.polynomial.num_spanning_trees()

        self.assertEqual(tutte_trees, kirchhoff,
                        f"Z({m},{t}) spanning tree mismatch: {tutte_trees} != {kirchhoff}")

    @unittest.skip("VF2 isomorphism too slow for large Zephyr graphs; requires tiling optimization")
    def test_z31(self):
        """Test Z(3,1)."""
        self._test_zephyr(3, 1)

    @unittest.skip("VF2 isomorphism too slow for large Zephyr graphs; requires tiling optimization")
    def test_z23(self):
        """Test Z(2,3)."""
        self._test_zephyr(2, 3)

    @unittest.skip("VF2 isomorphism too slow for large Zephyr graphs; requires tiling optimization")
    def test_z41(self):
        """Test Z(4,1)."""
        self._test_zephyr(4, 1)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    unittest.main()
