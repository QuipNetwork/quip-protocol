"""
Test Suite for Graph Composition and Tutte Polynomial Operations.

This module tests:
1. Graph composition operations (disjoint union, k-joins, clique sums)
2. Tutte polynomial calculations against networkx
3. Rainbow table consistency
4. Synthesis engine correctness

Run with: python -m tutte_test.test_graph_composition
"""

import sys
import os
import unittest
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

from tutte_test.tutte_utils import (
    TuttePolynomial,
    GraphBuilder,
    compute_tutte_polynomial,
    create_path_graph,
    create_cycle_graph,
    create_complete_graph,
    networkx_to_graphbuilder,
)
from tutte_test.graph_composition import (
    disjoint_union,
    cut_vertex_join,
    two_sum,
    parallel_connection,
    clique_sum,
    cartesian_product,
    edge_subdivision,
    identify_vertices,
    find_cut_vertices,
    find_bridges,
    decompose_at_cut_vertex,
    create_edge,
    create_k3,
    create_k4,
    create_star,
    tutte_disjoint_union,
    tutte_cut_vertex,
)


# =============================================================================
# NETWORKX TUTTE POLYNOMIAL COMPUTATION
# =============================================================================

def compute_tutte_networkx(G: nx.Graph) -> TuttePolynomial:
    """
    Compute Tutte polynomial using networkx's implementation.

    NetworkX returns a sympy polynomial, which we convert to our format.
    """
    try:
        from sympy import symbols, Poly

        # NetworkX tutte_polynomial returns a sympy expression
        x, y = symbols('x y')
        tutte_sympy = nx.tutte_polynomial(G)

        # Convert to our format
        poly = Poly(tutte_sympy, x, y)
        coeffs = {}
        for monom, coeff in poly.as_dict().items():
            # monom is (x_power, y_power)
            coeffs[monom] = int(coeff)

        return TuttePolynomial(coeffs)
    except ImportError:
        # sympy not available, skip networkx verification
        return None
    except Exception as e:
        print(f"Warning: networkx tutte_polynomial failed: {e}")
        return None


def graphbuilder_to_networkx(g: GraphBuilder) -> nx.Graph:
    """Convert our GraphBuilder to networkx Graph."""
    G = nx.Graph()
    for node in g.nodes:
        G.add_node(node)
    for edge_id, (u, v) in g.edges.items():
        G.add_edge(u, v)
    # Handle loops (self-edges)
    for edge_id, node in g.loops.items():
        G.add_edge(node, node)
    return G


# =============================================================================
# TEST CASES
# =============================================================================

class TestTuttePolynomialBasics(unittest.TestCase):
    """Test basic Tutte polynomial computations."""

    def test_single_edge(self):
        """T(K_2) = x"""
        g = create_edge()
        t = compute_tutte_polynomial(g)
        expected = TuttePolynomial({(1, 0): 1})
        self.assertEqual(t, expected)

    def test_triangle(self):
        """T(K_3) = x^2 + x + y"""
        g = create_k3()
        t = compute_tutte_polynomial(g)
        expected = TuttePolynomial({(2, 0): 1, (1, 0): 1, (0, 1): 1})
        self.assertEqual(t, expected)

    def test_path(self):
        """T(P_n) = x^{n-1}"""
        for n in range(2, 6):
            g = create_path_graph(n)
            t = compute_tutte_polynomial(g)
            expected = TuttePolynomial({(n-1, 0): 1})
            self.assertEqual(t, expected, f"Failed for P_{n}")

    def test_cycle(self):
        """T(C_n) = x^{n-1} + x^{n-2} + ... + x + y"""
        for n in range(3, 7):
            g = create_cycle_graph(n)
            t = compute_tutte_polynomial(g)
            # Should have x^{n-1} + x^{n-2} + ... + x + y
            expected_coeffs = {(i, 0): 1 for i in range(1, n)}
            expected_coeffs[(0, 1)] = 1
            expected = TuttePolynomial(expected_coeffs)
            self.assertEqual(t, expected, f"Failed for C_{n}")

    def test_spanning_trees(self):
        """T(1,1) = number of spanning trees."""
        # K_4 has 16 spanning trees
        g = create_k4()
        t = compute_tutte_polynomial(g)
        self.assertEqual(t.num_spanning_trees(), 16)

        # C_5 has 5 spanning trees
        g = create_cycle_graph(5)
        t = compute_tutte_polynomial(g)
        self.assertEqual(t.num_spanning_trees(), 5)


class TestCompositionOperations(unittest.TestCase):
    """Test graph composition operations."""

    def test_disjoint_union_formula(self):
        """T(G1 ∪ G2) = T(G1) × T(G2)"""
        g1 = create_k3()
        g2 = create_edge()

        t1 = compute_tutte_polynomial(g1)
        t2 = compute_tutte_polynomial(g2)

        union = disjoint_union(g1, g2)
        t_union = compute_tutte_polynomial(union)

        expected = tutte_disjoint_union(t1, t2)
        self.assertEqual(t_union, expected)

    def test_cut_vertex_formula(self):
        """T(G1 ·₁ G2) = T(G1) × T(G2) when joined at cut vertex."""
        g1 = create_k3()
        g2 = create_k3()

        t1 = compute_tutte_polynomial(g1)
        t2 = compute_tutte_polynomial(g2)

        joined = cut_vertex_join(g1, 0, g2, 0)
        t_joined = compute_tutte_polynomial(joined)

        expected = tutte_cut_vertex(t1, t2)
        self.assertEqual(t_joined, expected)

    def test_parallel_connection(self):
        """Test parallel connection preserves expected structure."""
        g1 = create_k3()
        g2 = create_k3()

        result = parallel_connection(g1, 0, g2, 0)

        # Should have 4 nodes (2 identified) and 5 edges
        self.assertEqual(result.num_nodes(), 4)
        self.assertEqual(result.num_edges(), 5)

    def test_two_sum(self):
        """Test 2-sum deletes the shared edge."""
        g1 = create_cycle_graph(4)
        g2 = create_cycle_graph(4)

        result = two_sum(g1, 0, g2, 0)

        # Two C_4 with one shared edge deleted = 6 nodes, 6 edges
        self.assertEqual(result.num_nodes(), 6)
        self.assertEqual(result.num_edges(), 6)

    def test_clique_sum_2(self):
        """Test 2-clique sum (edge sum)."""
        g1 = create_k4()
        g2 = create_k4()

        result = clique_sum(g1, [0, 1], g2, [0, 1], delete_clique_edges=True)

        # Two K_4 with 2 vertices identified, shared edge deleted
        # 4 + 4 - 2 = 6 nodes, 6 + 6 - 1 - 1 = 10 edges
        self.assertEqual(result.num_nodes(), 6)

    def test_edge_subdivision(self):
        """Test edge subdivision adds vertices correctly."""
        g = create_edge()

        subdiv = edge_subdivision(g, 0, k=2)

        # Original: 2 nodes, 1 edge
        # After subdividing with k=2: 4 nodes, 3 edges
        self.assertEqual(subdiv.num_nodes(), 4)
        self.assertEqual(subdiv.num_edges(), 3)


class TestCutAnalysis(unittest.TestCase):
    """Test cut vertex and bridge analysis."""

    def test_path_has_bridges(self):
        """Every edge in a path is a bridge."""
        g = create_path_graph(5)
        bridges = find_bridges(g)
        self.assertEqual(len(bridges), 4)

    def test_cycle_no_bridges(self):
        """Cycles have no bridges."""
        g = create_cycle_graph(5)
        bridges = find_bridges(g)
        self.assertEqual(len(bridges), 0)

    def test_k4_no_cut_vertices(self):
        """K_4 has no cut vertices."""
        g = create_k4()
        cuts = find_cut_vertices(g)
        self.assertEqual(len(cuts), 0)

    def test_barbell_has_cut_vertex(self):
        """Two triangles joined at a vertex has a cut vertex."""
        g1 = create_k3()
        g2 = create_k3()
        barbell = cut_vertex_join(g1, 0, g2, 0)

        cuts = find_cut_vertices(barbell)
        self.assertEqual(len(cuts), 1)

    def test_decompose_at_cut_vertex(self):
        """Test decomposition at cut vertex."""
        g1 = create_k3()
        g2 = create_k3()
        barbell = cut_vertex_join(g1, 0, g2, 0)

        cuts = find_cut_vertices(barbell)
        self.assertEqual(len(cuts), 1)

        components = decompose_at_cut_vertex(barbell, cuts[0])
        self.assertEqual(len(components), 2)


class TestNetworkxVerification(unittest.TestCase):
    """Verify our computations match networkx."""

    def _verify_polynomial(self, g: GraphBuilder, name: str):
        """Helper to verify polynomial against networkx."""
        our_poly = compute_tutte_polynomial(g)

        G_nx = graphbuilder_to_networkx(g)
        nx_poly = compute_tutte_networkx(G_nx)

        if nx_poly is None:
            self.skipTest("networkx tutte_polynomial not available")

        self.assertEqual(our_poly, nx_poly, f"Mismatch for {name}")

    def test_k3_matches_networkx(self):
        """K_3 polynomial matches networkx."""
        self._verify_polynomial(create_k3(), "K_3")

    def test_k4_matches_networkx(self):
        """K_4 polynomial matches networkx."""
        self._verify_polynomial(create_k4(), "K_4")

    def test_k5_matches_networkx(self):
        """K_5 polynomial matches networkx."""
        self._verify_polynomial(create_complete_graph(5), "K_5")

    def test_c5_matches_networkx(self):
        """C_5 polynomial matches networkx."""
        self._verify_polynomial(create_cycle_graph(5), "C_5")

    def test_petersen_matches_networkx(self):
        """Petersen graph polynomial matches networkx."""
        G_nx = nx.petersen_graph()
        g = networkx_to_graphbuilder(G_nx)
        self._verify_polynomial(g, "Petersen")


class TestRainbowTableConsistency(unittest.TestCase):
    """Test consistency with rainbow table."""

    @classmethod
    def setUpClass(cls):
        """Load rainbow table once for all tests."""
        try:
            from tutte_test.build_rainbow_table import RainbowTable
            table_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table.json')
            cls.rainbow_table = RainbowTable.load(table_path)
        except Exception as e:
            cls.rainbow_table = None
            print(f"Warning: Could not load rainbow table: {e}")

    def _verify_against_table(self, name: str, graph_fn):
        """Verify computed polynomial matches rainbow table entry."""
        if self.rainbow_table is None:
            self.skipTest("Rainbow table not available")

        entry = self.rainbow_table.get_entry(name)
        if entry is None:
            self.skipTest(f"{name} not in rainbow table")

        expected = self.rainbow_table._entry_to_polynomial(entry)
        g = graph_fn()
        computed = compute_tutte_polynomial(g)

        self.assertEqual(computed, expected, f"Mismatch for {name}")

    def test_k2_matches_table(self):
        self._verify_against_table('K_2', create_edge)

    def test_k3_matches_table(self):
        self._verify_against_table('K_3', create_k3)

    def test_k4_matches_table(self):
        self._verify_against_table('K_4', create_k4)

    def test_k5_matches_table(self):
        self._verify_against_table('K_5', lambda: create_complete_graph(5))

    def test_k6_matches_table(self):
        self._verify_against_table('K_6', lambda: create_complete_graph(6))

    def test_c4_matches_table(self):
        self._verify_against_table('C_4', lambda: create_cycle_graph(4))

    def test_c5_matches_table(self):
        self._verify_against_table('C_5', lambda: create_cycle_graph(5))

    def test_p3_matches_table(self):
        self._verify_against_table('P_3', lambda: create_path_graph(3))

    def test_p4_matches_table(self):
        self._verify_against_table('P_4', lambda: create_path_graph(4))

    def test_petersen_matches_table(self):
        if self.rainbow_table is None:
            self.skipTest("Rainbow table not available")

        entry = self.rainbow_table.get_entry('Petersen')
        if entry is None:
            self.skipTest("Petersen not in rainbow table")

        expected = self.rainbow_table._entry_to_polynomial(entry)

        # Build Petersen from networkx
        G_nx = nx.petersen_graph()
        g = networkx_to_graphbuilder(G_nx)
        computed = compute_tutte_polynomial(g)

        self.assertEqual(computed, expected, "Petersen graph mismatch")

    def test_wheel_matches_table(self):
        """Test wheel graphs match rainbow table."""
        for n in [5, 6]:
            with self.subTest(n=n):
                if self.rainbow_table is None:
                    self.skipTest("Rainbow table not available")

                entry = self.rainbow_table.get_entry(f'W_{n}')
                if entry is None:
                    self.skipTest(f"W_{n} not in rainbow table")

                expected = self.rainbow_table._entry_to_polynomial(entry)

                # Build wheel from networkx
                G_nx = nx.wheel_graph(n)
                g = networkx_to_graphbuilder(G_nx)
                computed = compute_tutte_polynomial(g)

                self.assertEqual(computed, expected, f"W_{n} mismatch")


class TestCompositionWithRainbowTable(unittest.TestCase):
    """Test composition operations using rainbow table graphs."""

    @classmethod
    def setUpClass(cls):
        """Load rainbow table."""
        try:
            from tutte_test.build_rainbow_table import RainbowTable
            table_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table.json')
            cls.rainbow_table = RainbowTable.load(table_path)
        except Exception:
            cls.rainbow_table = None

    def test_disjoint_union_k3_k4(self):
        """Test disjoint union of K_3 and K_4."""
        g1 = create_k3()
        g2 = create_k4()

        t1 = compute_tutte_polynomial(g1)
        t2 = compute_tutte_polynomial(g2)

        union = disjoint_union(g1, g2)
        t_union = compute_tutte_polynomial(union)

        # T(G1 ∪ G2) = T(G1) × T(G2)
        expected = t1 * t2
        self.assertEqual(t_union, expected)

    def test_cut_vertex_join_cycles(self):
        """Test cut vertex join of two cycles."""
        g1 = create_cycle_graph(4)
        g2 = create_cycle_graph(5)

        t1 = compute_tutte_polynomial(g1)
        t2 = compute_tutte_polynomial(g2)

        joined = cut_vertex_join(g1, 0, g2, 0)
        t_joined = compute_tutte_polynomial(joined)

        # Should equal product
        expected = t1 * t2
        self.assertEqual(t_joined, expected)

    def test_star_compositions(self):
        """Test composition with star graphs."""
        star3 = create_star(3)
        star4 = create_star(4)

        t3 = compute_tutte_polynomial(star3)
        t4 = compute_tutte_polynomial(star4)

        # Star graphs: T(S_n) = x^n (n bridges from center)
        self.assertEqual(t3, TuttePolynomial({(3, 0): 1}))
        self.assertEqual(t4, TuttePolynomial({(4, 0): 1}))

        # Join at center
        joined = cut_vertex_join(star3, 0, star4, 0)
        t_joined = compute_tutte_polynomial(joined)

        # Should be x^7
        expected = TuttePolynomial({(7, 0): 1})
        self.assertEqual(t_joined, expected)


class TestSynthesisEngine(unittest.TestCase):
    """Test the synthesis engine."""

    def test_synthesize_k3(self):
        """Test synthesis of K_3."""
        from tutte_test.tutte_synthesis import SynthesisEngine

        engine = SynthesisEngine(verbose=False)
        target = TuttePolynomial({(2, 0): 1, (1, 0): 1, (0, 1): 1})

        result = engine.synthesize(target)

        self.assertTrue(result.success)
        self.assertTrue(engine.verify_synthesis(result, target))

    def test_synthesize_product(self):
        """Test synthesis of product polynomial."""
        from tutte_test.tutte_synthesis import SynthesisEngine

        engine = SynthesisEngine(verbose=False)

        # K_3 × K_2
        k3_poly = TuttePolynomial({(2, 0): 1, (1, 0): 1, (0, 1): 1})
        k2_poly = TuttePolynomial({(1, 0): 1})
        target = k3_poly * k2_poly

        result = engine.synthesize(target)

        self.assertTrue(result.success)
        self.assertTrue(engine.verify_synthesis(result, target))

    def test_synthesize_theta(self):
        """Test synthesis of theta graph (parallel edges)."""
        from tutte_test.tutte_synthesis import SynthesisEngine

        engine = SynthesisEngine(verbose=False)

        # Theta graph: x + y^2 + y
        target = TuttePolynomial({(1, 0): 1, (0, 2): 1, (0, 1): 1})

        result = engine.synthesize(target)

        self.assertTrue(result.success)
        self.assertTrue(engine.verify_synthesis(result, target))


# =============================================================================
# Z(1,t) DECOMPOSITION TESTS
# =============================================================================

class TestZ1tDecomposition(unittest.TestCase):
    """Test the Z(1,t) Zephyr graph decomposition pattern."""

    @classmethod
    def setUpClass(cls):
        """Check if D-Wave libraries are available."""
        try:
            import dwave_networkx
            cls.dwave_available = True
        except ImportError:
            cls.dwave_available = False

    def test_z1t_edge_formula(self):
        """Test that edge formula E(Z(1,t)) = 16t² + 6t is correct."""
        from tutte_test.tutte_synthesis import z1t_edge_formula

        # Known values
        self.assertEqual(z1t_edge_formula(1), 22)   # Z(1,1)
        self.assertEqual(z1t_edge_formula(2), 76)   # Z(1,2)
        self.assertEqual(z1t_edge_formula(3), 162)  # Z(1,3)
        self.assertEqual(z1t_edge_formula(4), 280)  # Z(1,4)
        self.assertEqual(z1t_edge_formula(5), 430)  # Z(1,5)

    def test_z12_decomposition_pattern(self):
        """Test Z(1,2) decomposes into 2 Z(1,1) copies + connector."""
        if not self.dwave_available:
            self.skipTest("D-Wave libraries not available")

        from tutte_test.tutte_synthesis import decompose_zephyr_z1t

        decomp = decompose_zephyr_z1t(2)

        # Should have exactly 2 Z(1,1) copies
        self.assertEqual(len(decomp.z11_copies), 2)

        # Should have 1 pair with correct structure
        self.assertEqual(len(decomp.pair_connectors), 1)

        pair_info = list(decomp.pair_connectors.values())[0]
        self.assertEqual(pair_info['edges'], 32)
        self.assertEqual(pair_info['components'], 2)
        self.assertEqual(pair_info['trees'], [768, 768])

        # Pattern should be verified
        self.assertTrue(decomp.pattern_verified)

    def test_z13_decomposition_pattern(self):
        """Test Z(1,3) decomposes into 3 Z(1,1) copies + 3 pair connectors."""
        if not self.dwave_available:
            self.skipTest("D-Wave libraries not available")

        from tutte_test.tutte_synthesis import decompose_zephyr_z1t

        decomp = decompose_zephyr_z1t(3)

        # Should have exactly 3 Z(1,1) copies
        self.assertEqual(len(decomp.z11_copies), 3)

        # Should have 3 pairs (C(3,2) = 3)
        self.assertEqual(len(decomp.pair_connectors), 3)

        # Each pair should have same connector structure
        for pair_info in decomp.pair_connectors.values():
            self.assertEqual(pair_info['edges'], 32)
            self.assertEqual(pair_info['components'], 2)
            self.assertEqual(pair_info['trees'], [768, 768])

        # Pattern should be verified
        self.assertTrue(decomp.pattern_verified)

    def test_z14_decomposition_pattern(self):
        """Test Z(1,4) decomposes into 4 Z(1,1) copies + 6 pair connectors."""
        if not self.dwave_available:
            self.skipTest("D-Wave libraries not available")

        from tutte_test.tutte_synthesis import decompose_zephyr_z1t

        decomp = decompose_zephyr_z1t(4)

        # Should have exactly 4 Z(1,1) copies
        self.assertEqual(len(decomp.z11_copies), 4)

        # Should have 6 pairs (C(4,2) = 6)
        self.assertEqual(len(decomp.pair_connectors), 6)

        # Pattern should be verified
        self.assertTrue(decomp.pattern_verified)

    def test_connector_component_trees(self):
        """Test that connector component has exactly 768 spanning trees."""
        if not self.dwave_available:
            self.skipTest("D-Wave libraries not available")

        from tutte_test.tutte_synthesis import (
            decompose_zephyr_z1t,
            ZEPHYR_CONNECTOR_COMPONENT_TREES,
        )

        decomp = decompose_zephyr_z1t(2)

        self.assertIsNotNone(decomp.component_polynomial)
        self.assertEqual(
            decomp.component_polynomial.num_spanning_trees(),
            ZEPHYR_CONNECTOR_COMPONENT_TREES
        )
        self.assertEqual(ZEPHYR_CONNECTOR_COMPONENT_TREES, 768)

    def test_edge_count_matches_formula(self):
        """Test that actual edge counts match the formula for t=2,3,4."""
        if not self.dwave_available:
            self.skipTest("D-Wave libraries not available")

        from tutte_test.tutte_synthesis import decompose_zephyr_z1t, z1t_edge_formula

        for t in [2, 3, 4]:
            decomp = decompose_zephyr_z1t(t)
            expected = z1t_edge_formula(t)
            self.assertEqual(decomp.edges, expected,
                           f"Z(1,{t}) edge count mismatch: {decomp.edges} != {expected}")


# =============================================================================
# BENCHMARK FUNCTIONS (not unittest)
# =============================================================================

def benchmark_networkx_comparison():
    """Benchmark our implementation against networkx on various graphs."""
    import time

    print("\n" + "=" * 70)
    print("BENCHMARK: Our Implementation vs NetworkX")
    print("=" * 70)

    test_cases = [
        ("K_3", create_k3),
        ("K_4", create_k4),
        ("K_5", lambda: create_complete_graph(5)),
        ("C_5", lambda: create_cycle_graph(5)),
        ("C_8", lambda: create_cycle_graph(8)),
        ("P_6", lambda: create_path_graph(6)),
    ]

    print(f"\n{'Graph':<10} {'Our Time':<12} {'NX Time':<12} {'Match':<8} {'Trees'}")
    print("-" * 60)

    for name, graph_fn in test_cases:
        g = graph_fn()
        G_nx = graphbuilder_to_networkx(g)

        # Our implementation
        start = time.time()
        our_poly = compute_tutte_polynomial(g)
        our_time = (time.time() - start) * 1000

        # NetworkX
        start = time.time()
        nx_poly = compute_tutte_networkx(G_nx)
        nx_time = (time.time() - start) * 1000

        match = "✓" if (nx_poly is None or our_poly == nx_poly) else "✗"
        trees = our_poly.num_spanning_trees()

        print(f"{name:<10} {our_time:<12.2f} {nx_time:<12.2f} {match:<8} {trees}")


def run_all_tests():
    """Run all tests and benchmarks."""
    # Run unittest tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestTuttePolynomialBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestCompositionOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestCutAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestNetworkxVerification))
    suite.addTests(loader.loadTestsFromTestCase(TestRainbowTableConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestCompositionWithRainbowTable))
    suite.addTests(loader.loadTestsFromTestCase(TestSynthesisEngine))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Run benchmarks
    benchmark_networkx_comparison()

    return result


if __name__ == "__main__":
    run_all_tests()
