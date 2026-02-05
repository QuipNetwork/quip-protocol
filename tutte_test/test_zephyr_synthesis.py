"""Test Zephyr Z(m,t) synthesis using only basic graph primitives.

This test verifies that we can efficiently compute Tutte polynomials for
Zephyr topology graphs using only K_n, C_n, P_n, W_n, and S_n graphs
in the rainbow table - without any Zephyr-specific decomposition functions.

If this works, we can delete:
- tutte_synthesis.py (Zephyr-specific functions)
- tutte_utils.py (old infrastructure)
- graph_composition.py (unused by new synthesis)
"""

import time
import unittest
from typing import Dict, List, Tuple

import networkx as nx
from sympy import Poly, symbols

from .polynomial import TuttePolynomial
from .graph import Graph, complete_graph, cycle_graph, path_graph, wheel_graph, star_graph
from .rainbow_table import RainbowTable
from .synthesis import SynthesisEngine
from .validation import verify_spanning_trees


def sympy_to_tutte(nx_poly) -> TuttePolynomial:
    """Convert networkx/sympy polynomial to TuttePolynomial."""
    coeffs = {}

    # Handle different sympy types
    if hasattr(nx_poly, 'as_dict'):
        poly_dict = nx_poly.as_dict()
    elif hasattr(nx_poly, 'as_poly'):
        poly_dict = nx_poly.as_poly().as_dict()
    else:
        # Try to convert to Poly first
        x, y = symbols('x y')
        try:
            poly = Poly(nx_poly, x, y)
            poly_dict = poly.as_dict()
        except:
            # Single term or constant
            poly_dict = {(0, 0): int(nx_poly)}

    for monom, coeff in poly_dict.items():
        if len(monom) >= 2:
            coeffs[(monom[0], monom[1])] = int(coeff)
        elif len(monom) == 1:
            coeffs[(monom[0], 0)] = int(coeff)
        else:
            coeffs[(0, 0)] = int(coeff)

    return TuttePolynomial.from_coefficients(coeffs)


def build_basic_rainbow_table() -> RainbowTable:
    """Build rainbow table with only basic NetworkX graph primitives.

    Includes:
    - Complete graphs K_2 through K_8
    - Cycle graphs C_3 through C_12
    - Path graphs P_2 through P_10
    - Wheel graphs W_4 through W_8
    - Star graphs S_3 through S_8
    - Petersen graph
    - Grid graphs
    """
    table = RainbowTable()

    # Complete graphs K_n: well-known polynomials
    k_polys = {
        2: {(1, 0): 1},  # x
        3: {(2, 0): 1, (1, 0): 1, (0, 1): 1},  # x^2 + x + y
        4: {(3, 0): 1, (2, 0): 3, (1, 1): 4, (1, 0): 2, (0, 1): 2, (0, 2): 3, (0, 3): 1},
        5: {(4, 0): 1, (3, 0): 6, (2, 1): 10, (2, 0): 11, (1, 1): 20, (1, 2): 15,
            (1, 3): 5, (1, 0): 6, (0, 1): 6, (0, 2): 15, (0, 3): 15, (0, 4): 10,
            (0, 5): 4, (0, 6): 1},
    }

    for n in range(2, 6):
        g = complete_graph(n)
        poly = TuttePolynomial.from_coefficients(k_polys[n])
        table.add(g, f"K_{n}", poly)

    # For K_6, K_7, K_8 - compute via deletion-contraction (cached)
    for n in range(6, 9):
        G_nx = nx.complete_graph(n)
        g = Graph.from_networkx(G_nx)
        # Use networkx to compute (slow but accurate)
        nx_poly = nx.tutte_polynomial(G_nx)
        poly = sympy_to_tutte(nx_poly)
        table.add(g, f"K_{n}", poly)

    # Cycle graphs C_n: T(C_n) = x^{n-1} + x^{n-2} + ... + x + y
    for n in range(3, 13):
        g = cycle_graph(n)
        coeffs = {(i, 0): 1 for i in range(1, n)}
        coeffs[(0, 1)] = 1
        poly = TuttePolynomial.from_coefficients(coeffs)
        table.add(g, f"C_{n}", poly)

    # Path graphs P_n: T(P_n) = x^{n-1}
    for n in range(2, 11):
        g = path_graph(n)
        poly = TuttePolynomial.x(n - 1)
        table.add(g, f"P_{n}", poly)

    # Wheel graphs W_n (n spokes + hub)
    for n in range(4, 9):
        G_nx = nx.wheel_graph(n)
        g = Graph.from_networkx(G_nx)
        nx_poly = nx.tutte_polynomial(G_nx)
        poly = sympy_to_tutte(nx_poly)
        table.add(g, f"W_{n}", poly)

    # Star graphs S_n
    for n in range(3, 9):
        g = star_graph(n)
        poly = TuttePolynomial.x(n)  # Star is n bridges
        table.add(g, f"S_{n}", poly)

    # Petersen graph
    G_pet = nx.petersen_graph()
    g_pet = Graph.from_networkx(G_pet)
    nx_poly = nx.tutte_polynomial(G_pet)
    poly = sympy_to_tutte(nx_poly)
    table.add(g_pet, "Petersen", poly)

    # Small grid graphs
    for rows in range(2, 4):
        for cols in range(2, 5):
            G_grid = nx.grid_2d_graph(rows, cols)
            # Relabel nodes to integers
            G_grid = nx.convert_node_labels_to_integers(G_grid)
            g_grid = Graph.from_networkx(G_grid)

            if G_grid.number_of_edges() <= 12:  # Only small grids
                nx_poly = nx.tutte_polynomial(G_grid)
                poly = sympy_to_tutte(nx_poly)
                table.add(g_grid, f"Grid_{rows}x{cols}", poly)

    return table


def get_zephyr_graph(m: int, t: int) -> nx.Graph:
    """Get Zephyr graph Z(m,t) using dwave_networkx."""
    try:
        import dwave_networkx as dnx
        return dnx.zephyr_graph(m, t)
    except ImportError:
        raise ImportError("dwave_networkx required for Zephyr graphs. Install with: pip install dwave-networkx")


class TestZephyrSynthesis(unittest.TestCase):
    """Test Zephyr graph synthesis with basic rainbow table."""

    @classmethod
    def setUpClass(cls):
        """Build the basic rainbow table once for all tests."""
        print("\nBuilding basic rainbow table...")
        start = time.perf_counter()
        cls.table = build_basic_rainbow_table()
        elapsed = time.perf_counter() - start
        print(f"Built rainbow table with {len(cls.table.entries)} entries in {elapsed:.2f}s")

        cls.engine = SynthesisEngine(table=cls.table, verbose=False)

    def _test_zephyr(self, m: int, t: int, max_time_s: float = 60.0) -> Dict:
        """Test synthesis of Z(m,t) and return results."""
        try:
            G = get_zephyr_graph(m, t)
        except ImportError as e:
            self.skipTest(str(e))
            return {}

        graph = Graph.from_networkx(G)

        print(f"\n  Z({m},{t}): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Clear cache for fair timing
        self.engine._cache.clear()

        start = time.perf_counter()
        result = self.engine.synthesize(graph)
        elapsed = time.perf_counter() - start

        # Verify with Kirchhoff
        kirchhoff = int(round(nx.number_of_spanning_trees(G)))
        tutte_trees = result.polynomial.num_spanning_trees()
        verified = tutte_trees == kirchhoff

        print(f"    Time: {elapsed:.3f}s")
        print(f"    Method: {result.method}")
        print(f"    Spanning trees: {tutte_trees}")
        print(f"    Kirchhoff check: {kirchhoff}")
        print(f"    Verified: {verified}")

        self.assertTrue(verified, f"Z({m},{t}) spanning tree mismatch: {tutte_trees} != {kirchhoff}")
        self.assertLess(elapsed, max_time_s, f"Z({m},{t}) took too long: {elapsed:.1f}s > {max_time_s}s")

        return {
            'name': f'Z({m},{t})',
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'time_s': elapsed,
            'method': result.method,
            'spanning_trees': tutte_trees,
            'verified': verified
        }

    def test_z11(self):
        """Test Z(1,1) - smallest Zephyr unit cell."""
        self._test_zephyr(1, 1, max_time_s=10.0)

    def test_z12(self):
        """Test Z(1,2)."""
        self._test_zephyr(1, 2, max_time_s=30.0)

    def test_z13(self):
        """Test Z(1,3)."""
        self._test_zephyr(1, 3, max_time_s=60.0)

    def test_z14(self):
        """Test Z(1,4)."""
        self._test_zephyr(1, 4, max_time_s=120.0)

    def test_z21(self):
        """Test Z(2,1)."""
        self._test_zephyr(2, 1, max_time_s=60.0)

    def test_z22(self):
        """Test Z(2,2)."""
        self._test_zephyr(2, 2, max_time_s=120.0)


class TestLargeZephyr(unittest.TestCase):
    """Test larger Zephyr graphs - may be slow."""

    @classmethod
    def setUpClass(cls):
        """Build the basic rainbow table once for all tests."""
        cls.table = build_basic_rainbow_table()
        cls.engine = SynthesisEngine(table=cls.table, verbose=False)

    def _test_zephyr(self, m: int, t: int, max_time_s: float) -> Dict:
        """Test synthesis of Z(m,t)."""
        try:
            G = get_zephyr_graph(m, t)
        except ImportError as e:
            self.skipTest(str(e))
            return {}

        graph = Graph.from_networkx(G)

        print(f"\n  Z({m},{t}): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        self.engine._cache.clear()

        start = time.perf_counter()
        result = self.engine.synthesize(graph)
        elapsed = time.perf_counter() - start

        kirchhoff = int(round(nx.number_of_spanning_trees(G)))
        tutte_trees = result.polynomial.num_spanning_trees()
        verified = tutte_trees == kirchhoff

        print(f"    Time: {elapsed:.3f}s, Verified: {verified}")

        return {
            'name': f'Z({m},{t})',
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'time_s': elapsed,
            'verified': verified
        }

    def test_z31(self):
        """Test Z(3,1)."""
        self._test_zephyr(3, 1, max_time_s=300.0)

    def test_z23(self):
        """Test Z(2,3)."""
        self._test_zephyr(2, 3, max_time_s=300.0)

    def test_z41(self):
        """Test Z(4,1)."""
        self._test_zephyr(4, 1, max_time_s=300.0)


def run_zephyr_benchmark():
    """Run comprehensive Zephyr benchmark."""
    print("=" * 70)
    print("ZEPHYR Z(m,t) SYNTHESIS BENCHMARK")
    print("Using only basic graph primitives (K, C, P, W, S, grids)")
    print("=" * 70)

    # Build table
    print("\nBuilding basic rainbow table...")
    start = time.perf_counter()
    table = build_basic_rainbow_table()
    build_time = time.perf_counter() - start
    print(f"Built {len(table.entries)} entries in {build_time:.2f}s")

    engine = SynthesisEngine(table=table, verbose=False)

    # Test cases: (m, t, max_time)
    test_cases = [
        (1, 1, 10),
        (1, 2, 30),
        (1, 3, 60),
        (1, 4, 120),
        (2, 1, 60),
        (2, 2, 120),
        (2, 3, 300),
        (2, 4, 600),
        (3, 1, 300),
        (3, 2, 600),
        (4, 1, 600),
    ]

    results = []

    print("\n" + "-" * 70)
    print(f"{'Z(m,t)':<10} {'Nodes':>8} {'Edges':>8} {'Time (s)':>12} {'Trees':>15} {'Verified':>10}")
    print("-" * 70)

    for m, t, max_time in test_cases:
        try:
            G = get_zephyr_graph(m, t)
        except ImportError:
            print(f"Z({m},{t}): SKIPPED (dwave_networkx not installed)")
            continue

        graph = Graph.from_networkx(G)
        engine._cache.clear()

        start = time.perf_counter()
        try:
            result = engine.synthesize(graph)
            elapsed = time.perf_counter() - start

            if elapsed > max_time:
                print(f"Z({m},{t}): TIMEOUT after {elapsed:.1f}s")
                results.append({'name': f'Z({m},{t})', 'status': 'timeout', 'time': elapsed})
                continue

            kirchhoff = int(round(nx.number_of_spanning_trees(G)))
            tutte_trees = result.polynomial.num_spanning_trees()
            verified = tutte_trees == kirchhoff

            status = "PASS" if verified else "FAIL"
            print(f"Z({m},{t})    {G.number_of_nodes():>8} {G.number_of_edges():>8} {elapsed:>12.3f} {tutte_trees:>15} {status:>10}")

            results.append({
                'name': f'Z({m},{t})',
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'time': elapsed,
                'trees': tutte_trees,
                'verified': verified
            })

        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"Z({m},{t}): ERROR after {elapsed:.1f}s - {e}")
            results.append({'name': f'Z({m},{t})', 'status': 'error', 'error': str(e)})

    print("-" * 70)

    # Summary
    passed = sum(1 for r in results if r.get('verified', False))
    total = len([r for r in results if 'verified' in r])
    print(f"\nPassed: {passed}/{total}")

    if total > 0:
        avg_time = sum(r['time'] for r in results if 'time' in r and r.get('verified')) / passed if passed > 0 else 0
        print(f"Average time (passing): {avg_time:.3f}s")

    return results


if __name__ == "__main__":
    run_zephyr_benchmark()
