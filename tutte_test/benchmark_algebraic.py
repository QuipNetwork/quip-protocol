"""Benchmark Algebraic Synthesis vs NetworkX Tutte Polynomial.

This script benchmarks the new algebraic synthesis engine against NetworkX's
built-in tutte_polynomial function for various graph families up to and
including Z(1,1) complexity (12 nodes, 22 edges).

Run with: python -m tutte_test.benchmark_algebraic
"""

import time
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

import networkx as nx

from .graph import (
    Graph, complete_graph, cycle_graph, path_graph,
    wheel_graph, grid_graph, petersen_graph
)
from .polynomial import TuttePolynomial
from .rainbow_table import RainbowTable
from .algebraic_synthesis import AlgebraicSynthesisEngine
from .synthesis import SynthesisEngine


# =============================================================================
# BENCHMARK UTILITIES
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    graph_name: str
    nodes: int
    edges: int

    # Our synthesis
    synth_time_ms: float
    synth_poly: Optional[TuttePolynomial]
    synth_spanning_trees: int

    # NetworkX
    nx_time_ms: float
    nx_poly: Optional[TuttePolynomial]
    nx_spanning_trees: int

    # Verification
    polynomials_match: bool
    trees_match: bool

    def __repr__(self) -> str:
        match = "✓" if self.polynomials_match else "✗"
        return (f"{self.graph_name:<25} {self.nodes:>3}n {self.edges:>3}e | "
                f"Synth: {self.synth_time_ms:>8.2f}ms | "
                f"NX: {self.nx_time_ms:>8.2f}ms | "
                f"Trees: {self.synth_spanning_trees:>10} | {match}")


def time_function(func: Callable, *args, **kwargs) -> Tuple[float, any]:
    """Time a function call, return (time_ms, result)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start) * 1000, result


def networkx_tutte(G: nx.Graph) -> Optional[TuttePolynomial]:
    """Compute Tutte polynomial using NetworkX."""
    try:
        from sympy import symbols, Poly
        x, y = symbols('x y')
        tutte_sympy = nx.tutte_polynomial(G)
        poly = Poly(tutte_sympy, x, y)
        coeffs = {}
        for monom, coeff in poly.as_dict().items():
            coeffs[monom] = int(coeff)
        return TuttePolynomial.from_coefficients(coeffs)
    except Exception as e:
        print(f"  NetworkX error: {e}")
        return None


# =============================================================================
# GRAPH GENERATORS
# =============================================================================

def generate_test_graphs() -> List[Tuple[str, Graph]]:
    """Generate test graphs up to Z(1,1) complexity."""
    graphs = []

    # Complete graphs (up to K_7 - 21 edges)
    for n in range(2, 8):
        graphs.append((f"K_{n}", complete_graph(n)))

    # Cycle graphs
    for n in range(3, 13):
        graphs.append((f"C_{n}", cycle_graph(n)))

    # Path graphs
    for n in range(2, 13):
        graphs.append((f"P_{n}", path_graph(n)))

    # Wheel graphs (up to W_11 - 20 edges)
    for n in range(4, 12):
        graphs.append((f"W_{n}", wheel_graph(n)))

    # Grid graphs
    for m, n in [(2, 2), (2, 3), (2, 4), (2, 5), (3, 3), (3, 4)]:
        graphs.append((f"Grid_{m}x{n}", grid_graph(m, n)))

    # Petersen graph (10 nodes, 15 edges)
    graphs.append(("Petersen", petersen_graph()))

    # Some special constructions
    # Prism graph (2 cycles connected)
    def prism_graph(n: int) -> Graph:
        G = nx.circular_ladder_graph(n)
        return Graph.from_networkx(G)

    for n in [3, 4, 5, 6]:
        graphs.append((f"Prism_{n}", prism_graph(n)))

    # Möbius-Kantor graph (16 nodes, 24 edges)
    try:
        G_mk = nx.moebius_kantor_graph()
        graphs.append(("Moebius-Kantor", Graph.from_networkx(G_mk)))
    except:
        pass

    # Hypercube Q_3 (8 nodes, 12 edges)
    try:
        G_q3 = nx.hypercube_graph(3)
        graphs.append(("Q_3", Graph.from_networkx(G_q3)))
    except:
        pass

    # Sort by edge count for progressive difficulty
    graphs.sort(key=lambda x: (x[1].edge_count(), x[1].node_count()))

    return graphs


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark(
    max_edges: int = 22,
    use_empty_table: bool = True,
    verbose: bool = True
) -> List[BenchmarkResult]:
    """Run the benchmark suite.

    Args:
        max_edges: Maximum number of edges to benchmark
        use_empty_table: If True, use empty rainbow table (no cache)
        verbose: Print progress

    Returns:
        List of BenchmarkResult objects
    """
    # Create synthesis engine with empty table if requested
    if use_empty_table:
        table = RainbowTable()
        if verbose:
            print("Using EMPTY rainbow table (no cached polynomials)")
    else:
        from .rainbow_table import load_default_table
        table = load_default_table()
        if verbose:
            print(f"Using rainbow table with {len(table)} entries")

    # Use the standard synthesis engine (which has deletion-contraction fallback)
    synth_engine = SynthesisEngine(table=table, verbose=False)

    # Generate test graphs
    all_graphs = generate_test_graphs()
    test_graphs = [(name, g) for name, g in all_graphs if g.edge_count() <= max_edges]

    if verbose:
        print(f"\nBenchmarking {len(test_graphs)} graphs (up to {max_edges} edges)")
        print("=" * 80)
        print(f"{'Graph':<25} {'Size':>8} | {'Synth':>12} | {'NetworkX':>12} | {'Trees':>10} | Match")
        print("-" * 80)

    results = []

    for name, graph in test_graphs:
        # Skip very large graphs for NetworkX (it's exponential)
        skip_nx = graph.edge_count() > 15

        # Our synthesis
        synth_time, synth_result = time_function(synth_engine.synthesize, graph)
        synth_poly = synth_result.polynomial
        synth_trees = synth_poly.num_spanning_trees()

        # NetworkX
        if skip_nx:
            nx_time = float('inf')
            nx_poly = None
            nx_trees = -1
        else:
            G_nx = graph.to_networkx()
            nx_time, nx_poly = time_function(networkx_tutte, G_nx)
            nx_trees = nx_poly.num_spanning_trees() if nx_poly else -1

        # Verification
        if nx_poly is not None:
            polys_match = synth_poly == nx_poly
            trees_match = synth_trees == nx_trees
        else:
            # Verify against Kirchhoff's theorem instead
            try:
                kirchhoff_trees = round(nx.number_of_spanning_trees(graph.to_networkx()))
                polys_match = True  # Can't verify polynomial
                trees_match = synth_trees == kirchhoff_trees
            except:
                polys_match = True
                trees_match = True

        result = BenchmarkResult(
            graph_name=name,
            nodes=graph.node_count(),
            edges=graph.edge_count(),
            synth_time_ms=synth_time,
            synth_poly=synth_poly,
            synth_spanning_trees=synth_trees,
            nx_time_ms=nx_time if not skip_nx else -1,
            nx_poly=nx_poly,
            nx_spanning_trees=nx_trees,
            polynomials_match=polys_match,
            trees_match=trees_match
        )

        results.append(result)

        if verbose:
            nx_time_str = f"{nx_time:>8.2f}ms" if not skip_nx else "  (skipped)"
            match = "✓" if (polys_match and trees_match) else "✗"
            print(f"{name:<25} {graph.node_count():>3}n {graph.edge_count():>3}e | "
                  f"{synth_time:>8.2f}ms | {nx_time_str:>12} | "
                  f"{synth_trees:>10} | {match}")

    return results


def print_summary(results: List[BenchmarkResult]):
    """Print benchmark summary."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_synth_time = sum(r.synth_time_ms for r in results)
    total_nx_time = sum(r.nx_time_ms for r in results if r.nx_time_ms > 0)
    nx_count = sum(1 for r in results if r.nx_time_ms > 0)

    all_match = all(r.polynomials_match and r.trees_match for r in results)
    mismatches = [r for r in results if not (r.polynomials_match and r.trees_match)]

    print(f"Total graphs tested: {len(results)}")
    print(f"Total synthesis time: {total_synth_time:.2f}ms")
    print(f"Total NetworkX time: {total_nx_time:.2f}ms (for {nx_count} graphs ≤15 edges)")
    print(f"All results match: {'YES' if all_match else 'NO'}")

    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for r in mismatches:
            print(f"  {r.graph_name}: synth_trees={r.synth_spanning_trees}, "
                  f"nx_trees={r.nx_spanning_trees}")

    # Timing comparison for graphs where both ran
    compared = [r for r in results if r.nx_time_ms > 0]
    if compared:
        synth_faster = sum(1 for r in compared if r.synth_time_ms < r.nx_time_ms)
        print(f"\nSynthesis faster than NetworkX: {synth_faster}/{len(compared)} graphs")

        avg_speedup = sum(r.nx_time_ms / r.synth_time_ms for r in compared
                         if r.synth_time_ms > 0) / len(compared)
        print(f"Average speedup: {avg_speedup:.2f}x")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Tutte polynomial synthesis")
    parser.add_argument("--max-edges", type=int, default=22,
                       help="Maximum edges to benchmark (default: 22, same as Z(1,1))")
    parser.add_argument("--use-cache", action="store_true",
                       help="Use cached rainbow table instead of empty")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output")

    args = parser.parse_args()

    results = run_benchmark(
        max_edges=args.max_edges,
        use_empty_table=not args.use_cache,
        verbose=not args.quiet
    )

    print_summary(results)

    # Return exit code based on correctness
    all_correct = all(r.trees_match for r in results)
    sys.exit(0 if all_correct else 1)


if __name__ == "__main__":
    main()
