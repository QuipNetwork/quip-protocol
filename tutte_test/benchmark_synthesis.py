"""Benchmark Synthesis Methods and Rainbow Table Encoding.

Compares:
1. JSON rainbow table vs binary bitstring encoding sizes
2. New synthesis engine vs old synthesis vs networkx.tutte_polynomial()
"""

import json
import os
import time
import struct
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass

import networkx as nx

# Import our modules
from .polynomial import TuttePolynomial, encode_varuint, encode_varsint
from .graph import Graph, complete_graph, cycle_graph
from .rainbow_table import RainbowTable, load_default_table
from .synthesis import SynthesisEngine, compute_tutte_polynomial
from .validation import verify_spanning_trees

# Import old deletion-contraction for comparison
from .tutte_utils import GraphBuilder, compute_tutte_polynomial as old_compute_tutte


# =============================================================================
# BINARY RAINBOW TABLE FORMAT
# =============================================================================

def encode_rainbow_table_binary(table: RainbowTable) -> bytes:
    """Encode rainbow table to compact binary format.

    Format:
        Header:
            [magic: 4 bytes] = "RTBL"
            [version: 1 byte] = 1
            [num_entries: varuint]

        For each entry:
            [name_len: varuint]
            [name: bytes]
            [node_count: varuint]
            [edge_count: varuint]
            [spanning_trees: varuint]
            [polynomial_bytes_len: varuint]
            [polynomial_bytes: bytes]
    """
    result = bytearray()

    # Magic header
    result.extend(b"RTBL")
    result.append(1)  # version

    # Number of entries
    result.extend(encode_varuint(len(table.entries)))

    # Each entry
    for key, entry in table.entries.items():
        # Name
        name_bytes = entry.name.encode('utf-8')
        result.extend(encode_varuint(len(name_bytes)))
        result.extend(name_bytes)

        # Metadata
        result.extend(encode_varuint(entry.node_count))
        result.extend(encode_varuint(entry.edge_count))
        result.extend(encode_varuint(entry.spanning_trees))

        # Polynomial as bitstring
        poly_bytes = entry.polynomial.to_bytes()
        result.extend(encode_varuint(len(poly_bytes)))
        result.extend(poly_bytes)

    return bytes(result)


def save_binary_rainbow_table(table: RainbowTable, path: str) -> int:
    """Save rainbow table to binary format, return size in bytes."""
    data = encode_rainbow_table_binary(table)
    with open(path, 'wb') as f:
        f.write(data)
    return len(data)


# =============================================================================
# SIZE COMPARISON
# =============================================================================

def compare_rainbow_table_sizes():
    """Compare JSON vs binary rainbow table sizes."""
    print("=" * 60)
    print("RAINBOW TABLE SIZE COMPARISON")
    print("=" * 60)

    # Load existing table
    table_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table.json')
    table = RainbowTable.load(table_path)

    # Get JSON size
    json_size = os.path.getsize(table_path)

    # Create binary version
    binary_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table.bin')
    binary_size = save_binary_rainbow_table(table, binary_path)

    # Also create a minimal JSON (no polynomial_str, compact)
    minimal_json = create_minimal_json(table)
    minimal_path = os.path.join(os.path.dirname(__file__), 'tutte_rainbow_table_minimal.json')
    with open(minimal_path, 'w') as f:
        json.dump(minimal_json, f, separators=(',', ':'))
    minimal_size = os.path.getsize(minimal_path)

    print(f"\nNumber of entries: {len(table.entries)}")
    print(f"\nFormat sizes:")
    print(f"  Original JSON:  {json_size:>10,} bytes ({json_size/1024:.1f} KB)")
    print(f"  Minimal JSON:   {minimal_size:>10,} bytes ({minimal_size/1024:.1f} KB)")
    print(f"  Binary:         {binary_size:>10,} bytes ({binary_size/1024:.1f} KB)")
    print(f"\nCompression ratios (vs original JSON):")
    print(f"  Minimal JSON:   {json_size/minimal_size:.2f}x smaller")
    print(f"  Binary:         {json_size/binary_size:.2f}x smaller")

    # Analyze breakdown by component
    print("\nBinary format breakdown:")
    analyze_binary_breakdown(table)

    return {
        'json_size': json_size,
        'minimal_json_size': minimal_size,
        'binary_size': binary_size,
        'num_entries': len(table.entries)
    }


def create_minimal_json(table: RainbowTable) -> dict:
    """Create minimal JSON representation (no redundant fields)."""
    graphs = {}
    for key, entry in table.entries.items():
        # Only store what's needed to reconstruct
        coeffs = {}
        for (i, j), c in entry.polynomial.to_coefficients().items():
            coeffs[f"{i},{j}"] = c

        graphs[key] = {
            'n': entry.name,
            'v': entry.node_count,
            'e': entry.edge_count,
            'c': coeffs
        }

    return {'g': graphs}


def analyze_binary_breakdown(table: RainbowTable):
    """Analyze where bytes go in binary format."""
    header_size = 5  # magic + version
    entry_count_size = len(encode_varuint(len(table.entries)))

    name_bytes = 0
    metadata_bytes = 0
    polynomial_bytes = 0

    for entry in table.entries.values():
        name_b = entry.name.encode('utf-8')
        name_bytes += len(encode_varuint(len(name_b))) + len(name_b)

        metadata_bytes += len(encode_varuint(entry.node_count))
        metadata_bytes += len(encode_varuint(entry.edge_count))
        metadata_bytes += len(encode_varuint(entry.spanning_trees))

        poly_b = entry.polynomial.to_bytes()
        polynomial_bytes += len(encode_varuint(len(poly_b))) + len(poly_b)

    total = header_size + entry_count_size + name_bytes + metadata_bytes + polynomial_bytes

    print(f"  Header:       {header_size + entry_count_size:>6} bytes ({100*(header_size+entry_count_size)/total:.1f}%)")
    print(f"  Names:        {name_bytes:>6} bytes ({100*name_bytes/total:.1f}%)")
    print(f"  Metadata:     {metadata_bytes:>6} bytes ({100*metadata_bytes/total:.1f}%)")
    print(f"  Polynomials:  {polynomial_bytes:>6} bytes ({100*polynomial_bytes/total:.1f}%)")
    print(f"  Total:        {total:>6} bytes")


# =============================================================================
# SYNTHESIS BENCHMARKS
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    method: str
    graph_name: str
    nodes: int
    edges: int
    time_ms: float
    spanning_trees: int
    verified: bool
    num_terms: int


def build_z11_graph() -> nx.Graph:
    """Build Z(1,1) Zephyr topology graph.

    Z(1,1) has specific structure from quantum annealer topology.
    For simplicity, we use a known construction.
    """
    # Z(1,1) is a specific 8-node graph from D-Wave's Zephyr topology
    # It has 16*1^2 + 6*1 = 22 edges
    # Here we construct it from the definition

    G = nx.Graph()

    # Z(1,1) has 8 nodes in a unit cell
    # Internal K_4,4 bipartite structure plus additional edges
    # Left side: 0,1,2,3  Right side: 4,5,6,7

    # Complete bipartite K_4,4
    for i in range(4):
        for j in range(4, 8):
            G.add_edge(i, j)

    # Odd couplers (vertical chains within each side)
    G.add_edge(0, 1)
    G.add_edge(2, 3)
    G.add_edge(4, 5)
    G.add_edge(6, 7)

    # External couplers
    G.add_edge(0, 2)
    G.add_edge(1, 3)

    return G


def build_test_graphs() -> List[Tuple[str, nx.Graph]]:
    """Build collection of test graphs for benchmarking."""
    graphs = []

    # Complete graphs
    for n in [3, 4, 5, 6]:
        graphs.append((f"K_{n}", nx.complete_graph(n)))

    # Cycle graphs
    for n in [4, 5, 6, 8, 10]:
        graphs.append((f"C_{n}", nx.cycle_graph(n)))

    # Z(1,1) - Zephyr unit cell
    graphs.append(("Z(1,1)", build_z11_graph()))

    # Petersen graph
    graphs.append(("Petersen", nx.petersen_graph()))

    # Grid graphs
    graphs.append(("Grid_3x3", nx.grid_2d_graph(3, 3)))

    # Wheel graphs
    for n in [4, 5, 6]:
        graphs.append((f"W_{n}", nx.wheel_graph(n)))

    return graphs


def benchmark_networkx(G: nx.Graph, name: str) -> BenchmarkResult:
    """Benchmark networkx.tutte_polynomial()."""
    start = time.perf_counter()
    try:
        poly = nx.tutte_polynomial(G)
        elapsed = (time.perf_counter() - start) * 1000

        # Evaluate at (1,1) for spanning trees
        spanning_trees = int(poly.eval({poly.gens[0]: 1, poly.gens[1]: 1}))
        num_terms = len(poly.as_dict())

        # Verify with Kirchhoff
        kirchhoff = int(round(nx.number_of_spanning_trees(G)))
        verified = spanning_trees == kirchhoff

        return BenchmarkResult(
            method="networkx",
            graph_name=name,
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            time_ms=elapsed,
            spanning_trees=spanning_trees,
            verified=verified,
            num_terms=num_terms
        )
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return BenchmarkResult(
            method="networkx",
            graph_name=name,
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            time_ms=elapsed,
            spanning_trees=-1,
            verified=False,
            num_terms=0
        )


def benchmark_old_deletion_contraction(G: nx.Graph, name: str) -> BenchmarkResult:
    """Benchmark old tutte_utils.py deletion-contraction."""
    start = time.perf_counter()
    try:
        # Convert to GraphBuilder
        gb = GraphBuilder()
        node_map = {}
        for node in G.nodes():
            node_map[node] = gb.add_node()
        for u, v in G.edges():
            gb.add_edge(node_map[u], node_map[v])

        # Compute via deletion-contraction
        poly = old_compute_tutte(gb)
        elapsed = (time.perf_counter() - start) * 1000

        spanning_trees = poly.evaluate(1, 1)

        # Verify with Kirchhoff
        kirchhoff = int(round(nx.number_of_spanning_trees(G)))
        verified = spanning_trees == kirchhoff

        return BenchmarkResult(
            method="old_d-c",
            graph_name=name,
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            time_ms=elapsed,
            spanning_trees=spanning_trees,
            verified=verified,
            num_terms=len(poly.coefficients)
        )
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return BenchmarkResult(
            method="old_d-c",
            graph_name=name,
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            time_ms=elapsed,
            spanning_trees=-1,
            verified=False,
            num_terms=0
        )


def benchmark_new_synthesis(G: nx.Graph, name: str, engine: SynthesisEngine) -> BenchmarkResult:
    """Benchmark new synthesis.py engine."""
    graph = Graph.from_networkx(G)

    start = time.perf_counter()
    try:
        result = engine.synthesize(graph)
        elapsed = (time.perf_counter() - start) * 1000

        spanning_trees = result.polynomial.num_spanning_trees()

        # Verify with Kirchhoff
        kirchhoff = int(round(nx.number_of_spanning_trees(G)))
        verified = spanning_trees == kirchhoff

        return BenchmarkResult(
            method="new_synthesis",
            graph_name=name,
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            time_ms=elapsed,
            spanning_trees=spanning_trees,
            verified=verified,
            num_terms=result.polynomial.num_terms()
        )
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return BenchmarkResult(
            method="new_synthesis",
            graph_name=name,
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            time_ms=elapsed,
            spanning_trees=-1,
            verified=False,
            num_terms=0
        )


def run_synthesis_benchmarks():
    """Run all synthesis benchmarks."""
    print("\n" + "=" * 60)
    print("SYNTHESIS METHOD BENCHMARKS")
    print("=" * 60)

    # Initialize new engine
    engine = SynthesisEngine(verbose=False)

    graphs = build_test_graphs()
    results: List[BenchmarkResult] = []

    print(f"\nBenchmarking {len(graphs)} graphs with 3 methods...\n")

    # Header
    print(f"{'Graph':<12} {'|':^3} {'NetworkX':^12} {'|':^3} {'Old D-C':^12} {'|':^3} {'New Synth':^12} {'|':^3} {'Speedup':^10}")
    print(f"{'':12} {'|':^3} {'(ms)':^12} {'|':^3} {'(ms)':^12} {'|':^3} {'(ms)':^12} {'|':^3} {'vs NX':^10}")
    print("-" * 80)

    for name, G in graphs:
        # Skip very large graphs for networkx and old d-c (too slow)
        skip_slow = G.number_of_edges() > 15

        if skip_slow:
            nx_result = BenchmarkResult(
                method="networkx", graph_name=name,
                nodes=G.number_of_nodes(), edges=G.number_of_edges(),
                time_ms=-1, spanning_trees=-1, verified=False, num_terms=0
            )
            old_result = BenchmarkResult(
                method="old_d-c", graph_name=name,
                nodes=G.number_of_nodes(), edges=G.number_of_edges(),
                time_ms=-1, spanning_trees=-1, verified=False, num_terms=0
            )
        else:
            nx_result = benchmark_networkx(G, name)
            old_result = benchmark_old_deletion_contraction(G, name)

        new_result = benchmark_new_synthesis(G, name, engine)

        results.extend([nx_result, old_result, new_result])

        # Format output
        nx_time = f"{nx_result.time_ms:.2f}" if nx_result.time_ms >= 0 else "skip"
        old_time = f"{old_result.time_ms:.2f}" if old_result.time_ms >= 0 else "error"
        new_time = f"{new_result.time_ms:.2f}" if new_result.time_ms >= 0 else "error"

        if nx_result.time_ms > 0 and new_result.time_ms > 0:
            speedup = f"{nx_result.time_ms / new_result.time_ms:.1f}x"
        else:
            speedup = "-"

        # Verification markers
        nx_mark = "✓" if nx_result.verified else "✗" if nx_result.time_ms >= 0 else " "
        old_mark = "✓" if old_result.verified else "✗" if old_result.time_ms >= 0 else " "
        new_mark = "✓" if new_result.verified else "✗" if new_result.time_ms >= 0 else " "

        print(f"{name:<12} {'|':^3} {nx_time:>10}{nx_mark} {'|':^3} {old_time:>10}{old_mark} {'|':^3} {new_time:>10}{new_mark} {'|':^3} {speedup:^10}")

    print("-" * 80)
    print("✓ = T(1,1) matches Kirchhoff spanning tree count")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    nx_times = [r.time_ms for r in results if r.method == "networkx" and r.time_ms > 0]
    old_times = [r.time_ms for r in results if r.method == "old_d-c" and r.time_ms > 0]
    new_times = [r.time_ms for r in results if r.method == "new_synthesis" and r.time_ms > 0]

    if nx_times:
        print(f"\nNetworkX:     avg {sum(nx_times)/len(nx_times):.2f}ms, total {sum(nx_times):.2f}ms")
    if old_times:
        print(f"Old D-C:       avg {sum(old_times)/len(old_times):.2f}ms, total {sum(old_times):.2f}ms")
    if new_times:
        print(f"New Synthesis: avg {sum(new_times)/len(new_times):.2f}ms, total {sum(new_times):.2f}ms")

    # Verification summary
    nx_verified = sum(1 for r in results if r.method == "networkx" and r.verified)
    old_verified = sum(1 for r in results if r.method == "old_d-c" and r.verified)
    new_verified = sum(1 for r in results if r.method == "new_synthesis" and r.verified)

    print(f"\nVerification (T(1,1) = Kirchhoff):")
    print(f"  NetworkX:     {nx_verified}/{len([r for r in results if r.method == 'networkx'])}")
    print(f"  Old D-C:       {old_verified}/{len([r for r in results if r.method == 'old_d-c'])}")
    print(f"  New Synthesis: {new_verified}/{len([r for r in results if r.method == 'new_synthesis'])}")

    return results


def run_restricted_benchmark():
    """Benchmark using only K graphs and Cycle graphs (no rainbow table lookup)."""
    print("\n" + "=" * 60)
    print("RESTRICTED BENCHMARK (K and C graphs only)")
    print("=" * 60)
    print("\nBuilding graphs from K_n and C_n compositions only...")

    # Create a restricted rainbow table with only K and C graphs
    restricted_table = RainbowTable()

    # Add K_2 through K_6 using known polynomial formulas
    # K_2: T = x
    restricted_table.add(complete_graph(2), "K_2", TuttePolynomial.x())
    # K_3: T = x^2 + x + y
    restricted_table.add(complete_graph(3), "K_3",
        TuttePolynomial.from_coefficients({(2,0): 1, (1,0): 1, (0,1): 1}))
    # K_4: T = x^3 + 3x^2 + 4xy + 2x + y^3 + 3y^2 + 2y
    restricted_table.add(complete_graph(4), "K_4",
        TuttePolynomial.from_coefficients({
            (3,0): 1, (2,0): 3, (1,1): 4, (1,0): 2, (0,1): 2, (0,2): 3, (0,3): 1
        }))

    # Add C_3 through C_10 using formula: T(C_n) = x^{n-1} + x^{n-2} + ... + x + y
    for n in range(3, 11):
        coeffs = {(i, 0): 1 for i in range(1, n)}
        coeffs[(0, 1)] = 1
        restricted_table.add(cycle_graph(n), f"C_{n}",
            TuttePolynomial.from_coefficients(coeffs))

    print(f"Restricted table has {len(restricted_table.entries)} entries")

    # Create engine with restricted table
    engine = SynthesisEngine(table=restricted_table, verbose=False)

    # Test Z(1,1)
    print("\nBenchmarking Z(1,1) synthesis:")
    G = build_z11_graph()

    # Multiple runs for averaging
    n_runs = 5
    times = []

    for i in range(n_runs):
        # Clear cache between runs
        engine._cache.clear()
        graph = Graph.from_networkx(G)

        start = time.perf_counter()
        result = engine.synthesize(graph)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    spanning_trees = result.polynomial.num_spanning_trees()
    kirchhoff = int(round(nx.number_of_spanning_trees(G)))

    print(f"  Time: {avg_time:.2f}ms (avg of {n_runs} runs)")
    print(f"  Spanning trees: {spanning_trees}")
    print(f"  Kirchhoff check: {kirchhoff}")
    print(f"  Verified: {spanning_trees == kirchhoff}")
    print(f"  Method: {result.method}")
    print(f"  Recipe:")
    for step in result.recipe[:10]:  # First 10 steps
        print(f"    {step}")
    if len(result.recipe) > 10:
        print(f"    ... ({len(result.recipe) - 10} more steps)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all benchmarks."""
    # Size comparison
    size_results = compare_rainbow_table_sizes()

    # Synthesis benchmarks
    synthesis_results = run_synthesis_benchmarks()

    # Restricted benchmark
    run_restricted_benchmark()

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
