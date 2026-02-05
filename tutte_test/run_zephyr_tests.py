"""Run Zephyr tests individually with a shared rainbow table."""
import time
import sys

import networkx as nx
import dwave_networkx as dnx

from tutte_test.polynomial import TuttePolynomial
from tutte_test.graph import Graph
from tutte_test.synthesis import SynthesisEngine
from tutte_test.rainbow_table import load_default_table


def test_zephyr(m: int, t: int, engine: SynthesisEngine, max_time_s: float = 120.0):
    """Test synthesis of Z(m,t)."""
    try:
        G = dnx.zephyr_graph(m, t)
    except Exception as e:
        print(f"  Z({m},{t}): SKIPPED - {e}")
        return None

    graph = Graph.from_networkx(G)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    print(f"  Z({m},{t}): {n_nodes} nodes, {n_edges} edges")

    # Clear cache for fair timing
    engine._cache.clear()
    engine._multigraph_cache.clear()

    start = time.perf_counter()
    try:
        result = engine.synthesize(graph)
        elapsed = time.perf_counter() - start

        if elapsed > max_time_s:
            print(f"    TIMEOUT after {elapsed:.1f}s (limit: {max_time_s}s)")
            return {'status': 'timeout', 'time': elapsed}

        # Verify with Kirchhoff
        kirchhoff = int(round(nx.number_of_spanning_trees(G)))
        tutte_trees = result.polynomial.num_spanning_trees()
        verified = tutte_trees == kirchhoff

        status = "PASS" if verified else "FAIL"
        print(f"    Time: {elapsed:.3f}s")
        print(f"    Method: {result.method}")
        print(f"    Spanning trees: {tutte_trees}")
        print(f"    Kirchhoff: {kirchhoff}")
        print(f"    Status: {status}")

        return {
            'status': status,
            'time': elapsed,
            'method': result.method,
            'trees': tutte_trees,
            'verified': verified
        }

    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"    ERROR after {elapsed:.1f}s: {e}")
        return {'status': 'error', 'error': str(e)}


def main():
    print("=" * 60)
    print("ZEPHYR Z(m,t) SYNTHESIS TESTS")
    print("=" * 60)

    # Load default rainbow table (faster than building basic table)
    print("\nLoading default rainbow table...")
    start = time.perf_counter()
    table = load_default_table()
    elapsed = time.perf_counter() - start
    print(f"Loaded {len(table.entries)} entries in {elapsed:.2f}s")

    engine = SynthesisEngine(table=table, verbose=False)

    # Test cases: (m, t, max_time)
    test_cases = [
        (1, 1, 30),
        (1, 2, 60),
        (1, 3, 120),
        (1, 4, 300),
        (2, 1, 120),
        (2, 2, 300),
        (2, 3, 600),
        (2, 4, 900),
    ]

    results = []
    print("\n" + "-" * 60)

    for m, t, max_time in test_cases:
        result = test_zephyr(m, t, engine, max_time)
        if result:
            results.append({'name': f'Z({m},{t})', **result})
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.get('verified', False))
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    for r in results:
        status = r.get('status', 'unknown')
        time_s = r.get('time', 0)
        print(f"  {r['name']}: {status} ({time_s:.2f}s)")


if __name__ == "__main__":
    main()
