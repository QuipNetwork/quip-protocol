"""Benchmark for cotree DP on ALL graphs from every benchmark module.

Collects graphs from:
- Cotree DP targets (K_n, K_{a,b}, threshold graphs)
- Björklund vertex-exponential benchmark graphs
- Björklund hard graphs (engine timeouts)
- Standard test suite (test_tutte.py)
- Chord ordering benchmark graph types

For each graph: checks if cograph, runs cotree DP if yes,
validates against exact spanning tree count, reports non-cographs.

Usage:
    make benchmark-cotree
    uv run python -m tutte.tests.benchmark_cotree_dp
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import networkx as nx

from tutte.graph import (
    Graph, complete_graph, cycle_graph, path_graph,
    petersen_graph, wheel_graph, grid_graph,
)
from tutte.cotree_dp import is_cograph, compute_tutte_cotree_dp


# =============================================================================
# VALIDATION
# =============================================================================

def _spanning_tree_count(graph: Graph) -> int:
    """Exact spanning tree count (connected or disconnected)."""
    components = graph.connected_components()
    if len(components) > 1:
        result = 1
        for comp in components:
            result *= _spanning_tree_count_connected(comp)
        return result
    return _spanning_tree_count_connected(graph)


def _spanning_tree_count_connected(graph: Graph) -> int:
    """Exact spanning tree count for a connected graph via sympy."""
    if graph.node_count() <= 1:
        return 1

    try:
        from sympy import zeros  # type: ignore[import-unresolved]
    except ImportError:
        G = graph.to_networkx()
        return round(nx.number_of_spanning_trees(G))

    nodes = sorted(graph.nodes)
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    L = zeros(n, n)
    for u, v in graph.edges:
        i, j = idx[u], idx[v]
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1
    return int(L[1:, 1:].det())


def _exact_t11(poly) -> int:
    """Exact T(1,1) via integer coefficient sum."""
    return sum(poly.to_coefficients().values())


# =============================================================================
# GRAPH BUILDERS
# =============================================================================

def _random_graph(n: int, p: float, seed: int = 42) -> Graph:
    return Graph.from_networkx(nx.erdos_renyi_graph(n, p, seed=seed))


def _random_regular(n: int, d: int, seed: int = 42) -> Optional[Graph]:
    try:
        return Graph.from_networkx(nx.random_regular_graph(d, n, seed=seed))
    except nx.NetworkXError:
        return None


def _make_threshold(sequence: str) -> Graph:
    G = nx.Graph()
    G.add_node(0)
    for idx, op in enumerate(sequence, 1):
        G.add_node(idx)
        if op == 'd':
            for v in range(idx):
                G.add_edge(v, idx)
    return Graph.from_networkx(G)


def _build_all_graphs() -> List[Tuple[str, Graph]]:
    """Build ALL benchmark graphs from every module's test suite."""
    graphs = []

    def _add(name: str, g: Graph):
        graphs.append((name, g))

    def _try_add(name: str, g: Optional[Graph]):
        if g is not None:
            graphs.append((name, g))

    # ===== COTREE DP TARGETS (cographs) =====

    # Complete graphs K_3..K_20
    for n in range(3, 21):
        _add(f"K_{n}", complete_graph(n))

    # Complete bipartite K_{a,b}
    for a, b in [(2, 3), (2, 4), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
                 (8, 8), (9, 9), (10, 10), (6, 7), (6, 8), (7, 8),
                 (7, 9), (8, 9), (6, 10), (7, 10), (6, 12), (5, 14)]:
        _add(f"K_{{{a},{b}}}", Graph.from_networkx(nx.complete_bipartite_graph(a, b)))

    # Dense threshold graphs
    for seq, label in [
        ("ddd", "Thr_d3"),
        ("ddddd", "Thr_d5"),
        ("ddddddd", "Thr_d7"),
        ("dddddddddd", "Thr_d10"),
        ("ddi" * 4, "Thr_ddi4"),
        ("ddi" * 6, "Thr_ddi6"),
        ("dddi" * 4, "Thr_dddi4"),
        ("dddi" * 5, "Thr_dddi5"),
        ("d" * 10 + "i" * 5, "Thr_d10i5"),
        ("d" * 12, "Thr_d12"),
        ("d" * 15, "Thr_d15"),
        ("didi" * 3, "Thr_didi3"),
        ("ddid" * 3, "Thr_ddid3"),
        ("ddi", "Thr_ddi1"),
        ("ddid", "Thr_ddid1"),
    ]:
        _add(label, _make_threshold(seq))

    # ===== STANDARD TEST SUITE (test_tutte.py) =====

    _add("C_4", Graph.from_networkx(nx.cycle_graph(4)))
    _add("C_5", cycle_graph(5))
    _add("C_8", cycle_graph(8))
    _add("C_12", cycle_graph(12))
    _add("P_3", path_graph(3))
    _add("P_4", path_graph(4))
    _add("P_8", path_graph(8))
    _add("W_5", wheel_graph(5))
    _add("W_7", wheel_graph(7))
    _add("Petersen", petersen_graph())
    _add("Grid_3x3", grid_graph(3, 3))

    # ===== BJÖRKLUND BENCHMARK GRAPHS =====

    # Dense random G(n, p)
    for n in [10, 11, 12, 13, 14, 15, 16, 18]:
        for p in [0.5, 0.6, 0.7, 0.8]:
            _add(f"G({n},{p})", _random_graph(n, p, seed=42))

    # Dense random with different seeds
    for n in [12, 14, 16]:
        for seed in [7, 13, 29]:
            _add(f"G({n},0.6,s{seed})", _random_graph(n, 0.6, seed=seed))

    # Dense random, odd vertex counts
    for n in [11, 13, 15, 17]:
        for seed in [3, 17, 53]:
            _add(f"G({n},0.65,s{seed})", _random_graph(n, 0.65, seed=seed))

    # Random regular with high degree
    for n, d in [(10, 6), (10, 7), (12, 6), (12, 7), (14, 6), (14, 7),
                 (16, 5), (16, 6), (18, 5), (18, 6)]:
        _try_add(f"Reg({n},{d})", _random_regular(n, d, seed=42))

    # Barabási-Albert
    for n in [12, 14, 16, 18]:
        for m_attach in [4, 5, 6]:
            _add(f"BA({n},{m_attach})", Graph.from_networkx(
                nx.barabasi_albert_graph(n, m_attach, seed=42)))

    # Watts-Strogatz small-world
    for n in [12, 14, 16, 18]:
        for k, p in [(6, 0.3), (8, 0.5), (6, 0.7)]:
            if k < n:
                _add(f"WS({n},{k},{p})", Graph.from_networkx(
                    nx.watts_strogatz_graph(n, k, p, seed=42)))

    # Power-law cluster
    for n in [12, 14, 16]:
        for m_edges in [4, 5]:
            try:
                _add(f"PLC({n},{m_edges})", Graph.from_networkx(
                    nx.powerlaw_cluster_graph(n, m_edges, 0.3, seed=42)))
            except Exception:
                pass

    # ===== BJÖRKLUND HARD GRAPHS (engine timeouts) =====
    # These use different n/p/seed combinations not covered above

    _add("G(15,0.8)", _random_graph(15, 0.8, seed=42))
    _add("G(17,0.65,s17)", _random_graph(17, 0.65, seed=17))
    _add("G(17,0.65,s3)", _random_graph(17, 0.65, seed=3))
    _add("WS(18,8,0.5)", Graph.from_networkx(
        nx.watts_strogatz_graph(18, 8, 0.5, seed=42)))

    # ===== SPARSE / STRUCTURED (wheels, grids, low-degree regular) =====

    for n in [16, 20]:
        _add(f"W_{n}", wheel_graph(n))
    for r, c in [(4, 5), (5, 5), (4, 6)]:
        _add(f"Grid_{r}x{c}", grid_graph(r, c))
    for n, d in [(20, 3), (16, 4)]:
        _try_add(f"Reg({n},{d})", _random_regular(n, d, seed=42))

    # ===== CHORD ORDERING STYLE (random 2-connected) =====

    for m_target in [30, 50, 70]:
        n = max(int(2 * m_target / 3.5), 3)
        for seed in [42, 142]:
            G = nx.gnm_random_graph(n, m_target, seed=seed)
            if nx.is_connected(G):
                _add(f"Rand2c({n},{m_target},s{seed})",
                     Graph.from_networkx(G))

    # Deduplicate by name
    seen = set()
    unique = []
    for name, g in graphs:
        if name not in seen:
            seen.add(name)
            unique.append((name, g))

    return unique


# =============================================================================
# MAIN
# =============================================================================

def main():
    graphs = _build_all_graphs()

    print("Cotree DP Benchmark (all graph sources)")
    print(f"  Total graphs: {len(graphs)}")
    print()

    # Separate into cographs and non-cographs
    cograph_list = []
    non_cograph_list = []

    for name, g in graphs:
        try:
            is_cog = is_cograph(g)
        except TypeError:
            is_cog = False
        if is_cog:
            cograph_list.append((name, g))
        else:
            non_cograph_list.append((name, g.node_count(), g.edge_count()))

    print(f"  Cographs: {len(cograph_list)}")
    print(f"  Non-cographs: {len(non_cograph_list)}")
    print()

    # ===== COGRAPH RESULTS =====

    print("=== Cographs (cotree DP computed) ===")
    print(f"{'Graph':<20} {'n':>4} {'m':>5} {'Time':>10} {'Match':>6}")
    print("-" * 49)

    passed = 0
    failed = 0
    errors = 0
    total_time = 0.0

    for name, g in cograph_list:
        n = g.node_count()
        m = g.edge_count()

        t0 = time.perf_counter()
        try:
            poly = compute_tutte_cotree_dp(g)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"{name:<20} {n:>4} {m:>5} {elapsed:>9.2f}s ERROR: {e}")
            errors += 1
            continue

        total_time += elapsed
        t11 = _exact_t11(poly)
        kirchhoff = _spanning_tree_count(g)
        match = t11 == kirchhoff

        if match:
            passed += 1
        else:
            failed += 1

        print(f"{name:<20} {n:>4} {m:>5} {elapsed:>9.2f}s {'OK' if match else 'FAIL':>6}")
        if not match:
            print(f"  T(1,1)     = {t11}")
            print(f"  Kirchhoff  = {kirchhoff}")

    # ===== NON-COGRAPH LIST =====

    print()
    print("=== Non-cographs (skipped — not P₄-free) ===")
    print(f"{'Graph':<20} {'n':>4} {'m':>5}")
    print("-" * 33)
    for name, n, m in non_cograph_list:
        print(f"{name:<20} {n:>4} {m:>5}")

    # ===== SUMMARY =====

    print()
    print(f"Summary:")
    print(f"  Cographs computed: {passed + failed + errors}")
    print(f"    Passed:  {passed}")
    if failed:
        print(f"    Failed:  {failed}")
    if errors:
        print(f"    Errors:  {errors}")
    print(f"  Non-cographs skipped: {len(non_cograph_list)}")
    print(f"  Total cotree DP time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
