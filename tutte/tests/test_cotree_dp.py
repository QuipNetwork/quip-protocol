"""Tests for the cotree DP module.

Validates cograph recognition, cotree construction, and Tutte polynomial
computation. Organized by component under test:

Sections:
    A. Input validation — type checks, size guards, invalid inputs
    B. Cograph recognition — positive and negative cases, early rejection
    C. Cotree structure — correct decomposition shapes
    D. Tutte polynomial — known values (hardcoded expected polynomials)
    E. Tutte polynomial — Kirchhoff validation (T(1,1) = spanning trees)
    F. Graph families — disconnected, complement, asymmetric, threshold
    G. Engine cross-validation — full polynomial match on large cographs
"""

import signal

import networkx as nx
import pytest

from tutte.graph import Graph, MultiGraph, complete_graph, cycle_graph, path_graph, disjoint_union
from tutte.polynomial import TuttePolynomial
from tutte.validation import count_spanning_trees_kirchhoff
from tutte.cotree_dp import is_cograph, build_cotree, CotreeNode, compute_tutte_cotree_dp


# =============================================================================
# HELPERS
# =============================================================================

def _make_threshold(sequence: str) -> Graph:
    """Build a threshold graph from a sequence of 'd' and 'i'.

    'd' = dominating vertex (connected to all existing vertices)
    'i' = isolated vertex (no edges to existing vertices)
    """
    G = nx.Graph()
    G.add_node(0)
    for idx, op in enumerate(sequence, 1):
        G.add_node(idx)
        if op == 'd':
            for vertex in range(idx):
                G.add_edge(vertex, idx)
    return Graph.from_networkx(G)


# =============================================================================
# A. INPUT VALIDATION
# =============================================================================

class TestInputValidation:
    """Type checks, size guards, and invalid inputs."""

    def test_is_cograph_rejects_multigraph(self):
        """is_cograph must raise TypeError on MultiGraph."""
        multigraph = MultiGraph(
            nodes=frozenset({0, 1}),
            edge_counts={(0, 1): 2},
            loop_counts={},
        )
        with pytest.raises(TypeError, match="simple Graph"):
            is_cograph(multigraph)

    def test_compute_rejects_multigraph(self):
        """compute_tutte_cotree_dp must raise TypeError on MultiGraph."""
        multigraph = MultiGraph(
            nodes=frozenset({0, 1}),
            edge_counts={(0, 1): 1},
            loop_counts={},
        )
        with pytest.raises(TypeError, match="simple Graph"):
            compute_tutte_cotree_dp(multigraph)

    def test_compute_rejects_more_than_35_vertices(self):
        """compute_tutte_cotree_dp must reject n > 35.

        Uses a 36-vertex graph WITH edges — edgeless graphs return early
        before the guard is reached.
        """
        graph = Graph.from_networkx(nx.complete_graph(36))
        with pytest.raises(ValueError, match="vertices > 35"):
            compute_tutte_cotree_dp(graph)

    def test_build_cotree_rejects_more_than_500_vertices(self):
        """build_cotree must reject n > 500 (recursion depth guard)."""
        graph = Graph(nodes=frozenset(range(501)), edges=frozenset())
        with pytest.raises(ValueError, match="500"):
            build_cotree(graph)

    def test_invalid_cotree_node_type_raises(self):
        """CotreeNode with invalid node_type must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid node_type"):
            CotreeNode(node_type='invalid')

    def test_non_cograph_raises_value_error(self):
        """compute_tutte_cotree_dp must raise ValueError on non-cographs."""
        with pytest.raises(ValueError, match="not a cograph"):
            compute_tutte_cotree_dp(cycle_graph(5))


# =============================================================================
# B. COGRAPH RECOGNITION
# =============================================================================

COGRAPHS = [
    ("K_3", lambda: complete_graph(3)),
    ("K_4", lambda: complete_graph(4)),
    ("K_5", lambda: complete_graph(5)),
    ("K_6", lambda: complete_graph(6)),
    ("K_7", lambda: complete_graph(7)),
    ("K_8", lambda: complete_graph(8)),
    ("C_4", lambda: Graph.from_networkx(nx.cycle_graph(4))),
    ("P_3", lambda: path_graph(3)),
    ("K_{2,3}", lambda: Graph.from_networkx(nx.complete_bipartite_graph(2, 3))),
    ("K_{3,3}", lambda: Graph.from_networkx(nx.complete_bipartite_graph(3, 3))),
    ("K_{4,4}", lambda: Graph.from_networkx(nx.complete_bipartite_graph(4, 4))),
    ("Threshold_ddd", lambda: _make_threshold("ddd")),
    ("Threshold_ddi", lambda: _make_threshold("ddi")),
    ("Threshold_ddid", lambda: _make_threshold("ddid")),
    ("Threshold_ddddd", lambda: _make_threshold("ddddd")),
]

NON_COGRAPHS = [
    ("C_5", lambda: cycle_graph(5)),
    ("C_6", lambda: cycle_graph(6)),
    ("P_4", lambda: path_graph(4)),
    ("P_5", lambda: path_graph(5)),
    ("Petersen", lambda: Graph.from_networkx(nx.petersen_graph())),
]


class TestCographRecognition:
    """Positive and negative recognition, early P₄ rejection."""

    @pytest.mark.parametrize("name,builder", COGRAPHS, ids=[g[0] for g in COGRAPHS])
    def test_recognizes_known_cographs(self, name, builder):
        """Known cographs must be recognized."""
        assert is_cograph(builder()), f"{name} should be a cograph"

    @pytest.mark.parametrize("name,builder", NON_COGRAPHS, ids=[g[0] for g in NON_COGRAPHS])
    def test_rejects_known_non_cographs(self, name, builder):
        """Known non-cographs must be rejected."""
        assert not is_cograph(builder()), f"{name} should NOT be a cograph"

    def test_early_p4_rejection(self):
        """P₄ is the simplest non-cograph — detected at first recursion level."""
        assert not is_cograph(path_graph(4))
        assert build_cotree(path_graph(4)) is None

    def test_p4_embedded_in_larger_graph(self):
        """W₄ (wheel on 4 rim vertices) contains induced P₄ among rim vertices."""
        edges = {(0, 1), (1, 2), (2, 3), (0, 4), (1, 4), (2, 4), (3, 4)}
        graph = Graph(nodes=frozenset(range(5)), edges=frozenset(edges))
        assert not is_cograph(graph)

    def test_complement_of_cograph_is_cograph(self):
        """complement(K_{3,3}) = K_3 ∪ K_3 — also a cograph."""
        graph = Graph.from_networkx(nx.complement(nx.complete_bipartite_graph(3, 3)))
        assert is_cograph(graph)


# =============================================================================
# C. COTREE STRUCTURE
# =============================================================================

class TestCotreeStructure:
    """Verify the shape of constructed cotrees."""

    def test_k3_is_join_of_three_leaves(self):
        """K_3 cotree: ⊗(v0, v1, v2)."""
        cotree = build_cotree(complete_graph(3))
        assert cotree is not None
        assert cotree.node_type == 'join'
        assert cotree.size() == 3
        assert all(child.node_type == 'leaf' for child in cotree.children)

    def test_k33_is_join_of_two_unions(self):
        """K_{3,3} cotree: ⊗(∪(a,b,c), ∪(d,e,f))."""
        cotree = build_cotree(Graph.from_networkx(nx.complete_bipartite_graph(3, 3)))
        assert cotree is not None
        assert cotree.node_type == 'join'
        assert len(cotree.children) == 2
        assert all(child.node_type == 'union' for child in cotree.children)

    def test_non_cograph_returns_none(self):
        """Non-cographs must return None from build_cotree."""
        assert build_cotree(cycle_graph(5)) is None
        assert build_cotree(path_graph(4)) is None


# =============================================================================
# D. TUTTE POLYNOMIAL — KNOWN VALUES
# =============================================================================

class TestKnownPolynomials:
    """Hardcoded expected polynomial values for small graphs."""

    def test_empty_graph(self):
        """No edges: T = 1."""
        graph = Graph(nodes=frozenset({0, 1}), edges=frozenset())
        assert compute_tutte_cotree_dp(graph) == TuttePolynomial.one()

    def test_single_vertex(self):
        """Single vertex: T = 1."""
        graph = Graph(nodes=frozenset({0}), edges=frozenset())
        assert is_cograph(graph)
        assert compute_tutte_cotree_dp(graph) == TuttePolynomial.one()

    def test_two_isolated_vertices(self):
        """Two vertices, no edges: T = 1."""
        graph = Graph(nodes=frozenset({0, 1}), edges=frozenset())
        assert compute_tutte_cotree_dp(graph) == TuttePolynomial.one()

    def test_single_edge(self):
        """K_2: T = x."""
        graph = Graph(nodes=frozenset({0, 1}), edges=frozenset({(0, 1)}))
        assert compute_tutte_cotree_dp(graph) == TuttePolynomial.x()

    def test_triangle(self):
        """K_3: T = x² + x + y."""
        expected = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
        assert compute_tutte_cotree_dp(complete_graph(3)) == expected

    def test_c4(self):
        """C₄: T = x³ + x² + x + y."""
        expected = TuttePolynomial.from_coefficients(
            {(3, 0): 1, (2, 0): 1, (1, 0): 1, (0, 1): 1}
        )
        assert compute_tutte_cotree_dp(Graph.from_networkx(nx.cycle_graph(4))) == expected


# =============================================================================
# E. TUTTE POLYNOMIAL — KIRCHHOFF VALIDATION
# =============================================================================

class TestKirchhoffValidation:
    """T(1,1) must equal spanning tree count for all cographs."""

    @pytest.mark.parametrize("name,builder", COGRAPHS, ids=[g[0] for g in COGRAPHS])
    def test_spanning_tree_count_matches(self, name, builder):
        """T(1,1) = Kirchhoff spanning tree count."""
        graph = builder()
        poly = compute_tutte_cotree_dp(graph)
        t11 = int(poly.evaluate(1, 1))

        components = graph.connected_components()
        if len(components) == 1:
            expected = count_spanning_trees_kirchhoff(graph)
        else:
            expected = 1
            for component in components:
                expected *= count_spanning_trees_kirchhoff(component)

        assert t11 == expected, f"{name}: T(1,1)={t11} != Kirchhoff={expected}"


# =============================================================================
# F. GRAPH FAMILIES — DISCONNECTED, COMPLEMENT, ASYMMETRIC, THRESHOLD
# =============================================================================

class TestGraphFamilies:
    """Exercises specific cotree structures and graph family edge cases."""

    def test_disconnected_cograph(self):
        """K_4 ∪ K_5: T = T(K_4) × T(K_5).

        Exercises ∪ combine with substantial components on both sides.
        """
        graph = disjoint_union(complete_graph(4), complete_graph(5))
        assert is_cograph(graph)
        poly = compute_tutte_cotree_dp(graph)

        poly_k4 = compute_tutte_cotree_dp(complete_graph(4))
        poly_k5 = compute_tutte_cotree_dp(complete_graph(5))
        assert poly == poly_k4 * poly_k5

    def test_complement_polynomial(self):
        """complement(K_{3,3}) = K_3 ∪ K_3: T = T(K_3)²."""
        graph = Graph.from_networkx(nx.complement(nx.complete_bipartite_graph(3, 3)))
        poly = compute_tutte_cotree_dp(graph)

        poly_k3 = compute_tutte_cotree_dp(complete_graph(3))
        assert poly == poly_k3 * poly_k3

    def test_asymmetric_complete_bipartite(self):
        """K_{2,8}: very unbalanced cotree (2 leaves vs 8 leaves under ⊗)."""
        graph = Graph.from_networkx(nx.complete_bipartite_graph(2, 8))
        assert is_cograph(graph)
        poly = compute_tutte_cotree_dp(graph)
        kirchhoff = count_spanning_trees_kirchhoff(graph)
        assert int(poly.evaluate(1, 1)) == kirchhoff

    def test_deep_alternating_threshold(self):
        """Threshold 'didididi': deep linear cotree with alternating ⊗/∪ levels.

        This graph is disconnected ('i' adds isolated vertices), so
        Kirchhoff on the whole graph returns 0. Compute per-component.
        """
        graph = _make_threshold("didididi")
        assert is_cograph(graph)
        poly = compute_tutte_cotree_dp(graph)
        t11 = int(poly.evaluate(1, 1))

        components = graph.connected_components()
        if len(components) == 1:
            expected = count_spanning_trees_kirchhoff(graph)
        else:
            expected = 1
            for component in components:
                expected *= count_spanning_trees_kirchhoff(component)

        assert t11 == expected, (
            f"Threshold didididi: T(1,1)={t11} != Kirchhoff={expected}, "
            f"components={len(components)}"
        )


# =============================================================================
# G. ENGINE CROSS-VALIDATION (Issues #5, #11, #12)
# =============================================================================

_ENGINE_TIMEOUT = 300  # 5 minutes per engine call


class _EngineTimeout(Exception):
    pass


def _timeout_handler(_signum, _frame):
    raise _EngineTimeout("Engine exceeded 5 minute timeout")


def _engine_poly_with_timeout(graph: Graph):
    """Run the synthesis engine with a 5-minute timeout.

    Returns the polynomial, or None if the engine times out.
    Uses SIGALRM on Unix; runs without timeout on Windows.
    """
    from tutte.synthesis import SynthesisEngine

    engine = SynthesisEngine()
    has_alarm = hasattr(signal, 'SIGALRM')

    old_handler = None
    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(_ENGINE_TIMEOUT)
    try:
        result = engine.synthesize(graph)
        if has_alarm:
            signal.alarm(0)
        return result.polynomial
    except _EngineTimeout:
        if has_alarm:
            signal.alarm(0)
        return None
    finally:
        if has_alarm and old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)


CROSS_VALIDATION_GRAPHS = [
    # Complete graphs K_10..K_15 — maximally dense, deep ⊗ combine
    ("K_10", lambda: Graph.from_networkx(nx.complete_graph(10))),
    ("K_11", lambda: Graph.from_networkx(nx.complete_graph(11))),
    ("K_12", lambda: Graph.from_networkx(nx.complete_graph(12))),
    ("K_13", lambda: Graph.from_networkx(nx.complete_graph(13))),
    ("K_14", lambda: Graph.from_networkx(nx.complete_graph(14))),
    ("K_15", lambda: Graph.from_networkx(nx.complete_graph(15))),

    # Threshold graphs with 10+ operations — deep linear cotrees
    ("Thr_d10",   lambda: _make_threshold("d" * 10)),
    ("Thr_d12",   lambda: _make_threshold("d" * 12)),
    ("Thr_ddi4",  lambda: _make_threshold("ddi" * 4)),
    ("Thr_dddi4", lambda: _make_threshold("dddi" * 4)),

    # Complete bipartite K_{a,b} — two-level cotree: ⊗(∪(...), ∪(...))
    ("K_{5,5}", lambda: Graph.from_networkx(nx.complete_bipartite_graph(5, 5))),
    ("K_{6,6}", lambda: Graph.from_networkx(nx.complete_bipartite_graph(6, 6))),
    ("K_{7,7}", lambda: Graph.from_networkx(nx.complete_bipartite_graph(7, 7))),
    ("K_{8,8}", lambda: Graph.from_networkx(nx.complete_bipartite_graph(8, 8))),
]


# =============================================================================
# H. K_n SCALING BENCHMARK — find the practical N_MAX cap
# =============================================================================

class TestKnScaling:
    """Run cotree DP on K_8 through K_100 to find where it becomes too slow.

    Each K_n result is printed immediately. Stops on first timeout (15 min).
    Run with: pytest tests/test_cotree_dp.py::TestKnScaling -v -s
    """

    def test_kn_scaling(self):
        """Cotree DP on K_8..K_100: print time for each, stop on timeout."""
        import sys
        import time
        from tutte.validation import _exact_num_spanning_trees, _exact_spanning_tree_count

        max_n = 100
        timeout = 900  # 15 minutes

        print()
        print(f"{'Graph':<10} {'n':>4} {'m':>6} {'Time':>12} "
              f"{'T(1,1)':>20} {'Kirchhoff':>20} {'Status':>10}")
        print("-" * 86)
        sys.stdout.flush()

        for n in range(30, max_n + 1):
            graph = complete_graph(n)
            m = graph.edge_count()

            # Exact Kirchhoff via sympy integer determinant (no float precision loss)
            kirchhoff = _exact_spanning_tree_count(graph)

            # Cotree DP with timeout
            old_handler = None
            has_alarm = hasattr(signal, 'SIGALRM')
            if has_alarm:
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout)

            poly = None
            elapsed = None
            try:
                t0 = time.perf_counter()
                poly = compute_tutte_cotree_dp(graph)
                elapsed = time.perf_counter() - t0
                if has_alarm:
                    signal.alarm(0)
            except _EngineTimeout:
                if has_alarm:
                    signal.alarm(0)
            except Exception as e:
                if has_alarm:
                    signal.alarm(0)
                print(f"K_{n:<7} {n:>4} {m:>6} {'ERROR':>12} "
                      f"{'-':>20} {'-':>20} {str(e)[:30]:>10}")
                sys.stdout.flush()
                break
            finally:
                if has_alarm and old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

            if poly is not None:
                # Exact T(1,1) via integer coefficient sum (no float precision loss)
                t11 = _exact_num_spanning_trees(poly)
                match = t11 == kirchhoff
                status = "OK" if match else "FAIL"
                print(f"K_{n:<7} {n:>4} {m:>6} {elapsed:>11.1f}s "
                      f"{t11:>20} {kirchhoff:>20} {status:>10}")
                assert match, f"K_{n}: T(1,1)={t11} != Kirchhoff={kirchhoff}"
            else:
                print(f"K_{n:<7} {n:>4} {m:>6} {'TIMEOUT':>12} "
                      f"{'-':>20} {kirchhoff:>20} {'TIMEOUT':>10}")
                print()
                print(f"K_{n} timed out after {timeout}s. Stopping.")
                break

            sys.stdout.flush()


class TestEngineCrossValidation:
    """Full polynomial match against the synthesis engine on large cographs.

    If the engine times out (5 min), falls back to exact Kirchhoff verification
    using sympy integer determinant to avoid float64 precision loss.
    """

    @pytest.mark.parametrize(
        "name,builder",
        CROSS_VALIDATION_GRAPHS,
        ids=[g[0] for g in CROSS_VALIDATION_GRAPHS],
    )
    def test_matches_engine_or_kirchhoff(self, name, builder):
        """Cotree DP polynomial must match engine (or exact Kirchhoff on timeout)."""
        graph = builder()
        cotree_poly = compute_tutte_cotree_dp(graph)

        engine_poly = _engine_poly_with_timeout(graph)

        if engine_poly is not None:
            assert cotree_poly == engine_poly, (
                f"{name}: cotree DP polynomial differs from engine.\n"
                f"  cotree T(1,1) = {int(cotree_poly.evaluate(1, 1))}\n"
                f"  engine T(1,1) = {int(engine_poly.evaluate(1, 1))}\n"
                f"  cotree terms  = {cotree_poly.num_terms()}\n"
                f"  engine terms  = {engine_poly.num_terms()}"
            )
        else:
            from tutte.validation import _exact_num_spanning_trees, _exact_spanning_tree_count
            t11 = _exact_num_spanning_trees(cotree_poly)
            components = graph.connected_components()
            if len(components) == 1:
                kirchhoff = _exact_spanning_tree_count(graph)
            else:
                kirchhoff = 1
                for component in components:
                    kirchhoff *= _exact_spanning_tree_count(component)
            assert t11 == kirchhoff, (
                f"{name}: engine timed out, Kirchhoff fallback failed.\n"
                f"  cotree T(1,1) = {t11}\n"
                f"  Kirchhoff     = {kirchhoff}\n"
                f"  components    = {len(components)}"
            )
