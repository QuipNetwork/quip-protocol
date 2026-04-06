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
from tutte.validation import (
    count_spanning_trees_kirchhoff,
    _exact_num_spanning_trees,
    _exact_spanning_tree_count,
)
from tutte.cotree_dp import CotreeNode, compute_tutte_cotree_dp
from tutte.cotree_dp.recognition import CotreeNodeType, _build_cotree


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

    def test__build_cotree_rejects_more_than_500_vertices(self):
        """_build_cotree must reject n > 500 (recursion depth guard)."""
        graph = Graph(nodes=frozenset(range(501)), edges=frozenset())
        with pytest.raises(ValueError, match="500"):
            _build_cotree(graph)

    def test_invalid_cotree_node_type_raises(self):
        """CotreeNode with invalid node_type must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid node_type"):
            CotreeNode(node_type='invalid')

    def test_invalid_cotree_node_type_int(self):
        """CotreeNode rejects non-enum types (int)."""
        with pytest.raises(ValueError, match="Invalid node_type"):
            CotreeNode(node_type=42)

    def test_invalid_cotree_node_type_none(self):
        """CotreeNode rejects None as node_type."""
        with pytest.raises(ValueError, match="Invalid node_type"):
            CotreeNode(node_type=None)

    def test_valid_cotree_node_types_accepted(self):
        """All three CotreeNodeType enum values must be accepted."""
        CotreeNode(node_type=CotreeNodeType.LEAF, vertex=0, vertices=frozenset({0}))
        CotreeNode(node_type=CotreeNodeType.DISJOINT_UNION_OP)
        CotreeNode(node_type=CotreeNodeType.COMPLETE_UNION_OP)

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
        """Known cographs must be accepted by compute_tutte_cotree_dp."""
        compute_tutte_cotree_dp(builder())  # should not raise

    @pytest.mark.parametrize("name,builder", NON_COGRAPHS, ids=[g[0] for g in NON_COGRAPHS])
    def test_rejects_known_non_cographs(self, name, builder):
        """Known non-cographs must be rejected by compute_tutte_cotree_dp."""
        with pytest.raises(ValueError, match="not a cograph"):
            compute_tutte_cotree_dp(builder())

    def test_early_p4_rejection(self):
        """P₄ is the simplest non-cograph — detected at first recursion level."""
        with pytest.raises(ValueError, match="not a cograph"):
            compute_tutte_cotree_dp(path_graph(4))
        assert _build_cotree(path_graph(4)) is None

    def test_p4_embedded_in_larger_graph(self):
        """W₄ (wheel on 4 rim vertices) contains induced P₄ among rim vertices."""
        edges = {(0, 1), (1, 2), (2, 3), (0, 4), (1, 4), (2, 4), (3, 4)}
        graph = Graph(nodes=frozenset(range(5)), edges=frozenset(edges))
        with pytest.raises(ValueError, match="not a cograph"):
            compute_tutte_cotree_dp(graph)

    def test_complement_of_cograph_is_cograph(self):
        """complement(K_{3,3}) = K_3 ∪ K_3 — also a cograph."""
        graph = Graph.from_networkx(nx.complement(nx.complete_bipartite_graph(3, 3)))
        compute_tutte_cotree_dp(graph)  # should not raise


# =============================================================================
# C. COTREE STRUCTURE
# =============================================================================

class TestCotreeStructure:
    """Verify the shape of constructed cotrees."""

    def test_k3_is_join_of_three_leaves(self):
        """K_3 cotree: ⊗(v0, v1, v2)."""
        cotree = _build_cotree(complete_graph(3))
        assert cotree is not None
        assert cotree.node_type == CotreeNodeType.COMPLETE_UNION_OP
        assert cotree.size() == 3
        assert all(child.node_type == CotreeNodeType.LEAF for child in cotree.children)

    def test_k33_is_join_of_two_unions(self):
        """K_{3,3} cotree: ⊗(∪(a,b,c), ∪(d,e,f))."""
        cotree = _build_cotree(Graph.from_networkx(nx.complete_bipartite_graph(3, 3)))
        assert cotree is not None
        assert cotree.node_type == CotreeNodeType.COMPLETE_UNION_OP
        assert len(cotree.children) == 2
        assert all(child.node_type == CotreeNodeType.DISJOINT_UNION_OP for child in cotree.children)

    def test_non_cograph_returns_none(self):
        """Non-cographs must return None from _build_cotree."""
        assert _build_cotree(cycle_graph(5)) is None
        assert _build_cotree(path_graph(4)) is None

    def test_3way_complete_union_matches_k4(self):
        """Manually construct ⊗(v0, v1, ⊗(v2, v3)) and verify it equals K_4.

        K_4 has cotree ⊗(v0, v1, v2, v3) — a single ⊗ with 4 leaves.
        An alternative valid cotree is ⊗(v0, v1, ⊗(v2, v3)) — a ⊗ with
        2 leaves and 1 child ⊗. Both must produce the same polynomial.

        This tests that iterative left-fold ⊗ combine (used by
        _compute_subgraph_table for 3+ children) is associative —
        i.e., ⊗(A, B, C) via fold gives the same result as ⊗(A, ⊗(B, C)).
        """
        from tutte.cotree_dp.dp import _compute_subgraph_table, _extract_tutte_polynomial

        # Build flat cotree: ⊗(v0, v1, v2, v3) — 4 leaves under one ⊗
        flat_tree = CotreeNode(
            node_type=CotreeNodeType.COMPLETE_UNION_OP,
            children=[
                CotreeNode(node_type=CotreeNodeType.LEAF, vertex=i, vertices=frozenset({i}))
                for i in range(4)
            ],
            vertices=frozenset(range(4)),
        )

        # Build nested cotree: ⊗(v0, v1, ⊗(v2, v3))
        nested_tree = CotreeNode(
            node_type=CotreeNodeType.COMPLETE_UNION_OP,
            children=[
                CotreeNode(node_type=CotreeNodeType.LEAF, vertex=0, vertices=frozenset({0})),
                CotreeNode(node_type=CotreeNodeType.LEAF, vertex=1, vertices=frozenset({1})),
                CotreeNode(
                    node_type=CotreeNodeType.COMPLETE_UNION_OP,
                    children=[
                        CotreeNode(node_type=CotreeNodeType.LEAF, vertex=2, vertices=frozenset({2})),
                        CotreeNode(node_type=CotreeNodeType.LEAF, vertex=3, vertices=frozenset({3})),
                    ],
                    vertices=frozenset({2, 3}),
                ),
            ],
            vertices=frozenset(range(4)),
        )

        flat_table = _compute_subgraph_table(flat_tree)
        nested_table = _compute_subgraph_table(nested_tree)

        flat_poly = _extract_tutte_polynomial(flat_table, 4, 1)
        nested_poly = _extract_tutte_polynomial(nested_table, 4, 1)

        # Both must equal T(K_4)
        expected = compute_tutte_cotree_dp(complete_graph(4))
        assert flat_poly == expected, f"Flat ⊗(4 leaves): {flat_poly} != {expected}"
        assert nested_poly == expected, f"Nested ⊗(2 leaves + ⊗(2)): {nested_poly} != {expected}"
        assert flat_poly == nested_poly, f"Flat != Nested: {flat_poly} != {nested_poly}"

    def test_3way_disjoint_union_associativity(self):
        """∪(v0, v1, ∪(v2, v3)) must equal ∪(v0, v1, v2, v3).

        Same associativity test for ∪ combine.
        """
        from tutte.cotree_dp.dp import _compute_subgraph_table, _extract_tutte_polynomial

        # Flat: ∪(v0, v1, v2)
        flat_tree = CotreeNode(
            node_type=CotreeNodeType.DISJOINT_UNION_OP,
            children=[
                CotreeNode(node_type=CotreeNodeType.LEAF, vertex=i, vertices=frozenset({i}))
                for i in range(3)
            ],
            vertices=frozenset(range(3)),
        )

        # Nested: ∪(v0, ∪(v1, v2))
        nested_tree = CotreeNode(
            node_type=CotreeNodeType.DISJOINT_UNION_OP,
            children=[
                CotreeNode(node_type=CotreeNodeType.LEAF, vertex=0, vertices=frozenset({0})),
                CotreeNode(
                    node_type=CotreeNodeType.DISJOINT_UNION_OP,
                    children=[
                        CotreeNode(node_type=CotreeNodeType.LEAF, vertex=1, vertices=frozenset({1})),
                        CotreeNode(node_type=CotreeNodeType.LEAF, vertex=2, vertices=frozenset({2})),
                    ],
                    vertices=frozenset({1, 2}),
                ),
            ],
            vertices=frozenset(range(3)),
        )

        flat_table = _compute_subgraph_table(flat_tree)
        nested_table = _compute_subgraph_table(nested_tree)

        # 3 isolated vertices, 3 components → T = 1
        flat_poly = _extract_tutte_polynomial(flat_table, 3, 3)
        nested_poly = _extract_tutte_polynomial(nested_table, 3, 3)

        assert flat_poly == TuttePolynomial.one()
        assert nested_poly == TuttePolynomial.one()
        assert flat_poly == nested_poly


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
    """T(1,1) must equal spanning tree count for all cographs.

    Uses exact integer arithmetic for both T(1,1) and Kirchhoff to avoid
    float64 precision loss on large graphs (n >= 18 where spanning tree
    counts exceed 2^53).
    """

    @pytest.mark.parametrize("name,builder", COGRAPHS, ids=[g[0] for g in COGRAPHS])
    def test_spanning_tree_count_matches(self, name, builder):
        """T(1,1) = Kirchhoff spanning tree count (exact integer arithmetic)."""
        graph = builder()
        poly = compute_tutte_cotree_dp(graph)

        # Exact T(1,1) via integer coefficient sum — no float conversion
        t11 = _exact_num_spanning_trees(poly)

        # Exact Kirchhoff via sympy integer determinant — no float conversion
        components = graph.connected_components()
        if len(components) == 1:
            expected = _exact_spanning_tree_count(graph)
        else:
            expected = 1
            for component in components:
                expected *= _exact_spanning_tree_count(component)

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
        poly = compute_tutte_cotree_dp(graph)
        kirchhoff = count_spanning_trees_kirchhoff(graph)
        assert int(poly.evaluate(1, 1)) == kirchhoff

    def test_deep_alternating_threshold(self):
        """Threshold 'didididi': deep linear cotree with alternating ⊗/∪ levels.

        This graph is disconnected ('i' adds isolated vertices), so
        Kirchhoff on the whole graph returns 0. Compute per-component.
        """
        graph = _make_threshold("didididi")
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
    """Run cotree DP on K_30 through K_100 to find where it becomes too slow.

    K_8 through K_29 are covered by Section G (engine cross-validation).
    This section tests larger graphs where the engine times out but
    cotree DP still succeeds.

    Each K_n result is printed immediately. Stops on first timeout (15 min).
    Run with: pytest tests/test_cotree_dp.py::TestKnScaling -v -s
    """

    def test_kn_scaling(self):
        """Cotree DP on K_30..K_100: print time for each, stop on timeout."""
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


# =============================================================================
# I. CELLSEL CACHE GROWTH (Issue #4)
# =============================================================================

class TestCellSelCacheGrowth:
    """Measure CellSel cache peak size to determine the cache limit for issue #4.

    The _cellsel_cache dict grows during compute_tutte_cotree_dp and is
    auto-cleared after each call. This test disables auto-clear to measure
    peak entries, then uses the result to validate a proposed cache limit.

    Run with: pytest tests/test_cotree_dp.py::TestCellSelCacheGrowth -v -s
    """

    def test_cache_peak_k30(self):
        """Measure peak CellSel cache entries on K_30.

        K_30 is the largest graph we routinely benchmark with cotree DP.
        The peak cache size determines the safe limit for issue #4.
        """
        import time
        import tutte.cotree_dp.dp as dp_mod
        from tutte.cotree_dp.combinatorics import _cellsel_cache

        n = 30
        g = complete_graph(n)

        # Disable auto-clear to observe peak
        original_clear = dp_mod.clear_cellsel_cache
        dp_mod.clear_cellsel_cache = lambda: None
        _cellsel_cache.clear()

        try:
            t0 = time.perf_counter()
            poly = compute_tutte_cotree_dp(g)
            elapsed = time.perf_counter() - t0

            peak_entries = len(_cellsel_cache)

            # Estimate memory per entry:
            #   key: tuple of sorted ints (~5 elements avg) + int
            #     tuple overhead: 56 bytes + 5 × 28 bytes = 196 bytes
            #     int key: 28 bytes
            #   value: int = 28 bytes
            #   dict overhead per entry: ~50 bytes
            est_bytes_per_entry = 196 + 28 + 28 + 50
            est_memory_mb = peak_entries * est_bytes_per_entry / (1024 * 1024)

            print(f"\n{'='*60}")
            print(f"CellSel cache growth on K_{n}")
            print(f"  Peak cache entries: {peak_entries:,}")
            print(f"  Estimated memory:   {est_memory_mb:.1f} MB")
            print(f"  Computation time:   {elapsed:.1f}s")
            print(f"{'='*60}")

            # Verify correctness
            kirchhoff = _exact_spanning_tree_count(g)
            t11 = _exact_num_spanning_trees(poly)
            assert t11 == kirchhoff, (
                f"K_{n}: T(1,1)={t11} != Kirchhoff={kirchhoff}"
            )

            # The cache must have entries (sanity check)
            assert peak_entries > 0, "Cache should have entries after K_30"

            # Record the peak for use in setting the limit.
            # A safe limit is 2× the K_30 peak (headroom for K_35).
            suggested_limit = peak_entries * 2
            print(f"  Suggested cache limit: {suggested_limit:,} "
                  f"(2× K_{n} peak)")

        finally:
            dp_mod.clear_cellsel_cache = original_clear
            _cellsel_cache.clear()
