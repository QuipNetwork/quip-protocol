"""Integration tests for cotree DP in the synthesis pipeline.

Validates that the cotree DP module is correctly wired into
SynthesisEngine.synthesize() as step 9b (after treewidth DP, before k-sum).

The inner loop (base.py) is NOT modified -- chord addition produces
multigraphs that are caught by treewidth DP or series-parallel after
batch-parallel reduction. Cographs are caught at the outer loop before
CEJ ever starts.

The hybrid engine (hybrid.py) is NOT modified -- it is a secondary engine
used only in a few matroid tests. Cographs entering it still compute
correctly via the tiling fallback.

Sections:
    A. Engine produces correct polynomials for cographs
    B. Engine falls through correctly for non-cographs
    C. SynthesisResult metadata (method, recipe, verified)
    D. Event log contains COTREE_DP events
    E. Cache and rainbow table promotion
    F. Kirchhoff validation on engine results for cographs
    G. No regressions on non-cograph graphs
"""

from __future__ import annotations

import networkx as nx
import pytest

from tutte.graph import Graph, complete_graph, cycle_graph, path_graph
from tutte.polynomial import TuttePolynomial
from tutte.synthesis.engine import SynthesisEngine
from tutte.cotree_dp import compute_tutte_cotree_dp
from tutte.validation import (
    _exact_num_spanning_trees,
    _exact_spanning_tree_count,
)
from tutte.lookup.core import RainbowTable
from tutte.logs import get_log, reset_log, EventType


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


def _empty_table_engine() -> SynthesisEngine:
    """Create a SynthesisEngine with an empty rainbow table.

    Uses RainbowTable() (no precomputed entries) so that no test graph
    can be resolved by rainbow table lookup at step 3. This ensures
    that graphs must reach the cotree DP step to be caught.
    """
    return SynthesisEngine(table=RainbowTable())


# =============================================================================
# A. ENGINE PRODUCES CORRECT POLYNOMIALS FOR COGRAPHS
# =============================================================================

COGRAPH_VALIDATION = [
    ("K_3", lambda: complete_graph(3)),
    ("K_4", lambda: complete_graph(4)),
    ("K_5", lambda: complete_graph(5)),
    ("K_6", lambda: complete_graph(6)),
    ("K_8", lambda: complete_graph(8)),
    ("K_10", lambda: Graph.from_networkx(nx.complete_graph(10))),
    ("K_12", lambda: Graph.from_networkx(nx.complete_graph(12))),
    ("K_14", lambda: Graph.from_networkx(nx.complete_graph(14))),
    ("K_{3,3}", lambda: Graph.from_networkx(nx.complete_bipartite_graph(3, 3))),
    ("K_{4,4}", lambda: Graph.from_networkx(nx.complete_bipartite_graph(4, 4))),
    ("K_{5,5}", lambda: Graph.from_networkx(nx.complete_bipartite_graph(5, 5))),
    ("K_{6,6}", lambda: Graph.from_networkx(nx.complete_bipartite_graph(6, 6))),
    ("K_{7,7}", lambda: Graph.from_networkx(nx.complete_bipartite_graph(7, 7))),
    ("Threshold_ddd", lambda: _make_threshold("ddd")),
    ("Threshold_ddddd", lambda: _make_threshold("ddddd")),
    ("Threshold_dddddddd", lambda: _make_threshold("d" * 8)),
    ("Threshold_ddi4", lambda: _make_threshold("ddi" * 4)),
    ("Threshold_ddid3", lambda: _make_threshold("ddid" * 3)),
    ("C_4", lambda: Graph.from_networkx(nx.cycle_graph(4))),
    ("P_3", lambda: path_graph(3)),
]


class TestEngineCorrectness:
    """Engine polynomial must match standalone cotree DP on cographs."""

    @pytest.mark.parametrize(
        "name,builder",
        COGRAPH_VALIDATION,
        ids=[g[0] for g in COGRAPH_VALIDATION],
    )
    def test_engine_matches_cotree_dp(self, name, builder):
        """Engine result must exactly match standalone cotree DP polynomial."""
        graph = builder()
        engine = _empty_table_engine()

        engine_result = engine.synthesize(graph)
        cotree_poly = compute_tutte_cotree_dp(graph)

        assert engine_result.polynomial == cotree_poly, (
            f"{name}: engine polynomial differs from standalone cotree DP.\n"
            f"  engine T(1,1) = {int(engine_result.polynomial.evaluate(1, 1))}\n"
            f"  cotree T(1,1) = {int(cotree_poly.evaluate(1, 1))}\n"
            f"  engine method = {engine_result.method}"
        )


# =============================================================================
# B. ENGINE FALLS THROUGH CORRECTLY FOR NON-COGRAPHS
# =============================================================================

NON_COGRAPH_GRAPHS = [
    ("C_5", lambda: cycle_graph(5)),
    ("C_6", lambda: cycle_graph(6)),
    ("C_7", lambda: cycle_graph(7)),
    ("P_4", lambda: path_graph(4)),
    ("P_5", lambda: path_graph(5)),
    ("Petersen", lambda: Graph.from_networkx(nx.petersen_graph())),
    ("W_5", lambda: Graph.from_networkx(nx.wheel_graph(5))),
    ("Grid_3x3", lambda: Graph.from_networkx(nx.grid_2d_graph(3, 3))),
]


class TestNonCographFallthrough:
    """Engine must not crash on non-cographs -- should fall through to other methods."""

    @pytest.mark.parametrize(
        "name,builder",
        NON_COGRAPH_GRAPHS,
        ids=[g[0] for g in NON_COGRAPH_GRAPHS],
    )
    def test_engine_handles_non_cographs(self, name, builder):
        """Non-cographs must be synthesized by a method other than cotree_dp."""
        graph = builder()
        engine = _empty_table_engine()
        result = engine.synthesize(graph)

        assert result.polynomial is not None, f"{name}: engine returned None polynomial"
        assert result.method != "cotree_dp", (
            f"{name}: non-cograph was incorrectly handled by cotree_dp"
        )


# =============================================================================
# C. SYNTHESIS RESULT METADATA
# =============================================================================

# Graphs that family recognition does NOT handle and that have tw > 10,
# so they MUST reach cotree DP (with an empty rainbow table).
# - K_n: no closed-form in family_recognition/formulas.py
# - Threshold graphs: not in family_recognition
# - K_{a,b}: not in family_recognition
# These graphs are connected and 2-connected, so steps 5/6 do not fire.
# They are not cycles, not series-parallel, and have treewidth > 10.

# Treewidth of K_n is n-1, K_{a,b} is min(a,b), threshold "d"*k is k.
# Treewidth DP handles tw <= 10 (step 9), so we need tw > 10.
# K_12 (tw=11), K_14 (tw=13), K_15 (tw=14) all bypass treewidth DP.
# Threshold "d"*12 has 13 vertices, tw=12. K_{11,11} has tw=11.
MUST_HIT_COTREE_DP = [
    ("K_12", lambda: Graph.from_networkx(nx.complete_graph(12))),
    ("K_14", lambda: Graph.from_networkx(nx.complete_graph(14))),
    ("K_15", lambda: Graph.from_networkx(nx.complete_graph(15))),
    ("Threshold_d12", lambda: _make_threshold("d" * 12)),
    ("K_{11,11}", lambda: Graph.from_networkx(nx.complete_bipartite_graph(11, 11))),
]


class TestResultMetadata:
    """SynthesisResult must have correct method, recipe, and verified fields.

    Uses empty rainbow table and graphs outside family recognition coverage
    to guarantee cotree DP fires. Assertions are unconditional.
    """

    @pytest.mark.parametrize(
        "name,builder",
        MUST_HIT_COTREE_DP,
        ids=[g[0] for g in MUST_HIT_COTREE_DP],
    )
    def test_method_is_cotree_dp(self, name, builder):
        """Graphs outside family recognition with tw > 10 must be solved by cotree_dp."""
        graph = builder()
        engine = _empty_table_engine()
        result = engine.synthesize(graph)

        assert result.method == "cotree_dp", (
            f"{name}: expected method='cotree_dp', got '{result.method}'"
        )
        assert result.verified is True, (
            f"{name}: cotree DP result must be marked verified"
        )
        assert any("Cotree" in step or "cotree" in step for step in result.recipe), (
            f"{name}: recipe must mention cotree DP: {result.recipe}"
        )


# =============================================================================
# D. EVENT LOG CONTAINS COTREE_DP EVENTS
# =============================================================================

class TestEventLog:
    """Event log must record COTREE_DP events when cotree DP fires."""

    def test_cotree_dp_event_logged(self):
        """K_14 with empty table must produce a COTREE_DP event."""
        graph = Graph.from_networkx(nx.complete_graph(14))
        engine = _empty_table_engine()
        reset_log()

        result = engine.synthesize(graph)

        assert result.method == "cotree_dp", (
            f"Expected cotree_dp, got {result.method} -- event log test is invalid"
        )
        log = get_log()
        cotree_events = log.filter(EventType.COTREE_DP)
        assert len(cotree_events) > 0, (
            "No COTREE_DP events found in log despite method='cotree_dp'"
        )

    def test_no_cotree_dp_event_for_non_cograph(self):
        """Non-cographs must NOT produce COTREE_DP events."""
        graph = cycle_graph(5)
        engine = _empty_table_engine()
        reset_log()

        engine.synthesize(graph)

        log = get_log()
        cotree_events = log.filter(EventType.COTREE_DP)
        assert len(cotree_events) == 0, (
            f"COTREE_DP events found for non-cograph C_5: {cotree_events}"
        )


# =============================================================================
# E. CACHE AND RAINBOW TABLE PROMOTION
# =============================================================================

class TestCachePromotion:
    """Second synthesis of the same cograph must hit cache."""

    def test_second_call_hits_cache(self):
        """Synthesizing the same cograph twice: second call must hit cache."""
        graph = Graph.from_networkx(nx.complete_graph(10))
        engine = _empty_table_engine()
        reset_log()

        result1 = engine.synthesize(graph)
        result2 = engine.synthesize(graph)

        assert result1.polynomial == result2.polynomial

        # Verify cache hit via event log: the second call should not
        # produce a second COTREE_DP event.
        log = get_log()
        cotree_events = log.filter(EventType.COTREE_DP)
        assert len(cotree_events) <= 1, (
            f"Expected at most 1 COTREE_DP event, got {len(cotree_events)} -- "
            f"cache may not be working"
        )

    def test_polynomial_survives_cache_roundtrip(self):
        """Cached polynomial must be identical to the original."""
        graph = Graph.from_networkx(nx.complete_graph(12))
        engine = _empty_table_engine()

        result1 = engine.synthesize(graph)
        result2 = engine.synthesize(graph)

        assert result1.polynomial == result2.polynomial
        assert result2.polynomial == compute_tutte_cotree_dp(graph)


# =============================================================================
# F. KIRCHHOFF VALIDATION ON ENGINE RESULTS FOR COGRAPHS
# =============================================================================

KIRCHHOFF_COGRAPHS = [
    ("K_5", lambda: complete_graph(5)),
    ("K_8", lambda: complete_graph(8)),
    ("K_10", lambda: Graph.from_networkx(nx.complete_graph(10))),
    ("K_14", lambda: Graph.from_networkx(nx.complete_graph(14))),
    ("K_{3,3}", lambda: Graph.from_networkx(nx.complete_bipartite_graph(3, 3))),
    ("K_{5,5}", lambda: Graph.from_networkx(nx.complete_bipartite_graph(5, 5))),
    ("K_{7,7}", lambda: Graph.from_networkx(nx.complete_bipartite_graph(7, 7))),
    ("Threshold_d8", lambda: _make_threshold("d" * 8)),
    ("Threshold_ddi4", lambda: _make_threshold("ddi" * 4)),
]


class TestKirchhoffOnEngineResults:
    """T(1,1) from engine must equal exact Kirchhoff spanning tree count."""

    @pytest.mark.parametrize(
        "name,builder",
        KIRCHHOFF_COGRAPHS,
        ids=[g[0] for g in KIRCHHOFF_COGRAPHS],
    )
    def test_engine_kirchhoff(self, name, builder):
        """Engine T(1,1) must equal exact Kirchhoff count (integer arithmetic)."""
        graph = builder()
        engine = _empty_table_engine()
        result = engine.synthesize(graph)

        t11 = _exact_num_spanning_trees(result.polynomial)

        components = graph.connected_components()
        if len(components) == 1:
            kirchhoff = _exact_spanning_tree_count(graph)
        else:
            kirchhoff = 1
            for component in components:
                kirchhoff *= _exact_spanning_tree_count(component)

        assert t11 == kirchhoff, (
            f"{name}: T(1,1)={t11} != Kirchhoff={kirchhoff}, method={result.method}"
        )


# =============================================================================
# G. NO REGRESSIONS ON NON-COGRAPH GRAPHS
# =============================================================================

REGRESSION_GRAPHS = [
    ("C_5", lambda: cycle_graph(5)),
    ("C_8", lambda: cycle_graph(8)),
    ("P_4", lambda: path_graph(4)),
    ("P_8", lambda: path_graph(8)),
    ("Petersen", lambda: Graph.from_networkx(nx.petersen_graph())),
    ("W_5", lambda: Graph.from_networkx(nx.wheel_graph(5))),
    ("K_3", lambda: complete_graph(3)),
    ("K_4", lambda: complete_graph(4)),
    ("Single_edge", lambda: Graph(nodes=frozenset({0, 1}), edges=frozenset({(0, 1)}))),
    ("Empty", lambda: Graph(nodes=frozenset({0, 1}), edges=frozenset())),
]


class TestNoRegressions:
    """Engine must produce correct polynomials for non-cograph graphs.

    These graphs were handled correctly before integration. This section
    verifies that inserting cotree DP does not break existing behavior.
    """

    @pytest.mark.parametrize(
        "name,builder",
        REGRESSION_GRAPHS,
        ids=[g[0] for g in REGRESSION_GRAPHS],
    )
    def test_kirchhoff_regression(self, name, builder):
        """T(1,1) must equal Kirchhoff for all regression graphs."""
        graph = builder()
        engine = _empty_table_engine()
        result = engine.synthesize(graph)

        t11 = _exact_num_spanning_trees(result.polynomial)

        components = graph.connected_components()
        if len(components) == 1:
            kirchhoff = _exact_spanning_tree_count(graph)
        else:
            kirchhoff = 1
            for component in components:
                kirchhoff *= _exact_spanning_tree_count(component)

        assert t11 == kirchhoff, (
            f"{name}: T(1,1)={t11} != Kirchhoff={kirchhoff}, method={result.method}"
        )

    def test_triangle_exact(self):
        """K_3: T = x^2 + x + y -- must match exactly."""
        expected = TuttePolynomial.from_coefficients({(2, 0): 1, (1, 0): 1, (0, 1): 1})
        engine = _empty_table_engine()
        result = engine.synthesize(complete_graph(3))
        assert result.polynomial == expected

    def test_single_edge_exact(self):
        """Single edge: T = x."""
        graph = Graph(nodes=frozenset({0, 1}), edges=frozenset({(0, 1)}))
        engine = _empty_table_engine()
        result = engine.synthesize(graph)
        assert result.polynomial == TuttePolynomial.x()

    def test_empty_graph_exact(self):
        """Empty graph: T = 1."""
        graph = Graph(nodes=frozenset({0, 1}), edges=frozenset())
        engine = _empty_table_engine()
        result = engine.synthesize(graph)
        assert result.polynomial == TuttePolynomial.one()